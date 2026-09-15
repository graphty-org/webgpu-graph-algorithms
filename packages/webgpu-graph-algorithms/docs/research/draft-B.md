# @graphty/webgpu-graph-algorithms -- implementation plan (draft B, integration-first)

Status: plan draft for the owner, 2026-09-14. Planning only; no implementation
code exists beyond the short illustrative snippets in this document.

Angle of this draft: INTEGRATION-FIRST. Every section of the required outline
is complete, but sections 2 (runtime model), 3 (public API), 9 (how the
optional / detected acceleration plugs into `@graphty/algorithms`,
`@graphty/layout` and `@graphty/graphty-element`) and 12 (CI/CD lanes) go
deepest, because those are the places where a wrong decision is the most
expensive to reverse once the package is inside the monorepo.

Inputs (all read in full): research notes `tmp/webgpu-plan/01-layout-needs.md`
through `07-format-api-and-conventions.md` (cited below as "note NN section
X"), the accepted design
`/home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md`
sections 10, 14.3-14.6, 15, 16 (cited as "design S.S, line N"), the staged
`packages/graph-format` implementation under
`/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/`, and
the owner's request quoted in the brief.

Vocabulary: n = node count, E = logical edge count, A = arc count
(`snapshot.arcCount`; `2E - selfLoops` for undirected snapshots), dim = 2 or 3,
WG = workgroup size (256 by default), "ctx" = a `GpuContext`.

Every fact about existing code cites a path; every external fact cites a
research note (which carries the URL) or a URL directly. Estimates are
labelled with their basis; nothing is presented as measured unless a probe
in `tmp/webgpu-plan/probe/` or a cited file measured it.

---------------------------------------------------------------------------

## 1. Goals, non-goals and inherited decisions

### 1.1 Goals

| Id | Goal | Source |
| --- | --- | --- |
| G1 | GPU-accelerated force-directed layout is the first deliverable after a walking skeleton: a `LayoutSimulation` for ForceAtlas2 (and Fruchterman-Reingold on the same kernels) that graphty-element can step per frame with pins and drag, and that Node can run in batch. | owner request; design 14.3 lines 3977-3985 |
| G2 | One code base runs under Node (Dawn via the `webgpu` npm package) and in browsers; the package never touches `navigator` and never imports the native module from its root entry. | owner request; note 05 section 1; note 02 finding 5 |
| G3 | Consume `@graphty/graph-format` snapshots exactly as implemented (arena hot prefix, per-array, windowed uploads; `override USE_PERM`; `gpuView()`; `foldArcs`; `renumberPartition`; `INVALID_INDEX`). | design 10; note 07 sections 1-3 |
| G4 | High performance at 10^5-10^6 nodes is a design driver: data layouts, kernels and the host loop are chosen for that scale (batched iterations per submit, no per-iteration readback, approximate repulsion, degree-tier load balancing). | owner request; design 15.3 |
| G5 | Optional / detected acceleration for the EXISTING algorithm and layout packages, with the GPU package injected by the caller and detection done by the app; the GPU package itself never falls back. | owner request; design line 2324-2325; `/home/apowers/Projects/webgpu-graph-algorithms/CLAUDE.md` |
| G6 | Tests primarily in Node (Dawn), a light browser suite (Playwright Chromium) to prove the browser path, both on the real GPU locally. | owner request; note 05 section 9 |
| G7 | CI with a default lane (no GPU: lavapipe + SwiftShader) and a GPU lane (self-hosted RTX 4070 SUPER), designed to slot into the monorepo's `ci.yml` shard matrix later as "one runner for GPU and a default runner for other tests". | owner request; note 06 |
| G8 | A GPU algorithm library grouped by primitive family (SpMV, frontier, edge-parallel, sort-based) with the NVIDIA / cuGraph / Gunrock / GAP techniques translated to WGSL's constraints (no float atomics, no cross-workgroup locks, subgroup size not constant). | owner request; note 04 |

### 1.2 Non-goals (v1)

- No CPU, WebGL or "software mode" inside this package. A missing adapter is
  an error (`E_NO_ADAPTER`); the CPU packages are the default path, not a
  fallback (project rule; design 10.8 lines 2545-2550).
- No rendering. The package computes positions and results; graphty-element
  renders them (Babylon on WebGL today, note 01 section 4.8).
- No `SharedArrayBuffer`, no worker orchestration inside the package (the
  format has no SAB in v1; a consumer may run a `GpuContext` in a worker and
  transfer snapshots with `toWire`).
- No `shader-f16` requirement, no compat-mode support work beyond the
  defaults table (note 05 section 4); optional features are fast paths only.
- Not GPU targets (kept on the CPU path by the adapters): DFS, topological
  sort, cycle detection, Prim, Girvan-Newman, hierarchical / TeraHAC / GRSBM /
  SynC, MCL, max-flow / min-cut, bipartite matching, isomorphism, A* (note 02
  section 7.2 last row).
- No dense-matrix betweenness (the Buffalo 2023-06 thesis is a negative
  result for sparse graphs, note 04 section 0 item 8) and nothing from the
  GPL `jaredmcqueen/analytics` repository (note 03 section 3).

### 1.3 Decisions inherited from the graph-format design (not relitigated)

| Design section | Inherited decision | How this plan honours it |
| --- | --- | --- |
| 10.1 (lines 2327-2350) | The arrays a GPU package binds: `rowPtr`, `colIdx`, `weights`, `arcToEdge`, `edgeToArc`, `reverse()`, `coo().src`, `edgeList()`, degrees, `degreeOrder(opts).perm / segmentOffsets`, `mate()`, `gpuView()` columns; the weighted-degree normaliser is computed on the device; `override USE_PERM` with `colIdx` / `rowPtr` as dummy bindings for identity permutations. | Section 4.2 (residency keys), section 6 (segmented reduce), section 3.5 (WGSL prelude declares `USE_PERM`). |
| 10.2 (2352-2373) | 4-byte arrays over plain `ArrayBuffer`; `u8` via `paddedU32View()` + `unpack4xU8`; `bool` / masks as u32 bitmaps LSB-first; `components: 3` never `array<vec3<f32>>`; `f64` via cached f32 `gpuView()`. | Section 4.3 (column uploads), section 7.3 (positions as `array<f32>` with `3*i`). |
| 10.3 (2375-2441) | Upload plan: whole arena hot prefix when `arena.byteLength <= maxBufferSize` AND every segment `<= maxStorageBufferBindingSize`; else per array (also the `arena === null` path); else windows. | Section 4.2 upload planner, with the decision table and faked-limit tests. |
| 10.4 (2443-2465) | `gpuEligibility`; results attach by reference through `nodes.set()`. | Section 9.7 (adapters attach GPU readbacks by reference). |
| 10.5 (2467-2491) | Invariants the GPU may assume; never bind a zero-length array; guarded division by weight sums. | Section 5.6 (empty-range dispatch rule), section 8.2 (PageRank normaliser). |
| 10.6 (2493-2520) | 1D dispatch legal iff `ceil(count / WG) <= maxComputeWorkgroupsPerDimension` (65,535 x 256 = 16,776,960, NOT 2^24); 2D grid or grid-stride above; windows start at 64-arc boundaries computed with `%`. | Section 5.2 DispatchPlanner, section 4.2 window planner. |
| 10.7 (2522-2543) | Readback conventions: index-aligned `Uint32Array` / `Float32Array`, `foldArcs` for per-arc results, copy out before `unmap()`, `dest?` on every algorithm. | Section 3.3 signatures (`dest?`), section 4.4 Readback. |
| 10.8 (2545-2550) | Device queries, chunking, 2D dispatch, frontier queues, scans, dense relabelling and "GPU unavailable" behaviour belong to the GPU package; it never falls back. | Sections 2.4, 5, 6. |
| 14.3 (3959-4019) | `LayoutSimulation { load, step, settled, setFixed, setPosition, dispose }`; positions are the owner's stride-3 scene-unit array read AND written in place; the GPU buffer is authoritative while stepping; FA2 default mass `outDegree() + 1`; weights via `snapshot.weights` (`weight === true`) or a named edge column; `LayoutSimulation` and the position helpers are OWNED by `@graphty/layout`. | Section 7.18, section 9.3. |
| 14.4 (4048-4211) | `DataManager` owns the builder and the position array; `snapshot-replaced { previous, next, report }` triggers `accelerator.release(previous)`; adapters call `getSnapshot()` once and run `indexed.*` "(or the injected GPU accelerator)"; layouts get `dm.undirected(s).snapshot`; drag writes `simulation.setPosition`; `column.markDirty()` once per frame. | Section 9.4. |
| 14.5 (4212-4243) | Move-in as `webgpu-graph-algorithms/`; delete `CSRGraph` / `EdgeListGraph`; every entry takes `GraphSnapshot`; `noUncheckedIndexedAccess` OFF; upload cache is a `WeakMap` keyed on the typed-array object, dropped via `column.version`, released by `release(snapshot)`; `parents` are `Uint32Array` with `INVALID_INDEX`; injected as `runAlgorithm(snapshot, { accelerator: gpu })`. | Sections 3, 4, 9. ONE explicit departure: 14.5 says "a browser-only vitest project"; the owner's request makes Node the primary project (section 1.4, D1). |
| 14.6 (4245-4275) | Landing order F1 -> A1 -> F2 -> A2 / L1 / E1 -> W1 -> D1 -> IO1 -> 2.0; ownership of shared helpers (`foldArcs`, `renumberPartition` -> graph-format; `LayoutSimulation` -> layout; upload planning, chunking, readback, `release` -> webgpu-graph-algorithms). | Section 9.8 maps every integration change to a landing step. |
| 13.5 rule 3 (via note 02 section 3) | Consumers declare `@graphty/graph-format` in BOTH `dependencies` (`workspace:*`) and `peerDependencies`; `isGraphSnapshot()` is a brand check. | Section 3.1 package.json. |
| 15.3 (4349-4357) | Target tiers: 100k / 1M (mobile), 1M / 10M (desktop interactive), 10M / 100M (batch, raised limits and windowed bindings). | Section 10 tables use exactly these tiers. |
| 16.2 (4535-4551) | Differential tests: `1e-5` relative for f32 GPU parity, order-agnostic for component lists; `validate({ checksum: true })` after each call so a write into a shared view fails. | Section 11.3. |

### 1.4 Decisions this plan makes that the design left open

| Id | Decision | Where |
| --- | --- | --- |
| D1 | Node (Dawn) is the PRIMARY test project; the browser project is a light smoke suite. Departs from design 14.5 "browser-only vitest project" on the owner's explicit instruction; 14.5 and 16.7 are amended when this plan is accepted. | 11, 12 |
| D2 | Runtime model: the core takes a `GPU` / `GPUAdapter` / `GPUDevice` from the caller; `./browser` and `./node` subpath entries do acquisition; `webgpu` (Dawn) is an optional peer loaded only by dynamic import inside `./node`. | 2 |
| D3 | Plug-in contract: `@graphty/algorithms` and `@graphty/layout` OWN structural `AlgorithmAccelerator` / `LayoutAccelerator` interfaces plus one async dispatcher each; the GPU package implements them with no runtime import of the CPU packages; no registry, no global state. | 9 |
| D4 | Detection lives in the graphty APP (probe + import + create + `element.accelerator = gpu`); graphty-element only exposes the property and the bridges; an element-level `"auto"` loader is a later convenience using the web-llm isolation pattern. | 9.4-9.5 |
| D5 | ForceAtlas2 reference formulas for BOTH the CPU rewrite and the GPU kernel are the published algorithm as Gephi and cuGraph implement it: `1/d` repulsion magnitude (`k m_i m_j / d` along the unit vector), per-node swing / traction from `F(t)` and `F(t-1)`, global sums fresh every iteration (not NetworkX's accumulation), symmetric `adjustSizes` (deferred); the port's `estimateFactor` and local-speed apply rule are unchanged. | 7.2 |
| D6 | Async step bridge: fire-and-forget, at most `maxInFlight` (default 2) batches in flight, positions copied into the owner's array when each batch's readback resolves; `settled` reflects the last completed batch. | 7.18 |
| D7 | Two repulsion back-ends selected by node count: exact tiled all-pairs (`n <= exactMaxNodes`, default 16,384, to be re-fixed by measurement) and a cell-sorted grid pyramid (cosmos P3M re-expressed as compute kernels) above; a Hilbert-sorted cluster tree (GraphWaGu style) is the documented second experiment, not v1. | 7.6-7.7 |
| D8 | Settlement: `settled` becomes true at `maxIter` iterations OR when the mean per-node displacement over a window is below a threshold relative to the layout radius; `setPosition`, unpin and `load` reheat. | 7.16 |
| D9 | WGSL lives in `src/wgsl/*.wgsl.ts` template-string modules composed by string concatenation (tsc-only dist, knip, eslint all work); no `?raw`. | 3.5 |
| D10 | CI: default lane on `ubuntu-latest` (Dawn on Mesa lavapipe + Chromium on SwiftShader) on every PR; GPU lane on an ephemeral self-hosted runner on the dev box, never a required check; GitHub-hosted GPU runners rejected (org on the free plan). | 12 |
| D11 | `webgpu` pinned to `0.4.0` until the dev container / runner image moves to Ubuntu 24.04 (0.5+ needs glibc 2.38). | 2.5, 12 |
| D12 | The package throws its own `WebGpuGraphError { code, details }` (never `GraphFormatError`), with a stable code list. | 3.3 |
| D13 | GPU scores are `Float32Array`; the accelerator interfaces type scores as `NumericVector` (`F32 | F64`) so CPU and GPU results satisfy one interface; labels / parents are `Uint32Array` with `INVALID_INDEX`; labels are renumbered on the CPU in first-seen order with `renumberPartition`. | 9.7 |
| D14 | Layouts simulate in LAYOUT units on the device and apply `scalingFactor` / `center` in the write-back; never `rescaleLayout` per step. | 7.17 |
| D15 | FA2's swing / traction reductions and the adaptive-speed controller run on the device (reduce kernel + one-workgroup finalize kernel), so k iterations are one submission with no host round trip. | 7.9 |
| D16 | Every kernel reads `subgroup_size` or takes it as an `override`; subgroup fast paths are separate module variants selected by `device.features`; nothing assumes width 32. | 5.1, 6 |
| D17 | Iterative algorithms run k rounds per `queue.submit` with `dispatchWorkgroupsIndirect` driven by device-side counters; host readback of convergence flags only every k rounds. | 5.4, 8 |
| D18 | Package skeleton mirrors `packages/graph-io` (both deps and peers on graph-format, `@webgpu/types` as a dependency with a triple-slash reference, `./node` subpath, coverage port 9058). | 3.1 |

---------------------------------------------------------------------------

## 2. Runtime model

### 2.1 One code base, three entry points

The library is split into a runtime-agnostic CORE and two thin ACQUISITION
entries. The core contains every kernel, planner and algorithm; it never
references `navigator`, `window`, `process` or the `webgpu` module. This is
the single property that makes Node and browser share one code base (note
02 finding 5; note 05 section 1 item 1).

```
@graphty/webgpu-graph-algorithms           (".")   core: GpuContext.from/create, GraphResidency, algorithms, layouts, errors, types
@graphty/webgpu-graph-algorithms/browser   uses globalThis.navigator.gpu; requestGpuContext(), probeBrowserWebGpu()
@graphty/webgpu-graph-algorithms/node      dynamic import("webgpu") (Dawn); createNodeGpu(), createNodeGpuContext(), probeNodeWebGpu()
```

Two rules keep the core import-order independent under Dawn-node, where
`GPUBufferUsage`, `GPUMapMode` and `GPUShaderStage` do not exist on
`globalThis` until `Object.assign(globalThis, dawn.globals)` has run (note 05
section 2.2, verified against the installed `webgpu@0.4.0`
`packages/graph-format/node_modules/webgpu/index.js`):

1. The core never reads those namespaces at module top level. It keeps its
   own numeric constants in `src/device/webgpu-constants.ts`
   (`BufferUsage.STORAGE = 0x0080`, `COPY_SRC = 0x0004`, `COPY_DST = 0x0008`,
   `MAP_READ = 0x0001`, `UNIFORM = 0x0040`, `INDIRECT = 0x0100`,
   `MapMode.READ = 0x0001`, `ShaderStage.COMPUTE = 0x4`), each with a comment
   naming the spec constant. A test compares them with the browser /
   Dawn globals at runtime so a drift is caught.
2. The `./node` entry installs `dawn.globals` anyway (user code and
   `@webgpu/types`-typed helpers expect them), but the core does not depend
   on it having happened.

### 2.2 Device acquisition (core)

```ts
// src/device/gpu-context.ts (public)
export type LimitPolicy = "default" | "raise" | Readonly<Partial<Record<RaisableLimit, number>>>;
export type RaisableLimit =
    | "maxBufferSize" | "maxStorageBufferBindingSize" | "maxStorageBuffersPerShaderStage"
    | "maxComputeWorkgroupStorageSize" | "maxComputeInvocationsPerWorkgroup" | "maxComputeWorkgroupSizeX"
    | "maxComputeWorkgroupsPerDimension";

export interface GpuContextOptions {
    readonly gpu?: GPU | undefined;               // browser: navigator.gpu; Node: dawn.create([...])
    readonly adapter?: GPUAdapter | undefined;    // skip requestAdapter()
    readonly device?: GPUDevice | undefined;      // adopt a device the caller owns (e.g. a future Babylon WebGPUEngine device)
    readonly powerPreference?: GPUPowerPreference | undefined;   // default "high-performance" (Chrome 145 needs it, note 05 section 3.2)
    readonly limits?: LimitPolicy | undefined;    // default "raise"
    readonly optionalFeatures?: readonly GPUFeatureName[] | undefined;   // default ["subgroups", "timestamp-query"]; requested only when the adapter has them
    readonly requiredFeatures?: readonly GPUFeatureName[] | undefined;   // default []; missing -> E_NO_DEVICE
    readonly label?: string | undefined;
    readonly onError?: ((error: WebGpuGraphError) => void) | undefined;  // uncapturederror sink; default: keep the last error and throw it from the next public call
}
```

`GpuContext.create(options)` runs, in order:

1. `options.device` given -> adopt (`ownsDevice = false`); else
   `options.adapter` given -> step 3; else `options.gpu.requestAdapter({
   powerPreference })`; `null` -> throw `E_NO_ADAPTER`.
2. Read `adapter.info` (vendor, architecture, device, description,
   `subgroupMinSize`, `subgroupMaxSize`); compute `software =
   isSoftwareAdapter(info)` = `architecture === "software"` (Dawn-node
   llvmpipe) OR `architecture === "swiftshader"` OR `isFallbackAdapter ===
   true` (Chromium; `undefined` in Dawn-node 0.4.0, note 05 section 2.2).
3. Build `requiredLimits` from the policy: `"raise"` takes each
   `RaisableLimit` from `adapter.limits` (clamped to what the adapter
   reports, never above), `"default"` requests nothing, an object requests
   exactly those values (rejecting a value above `adapter.limits` with
   `E_NO_DEVICE` before calling `requestDevice`, so the failure is
   diagnosable). Build `requiredFeatures = required + (optional intersect
   adapter.features)`. `requestDevice({ requiredLimits, requiredFeatures,
   label })`; a rejection -> `E_NO_DEVICE` with the adapter summary in
   `details`.
4. Capture `GpuCaps` from `device.limits` (never `adapter.limits`: Dawn-node
   reports 1 TiB / alignment 16 on the adapter while a default device is
   the spec default, note 05 section 2.4), `device.features`, the adapter
   info, `runtime: "browser" | "node"` (decided by the ENTRY that created
   the context, not by sniffing globals; `GpuContext.from(device)` sets
   `"unknown"`).
5. Install `device.addEventListener("uncapturederror", ...)` (works
   identically in Dawn-node and browsers, note 05 section 7.3) and chain
   `device.lost` into `ctx.lost`; on loss the context enters `state =
   "lost"`, every pending promise rejects with `E_DEVICE_LOST`, and every
   residency record is dropped (the buffers are gone with the device).
6. Create the singletons: `PipelineCache`, `BufferPool`, `Readback`
   staging ring, `GraphResidency`.

`GpuContext.from(device, info?)` is the zero-cost adoption path for callers
that already have a device (tests, a future shared Babylon device). `probe`
never creates a device:

```ts
export interface ProbeResult {
    readonly ok: boolean;
    readonly code: "OK" | "E_NO_WEBGPU" | "E_NO_ADAPTER" | "E_SOFTWARE_ONLY";
    readonly reason: string | null;
    readonly adapter: AdapterSummary | null;     // vendor, architecture, device, description, software, subgroupMinSize/MaxSize, features[], limits{}
}
export interface ProbeOptions { readonly gpu: GPU; readonly powerPreference?: GPUPowerPreference; readonly rejectSoftware?: boolean; }  // rejectSoftware default false
GpuContext.probe(options: ProbeOptions): Promise<ProbeResult>
```

`rejectSoftware: true` turns a lavapipe / SwiftShader adapter into
`E_SOFTWARE_ONLY` so an APP can decide "no acceleration on software
adapters" without the package deciding for it (the package itself runs on
software adapters -- that is how the default CI lane works, section 12).

### 2.3 The detection helper, per runtime

| Runtime | Entry | Helper | What it does |
| --- | --- | --- | --- |
| Browser | `./browser` | `probeBrowserWebGpu(opts?)` | `navigator.gpu` absent -> `{ ok: false, code: "E_NO_WEBGPU" }`; else `GpuContext.probe({ gpu: navigator.gpu, ...opts })`. Never throws. |
| Browser | `./browser` | `requestGpuContext(opts?)` | `GpuContext.create({ gpu: navigator.gpu, powerPreference: "high-performance", ...opts })`; throws `E_NO_WEBGPU` / `E_NO_ADAPTER` / `E_NO_DEVICE`. |
| Node | `./node` | `createNodeGpu(opts?)` | `await import("webgpu")` (throws `E_NO_WEBGPU` with the loader message when the native module is missing or its glibc is too old), `Object.assign(globalThis, dawn.globals)` unless `installGlobals: false`, `dawn.create(flags)` where flags come from `{ adapter?: string; backend?: string; dawnFeatures?: string[]; software?: boolean }` (`software: true` -> `adapter=llvmpipe`, documented as Linux / Mesa specific; note 05 section 2.5). Returns `{ gpu, dispose() }`; `dispose()` drops the reference so the process can exit (note 05 section 2.2, lifetime). |
| Node | `./node` | `probeNodeWebGpu(opts?)` | `createNodeGpu` + `GpuContext.probe` + `dispose`; never throws. |
| Node | `./node` | `createNodeGpuContext(opts?)` | `createNodeGpu` + `GpuContext.create`; `ctx.dispose()` also disposes the `GPU` handle. |
| Any | `.` | `GpuContext.probe({ gpu })`, `GpuContext.create({ gpu | adapter | device })`, `isSoftwareAdapter(info)` | the runtime-agnostic primitives the two entries are built on. |

Environment variables are read ONLY by the test setup (`GRAPHTY_GPU_ADAPTER`,
`GRAPHTY_GPU_REQUIRE`, `GRAPHTY_BROWSER_GPU`, `GRAPHTY_EGL_LIB_DIR`, section
11.2), never by `src/`.

### 2.4 The no-fallback rule inside the package versus optional / detected acceleration in consumers

The rule is enforced by WHERE decisions are made, not by discipline:

| Question | Who answers | How |
| --- | --- | --- |
| Is WebGPU present, is the adapter hardware, is it worth using? | The APP (graphty) at start-up, or a Node script | `probe*()` then `create*()`; the app sets `element.accelerator = gpu` or leaves it `null`. A node-count threshold for "use the GPU layout only above N nodes" is an app / element product setting, not package logic (note 02 section 7.1, L3). |
| No accelerator injected | The CPU packages' dispatcher (`indexed.accelerated(null)`) | runs the CPU implementation. This is the ONLY branch that chooses CPU, evaluated BEFORE any GPU work. |
| Accelerator injected, method missing (not yet implemented on the GPU) | The dispatcher | `acc.pageRank === undefined` -> CPU. Also evaluated before any GPU work; the GPU package declares only the methods it implements. |
| Accelerator injected, method throws (`E_DEVICE_LOST`, `E_OUT_OF_MEMORY`, `E_VALIDATION`, `E_UNSUPPORTED`) | The caller (graphty-element's operation queue reports a failed run; a Node script sees the rejection) | the error propagates. The element MAY offer "disable GPU acceleration" as a user action; that is a user decision, not a fallback. |
| Graph too large for the device (needs windowing the algorithm does not support yet) | The GPU package | throws `E_TOO_LARGE` with `details { needed, limit, path }` BEFORE allocating; never silently degrades. |
| Software adapter (lavapipe / SwiftShader) | The GPU package runs on it (correctness); the APP decides whether to inject it (`rejectSoftware`) | see 2.2. |

Consequences written into the code: `src/` contains no `try { gpu } catch {
cpu }`, no import of `@graphty/algorithms` or `@graphty/layout`, and no
reference to `navigator`. A lint rule (`no-restricted-imports` /
`no-restricted-globals` in the package eslint block) enforces the last two.

### 2.5 Subpath exports and keeping the native module out of browser bundles

```jsonc
// package.json (excerpt; full file in section 3.1)
"exports": {
    ".":         { "types": "./dist/webgpu-graph-algorithms.d.ts", "import": "./dist/webgpu-graph-algorithms.js", "default": "./dist/webgpu-graph-algorithms.js" },
    "./browser": { "types": "./dist/browser.d.ts",                 "import": "./dist/browser.js",                 "default": "./dist/browser.js" },
    "./node":    { "types": "./dist/node.d.ts",                    "import": "./dist/node.js",                    "default": "./dist/node.js" }
},
"sideEffects": false,
"dependencies":     { "@graphty/graph-format": "workspace:*", "@webgpu/types": "^0.1.72" },
"peerDependencies": { "@graphty/graph-format": "^0.1.0", "webgpu": "^0.4.0" },
"peerDependenciesMeta": { "webgpu": { "optional": true } }
```

Mechanisms, each with the test that guards it:

1. `webgpu` appears in exactly one source file, `src/node/index.ts`, and
   only inside `await import("webgpu")` within a function body. `test/build-
   output.test.ts` reads `dist/webgpu-graph-algorithms.js` and
   `dist/browser.js` and asserts the string `"webgpu"` (the module
   specifier) does not occur; it asserts `dist/node.js` contains it only as
   a dynamic import.
2. The vite library build (`scripts/build-bundle.js`, copied from
   `packages/graph-io/scripts/build-bundle.js` lines 33-47 which externalise
   every `dependencies` + `peerDependencies` name) marks `webgpu` and
   `@graphty/graph-format` external, so a consumer's bundler resolves them
   -- and only resolves `webgpu` if the consumer imported `./node`.
3. `"sideEffects": false` lets bundlers drop unused algorithms; nothing in
   the package registers itself anywhere (D3 rejects the registry pattern,
   note 02 section 4.2).
4. The optional peer means `npm` / `pnpm` do not install `webgpu` for
   browser consumers ("Npm will not automatically install optional peer
   dependencies", note 02 sources). A Node consumer that forgets it gets
   `E_NO_WEBGPU` with the message "install the optional peer dependency
   webgpu@0.4.0".
5. graphty-element learned that Safari fails on dynamic imports of
   non-existent modules even before the import is called
   (`graphty-element/src/ai/providers/index.ts:9-13`, note 02 finding 3).
   That concerns a browser importing a MISSING module; `./node` is never
   reachable from browser code because no browser-side file imports it.
   The rule is written into the package `CLAUDE.md`: "`./node` is imported
   only by Node entry points and tests".
6. No `"browser"` field and no conditional `"node"` export condition: an
   explicit subpath is unambiguous, while a condition can be flipped by
   bundler configuration and would let a browser bundle pull `node.js`.

### 2.6 Runtime differences the device layer absorbs

| Difference (note 05) | Dawn-node 0.4.0 | Browsers | What the device layer does |
| --- | --- | --- | --- |
| `adapter.isFallbackAdapter` | `undefined` | boolean | `isSoftwareAdapter(info)` tests `architecture` first, `isFallbackAdapter` second. |
| `forceFallbackAdapter`, `powerPreference` | ignored | honoured | Node adapter choice is `adapter=` / `backend=` strings only. |
| `minStorageBufferOffsetAlignment` | 16 | 256 | always align to 256 (graph-format arena is 256-aligned); never assert `=== 256`. |
| `maxBufferSize` on the adapter | 1 TiB (driver value) | 4 GiB | plan against `device.limits`; wrap large `createBuffer` in `pushErrorScope("out-of-memory")`. |
| `wgslLanguageFeatures` | 9 incl. `uniform_buffer_standard_layout` | 4 in Chromium 139 | uniform structs written with strict 16-byte layout; no reliance on relaxed layout. |
| `shader-f16` on NVIDIA | absent | absent (present on lavapipe) | never required; not used in v1. |
| `subgroups` size | 32 (NVIDIA), 8 (lavapipe) | 32 (NVIDIA), 4 (SwiftShader) | kernels take `SUBGROUP_SIZE` as an `override` from `adapter.info.subgroupMinSize` and are compiled per size (D16). |
| `timestamp-query` resolution | unquantised (assumed; unverified) | 100 us quantised (Chrome 121+) | profiler reports "quantised" in its output; per-kernel profiling is a Node activity. |
| `getMappedRange()` after `device.destroy()` | stays attached in 0.4.0 (0.6.1 adds a shim) | detached | `Readback` always `unmap()`s / `destroy()`s its staging buffers itself. |
| `mapAsync` round trip | 0.03-0.04 ms | 0.10-0.15 ms | both are far below one frame; batching decisions are made for the browser number. |
| Uncaptured validation errors | printed to stderr, no throw | console, no throw | `uncapturederror` listener + error scopes in both. |
| `adapter.requestAdapterInfo()` | absent | removed (Chrome 131) | `adapter.info` only. |
| Process lifetime | process stays alive while a `GPU` reference is reachable | n/a | `createNodeGpu().dispose()` drops the reference; `ctx.dispose()` calls it. |

### 2.7 Threads and workers

`navigator.gpu` exists in dedicated workers, so a `GpuContext` may be
created in a worker and fed snapshots through `toWire({ transfer: true })`
(design 16.3). Nothing in the package assumes the main thread. v1 does not
ship a worker wrapper; graphty-element runs the layout on the main thread
because its frame loop is synchronous (note 01 section 4.1) and the GPU
work is asynchronous anyway.

### 2.8 Lifetime

`ctx.dispose()`: rejects pending work with `E_DISPOSED`, destroys every
buffer in the residency, the pool and the staging ring, destroys the device
if `ownsDevice`, and (Node) disposes the `GPU` handle if the context created
it. `ctx.release(snapshot)` destroys only that snapshot's buffers (section
4.5). Nothing is freed by garbage collection (design 14.4 line 4123: "which
no `WeakMap` can do for it").

---------------------------------------------------------------------------

## 3. Package architecture

### 3.1 Skeleton and manifest

The package lives at `packages/webgpu-graph-algorithms/` in this staging repo
and moves to `webgpu-graph-algorithms/` in the monorepo at W1 (design 14.5).
It mirrors `packages/graph-io` file for file (note 07 section 4; the
`packages/README.md` checklist lines 44-198 then applies verbatim).

```
packages/webgpu-graph-algorithms/
+-- package.json  project.json  webgpu-graph-algorithms.ts  tsconfig.json  tsconfig.build.json  tsconfig.strict-consumer.json
+-- vitest.config.ts              # projects: node (primary), node-limits, bench, browser (smoke)   -- section 11
+-- scripts/entries.js  build-bundle.js  bundle-types.js  gpu-report.mjs
+-- README.md  CLAUDE.md  LICENSE  docs/HEADLESS_GPU_REPORT.md (moved from the repo root)
+-- src/
|   +-- index.ts                  # the only public barrel; explicit named exports; /// <reference types="@webgpu/types" />
|   +-- browser/index.ts          # ./browser entry (section 2.3)
|   +-- node/index.ts             # ./node entry; the only file that mentions "webgpu"
|   +-- errors.ts                 # WebGpuGraphError
|   +-- constants.ts              # WORKGROUP_SIZE = 256, MAX_WORKGROUPS_PER_DIM = 65535, ARC_WINDOW_ALIGN = 64, STORAGE_ALIGN = 256
|   +-- types/                    # public option / result / accelerator types (types only)
|   +-- device/                   # gpu-context.ts, caps.ts, webgpu-constants.ts, errors-scope.ts, profiler.ts
|   +-- memory/                   # residency.ts, upload-plan.ts, buffer-pool.ts, readback.ts, lease.ts
|   +-- kernel/                   # pipeline-cache.ts, kernel.ts, dispatch.ts, uniforms.ts, wgsl.ts (composer), batch.ts (command recording)
|   +-- wgsl/                     # *.wgsl.ts template-string modules (prelude.wgsl.ts, reduce.wgsl.ts, scan.wgsl.ts, ..., fa2-*.wgsl.ts)
|   +-- primitives/               # reduce.ts scan.ts segmented-reduce.ts compact.ts histogram.ts radix-sort.ts frontier.ts advance.ts spmv.ts coo-to-csr.ts bbox.ts
|   +-- algorithms/               # spmv/ (pagerank, hits, eigenvector, katz), traversal/ (bfs, sssp, closeness, betweenness), components/ (wcc), structure/ (kcore, triangles), community/ (lpa, louvain)
|   +-- layouts/                  # force-simulation.ts (shared core), forceatlas2.ts, fruchterman-reingold.ts, repulsion-exact.ts, repulsion-grid.ts
|   +-- accelerator.ts            # GpuAccelerator: the object implementing AlgorithmAccelerator & LayoutAccelerator (section 9)
+-- test/
|   +-- setup/gpu.ts              # acquire() per project; requireGpu(); env policy (section 11.2)
|   +-- helpers/device.ts graphs.ts oracle.ts caps.ts matchers.ts
|   +-- device/ memory/ kernel/ primitives/ algorithms/ layouts/   # node project
|   +-- limits/                   # node-limits project (real large limits only)
|   +-- browser/                  # browser smoke project (*.test.ts)
|   +-- types/*.test-d.ts  index.test.ts  build-output.test.ts
+-- benchmarks/                   # harness.ts datasets.ts run.ts <group>.bench.ts results/
```

`package.json` is the note 07 section 4.2 manifest with one addition (the
`./browser` export) and the name `webgpu-graph-algorithms` in `repository.
directory`. `@graphty/graph-format` is in both `dependencies` (`workspace:*`)
and `peerDependencies` (`^0.1.0`, bumped to `^1.0.0` at F2); `@webgpu/types
^0.1.72` is a `dependency` because exported signatures name `GPUDevice`;
`webgpu ^0.4.0` is an optional peer and a devDependency; no `browserslist`;
not `private`; `coverage:preview` on port 9058.

`tsconfig.json`: `noUncheckedIndexedAccess: false` (design 14.5 line 4218),
`lib: ["ES2020", "DOM", "DOM.Iterable"]` (the browser entry needs `navigator`
typings; the core and node entries never use them), `types: ["node",
"vitest/globals", "@webgpu/types"]`. `tsconfig.strict-consumer.json`
compiles `test/types/*.test-d.ts` against `dist/*.d.ts` with the strict flags
ON (design 16.6).

### 3.2 Layers and their one-line responsibilities

| Layer | Module(s) | Responsibility |
| --- | --- | --- |
| device | `GpuContext`, `GpuCaps`, `Profiler` | acquisition (section 2), capability record, error scopes, uncaptured-error routing, device-loss state, optional timestamp profiling. |
| memory | `GraphResidency`, `UploadPlanner`, `BufferPool`, `Readback`, `Lease` | upload cache keyed on typed-array objects + per-snapshot record; arena / per-array / windowed plans; pooled scratch; staging ring readback; `release(snapshot)`. |
| kernel | `PipelineCache`, `Kernel`, `DispatchPlanner`, `UniformRing`, `WgslComposer`, `CommandBatch` | compile-once pipelines keyed by module + overrides + features; bind-group layout from a binding spec; 1D/2D/indirect dispatch math; 16-byte-safe uniform packing with dynamic offsets; WGSL composition; recording k iterations into one command buffer. |
| primitives | `reduce`, `scan`, `segmentedReduce`, `compact`, `histogram`, `radixSort`, `Frontier`, `advance`, `spmv`, `cooToCsr`, `bbox` | the reusable building blocks (section 6), each with a CPU reference in `test/helpers/oracle.ts`. |
| algorithms | one async function per algorithm | `(ctx, snapshot, options?, dest?) => Promise<Result>`; index-aligned typed arrays; no id mapping (section 8). |
| layouts | `ForceSimulation` (shared), `ForceAtlas2Simulation`, `FruchtermanReingoldSimulation`, `RepulsionExact`, `RepulsionGrid` | `LayoutSimulation` implementations over the owner's stride-3 array (section 7). |
| accelerator | `GpuAccelerator` | the injectable object: implements `AlgorithmAccelerator & LayoutAccelerator` structurally, plus `release(snapshot)` and `dispose()` (section 9). |

Class list (kept deliberately short; algorithms are plain functions):

| Class | One line |
| --- | --- |
| `WebGpuGraphError` | `Error` with stable `code` and frozen `details`; the only error type the package throws. |
| `GpuContext` | owns one device, its caps, pipeline cache, buffer pool, staging ring and residency; `probe` / `create` / `from` / `accelerator()` / `release` / `dispose`. |
| `GraphResidency` | WeakMap-keyed upload cache for cores, views and columns; per-snapshot record for `release`. |
| `BufferPool` | size-class pool of `GPUBuffer`s by usage; `acquire` / `release` / `trim`. |
| `Readback` | ring of `MAP_READ` staging buffers; `read(src, byteLength, dest?)` copies out before `unmap`. |
| `PipelineCache` | `get(moduleId, overrides, features) -> GPUComputePipeline` with error scopes and compilation-info diagnostics. |
| `Kernel` | a compiled pipeline plus its binding spec; `bind(resources)` creates / caches bind groups; `dispatch(pass, plan)`. |
| `DispatchPlanner` | pure functions of caps: `plan1d`, `plan2d`, `planIndirect`. |
| `UniformRing` | one `UNIFORM` buffer with 256-byte-stride slots for per-iteration params inside a batch. |
| `CommandBatch` | records N dispatches / copies into one command encoder and submits once; carries the staging buffer for the batch's readback. |
| `Frontier` | vertex / edge queues, length counters, indirect-args buffer, ping-pong. |
| `ForceSimulation` | the shared layout state machine (buffers, in-flight batches, readback, settle, fixed mask, setPosition overrides); subclasses supply the per-iteration kernel sequence. |
| `GpuAccelerator` | thin adapter from the CPU packages' interfaces to the functions above. |

### 3.3 Public TypeScript API (root entry)

Errors:

```ts
export type WebGpuGraphErrorCode =
    | "E_NO_WEBGPU" | "E_NO_ADAPTER" | "E_NO_DEVICE" | "E_SOFTWARE_ONLY" | "E_DEVICE_LOST" | "E_DISPOSED"
    | "E_VALIDATION" | "E_SHADER_COMPILE" | "E_OUT_OF_MEMORY" | "E_TOO_LARGE" | "E_UNSUPPORTED"
    | "E_INVALID_ARGUMENT" | "E_SNAPSHOT" | "E_NOT_LOADED" | "E_IN_FLIGHT" | "E_ABORTED";
export class WebGpuGraphError extends Error {
    readonly code: WebGpuGraphErrorCode;
    readonly details: Readonly<Record<string, unknown>>;
    constructor(code: WebGpuGraphErrorCode, message: string, details?: Record<string, unknown>);
}
```

Context and capabilities:

```ts
export interface GpuCaps {
    readonly limits: GPUSupportedLimits;            // device.limits
    readonly features: ReadonlySet<string>;         // device.features
    readonly subgroupMinSize: number;               // 0 when the feature is absent
    readonly subgroupMaxSize: number;
    readonly software: boolean;
    readonly runtime: "browser" | "node" | "unknown";
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
}
export declare class GpuContext {
    static probe(options: ProbeOptions): Promise<ProbeResult>;
    static create(options: GpuContextOptions): Promise<GpuContext>;
    static from(device: GPUDevice, info?: Partial<GpuCaps>): GpuContext;
    readonly device: GPUDevice;
    readonly caps: GpuCaps;
    readonly state: "ready" | "lost" | "disposed";
    readonly lost: Promise<GPUDeviceLostInfo>;
    readonly residency: GraphResidency;
    readonly profiler: Profiler | null;             // non-null when "timestamp-query" was granted
    accelerator(): GpuAccelerator;                  // one per context, cached
    release(snapshot: GraphSnapshot): void;
    dispose(): void;
}
export function isSoftwareAdapter(info: GPUAdapterInfo & { isFallbackAdapter?: boolean }): boolean;
```

Algorithms (section 8 gives the full list; the signature shape is uniform):

```ts
export interface GpuRunOptions { readonly dest?: Float32Array | Uint32Array | undefined; readonly signal?: AbortSignal | undefined; readonly onProgress?: ((done: number, total: number) => void) | undefined; }

export function pageRank(ctx: GpuContext, s: GraphSnapshot, options?: PageRankOptions & GpuRunOptions): Promise<GpuPageRankResult>;
export function personalizedPageRank(ctx: GpuContext, s: GraphSnapshot, personalization: F32, options?: PageRankOptions & GpuRunOptions): Promise<GpuPageRankResult>;
export function hits(ctx: GpuContext, s: GraphSnapshot, options?: HitsOptions & GpuRunOptions): Promise<GpuHitsResult>;
export function eigenvectorCentrality(ctx: GpuContext, s: GraphSnapshot, options?: EigenvectorOptions & GpuRunOptions): Promise<GpuScoresResult>;
export function katzCentrality(ctx: GpuContext, s: GraphSnapshot, options?: KatzOptions & GpuRunOptions): Promise<GpuScoresResult>;
export function connectedComponents(ctx: GpuContext, s: GraphSnapshot, options?: GpuRunOptions): Promise<GpuLabelResult>;          // WCC semantics on directed input
export function breadthFirstSearch(ctx: GpuContext, s: GraphSnapshot, source: number, options?: BfsOptions & GpuRunOptions): Promise<GpuBfsResult>;
export function sssp(ctx: GpuContext, s: GraphSnapshot, source: number, options?: SsspOptions & GpuRunOptions): Promise<GpuSsspResult>;
export function bellmanFord(ctx: GpuContext, s: GraphSnapshot, source: number, options?: BellmanFordOptions & GpuRunOptions): Promise<GpuBellmanFordResult>;
export function closenessCentrality(ctx: GpuContext, s: GraphSnapshot, options?: ClosenessOptions & GpuRunOptions): Promise<GpuScoresResult>;
export function betweennessCentrality(ctx: GpuContext, s: GraphSnapshot, options?: BetweennessOptions & GpuRunOptions): Promise<GpuScoresResult>;
export function edgeBetweennessCentrality(ctx: GpuContext, s: GraphSnapshot, options?: BetweennessOptions & GpuRunOptions): Promise<GpuEdgeScoresResult>;
export function allPairsShortestPath(ctx: GpuContext, s: GraphSnapshot, options?: ApspOptions & GpuRunOptions): Promise<GpuApspResult>;
export function kCoreDecomposition(ctx: GpuContext, s: GraphSnapshot, options?: GpuRunOptions): Promise<GpuCorenessResult>;
export function triangleCount(ctx: GpuContext, s: GraphSnapshot, options?: GpuRunOptions): Promise<GpuTriangleResult>;
export function labelPropagation(ctx: GpuContext, s: GraphSnapshot, options?: LabelPropagationOptions & GpuRunOptions): Promise<GpuLabelResult>;
export function louvain(ctx: GpuContext, s: GraphSnapshot, options?: LouvainOptions & GpuRunOptions): Promise<GpuCommunityResult>;
export function degree(ctx: GpuContext, s: GraphSnapshot, options?: GpuRunOptions): Promise<U32>;   // walking-skeleton kernel; kept as a diagnostic
```

Result types (all index-aligned; `F32`, `U32` are the graph-format aliases
`Float32Array<ArrayBuffer>` / `Uint32Array<ArrayBuffer>`, note 07 section
1.2):

```ts
export interface GpuScoresResult      { readonly scores: F32; readonly iterations: number; readonly converged: boolean; }
export interface GpuPageRankResult    extends GpuScoresResult { readonly danglingMass: number; }
export interface GpuHitsResult        { readonly hubs: F32; readonly authorities: F32; readonly iterations: number; readonly converged: boolean; }
export interface GpuLabelResult       { readonly labels: U32; readonly count: number; groups(): U32[]; }      // labels dense 0..count-1, first-seen order (renumberPartition)
export interface GpuBfsResult         { readonly depth: U32; readonly parent: U32; readonly order: U32; readonly visitedCount: number; readonly levels: number; }   // INVALID_INDEX = unreached / root
export interface GpuSsspResult        { readonly dist: F32; readonly predArc: U32; readonly reachedCount: number; }   // +Infinity = unreached
export interface GpuBellmanFordResult extends GpuSsspResult { readonly hasNegativeCycle: boolean; }
export interface GpuEdgeScoresResult  { readonly scores: F32; }        // length edgeCount (folded with foldArcs "first")
export interface GpuApspResult        { readonly dist: F32; readonly n: number; }   // n * n row-major
export interface GpuCorenessResult    { readonly coreness: U32; readonly maxCore: number; }
export interface GpuTriangleResult    { readonly perNode: U32; readonly total: number; }
export interface GpuCommunityResult   extends GpuLabelResult { readonly modularity: number; readonly levels: number; }
```

Layouts:

```ts
export interface GpuLayoutSimulation extends LayoutSimulation {            // LayoutSimulation is the design-14.3 interface (owned by @graphty/layout after L1; a structural copy lives in src/types until W1)
    load(snapshot: GraphSnapshot, positions: F32): void;
    step(iterations?: number): Promise<void>;                             // resolves when `positions` holds the result of these iterations
    readonly settled: boolean;
    setFixed(mask: NodeMask): void;
    setPosition(index: number, x: number, y: number, z: number): void;
    dispose(): void;
    // GPU-specific additions (not part of LayoutSimulation):
    readonly inFlight: number;                                            // batches submitted but not yet read back
    readonly iterationsDone: number;
    readonly stats: LayoutStats;                                          // last completed batch: swing, traction, speed, meanDisplacement, layoutRadius, msPerIteration (when profiled)
    flush(): Promise<void>;                                               // wait for every in-flight batch
    reheat(): void;                                                       // reset the settle window and iteration budget
    setParams(patch: Partial<ForceAtlas2Params | FruchtermanReingoldParams>): void;   // live tuning without reload
    run(options?: { readonly maxIter?: number; readonly batch?: number; readonly signal?: AbortSignal }): Promise<LayoutStats>;   // Node batch driver: step until settled
}
export function createForceAtlas2(ctx: GpuContext, options?: ForceAtlas2Options): GpuLayoutSimulation;
export function createFruchtermanReingold(ctx: GpuContext, options?: FruchtermanReingoldOptions): GpuLayoutSimulation;
```

The accelerator object (section 9 defines the two interfaces it satisfies):

```ts
export interface GpuAccelerator {
    readonly kind: "webgpu";
    readonly ctx: GpuContext;
    // AlgorithmAccelerator members (each delegating to the function above; only implemented ones are present):
    pageRank(s: GraphSnapshot, o?: PageRankOptions): Promise<GpuPageRankResult>;
    connectedComponents(s: GraphSnapshot): Promise<GpuLabelResult>;
    breadthFirstSearch(s: GraphSnapshot, source: number, o?: BfsOptions): Promise<GpuBfsResult>;
    // ... one per shipped algorithm
    // LayoutAccelerator members:
    forceAtlas2(o?: ForceAtlas2Options): GpuLayoutSimulation;
    fruchtermanReingold(o?: FruchtermanReingoldOptions): GpuLayoutSimulation;
    // lifecycle:
    release(s: GraphSnapshot): void;
    dispose(): void;
}
```

Option types re-declare the CPU packages' option shapes STRUCTURALLY in
`src/types/options.ts` (same field names and defaults as
`layout/src/layouts/force-directed/forceatlas2.ts` lines 26-42 and the
`indexed.*` options of design 14.2); at W1 a type test asserts mutual
assignability with the real `@graphty/algorithms` / `@graphty/layout` types
(devDependencies, types only, section 9.8).

### 3.4 Browser and Node entries

```ts
// ./browser
export interface BrowserGpuOptions extends Omit<GpuContextOptions, "gpu" | "adapter" | "device"> { readonly rejectSoftware?: boolean | undefined; }
export function probeBrowserWebGpu(options?: BrowserGpuOptions): Promise<ProbeResult>;
export function requestGpuContext(options?: BrowserGpuOptions): Promise<GpuContext>;

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
tsconfig.build.json` only, which neither copies `.wgsl` files nor
understands `?raw`; knip's project globs are `src/**/*.ts`; eslint lints
`.ts` only. If `.wgsl` files are ever wanted, a generator script emits the
`.wgsl.ts` files with the owner's auto-generated header.

Composition is string concatenation in `src/kernel/wgsl.ts`:

```ts
export interface WgslModuleSpec {
    readonly id: string;                                  // "fa2-attraction"
    readonly body: string;                                // the kernel source with `@compute @workgroup_size(WG)`
    readonly overrides: Readonly<Record<string, number | boolean>>;   // WG, USE_PERM, HAS_WEIGHTS, DIM, LINLOG, ...
    readonly needs: readonly ("subgroups")[];             // splice `enable subgroups;` + subgroup helpers only when the device has it
    readonly snippets?: Readonly<Record<string, string>>; // operator bodies substituted at `//@@NAME@@` markers (Gunrock-style advance / filter functors)
}
```

The prelude every module receives:

```wgsl
// prelude.wgsl.ts (excerpt)
const INVALID_INDEX: u32 = 0xFFFFFFFFu;
override WG: u32 = 256u;
override USE_PERM: bool = false;      // design 10.1: select(a, arcToEdge[a], USE_PERM)
override HAS_WEIGHTS: bool = false;   // weights === null -> 1.0; the weights slot is bound to colIdx and never read
override SUBGROUP_SIZE: u32 = 0u;     // 0 = no subgroups
fn linear_id(wid: vec3<u32>, lid: u32) -> u32 { return (wid.x + wid.y * 65535u) * WG + lid; }   // 2D grid linearisation (section 5.2)
```

Pipeline identity is `(id, overrides, needs present on device)`; the cache
key is the JSON of that triple (section 5.1). Operator snippets are part of
the key too (the same advance kernel with a BFS functor and an SSSP functor
are two pipelines).

Bind-group conventions (default limit: 8 storage buffers per stage, 4 bind
groups; note 05 section 4): group 0 = the graph (immutable per snapshot:
`rowPtr`, `colIdx`, `weights` or dummy, `perm` or dummy), group 1 =
algorithm state (ping-pong vectors, queues, partials), group 2 = the
params uniform (dynamic offset into the `UniformRing`), group 3 = optional
/ cold arrays (`arcToEdge`, `edgeToArc`, columns). A kernel that needs more
than 8 storage buffers in one stage is split, never given a raised limit as
a requirement.

### 3.6 Naming and conventions carried from the sibling packages

JSDoc on every export, explicit return types, `.js` suffixes on relative
imports, no default exports, no `console.log` in `src/`, plain ASCII,
prettier 4 / 120 / all, knip clean, no `eslint-disable`, no
`ts-expect-error` (note 07 section 7). Results are
`Uint32Array<ArrayBuffer>` / `Float32Array<ArrayBuffer>`; `INVALID_INDEX` is
the only sentinel; never a bitwise operator on an arc index or byte offset
(I3); never a write into a view; never a zero-length binding.

---------------------------------------------------------------------------

## 4. Memory and upload

### 4.1 GraphResidency: what is cached and under which key

```ts
export declare class GraphResidency {
    core(s: GraphSnapshot, need?: readonly CoreArrayName[]): CoreBinding;          // rowPtr, colIdx, weights (+ arcToEdge, edgeToArc on demand)
    view(s: GraphSnapshot, name: "reverse" | "coo" | "edgeList" | "outDegree" | "inDegree" | "degreeOrder" | "reverseDegreeOrder" | "mate"): ViewBinding;
    column(table: AttributeTable, name: string): ColumnBinding;                   // gpuView(name) + column.version
    array(key: TypedArrayData, label: string): ArrayBinding;                     // any format array (expanded weights, a mask), keyed on the object
    release(s: GraphSnapshot): void;                                             // destroys every buffer recorded for s
    stats(): ResidencyStats;                                                     // buffers, bytes, per-snapshot breakdown (for tests and a dev overlay)
}
export interface Binding { readonly buffer: GPUBuffer; readonly offset: number; readonly size: number; readonly window: ArcWindow | null; }
```

Keys (design 14.5 lines 4231-4234; note 07 sections 1.7, 2.3):

| Thing | Key | Invalidation |
| --- | --- | --- |
| core arena | `snapshot.rowPtr` (the array object; two snapshots sharing a core via `withColumns()` share `rowPtr`) | `release(snapshot)`; the record is also indexed by `snapshot.serial` so `withColumns()` siblings find it |
| per-array core upload | each of `rowPtr`, `colIdx`, `weights`, `arcToEdge`, `edgeToArc` array objects | `release(snapshot)` |
| view array | the view array object (`reverse().colIdx`, `coo().src`, `edgeList().src`, `degreeOrder().perm`, ...) | `release(snapshot)`; `dropCaches()` changes the object (note 07 section 2.3) -- the OLD buffer stays recorded in the per-snapshot record and is freed by `release`, so no leak and no stale read |
| column | the `gpuView(name)` array object PLUS `column.version` | a version bump (`markDirty()` / `setAll()`) re-uploads into the same buffer when the byte length is unchanged (a `writeBuffer`, no realloc); `release(snapshot)` frees it |
| ad hoc array (expanded named-column weights from graphty-element's `expandEdges` cache, a `NodeMask`) | the array object | `release(snapshot)` when registered against a snapshot, else `binding.destroy()` by the caller |

Two `WeakMap`s: `WeakMap<TypedArrayData, ResidentBuffer>` (fast lookup) and
`WeakMap<GraphSnapshot, ResidencyRecord>` (enumeration for `release`, also
keyed by `serial` in a `Map<number, ResidencyRecord>` cleared by `release`).
A `ResidentBuffer` records `{ buffer, byteLength, refs: Set<serial>, kind }`;
a core shared by sibling snapshots is destroyed when the last sibling is
released. `release` is idempotent and safe on a snapshot that was never
uploaded.

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
2425-2438; note 07 section 3):

| Plan | Condition | Buffers | Bindings |
| --- | --- | --- | --- |
| `arena` | `s.arena !== null` AND `arena.byteLength <= limits.maxBufferSize` AND every non-null `segment.byteLength <= limits.maxStorageBufferBindingSize` | ONE `GPUBuffer` of `hotByteLength` bytes (or `byteLength` when a cold segment is in `need`); ONE `writeBuffer(gbuf, 0, new Uint8Array(arena.buffer, arena.byteOffset, len))` | `{ buffer: gbuf, offset: seg.byteOffset - arena.byteOffset, size: seg.byteLength }` per segment (offsets are 256-multiples relative to `arena.byteOffset`, satisfying any `minStorageBufferOffsetAlignment <= 256`) |
| `perArray` | `arena === null` (every `fromCsr` on separate arrays, every `transpose()`, note 07 section 2.5) OR the arena exceeds `maxBufferSize` but each array fits its binding limit | one buffer per array, `writeBuffer(bufX, 0, s.colIdx)` etc. (no cast: `Uint32Array<ArrayBuffer>` is a `BufferSource`) | whole-buffer bindings |
| `windowed` | some array's byte length exceeds `maxStorageBufferBindingSize` (33,554,432 arcs per binding at the 128 MiB default; 536,870,911 at the 4070's 2 GiB under Dawn) | per-array buffers as above (a buffer may exceed the BINDING limit while staying under `maxBufferSize`; if it exceeds `maxBufferSize` too, the array is split across buffers at the same window boundaries) | a list of `ArcWindow { start, end, rowFirst, rowLast, bufferIndex, offset }` where `start = rowPtr[v0] - (rowPtr[v0] % 64)` (256-byte aligned; `%`, never `& ~63`, design 10.6); kernels get `start` as a rebase uniform and iterate `[max(rowPtr[u], start), min(rowPtr[u+1], end))`; a row longer than one window is split across windows with the row's contribution accumulated across dispatches |

Which algorithms support `windowed` in v1 is explicit (section 4.6): the
per-row gather family (attraction, SpMV, degree, segmented reduce) does; the
frontier family and sort-based algorithms throw `E_TOO_LARGE` with `{
path: "windowed", algorithm }` until they are extended. Raised limits are
requested from the adapter first (section 2.2), so on the 4070 SUPER the
windowed path is reached only above ~500M arcs.

Upload cost model (basis: no measurement yet; `writeBuffer` copies
synchronously from the V8 backing store, note 05 section 2.3; PCIe 4.0 x16
staging typically lands at 5-12 GB/s host-to-device): the 100k / 1M
undirected weighted arena (hot prefix 16.4 MB, design 10.3) is ~2-4 ms;
1M / 10M (164 MB) ~20-40 ms; 10M / 100M (1.64 GB, per-array) ~200-400 ms.
The walking skeleton (section 13, P0) measures these and records them in
`benchmarks/results/`.

### 4.3 View and column uploads

Views upload through the same planner in `perArray` mode (they are never in
the arena, design 10.3). `reverse()` on an undirected snapshot returns the
forward array objects (note 07 section 1.6), so `residency.view(s,
"reverse")` resolves to the SAME buffers as `core()` with no upload -- the
WeakMap key is the array object. `degreeOrder().segmentOffsets` is read on
the CPU (5 numbers) and passed as uniform scalars (design 10.1 line 2342).
`weightedOutDegree()` is never uploaded (F64); the normaliser is computed on
the device (section 6.3).

Columns: `column(table, name)` calls `table.gpuView(name)` (throws
`E_GPU_INELIGIBLE` from the format for string / list / json; the GPU package
lets it propagate) and records `column.version`. `u8` columns bind
`paddedU32View()` and kernels bound-check `i < rows * components`; `bool`
columns and masks bind their `data` words. A mutable column whose `version`
changed since upload is re-uploaded on the next `column()` call (same buffer
when the byte length matches). The position column is NOT uploaded through
this path: `LayoutSimulation.load()` takes the owner's array directly
(section 7.3).

### 4.4 BufferPool, Lease and Readback

`BufferPool.acquire(byteLength, usage, label)` rounds up to a size class
(powers of two from 4 KiB to 64 MiB, then 16 MiB steps; classes above
`maxBufferSize` are never created) and keeps at most `maxIdlePerClass = 4`
idle buffers per (class, usage). `release(buf)` returns it; `trim()`
destroys idle buffers (called by `ctx.release` and by layouts on `dispose`).
Large allocations run inside `pushErrorScope("out-of-memory")`; an OOM error
becomes `E_OUT_OF_MEMORY { requested, resident }` (never a silent smaller
buffer).

`Lease` is the scope object algorithms use: `using lease = pool.lease();
const a = lease.storage(n * 4, "sigma");` -- every buffer acquired through
the lease is released when the algorithm resolves or rejects (a
`try/finally`, not `Symbol.dispose`, until the monorepo's TS target
supports `using`). Persistent buffers (layout state) are owned by the
simulation object and freed in `dispose()`.

`Readback` owns a ring of `MAP_READ | COPY_DST` staging buffers (default 3;
the browser needs more than one because `mapAsync` on a buffer in use by a
queued copy is a validation error, note 05 section 7.2). `read(src,
byteLength, dest?)`: pick a free staging buffer (`mapState === "unmapped"`)
or grow the ring, `copyBufferToBuffer` inside the batch encoder, `submit`,
`await mapAsync(READ)`, `dest.set(new Float32Array(getMappedRange()))` (copy
BEFORE `unmap`, design 10.7), `unmap()`. Requests above the staging size
are split into chunks. A `readU32(src, offset)` helper reads a single
counter (BFS frontier length, convergence flag) through the same ring.

### 4.5 Release lifecycle

```
DataManager.getSnapshot()  --freeze-->  snapshot-replaced { previous, next, report }
    graphty-element listener: accelerator.release(previous)   -> GraphResidency.release(previous): destroy core, views, columns, ad hoc arrays recorded for previous
    LayoutSimulation engines: engine.reload(undirected(next), report, positions)  -> simulation.load(next, positions) re-uploads next's core; the simulation's OWN scratch (forces, partials, grid) is resized in place when nodeCount changed
    accelerator.dispose()  -> ctx.dispose(): everything, then device.destroy() if owned
```

Rules: `release` is called by the OWNER of the snapshot lifecycle (the
element, a Node script), never by an algorithm; an algorithm that receives a
snapshot with no residency uploads it and leaves it resident (the common
case: the next algorithm on the same snapshot pays nothing); a
`LayoutSimulation.load(next)` on a different snapshot releases nothing by
itself (the element releases `previous`), but the simulation drops its
references to the previous core bindings so a later `release(previous)`
finds no live user.

### 4.6 Chunking and dispatch limits per family

| Family | > 16,776,960 items per dispatch (2D grid) | Windowed bindings (arc ranges) | Notes |
| --- | --- | --- | --- |
| per-node map / reduce / integrate | yes (n > 16.7M nodes) | n/a | trivially chunked |
| per-row gather (attraction, SpMV, segmented reduce, degree) | yes | yes (v1) | window loop on the host: one dispatch per window, accumulating into the same output |
| per-arc map (coo / edgeList kernels: CC hook, Bellman-Ford relax) | yes (A > 16.7M arcs -- the 1M / 10M tier undirected has 20M arcs, so this is the NORMAL case at the desktop tier) | yes (v1) | the arc range of a window is the dispatch range |
| frontier advance / compaction | yes (edge frontier > 16.7M) | no (v1: `E_TOO_LARGE`) | needs the whole `colIdx` bound |
| radix sort / scan | yes | n/a (scratch is the package's own) | |
| grid pyramid | yes | n/a | fixed-size grid buffers |

### 4.7 Bytes per node and per edge on the device

Core (undirected, doubled arcs, weights present unless noted; design 15.1):

| Component | Bytes | 100k / 1M | 1M / 10M | 10M / 100M |
| --- | --- | --- | --- | --- |
| `rowPtr` | 4(n + 1) | 0.4 MB | 4 MB | 40 MB |
| `colIdx` | 4A = 8E | 8 MB | 80 MB | 800 MB |
| `weights` | 4A (0 when null) | 8 MB | 80 MB | 800 MB |
| hot prefix total | | 16.4 MB | 164 MB | 1.64 GB (per-array; `colIdx` alone exceeds a 128 MiB default binding above 33.5M arcs, so raised limits are required; on lavapipe this tier is out of reach) |
| `arcToEdge` (cold; only for edge-column gathers) | 4A | 8 MB | 80 MB | 800 MB |
| `edgeToArc` (cold; per-edge writeback) | 4E | 4 MB | 40 MB | 400 MB |
| `coo().src` | 4A | 8 MB | 80 MB | 800 MB |
| `edgeList().src/.dst` | 8E | 8 MB | 80 MB | 800 MB |
| `degreeOrder().perm` | 4n | 0.4 MB | 4 MB | 40 MB |
| `reverse()` (directed only) | 4(n+1) + 8A (+4A weights) | 12.4 MB | 124 MB | 1.24 GB |

Per-algorithm scratch (per node unless stated): PageRank 2 x 4 (ping-pong)
+ 4 (out-weight sum) + partials = ~12 B/node; BFS 8 (depth, parent) + 2 x 4
(queues) + 1/8 (bitset) = ~16 B/node + optional 4A edge queue; CC 4 B/node +
readback; betweenness with batch k sources: (4 sigma + 4 depth + 4 delta) x
k per node; FA2 exact ~44 B/node, FA2 grid ~60 B/node plus a fixed pyramid
(5.6 MB at 512^2 in 2D, ~38 MB at 128^3 in 3D) (section 7.3, note 03
section 8.2). Memory is never the binding constraint below 10M nodes on a
12 GB card; time is (section 10).

---------------------------------------------------------------------------

## 5. Kernel infrastructure

### 5.1 PipelineCache and Kernel

```ts
export declare class PipelineCache {
    get(spec: WgslModuleSpec, layout: GPUPipelineLayout | "auto"): Promise<GPUComputePipeline>;   // createComputePipelineAsync inside pushErrorScope("validation")
    warm(specs: readonly WgslModuleSpec[]): Promise<void>;                                        // called by load() so the first step() does not compile
    readonly size: number;
}
```

Key = `spec.id + "|" + stableJson(spec.overrides) + "|" + spec.needs.filter(
f => caps.features.has(f)).join(",") + "|" + hash(spec.snippets)`. Every
distinct `override` set is a distinct pipeline (WGSL 7.2.2, note 05 section
6 item 3), so overrides are limited to things that genuinely change the
code (`USE_PERM`, `HAS_WEIGHTS`, `DIM`, `LINLOG`, `STRONG_GRAVITY`,
`SUBGROUP_SIZE`, `WG`); everything numeric that varies per iteration is a
uniform. Compilation failure: `getCompilationInfo()` messages are formatted
with line numbers against the COMPOSED source and thrown as
`E_SHADER_COMPILE { id, messages }`; `test/kernel/wgsl-compile.test.ts`
compiles every module in every override combination the package uses on
the `null` backend where available (Dawn-node `backend=null`, note 05
section 2.2) and otherwise on the real device.

`Kernel` binds a compiled pipeline to a `BindingSpec` (ordered list of `{
group, binding, kind: "storage" | "storage-ro" | "uniform", name }`),
creates bind groups from a `Record<name, Binding>` and caches them by the
identity of the buffers and offsets (a layout's bind groups are created once
per `load()`, not per iteration).

### 5.2 DispatchPlanner: the 16,776,960 rule, 2D grids, grid-stride

```ts
export interface DispatchPlan { readonly x: number; readonly y: number; readonly z: 1; readonly items: number; readonly stride: number | null; }
export function plan1d(items: number, wg: number, caps: GpuCaps): DispatchPlan;      // throws E_TOO_LARGE if a 2D grid is also insufficient
export function planGridStride(items: number, wg: number, caps: GpuCaps, maxGroups?: number): DispatchPlan;
```

Rules (design 10.6 lines 2500-2507):

- `groups = ceil(items / wg)`. If `groups <= limits.maxComputeWorkgroupsPerDimension`
  (65,535 at the core default): `{ x: groups, y: 1 }`. The boundary test in
  `test/kernel/dispatch.test.ts` asserts `items = 16,776,960` is 1D and
  `16,776,961` is 2D with `wg = 256` -- NOT the round 2^24.
- Else 2D: `x = 65535`, `y = ceil(groups / 65535)`, and every kernel
  computes its item index with `linear_id()` from the prelude
  (`(wid.x + wid.y * 65535u) * WG + lid`) and returns early when `id >=
  params.items`. `y` above the limit -> `E_TOO_LARGE` (that is 65,535^2 x
  256 = 1.1e12 items, beyond any snapshot).
- Grid-stride is used for kernels whose per-item work is tiny and whose
  item count is huge (per-arc maps at 100M arcs): `groups = min(groups,
  maxGroups)` and the kernel loops `for (i = id; i < items; i += stride)`.
  The planner is pure so both branches are unit-tested with faked caps
  (spec defaults, SwiftShader-like, lavapipe-like, NVIDIA-like tables from
  note 05 section 4).

### 5.3 Uniforms

Chromium 139 lacks `uniform_buffer_standard_layout` (note 05 section 2.4),
so every params struct obeys the strict rules: scalars grouped into
16-byte-aligned members, `vec3` never used in uniforms, arrays of scalars
avoided (a 20-byte `array<u32, 5>` is illegal, design 10.1). `UniformRing`
packs a params object through a declared layout (`u32`/`f32`/`vec4f`
fields with explicit offsets, checked at construction to be a multiple of
16 in total), writes it into slot `k` of one `UNIFORM` buffer with a
256-byte stride (`minUniformBufferOffsetAlignment` is 256 in browsers, 64
/ 16 in Dawn-node; 256 satisfies all), and binds it with a dynamic offset.
A batch of k iterations therefore writes k slots once (`writeBuffer` of `k
* 256` bytes) and records k dispatches with different dynamic offsets --
no per-iteration `writeBuffer` calls and no host round trip. Values that
change on the device between iterations (FA2 speed) live in a small
STORAGE block written by the finalize kernel (section 7.9), not in the
uniform.

### 5.4 Indirect dispatch

Frontier-driven kernels use `dispatchWorkgroupsIndirect(argsBuffer,
offset)`. A one-workgroup `finalize` kernel turns a device-side count into
`(x, y, 1)` using the same rule as `plan1d` (clamped to 65,535 with a `y`
split; the kernel side uses `linear_id`), so an over-limit count never
reaches the API (the spec's "does nothing" behaviour for over-limit indirect
counts is noted as unverified in note 05 section 12 item 7; the finalize
kernel makes it moot). One `INDIRECT` buffer holds `4 * k` slots for a
batch of k rounds, so k BFS levels are recorded into one command buffer
(D17); a level whose frontier is empty dispatches `(0, 0, 1)` and the
following levels are no-ops, which is what makes "record 32 levels, read
the done flag once" correct without knowing the diameter.

### 5.5 Timestamp queries and profiling

When `timestamp-query` was granted, `ctx.profiler` wraps compute passes
with `timestampWrites` (begin / end per pass) into a query set of 256
slots, resolves into a buffer read back with the batch's staging buffer,
and reports `{ label, ns }` per pass. Chromium quantises to 100 us (note 05
section 3.1); the profiler reports `quantised: true` in browsers. Per-kernel
profiling is a Node activity (benchmarks, section 11.6); the layout's
`stats.msPerIteration` is derived from it when present, else from
`performance.now()` around the batch (wall time including queue latency,
labelled as such).

### 5.6 Empty ranges, zero-length arrays

`colIdx`, `weights`, `arcToEdge` have length 0 when `arcCount === 0`;
`edgeToArc` when `edgeCount === 0` (design 10.5). The residency never
creates a zero-byte buffer and the kernel layer never dispatches a plan with
`items === 0`: `plan1d(0, ...)` returns `{ x: 0 }` and `Kernel.dispatch`
skips it. A snapshot with `nodeCount === 0` short-circuits every algorithm
to its empty result on the CPU (an empty `Uint32Array(0)`), which is not a
fallback -- there is no work.

### 5.7 Error handling: validation, out-of-memory, device loss, cancellation

| Event | Detection | Behaviour |
| --- | --- | --- |
| Validation error during pipeline / bind-group creation | `pushErrorScope("validation")` around creation | thrown synchronously from the awaiting call as `E_VALIDATION { label, message }`; the message includes the buffer / pipeline label (every object is labelled) |
| Validation error at submit time (a kernel bug) | `uncapturederror` listener | routed to `options.onError`; if none, stored and thrown from the NEXT public call as `E_VALIDATION` so a silent stderr block in Dawn-node cannot pass a test (note 05 section 2.2) |
| Out of memory | `pushErrorScope("out-of-memory")` around every `createBuffer` above 16 MiB | `E_OUT_OF_MEMORY { requested, resident }`; the layout / algorithm releases what it allocated in the failing call |
| Device lost | `device.lost` | `ctx.state = "lost"`, all pending promises reject with `E_DEVICE_LOST { reason, message }`, residency cleared, every simulation enters `disposed`; the CALLER decides whether to create a new context and `load()` again (note 05 section 7.4). A test destroys the device mid-batch on both runtimes. |
| `AbortSignal` (`GpuRunOptions.signal`) | checked between batches | the algorithm stops recording, releases its lease and rejects with `E_ABORTED` (the caller cancelled; the package did not fail); `GpuLayoutSimulation.run()` honours the same signal; a batch already submitted completes on the device and its readback is discarded |
| Wrong argument (source index >= n, mask too short, `dest` too small, directed snapshot to an undirected-only kernel) | argument checks before any GPU work | `E_INVALID_ARGUMENT` / `E_SNAPSHOT` with the offending value in `details`; state unchanged |
| Snapshot too large for the current plan | `planUpload` / planner | `E_TOO_LARGE { needed, limit, path, algorithm }` before allocation |

Aborting never leaves a residency inconsistent because uploads are atomic per
array and scratch is lease-scoped.

### 5.8 Submission model

`CommandBatch` records one `GPUCommandEncoder` with one compute pass per
"phase" (a pass may contain many dispatches; implicit barriers between
dispatches in a pass order storage writes, note 05 section 7.1),
`copyBufferToBuffer` calls for readbacks at the end, and submits ONCE.
`queue.writeBuffer` calls issued before the submit (uniform slots,
`setPosition` writes, mask updates) are ordered before it by the queue
semantics. The batch returns `{ submitted: number (batch id), readback:
Promise<void> }`. Layouts and iterative algorithms never call
`onSubmittedWorkDone` (an extra ~0.1 ms promise in Chromium, note 05 section
7.2); `mapAsync` on the batch's staging buffer is the completion signal.

---------------------------------------------------------------------------

## 6. Primitives

Every primitive has a TypeScript interface in `src/primitives/`, one or
more WGSL modules, a stated complexity, and a CPU reference implementation
in `test/helpers/oracle.ts` used by its differential test. The primitive
order matches note 04 section 15 item 1 (what unblocks the most
algorithms), reordered so the layout slice's needs come first.

| # | Primitive | Interface (all take a `CommandBatch` and record into it) | WGSL strategy | Complexity | CPU reference |
| --- | --- | --- | --- | --- | --- |
| 1 | `reduce` | `reduce(batch, src: Binding, count, op: "sum" | "min" | "max", dtype: "f32" | "u32" | "vec4f", out: Binding, outOffset)` | two dispatches: workgroup tree reduce of 256 items into `partials[groups]` (subgroup variant: `subgroupAdd` then one cross-subgroup pass), then ONE workgroup reduces the partials with a grid-stride loop (partials <= 65,535 so one workgroup finishes in `ceil(65535/256)` = 256 iterations); deterministic order | O(count), 2 dispatches | `Array.reduce` in f64, compared with tolerance scaled by count |
| 2 | `scan` | `exclusiveScan(batch, src, count, out, totalOut?)` (u32) | reduce-then-scan: (a) workgroup scan of 256 (Hillis-Steele in workgroup memory, or `subgroupExclusiveAdd` + cross-subgroup fixup) writing block sums, (b) scan of block sums (recursive when > 256 blocks: at most 3 levels for 16.7M items), (c) add-back; no decoupled look-back in v1 (WGSL atomics are relaxed; note 04 section 2.1) | O(count), 3-7 dispatches | sequential prefix sum |
| 3 | `segmentedReduce` | `segmentedReduce(batch, graph: CoreBinding, tiers: DegreeTiers, per: "arc" -> value snippet, op, out)` | THREE pipelines from one module with a `TIER` override: thread-per-row for `[midEnd, lowEnd)` (degree < 32), subgroup-per-row for `[hiEnd, midEnd)` when `subgroups` exists (without the feature the mid tier is handled by a 32-invocation-per-row variant of the workgroup kernel, 8 rows per 256-wide workgroup, so no device ever runs a 1,000-arc row on one thread), workgroup-per-row for `[0, hiEnd)` (degree >= 1024) with a 256-wide workgroup reduce; rows visited through `degreeOrder(opts).perm` with `override USE_PERM` (identity when the caller passes no tiers); the value snippet is Gunrock's `neighborreduce` functor (note 04 section 2.2) | O(A), 3 dispatches | per-row loop in f64 |
| 4 | `compact` | `compact(batch, flags: Binding, count, out, outCount)` and `dedupe(batch, queue, count, owner: Binding, out, outCount)` | flag + scan + scatter (3-7 dispatches); dedupe by Davidson's ownership trick (write my queue index into `owner[v]`, read it back, keep iff equal; note 04 section 2.3) -- no atomics, last-writer-wins is correct | O(count) | filter / Set |
| 5 | `histogram` / counting sort | `histogram(batch, keys, count, bins, out)`; `countingSortByKey(batch, keys, count, bins, outIndex, outStart)` | `atomicAdd(&count[key], 1u)` (u32), scan, scatter with a per-bin `atomicAdd` cursor; order inside a bin is nondeterministic (documented); used by COO->CSR and the grid build's non-deterministic fast path | O(count + bins) | bucket loop |
| 6 | `radixSort` | `radixSort(batch, keys, values, count, bits: 8 | 16 | 24 | 32)` (LSD, key-value, stable) | 8 bits per pass, per pass: per-workgroup 256-bin histogram, scan of `groups x 256` (a `scan` call), stable scatter using per-workgroup local ranking in workgroup memory (the GraphWaGu / Fuchsia structure, note 03 section 2.2, re-derived, MIT); `bits` limits passes (grid cell keys of 18 bits need 3 passes); scratch `2 x (keys + values)` from the pool | O(count x passes), ~4 dispatches per pass | `Array.sort` with a stable comparator |
| 7 | `Frontier` | `class Frontier { readonly vertices: [Binding, Binding]; readonly count: [Binding, Binding]; readonly args: Binding; swap(); reset(batch, seed: number[]) }` | 2 x n-slot vertex queues, 2 x 4-byte counters (`atomic<u32>`), one `INDIRECT` args buffer with k slots; `finalizeArgs` kernel (section 5.4) | O(1) per round + the advance | JS arrays |
| 8 | `advance` | `advance(batch, graph, frontier, functor: { visit: snippet, filter: snippet }, tiers?)` | Gunrock `block_mapped` (note 04 section 2.5): each workgroup loads 256 frontier vertices, scans their degrees in workgroup memory, then every invocation strips `[local, aggregate)` with a binary search (`upper_bound`) over the scanned degrees to find its source; a `WORKGROUP_PER_ROW` tier for rows above 1,024 arcs (from `degreeOrder().segmentOffsets`, or a per-frontier degree check when the frontier is small); subgroup tier only with the feature; output appended with one `atomicAdd` per workgroup on the queue counter (workgroup-granular allocation, note 04 section 1 table); fused expand-contract variant for tiny frontiers (< 4,096 entries) | O(frontier degree sum) | edge loop |
| 9 | `spmv` (pull) | `spmvPull(batch, rev: CoreBinding, x: Binding, y: Binding, normaliser?: Binding, alpha, beta, tiers)` | `segmentedReduce` specialised: `y[v] = beta + alpha * sum_{u in in(v)} w * x[u] / norm[u]`; `Kahan` compensation in the workgroup-per-row tier for hubs (cheap, note 04 section 5) | O(A) | f64 loop |
| 10 | `cooToCsr` | `cooToCsr(batch, src, dst, w?, count, n) -> { rowPtr, colIdx, weights }` on the device | histogram by `src`, scan, scatter with cursors; rows NOT sorted by target (the format's I4 does not apply to device-internal graphs; a per-row sort is added only where an algorithm needs it -- Louvain's contraction sorts by (community(src), community(dst)) first with `radixSort` so rows come out sorted) | O(count + n) | `fromEdgeArrays` then compare |
| 11 | `bbox` | `bbox(batch, positions, n, dim, out: vec4f min / max)` | `reduce` with `op: "min"` / `"max"` over `vec4f`-packed positions (two reductions, deterministic) -- preferred over GraphWaGu's i32 fixed-point `atomicMin/Max` (note 03 section 2.2 item 6), which is kept as the documented cheaper alternative | O(n), 4 dispatches | `Math.min/max` loop |
| 12 | grid build | `buildGrid(batch, positions, mass, n, dim, spec: GridSpec) -> GridBinding` | cell id from `bbox` (`floor((p - min) / cellSize)` per axis, clamped), stable `radixSort` by cell id with node index as value (3 passes; bitwise deterministic within a cell) OR `countingSortByKey` (`deterministic: false`, faster), `cellStart` from the sort (a `mark boundaries + scan` kernel pair), per-cell `[sum m*x, sum m*y, sum m*z, sum m]` as `array<vec4f>` and `count` as `array<u32>` by a segmented reduce over the sorted ranges (no atomics), then one `downsample` dispatch per coarser level (each parent sums 4 / 8 children) | O(n log-ish + cells x levels), ~12 + levels dispatches | JS grid |

Subgroup variants (D16): reduce, scan, segmentedReduce and advance have a
`needs: ["subgroups"]` variant using `subgroupAdd`, `subgroupExclusiveAdd`,
`subgroupBallot`, `subgroupBroadcast` with `SUBGROUP_SIZE` from
`adapter.info.subgroupMinSize`; CI runs them at sizes 4 (SwiftShader), 8
(lavapipe) and 32 (NVIDIA) (note 05 section 3.2). The non-subgroup variant
is always compiled and tested too (`GRAPHTY_GPU_NO_SUBGROUPS=1` in the test
matrix forces it).

Determinism policy: reduce, scan, segmented reduce, radix sort, grid
centroids and the exact repulsion tile are bitwise reproducible on the same
device and dispatch shape (fixed tree order, no atomics on values). Counting
sort, frontier append order and BFS `order` are set-deterministic only,
which is documented per algorithm (section 8) and asserted by the tests as
set / partition equality.

---------------------------------------------------------------------------

## 7. Force-directed layouts -- FIRST DELIVERABLE

### 7.1 Scope and contract

The deliverable is `createForceAtlas2(ctx, options)` returning a
`GpuLayoutSimulation` (section 3.3) that:

1. implements design 14.3's `LayoutSimulation` over the element's stride-3
   scene-unit `Float32Array`, with the GPU buffer authoritative while
   stepping;
2. reproduces the ForceAtlas2 force model and adaptive speed controller of
   `@graphty/layout`'s `forceatlas2.ts` as it will exist after the L1
   rewrite (with the three reference decisions of 7.2), so the
   `forceatlas2` layout type gives the same family of pictures with or
   without a GPU;
3. is steppable, settle-reporting, pinnable and draggable the way
   `ngraph.forcelayout` is used by graphty-element today (note 01 section
   3), i.e. it can replace the DEFAULT engine at large n, not only the
   one-shot `forceatlas2Layout`;
4. scales from 10^2 nodes (parity with the CPU) through 10^4 (exact
   repulsion, interactive) to 10^5-10^6 (approximate repulsion; interactive
   at 10^5, batch at 10^6);
5. runs identically in Node (batch: `run()` until settled) and in the
   browser (per frame: `step(stepMultiplier)`).

Fruchterman-Reingold (`createFruchtermanReingold`) ships in the same phase
because it is the same kernel family with a simpler force law and a
host-side temperature (7.19).

### 7.2 Reference semantics (decision D5)

The CPU port (`layout/src/layouts/force-directed/forceatlas2.ts`) is a
transcription of NetworkX `forceatlas2_layout` with three measurable
deviations from the published algorithm (note 01 section 2.1.9). The GPU
kernel and the L1 CPU rewrite adopt ONE reference -- the published
ForceAtlas2 (Jacomy et al. 2014) as implemented by Gephi's
`ForceAtlas2.java` / `ForceFactory.java` and by cuGraph's
`fa2_kernels.cuh` (note 03 sections 4.3 and 5) -- with these settlements:

| Quantity | Port today | Reference adopted | Why |
| --- | --- | --- | --- |
| Repulsion magnitude | `k m_i m_j / d^2` (lines 322-329) | `k m_i m_j / d` (paper; Gephi `factor = coef * m1 * m2 / d / d` applied to the component vector) | the published law; Gephi / NetworkX / cuGraph users expect it; the port's law decays one power faster and collapses hubs |
| Distance floor | `max(d, 0.01)` | keep `max(d, 0.01)` plus a softening `d^2 + eps^2` with `eps = 0.05` in the approximate tier only (cuGraph `epssq`) | harmless; prevents the near-field blow-up in cells |
| Swing / traction per node | `swing_i = m_i |update_i|`, `traction_i = 0.5 m_i |2 p_i + update_i|` from positions (lines 379-390) | `swing_i = m_i |F_i(t) - F_i(t-1)|`, `traction_i = 0.5 m_i |F_i(t) + F_i(t-1)|` (paper; Gephi; cuGraph `compute_local_speed`) | the port's traction mixes positions and forces (a transcription artefact); the force form is what the speed controller was designed for |
| Global swing / traction | summed fresh every iteration (port) versus accumulated from 1 across iterations (NetworkX) | fresh every iteration (port = Gephi = cuGraph) | NetworkX's accumulation is the outlier |
| `estimateFactor` (jitter, speed efficiency, `speed += min(target - speed, 0.5 speed)`) | port lines 184-230 | unchanged (it is Gephi's `ForceAtlas2.java` lines 296-328 line for line, note 03 section 4.3 step 9) | |
| Local speed / apply | `factor = speed / (1 + sqrt(speed * swing_i))`, `p += F * factor` (lines 403-428) | unchanged | |
| `adjustSizes` correction | `d - (size_i - size_j)` (line 318) | `d - size_i - size_j` (symmetric; paper) -- DEFERRED to a later slice, not in v1 | rarely used; sign bug in the port |
| Gravity centre | centroid of positions (port lines 335-364) | centroid (port), not the origin (Gephi) | with drag and pins a centroid pull is what the element wants; documented |
| Mass | `degree + 1` via an O(n m) scan | `outDegree()[i] + 1` (design 14.3 line 4015; cuGraph `barnes_hut.cuh` lines 176-187) | |
| Initial positions | LCG in `[-1, 1)` per axis (port lines 57-65; `utils/random.ts` `m = 2^35 - 31, a = 185852, c = 1`) | the same LCG on the CPU in index order for NaN rows (note 01 section 2.5.1) so a seed gives the same start on both paths | |
| Termination | `totalMovement < 1e-10` (never fires) or `maxIter` | `maxIter` OR the settle rule of 7.16 | |

These are documented behaviour changes for the CPU FA2 too; the L1
Chromatic re-baseline commit already planned by design 14.3 lines 4041-4046
carries them. The GPU kernel takes them as constants, never rediscovers
them (note 01 section 2.1.9 recommendation). Open question 14.3 lets the
owner veto individual rows.

### 7.3 Buffers

All per-node vectors are stored as `array<f32>` with `3 * i + k` indexing
(design C14 / 10.2), even in 2D, so one buffer layout serves both
dimensions and the element's stride-3 array copies in with one `set`.

| Buffer | Bytes | Usage | Source / owner | Notes |
| --- | --- | --- | --- | --- |
| `rowPtr`, `colIdx`, `weights` (or dummy) | `4(n+1) + 4A (+4A)` | storage read | `ctx.residency.core(snapshot)` (undirected snapshot: both arcs present, design 10.5) | shared with every other kernel on the snapshot; freed by `release(snapshot)` |
| `perm` (degree tiers) | `4n` | storage read | `residency.view(snapshot, "degreeOrder")` when `flags`-derived skew warrants it (max degree > 1,024) else `USE_PERM = false` | |
| `positions` (layout units) | `12n` | storage read_write, COPY_SRC/DST | the simulation; seeded from the owner's array at `load()` | authoritative while stepping |
| `scenePositions` | `12n` | storage write, COPY_SRC | the simulation | `toScene` kernel output = `positions * scale + center`; what the staging ring copies |
| `force` | `12n` | storage read_write | the simulation | attraction + repulsion + gravity |
| `oldForce` | `12n` | storage read_write | the simulation | `F(t-1)` for swing / traction (7.2) |
| `mass` | `4n` | storage read | `outDegree()[i] + 1` computed on the CPU at `load()` (or `nodeMass`) | uploaded once per load |
| `fixed` mask | `4 ceil(n/32)` | storage read | `setFixed(mask)` (the `NodeMask` bitmap, LSB-first, `packages/graph-format/src/util/mask.ts`) | re-uploaded when dirty |
| `partials` | `groups x 64` | storage read_write | the simulation | per-workgroup: sum p (vec4), min (vec4), max (vec4), swing / traction (vec2), displacement (f32), padded |
| `state` | 128 | storage read_write, COPY_SRC | the simulation | `speed, speedEfficiency, swing, traction, centroid xyz, radius, meanDisplacement, iteration, settledCount` written by the finalize kernels; read by every kernel; copied to staging each batch |
| `params` uniform ring | `256 x k` per batch | uniform (dynamic offset) | `UniformRing` | `n, dim, flags, scalingRatio, gravity, jitterTolerance, scale, center, iterationIndex, seed` |
| grid tier only: `cellId` | `4n` | storage | the simulation | |
| grid tier only: sort scratch | `2 x (4n + 4n)` | storage | pool lease per batch | keys + values ping-pong |
| grid tier only: `sortedIdx`, `cellStart`, `cellCount` | `4n + 4(cells+1) + 4 cells` | storage | the simulation | |
| grid tier only: pyramid | `16 x sum(cells per level) + 4 x cells` | storage | the simulation | `[sum m*x, sum m*y, sum m*z, sum m]` per cell per level; count at the finest level; 5.6 MB at 512^2 (2D), ~38 MB at 128^3 (3D) |
| staging ring | `3 x 12n` (+ state) | MAP_READ, COPY_DST | `Readback` | one per in-flight batch |

Per-node total: exact tier `12 + 12 + 12 + 12 + 4 + 0.125 + partials` = ~53
B/node; grid tier ~70 B/node plus the fixed pyramid (note 03 section 8.2
estimates 40-64 B; ours carries `scenePositions` and `oldForce`).

### 7.4 Per-iteration kernel sequence

One `CommandBatch` records `k` iterations (the `iterations` argument of
`step`, the element's `stepMultiplier`) as `k` repetitions of the sequence
below, followed by `toScene` + the staging copies. Everything stays on the
device; the host sees one `mapAsync` per batch.

Exact tier (n <= `exactMaxNodes`):

| # | Kernel (`src/wgsl/`) | Dispatch | Reads | Writes | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | `fa2-stats-reduce` | `ceil(n / WG)` | positions, oldForce/force (previous iteration's displacement) | partials (sum p, min, max, displacement) | one pass over positions |
| 2 | `fa2-stats-finalize` | 1 workgroup | partials | state.centroid, state.radius, state.meanDisplacement, state.settledCount | grid-stride over <= 65,535 partials |
| 3 | `fa2-attraction` | 1-3 dispatches (tiers) | rowPtr, colIdx, weights, positions, mass, perm | force | 7.5 |
| 4 | `fa2-repulsion-exact` | `ceil(n / WG)` | positions, mass, state.centroid, force | force (+= repulsion + gravity), partials (swing, traction) | 7.6; gravity and the per-node swing / traction reduction are fused in the epilogue |
| 5 | `fa2-speed-finalize` | 1 workgroup | partials, state | state.swing, state.traction, state.speed, state.speedEfficiency | 7.9 |
| 6 | `fa2-integrate` | `ceil(n / WG)` | force, oldForce, mass, fixed, state | positions, oldForce | 7.10 |

Six dispatches per iteration; at `k = 4` a batch is 24 dispatches + 1
`toScene` + 2 copies.

Grid tier (n > `exactMaxNodes`): kernel 4 is replaced by the sequence of
7.7 (bbox from step 2's state; `cellId`; stable sort; `cellStart`;
`centroid`; `downsample` x levels; `farField`; `nearField` with the fused
gravity / swing / traction epilogue), i.e. `~12 + levels` dispatches per
iteration (2D: up to 8 levels; 3D: up to 6).

### 7.5 Attraction over CSR rows

Per node gather over the undirected snapshot (both arcs present, so the
sum is symmetric with no atomics; cosmos's two in/out passes and cuGraph's
four float `atomicAdd`s become one loop, note 03 section 1.6 / 4.5):

```wgsl
// fa2-attraction.wgsl.ts (core of the thread-per-row tier)
override LINLOG: bool = false;
override DISTRIBUTED: bool = false;
fn load_pos(i: u32) -> vec3f { return vec3f(pos[3u*i], pos[3u*i+1u], pos[3u*i+2u]); }

@compute @workgroup_size(WG)
fn attraction(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = linear_id(wid, lid.x) + P.tierStart;
    if (row >= P.tierEnd) { return; }
    let i = select(row, perm[row], USE_PERM);
    let pi = load_pos(i);
    var f = vec3f(0.0);
    for (var a = rowPtr[i]; a < rowPtr[i + 1u]; a = a + 1u) {
        let j = colIdx[a];
        if (j == i) { continue; }                              // a self-loop exerts no force (one arc, design 10.5)
        let w = select(1.0, weights[a], HAS_WEIGHTS);          // weights === null -> 1.0; the slot holds colIdx, never read
        let d = pi - load_pos(j);
        let len = max(length(d), 0.01);
        let mag = select(w, w * log(1.0 + len) / len, LINLOG); // linear: F = -d * w  (|F| = w * len); linlog: |F| = w * log(1 + len)
        f = f - d * mag;
    }
    if (DISTRIBUTED) { f = f / mass[i]; }
    store_force(i, f);                                         // overwrites: attraction is the first writer of `force` each iteration
}
```

Load balancing: the same module compiles to three tiers by `override
TIER` (thread-per-row, subgroup-per-row, workgroup-per-row) exactly as the
`segmentedReduce` primitive (section 6 row 3), with `P.tierStart /
tierEnd` from `degreeOrder().segmentOffsets` read on the CPU. The first
slice (P2) ships the thread-per-row tier with `USE_PERM = false` and adds
the tiers in P3 with the 10k-degree-hub fixture (note 03 section 8.6);
the API takes the permutation from day one so no signature changes.

Weights: `weight === true` binds `snapshot.weights` (`HAS_WEIGHTS =
flags.weighted`); `weight === "<edge column>"` binds the per-arc array the
caller expanded with `expandEdges` (graphty-element caches it per `(serial,
column, version)`, design 14.4 line 4118-4120); `weight === null` -> ones.
Parallel arcs sum (design 14.3 line 4037 documents the change).

### 7.6 Repulsion, exact tier: tiled all-pairs

```wgsl
// fa2-repulsion-exact.wgsl.ts (core)
var<workgroup> tile: array<vec4f, WG>;          // xyz + mass, 4 KiB at WG = 256 (16 KiB default limit)

@compute @workgroup_size(WG)
fn repulsion(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = linear_id(wid, lid.x);
    let valid = i < P.n;
    let pi = select(vec3f(0.0), load_pos(i), valid);
    let mi = select(0.0, mass[i], valid);
    var f = vec3f(0.0);
    let tiles = (P.n + WG - 1u) / WG;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let j = t * WG + lid.x;
        tile[lid.x] = select(vec4f(0.0), vec4f(load_pos(j), mass[j]), j < P.n);
        workgroupBarrier();
        for (var s = 0u; s < WG; s = s + 1u) {
            let jj = t * WG + s;
            if (jj < P.n && jj != i) {
                let d = pi - tile[s].xyz;
                let len = max(length(d), 0.01);
                f = f + d * (P.scalingRatio * mi * tile[s].w / (len * len));   // |F| = k m_i m_j / len, along d / len (7.2)
            }
        }
        workgroupBarrier();
    }
    if (valid) { epilogue(i, pi, mi, f); }      // gravity, force += , swing / traction partials (7.8, 7.9)
}
```

Properties: no atomics, deterministic (fixed summation order), `O(n^2)`
pair evaluations, memory-free, 2D and 3D identical (z is 0 for 2D and
never integrated). This kernel is also the ORACLE for the approximate tier
(7.7) and the differential target for the CPU FA2 (11.3). The measured
probe of this exact shape (3-component positions, 256-tile) ran 1.11
ms/iteration at 20k nodes on the RTX 4070 SUPER under Dawn-node
(`tmp/webgpu-plan/probe/dawn-perf.mjs`, note 05 section 2.5) -- 3.6e11
pair evaluations per second -- and 388 ms on lavapipe.

### 7.7 Repulsion, approximate tier: cell-sorted grid pyramid (option A) and cluster tree (option B)

Every large-scale GPU force layout approximates repulsion (note 03 section
0): cuGraph / Brinkmann with Burtscher's locked quadtree (not portable:
CAS spin-locks and cross-block spin-waits, note 03 section 4.5 / 8.1),
GraphWaGu with a Hilbert-sorted 4-ary cluster tree (portable; 95k nodes /
6.6M edges in 5.48 ms/iter, 1.13M nodes in ~160 ms/iter on an RTX 4070
Laptop at theta 2; note 03 section 2.3), cosmos.gl with a grid pyramid
plus a Monte-Carlo near field (portable; 100k in 6.6 ms/step, 200k in
13.8 ms/step in WebGL; note 03 section 1.3). Decision D7 takes the grid
pyramid as the primary back-end (note 03 section 8.3: every build stage is
a primitive the package needs anyway, no locks, fixed traversal loops,
production failure modes documented, extends to 3D) and keeps the cluster
tree as the documented experiment for pathological distributions.

Grid specification (`GridSpec`, computed on the CPU from `n`, `dim` and
caps):

- finest resolution per axis `G = clamp(nextPow2(2 * n^(1/dim)), 8, dim == 2
  ? 512 : 128)` (cosmos: `2 sqrt(n)` capped at 512; 3D capped at 128^3 =
  2.1M cells ~ 34 MB, note 03 section 8.2);
- levels `G, G/2, ..., 4`; cell size from the bounding box of the current
  iteration (`state.min / max` with a 1% margin), never a fixed space
  (cosmos clamps to a fixed `spaceSize`; we do not clamp scene positions).

Kernels per iteration (replacing 7.4 row 4):

| # | Kernel | Dispatch | What |
| --- | --- | --- | --- |
| 4a | `grid-cell-id` | `ceil(n / WG)` | `cell[i] = linearise(clamp(floor((p - min) / cellSize)))`; key = cell id (18 bits in 2D at 512^2, 21 bits in 3D at 128^3); value = i |
| 4b | `radixSort` (3 passes) | ~12 | stable sort by cell id -> `sortedIdx`; within a cell nodes are in index order (deterministic); `countingSortByKey` when `deterministic: false` (fewer dispatches, set-deterministic) |
| 4c | `grid-cell-start` | `ceil(n / WG)` + scan | mark cell boundaries in the sorted keys, scan -> `cellStart[cells + 1]`, `cellCount[cells]` |
| 4d | `grid-centroid` | `ceil(cells / WG)` | per cell: loop `sortedIdx[cellStart[c] .. cellStart[c+1])` summing `mass * p` and `mass` -> `level0[c] = vec4f(sum m*x, sum m*y, sum m*z, sum m)`; bounded by occupancy (segmented reduce shape; the workgroup-per-cell tier handles hub cells above 1,024 entries) |
| 4e | `grid-downsample` x (levels - 1) | `ceil(cells_L / WG)` each | `parent = sum of 4 (8) children` -- one dispatch per level, no atomics |
| 4f | `grid-far-field` | `ceil(n / WG)` | per node: at the coarsest level sum over every cell except the 3x3 (3x3x3) neighbourhood; at each finer level sum the 6x6 (6x6x6) block aligned to the parent's 3x3 minus this level's own 3x3 (3x3x3): space tiled exactly once, no theta (cosmos `force-level.frag` lines 60-96); per cell `F += d * (k * m_i * M_cell / (|d|^2 + eps^2))` with `d = p_i - centroid_cell`; 2D: `7 + 27 (levels-1)` ~ 196 evaluations per node at 512^2; 3D: `37 + 189 (levels-1)` ~ 982 at 128^3 |
| 4g | `grid-near-field` | `ceil(n / WG)` | per node: for each of the 9 (27) finest cells, iterate `sortedIdx[cellStart .. cellStart + count)` with the EXACT pair force, bounded by `NEAR_MAX = 64` entries per cell; above the cap take entries `[h, h + NEAR_MAX)` with a per-iteration hashed offset `h` (lowbias32 of `(cell, iteration, seed)`) and scale the cell's sum by `count / NEAR_MAX` (Horvitz-Thompson, cosmos `force-nearfield.frag` line 139; unbiased, no depth peeling because the sorted range is indexable); coincident points get a deterministic kick from an integer hash (cosmos found `sin()` hashes diverge across GPU vendors, note 03 section 8.4); the per-iteration near-field step is clamped to `2 * cellSize`; fused epilogue (gravity, swing / traction partials) |

Cost model (note 03 section 8.2; unverified until P3 measures): far field
is `O(n x evaluations)` with coherent, branch-free access (`2 x 10^8` at
1M nodes in 2D, `10^9` in 3D); near field is `O(n x mean occupancy)` with
hub cells as the tail, capped by `NEAR_MAX`; build is a 3-pass sort plus
`O(cells x levels)`. Expected on the 4070 SUPER: 100k nodes 3-8 ms/iter
(2D), 1M nodes 30-80 ms/iter (2D), 3D 1.5-2x -- bracketed by cosmos's
WebGL numbers below and GraphWaGu's WebGPU tree numbers above.

Option B, Hilbert-sorted cluster tree (GraphWaGu 2025; note 03 section
2.2): reuses `radixSort` and the level-wise `downsample` shape (`log_4 n`
merge dispatches, one per level), traversal with a private 64-entry stack
and `theta`. Reserved for the case the P3 hub-heavy fixtures show the
grid's near field degrading (very clumpy layouts); shares every primitive,
adds ~110 B/node. Not built unless measurements demand it.

Crossover (`repulsion: "exact" | "grid" | "auto"`, `exactMaxNodes`
default 16,384): the exact tile at 16k is ~0.7-1.7 ms/iter on the 4070 by
the measured 3.6e11 pairs/s (and Burtscher: O(n^2) fastest below ~10k
bodies in 2009; GraphWaGu 2022: FR bitmap best below ~5k on an RTX 2060;
cosmos: 4,096 because each WebGL peel pass costs ~0.1 ms; note 03 section
8.2), while the grid tier's fixed build cost is of the same order; on
integrated GPUs (5-10x slower on the exact tile) the crossover moves to
~8k. `"auto"` picks by `n` only (never by `caps.software`); the default is
re-fixed from the P3 benchmark on the dev GPU and one integrated GPU.

### 7.8 Gravity and the centroid

`q = p_i - state.centroid` (centroid from kernel 2, the unweighted mean of
positions as in the port); regular gravity `F += -gravity * m_i * q / |q|`
when `|q| > 0.01`, strong gravity `F += -gravity * m_i * q`
(`STRONG_GRAVITY` override). `gravity = 0` is accepted (the element schema
forbids it, the CPU accepts it; note 01 section 7.1). Fused into the
repulsion epilogue (exact tier) or the near-field epilogue (grid tier).

### 7.9 Swing, traction and the global speed: on the device

The epilogue of the repulsion / near-field kernel computes per node
`swing_i = m_i |F_i - Fold_i|` and `traction_i = 0.5 m_i |F_i + Fold_i|`
(7.2), reduces them over the workgroup (256 -> 1, subgroup variant when
available) and writes `partials[group].st = vec2f(swing, traction)`.
`fa2-speed-finalize` (one workgroup) sums the partials and runs
`estimateFactor` -- ~20 scalar operations, a line-for-line port of the CPU
`estimateFactor` (`forceatlas2.ts` lines 184-230):

```wgsl
// fa2-speed-finalize.wgsl.ts (single workgroup; partial reduction elided)
if (lid.x == 0u) {
    let n = f32(P.n);
    let optJitter = 0.05 * sqrt(n);
    let minJitter = sqrt(optJitter);
    let maxJitter = 10.0;
    let other = min(maxJitter, optJitter * traction / (n * n));
    var jitter = P.jitterTolerance * max(minJitter, other);
    var eff = S.speedEfficiency;
    if (swing / traction > 2.0) { eff = max(eff * 0.5, 0.05); jitter = max(jitter, P.jitterTolerance); }
    let target = select(jitter * eff * traction / swing, 1e30, swing == 0.0);
    if (swing > jitter * traction) { eff = max(eff * 0.7, 0.05); } else if (S.speed < 1000.0) { eff = eff * 1.3; }
    S.speed = S.speed + min(target - S.speed, 0.5 * S.speed);
    S.speedEfficiency = eff; S.swing = swing; S.traction = traction;
}
```

cuGraph brings these two sums to the host with `thrust::reduce` per
iteration (note 03 section 4.3 step 9); with WebGPU's asynchronous
submission that would cost a `mapAsync` per iteration, so the controller
stays on the device (D15). The `state` block is copied to staging with
every batch so `stats` (and the differential test of 11.3) can read the
per-iteration trace: the batch keeps a `traceSlot` of `k x 16` bytes that
the finalize kernel appends to.

### 7.10 Position update

```wgsl
// fa2-integrate.wgsl.ts (core)
let i = linear_id(wid, lid.x); if (i >= P.n) { return; }
let f = load_force(i);
let swing_i = mass[i] * length(f - load_old(i));
let factor = S.speed / (1.0 + sqrt(S.speed * swing_i));
let fixed = ((fixedMask[i >> 5u] >> (i & 31u)) & 1u) == 1u;
var p = load_pos(i);
var dp = select(f * factor, vec3f(0.0), fixed);
if (P.dim == 2u) { dp.z = 0.0; }                       // 2D never touches z (note 01 section 4.6)
store_pos(i, p + dp);
store_old(i, f);
partial_displacement(length(dp));                      // workgroup reduce -> partials; excluded from the mean when fixed
```

Fixed nodes still exert forces (they are in every gather and tile) and are
simply not moved; their `oldForce` is still updated so a later unpin does
not see a stale swing. `adjustSizes`'s `0.1 * speed` factor and the
10-unit cap are the deferred variant (7.2).

### 7.11 Fixed nodes and drag

`setFixed(mask)`: validates `mask.length >= ceil(n / 32)` (`E_INVALID_
ARGUMENT`), copies into the simulation's own words, marks the buffer dirty
(re-uploaded by `writeBuffer` before the next submit, `4 ceil(n/32)`
bytes), and reheats if any bit went from 1 to 0 (an unpin, like d3's
`unpin`, note 01 section 3.2).

`setPosition(i, x, y, z)` (scene units): validates `i < n`; writes the
three floats into the OWNER's array immediately (the renderer reads it
this frame); converts to layout units and `queue.writeBuffer(positions,
12 * i, ...)` immediately (queue order places it before the next submit,
after every batch already submitted); records `{ i, afterBatch:
lastSubmittedBatchId }` in an override list; reheats. When a batch's
readback resolves, rows with an override whose `afterBatch >= batchId`
are NOT copied from the readback (the batch was computed before the write
and would move the node back); the override is cleared once a batch
submitted after the write completes. During a drag graphty-element's
`NodeBehavior.onDragUpdate` calls `setNodePosition` on every pointer move
(note 01 section 4.5); the element bridge (9.4) additionally marks the
dragged node fixed for the duration of the drag (a temporary bit in the
mask) so the integrate kernel does not fight the pointer; `pinOnDrag`
decides whether the bit stays at drag end. Neighbours lag by one batch, as
they do with ngraph's one-step-per-frame today.

### 7.12 2D and 3D

`dim` is a uniform; positions are stride 3 in both. 2D: z is uploaded as
0, never integrated (7.10), and the grid is 2D (`G^2` cells, 3x3 / 6x6
loops). 3D: full vector maths, `G^3` cells capped at 128 per axis, 27 /
216 loops. graphty-element re-creates the engine on a view-mode switch
(`LayoutManager.updateLayoutDimension`, note 01 section 4.6), so the
simulation never changes `dim` after `load()`; `setParams({ dim })` is
rejected (`E_INVALID_ARGUMENT`).

### 7.13 Parameter parity with the CPU forceatlas2.ts and the element schema

| Option | CPU default (`forceatlas2.ts` lines 26-42) | Element schema (`ForceAtlas2LayoutEngine.ts` lines 99-115) | GPU v1 | Binding |
| --- | --- | --- | --- | --- |
| `maxIter` | 100 | int > 0, default 100 | honoured: total iteration budget across `step` calls (7.16) | host counter |
| `jitterTolerance` | 1.0 | > 0, default 1.0 | honoured | uniform |
| `scalingRatio` | 2.0 | > 0, default 2.0 | honoured | uniform |
| `gravity` | 1.0 | > 0, default 1.0 | honoured; 0 accepted | uniform |
| `strongGravity` | false | bool | honoured | `override STRONG_GRAVITY` |
| `distributedAction` | false | bool | honoured | `override DISTRIBUTED` |
| `linlog` | false | bool | honoured | `override LINLOG` |
| `nodeMass` | null -> degree + 1 | `Record \| null` | honoured: `Float32Array(n)`, a node column name (`gpuView`), or a `Record` resolved by the layout package's `resolveNodeVector` (design 14.3 line 4009) | `mass` buffer |
| `weight` | null | `weightPath: string \| null` (inert today, note 01 section 7.1) | `true` (snapshot weights), a column name (expanded per-arc array supplied by the caller), or `null`; BECOMES LIVE (documented change) | `weights` binding |
| `seed` | null | `number \| null` | honoured for NaN rows via the CPU LCG; `0` means unseeded (LCG quirk preserved) | CPU |
| `dim` | 2 | 2..3 | honoured; fixed at `load()` | uniform |
| `pos` | null | `Record \| null` | replaced by the owner's array: finite rows are kept (inverse-scaled), NaN rows seeded (design 14.4 line 4075-4077) | `positions` |
| `nodeSize` / `adjustSizes` | null | `Record \| null` | DEFERRED: throws `E_UNSUPPORTED` when set (sign-suspect on the CPU, 7.2) | -- |
| `dissuadeHubs` | ignored (`_dissuadeHubs`) | bool (schema lines 66-73) | accepted and ignored, exactly like the CPU | -- |
| `scale`, `center` (`CommonLayoutOptions`) | n/a (always unit ball) | `scalingFactor` 100 | applied at `load()` (inverse) and in `toScene` (7.17); never per-step rescaling | uniform |
| new: `repulsion`, `exactMaxNodes`, `nearMax`, `deterministic` | -- | -- | GPU-only tuning (7.7) | overrides / uniform |
| new: `settleThreshold`, `settleWindow` | -- | -- | 7.16 | host |
| new: `iterationsPerStep` (default = `stepMultiplier`), `maxInFlight` (2) | -- | -- | 7.18 | host |

### 7.14 The float-atomics workaround, summarised

WGSL has `atomic<u32>` / `atomic<i32>` only (note 05 section 6 item 1).
Every accumulation in this layout is therefore a GATHER owned by one
invocation (attraction over CSR rows, repulsion over tiles or cells,
near-field over sorted ranges) or a tree REDUCTION (centroid, bbox, swing,
traction, displacement, grid centroids, downsampling). No kernel scatters
a float. The only atomics are `u32` counters in the optional counting-sort
path and the radix sort's histograms. cuGraph's edge-parallel
`atomicAdd(float)` attraction, Burtscher's spin-wait summarisation and
d3-force-webgpu's racy link kernel are explicitly NOT ported (note 03
section 8.1, note 04 section 1 table).

### 7.15 Determinism

Given the same snapshot, seed, options, device and dispatch shape, the
exact tier is bitwise reproducible (fixed loop orders, tree reductions).
The grid tier is bitwise reproducible with `deterministic: true` (stable
radix sort; near-field hash seeded from `(seed, iteration)`) and
set-deterministic with `deterministic: false` (counting sort; the
`NEAR_MAX` subset of an over-full cell may differ). Across GPUs (different
`SUBGROUP_SIZE`, different fma contraction) coordinates differ at f32
noise level; parity tests compare distributions and the swing / traction
/ speed trace, never coordinates (note 01 section 8.7).

### 7.16 Settlement

`settled` is `iterationsDone >= maxIter` OR `settledCount >= settleWindow`,
where the finalize kernel increments `settledCount` when
`meanDisplacement <= settleThreshold * radius` (`radius` = max
`|p - centroid|` from the bbox; `meanDisplacement` over non-fixed nodes)
and resets it otherwise. Defaults: `settleThreshold = 1e-3`, `settleWindow
= 10` (ngraph's `0.01` per body and the element's 10-step average of
`0.05` in scene units are the models, note 01 section 3.1, made
scale-relative because layout units are not scene units). `setPosition`,
an unpin, `setParams` and `load` call `reheat()` (`iterationsDone = 0`,
`settledCount = 0`, `speed = 1`, `speedEfficiency = 1`), matching d3's
reheat on `setNodePosition` / `unpin` and ngraph's counter reset on
topology change (note 01 sections 3.1-3.2). The element must see `settled
=== true` within `maxIter` steps regardless of the threshold (screenshots
and label animations wait for it, note 01 section 4.3).

### 7.17 Units: layout units on the device, scene units in the array

FA2's forces are not scale-invariant (linear attraction vs `1/d`
repulsion), so the simulation runs in LAYOUT units (the CPU's `[-1, 1)`
seed scale) in `positions`, and the `toScene` kernel writes `scene = p *
scale + center` into `scenePositions` for the staging copy; `load()`
applies the inverse on the CPU to finite rows (`fromPositionColumn`
semantics, design 14.3 line 3972) and `setPosition` inverts its three
floats. No `rescaleLayout` per step (it would move pinned / dragged nodes
and change the camera framing every frame, note 01 section 8.4). The CPU
FA2's final unit-ball normalisation is therefore NOT reproduced by the
steppable engine; the element's `scalingFactor` (default 100) sets the
scene scale as it does for `SimpleLayoutEngine` today.

### 7.18 The LayoutSimulation contract in the element's frame loop

Facts: `UpdateManager.updateLayout()` calls `layoutManager.step()`
`stepMultiplier` times per render frame, synchronously, from Babylon's
render loop; nothing awaits (note 01 section 4.1). A Promise-returning
`step()` therefore needs the fire-and-forget bridge of decision D6:

State machine of `ForceSimulation` (`src/layouts/force-simulation.ts`):

```
created --load()--> loaded --step(k)--> loaded (inFlight 1..maxInFlight) --readback resolves--> loaded
loaded --dispose()--> disposed;   any --device lost--> disposed (pending promises reject E_DEVICE_LOST)
```

`step(k = iterationsPerStep)`:

1. `state !== "loaded"` -> reject `E_NOT_LOADED` / `E_DISPOSED`.
2. `settled` -> resolve immediately (no submission).
3. `inFlight >= maxInFlight` -> return the OLDEST pending promise (the
   call coalesces; the element's per-frame call is naturally throttled to
   the GPU's pace and never queues unbounded work).
4. Flush host writes: dirty mask -> `writeBuffer`; changed params -> the
   batch's uniform slots; `setPosition` writes already went to the queue.
5. Record `k` iterations + `toScene` + copies into a `CommandBatch`;
   submit; `inFlight++`; `iterationsSubmitted += k`.
6. Return a promise that awaits the batch's `mapAsync`, then: copy
   `scenePositions` into the owner's array with `set` (skipping overridden
   rows, 7.11), copy the state trace into `stats`, `iterationsDone += k`,
   update `settled`, `inFlight--`, `column.markDirty()` is the CALLER's
   job (the simulation does not know the column; the element bridge does
   it once per frame, design 14.4 M12).

The element bridge (section 9.4) calls `void sim.step(stepMultiplier)`
once per frame with a `.catch` that routes the error to the element's
error channel and stops the layout. With `maxInFlight = 2` the GPU always
has a batch queued while the CPU renders; positions the renderer draws lag
the simulation by one batch, invisible for a settling layout (note 01
section 5.3 / 8.5).

Node batch API: `await sim.run({ maxIter, batch: 8 })` loops `step(batch)`
until `settled` (or `maxIter`), returning the final `LayoutStats`. The same
kernels, no frame loop.

Readback cadence: one `12n`-byte copy per batch. At 100k nodes that is 1.2
MB: ~3 ms in Chromium (measured 2.65 ms for 1 MiB copy + map + slice, note
05 section 7.2) and well under 1 ms in Node; at 1M nodes 12 MB is ~30 ms
in Chromium -- longer than a frame -- so above ~250k nodes the element
should raise `iterationsPerStep` (fewer readbacks per iteration) and
accept a lower position refresh rate; sharing a device with a future
Babylon `WebGPUEngine` (`engine._device`, note 01 section 4.8) removes the
readback entirely and is the documented follow-up.

Topology change: `reload(undirected, report, positions)` (design 14.4
M3/M4/M5) maps to `load(next, positions)` on the same simulation: the
element has already remapped / grown the array (finite rows kept, new rows
NaN), the simulation re-uploads the new core (the old one is released by
the element's `snapshot-replaced` listener), resizes its scratch, seeds
NaN rows with the LCG (or, as a product option, at the neighbours'
centroid -- left to the element), and reheats.

### 7.19 Fruchterman-Reingold, the ngraph-like preset, ARF, Kamada-Kawai

Fruchterman-Reingold (`fruchterman-reingold.ts` lines 24-160, note 01
section 2.2) on the same skeleton: `k` default `1 / sqrt(n)`; initial
positions uniform `[0, 1)` per axis from ONE LCG (the single-RNG fix of
design 14.3 line 4045); repulsion `k^2 / d` (`d = |delta| || 0.1`: an
exact 0 becomes 0.1, otherwise unclamped -- replicated) through the same
exact tile / grid kernels with the force-law override `LAW = FR`
(magnitude `k^2 / len` = `d * k^2 / len^2`); attraction `d^2 / k` over
the CSR row (unweighted; the CPU ignores weights); temperature `t = 0.1`,
`dt = t / (iterations + 1)`, `t -= dt` per iteration written into the
uniform slots of the batch on the host (a batch knows its `k` temperatures
in advance); displacement `min(|disp|, t)` along the displacement
direction; `fixed` skipped; output NOT rescaled when `fixed` was given
(the steppable engine never rescales, 7.17). No swing / traction; kernels
1, 3, 4, 6 of 7.4 with a trivial integrate.

ngraph-like preset (note 02 section 7.1 row L3): graphty-element's default
engine is `ngraph.forcelayout` (`config/GraphBehavior.ts` line 13), a
spring-electrical simulation with Coulomb `1/d^2` repulsion, Hooke
springs, drag and a velocity integrator (note 01 section 3.1). The plan
reserves `forceLaw: "fa2" | "fr" | "coulomb"` on the repulsion kernels
(the grid's centroid approximation is valid for any radial law) and a
`velocityVerlet` integrate variant so a `"spring-electrical"` preset can
mimic ngraph's feel; whether the element routes `ngraph` to it above a
node-count threshold is a product decision for the element / app, not a
v1 kernel.

ARF (`arf.ts`, 2D only, note 01 section 2.3): the pair sum is an all-pairs
term plus a CSR-row correction, i.e. the exact tile with `LAW = ARF` plus
the attraction kernel with a constant; low priority, not scheduled.

Kamada-Kawai: not a steppable force simulation (L-BFGS line search needs a
cost readback per evaluation); its GPU pieces are the APSP primitive (A9,
section 8) and a dense cost / gradient kernel; scheduled after APSP,
bounded to n <= ~10k by the `n x n` distance matrix (note 01 section 2.4).

### 7.20 Scaling table

Basis: `E` = measured exact tile throughput 3.6e11 pairs/s on the RTX 4070
SUPER (`dawn-perf.mjs`, 1.11 ms at 20k); `G` = grid-tier extrapolation
bracketed by cosmos's WebGL measurements (100k: 6.6 ms, 200k: 13.8 ms) and
GraphWaGu's WebGPU tree (95k: 5.5 ms, 1.13M: 160 ms on an RTX 4070
Laptop); `A` = measured CSR gather on 100k / 1M arcs 0.1-0.7 ms/iter (note
06 section 3.5) scaled linearly in arcs; `R` = measured Chromium readback
2.65 ms per MiB; integrated GPU = 8x slower than the 4070 on compute
(GraphWaGu's Iris Xe vs RTX 4070 Laptop ratios and note 03's 5-10x), 2x
slower on readback. Edges = 5n (average degree 10, undirected doubled).
All numbers are per ITERATION unless stated; unverified until P2 / P3
measure them.

| Nodes / edges | Tier | Repulsion 4070 (E / G) | Attraction (A) | Total / iter 4070 | Total / iter integrated | Readback per batch (Chromium / Node) | Frames at 60 fps, `k = 1` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1k / 5k | exact | 0.03 ms (overhead-bound ~0.1) | < 0.05 ms | ~0.3 ms (6 dispatches) | ~1 ms | 12 KB: < 0.2 / < 0.1 ms | trivially 60 fps |
| 10k / 50k | exact | 0.3 ms | 0.05 ms | ~0.6 ms | ~3 ms | 120 KB: 0.4 / 0.1 ms | 60 fps with `k` up to 8 |
| 16k / 80k | exact (crossover) | 0.7 ms | 0.06 ms | ~1.0 ms | ~6 ms | 192 KB: 0.6 / 0.1 ms | 60 fps, `k` up to 4 |
| 65k / 325k | grid (2D) | 2-4 ms (G) | 0.2 ms | ~3-5 ms | ~20-40 ms | 780 KB: 2 / 0.3 ms | 60 fps at `k = 1-2`; integrated ~20 fps |
| 100k / 1M | grid (2D) | 3-8 ms (G) | 0.3-0.7 ms | ~4-9 ms | ~30-70 ms | 1.2 MB: 3 / 0.4 ms | 60 fps at `k = 1`; integrated 10-15 fps |
| 100k / 1M | exact (for comparison) | 28 ms (E) | 0.3-0.7 ms | ~29 ms | ~230 ms | same | 30 fps -- the reason for the grid tier |
| 1M / 5M | grid (2D) | 30-80 ms (G) | 2-4 ms | ~35-90 ms | ~300-700 ms | 12 MB: 30 / 3 ms | 10-25 fps in Node; browser readback-bound -> `k` >= 4; batch use |
| 1M / 5M | grid (3D) | 50-160 ms (G x 1.5-2) | 2-4 ms | ~55-165 ms | ~0.5-1.3 s | same | batch |
| 10M / 50M | grid (2D) | 0.3-0.8 s (G, cap 512^2 saturated: near field dominates) | 20-40 ms | ~0.4-0.9 s | not targeted | 120 MB: 0.3 / 0.03 s | Node batch only; 500 iterations ~5 min |

The CPU FA2 for comparison allocates `n x n x dim` per iteration and is
practical only to a few thousand nodes (note 01 section 2.1.6); the CPU
ngraph engine settles a 150-node story graph today (note 01 section 6).

---------------------------------------------------------------------------

## 8. Algorithms

Every algorithm is an async function `(ctx, snapshot, ...args, options?)`
returning index-aligned typed arrays (section 3.3), grouped by the
primitive family it needs. References are to note 04 (which read the
cuGraph, Gunrock, GAP sources and the Merrill / Beamer / Davidson /
McLaughlin-Bader papers) unless a URL is given. Result-shape parity with
`indexed.*` is in section 9.7; priority scores are note 02 section 7.2's
`value x speedup / risk`.

### 8.1 Family overview

| Family | Primitives | Algorithms | Views bound | Host loop |
| --- | --- | --- | --- | --- |
| SpMV / power iteration | `spmvPull`, `segmentedReduce`, `reduce` | PageRank, personalized PageRank, HITS, eigenvector, Katz | `reverse()` (aliases forward arrays when undirected), device out-weight sums, `degreeOrder({ of: "reverse" })`, `gpuView` personalization | k iterations per submit; convergence read every k |
| Edge-parallel (each edge once) | per-arc map, `compact`, `reduce`, `histogram` | WCC (Afforest), Bellman-Ford, Boruvka MST (later) | `edgeList().src/.dst/.weights` | fixed rounds + changed-flag every k |
| Frontier | `Frontier`, `advance`, `compact` / `dedupe`, `scan`, bitset | BFS, direction-optimizing BFS, SSSP (near-far), closeness, betweenness (multi-source), k-core peeling | `rowPtr`, `colIdx`, `weights`, `reverse()` (bottom-up on directed), `degreeOrder()` | k levels per submit with indirect dispatch |
| Sort / group-by | `radixSort`, `segmentedReduce`, `cooToCsr` | Louvain, label propagation (hub rows), triangle counting orientation | `rowPtr`, `colIdx`, `weights`, `edgeList()`, `outDegree()` | per level / pass readback of Q and move count |
| Dense | tiled matrix kernels | APSP / Floyd-Warshall (n <= ~8k at default limits) | `rowPtr`, `colIdx`, `weights` | one submit per block sweep |

### 8.2 SpMV family: PageRank first (A1, score 25)

Strategy (cuGraph `pagerank_impl.cuh` lines 222-320, note 04 section 5):
pull over `reverse()`; per iteration (a) `outWeightSum` computed ONCE by
`segmentedReduce` over forward `rowPtr` / `weights` (the device-side
normaliser design 10.1 line 2340 prescribes; guard `sum == 0`: a node with
out-arcs and zero weight sum is dangling), (b) dangling mass = `reduce` of
`rank[u]` where `outWeightSum[u] == 0`, (c) `spmvPull`: `rankOut[v] = (1 -
alpha) / n + alpha * (danglingMass / n + sum_{u in in(v)} w * rankIn[u] /
outWeightSum[u])` (+ personalization vector term when given), (d) L1 delta
= `reduce(|rankOut - rankIn|)`; ping-pong buffers; `k = 8` iterations per
submit, convergence read back every k (typical 20-60 iterations). Bindings
fit the default 8 storage buffers exactly (revRowPtr, revColIdx,
revWeights, outWeightSum, rankIn, rankOut, personalization-or-dummy,
partials; note 04 section 5). f32 accumulation with Kahan in the
workgroup-per-row tier; documented tolerance `1e-5` relative (design
16.2). HITS alternates two pulls (forward and reverse) with sum
normalisation; eigenvector adds an L2 normalise; Katz is `alpha * SpMV +
beta`. WebGPU adjustment: no push / float atomics (Gunrock's `pr.hxx` push
form is the counter-example).

### 8.3 Connected components (A3, score 8)

Afforest from GAP `gapbs/cc.cc` lines 40-150 (note 04 section 8): `comp[v]
= v`; 2 sampled link rounds over the r-th neighbour of every vertex
(`colIdx[rowPtr[v] + r]` when `r < degree`), compress (pointer jumping;
reads through `atomicLoad` on the same `array<atomic<u32>>` because WGSL
forbids mixing atomic and plain access), a 1,024-entry histogram readback
to find the giant component, then link the remaining edges of vertices not
in it (`edgeList()` each edge once, correct for directed and undirected
alike, design 10.1) until a device-side changed flag stays 0. All `u32`
CAS (`atomicCompareExchangeWeak`). Readback labels -> `renumberPartition`
(graph-format `src/snapshot/derived.ts` line 1155) on the CPU in
first-seen order so `groups()` is identical to the CPU's. cuGraph's
multi-root frontier expansion is more machinery for the same asymptotics
(note 04 section 8) and is not used.

### 8.4 Frontier family: BFS (A4), SSSP (A6), closeness (A5), betweenness (A7)

BFS (Merrill-Garland-Grimshaw 2011, note 04 section 3;
https://research.nvidia.com/publication/2011-08_high-performance-and-scalable-gpu-graph-traversal):
two-phase (expand into an edge frontier, contract into the next vertex
frontier) as the workhorse, a fused expand-contract kernel for frontiers
below 4,096 entries, `atomicCompareExchangeWeak(&depth[v], INVALID_INDEX,
level)` as the visit claim with `parent[v] = u` written by the winner,
Davidson's ownership dedupe as the exact safety net behind any workgroup
hash culling. Direction-optimizing variant (Beamer SC12 via cuGraph
`bfs_impl.cuh` lines 291-297, 637-638, 843-846): `alpha = m / n`, `beta =
24`, switch to bottom-up (over `reverse()`; the forward arrays when
undirected) when the frontier's degree sum exceeds the unvisited degree
estimate `/ alpha` and is growing, back when `next * 24 < unvisited` and
shrinking; bottom-up iterates the non-zero-degree unvisited list. Host
loop: 32 levels per submit with indirect args (5.4), one 4-byte readback of
the frontier length every 32 levels -- a per-level `mapAsync` on a road
network (europe.osm ~19,000 levels) would be slower than the CPU (note 04
section 3). Result parity: `depth` exact; `parent` any valid level-1
predecessor; `order` grouped by level (note 02 section 5).

SSSP (Davidson 2014 near-far, cuGraph `sssp_impl.cuh` lines 189-262, note
04 section 4): `dist` as `array<atomic<u32>>` holding f32 bit patterns
(`atomicMin` is exact for non-negative floats; `0x7F800000` = +Inf =
unreached), `delta = 32 * avgWeight / avgDegree`, near / far piles with the
ownership dedupe, a two-level near queue; requires `flags.nonNegativeWeights`
(else `E_UNSUPPORTED` pointing at `bellmanFord`); `flags.allWeightsOne` ->
runs BFS. `predArc` by a second pass `atomicMin(&pred[v], arc)` over the
settled frontier (the @antv/webgpu-graph `updatePred` idea, note 04 section
4) -- ties differ from the CPU. Bellman-Ford (A11): edge-parallel relax
over `edgeList()` both directions on undirected, `n - 1` rounds with a
changed flag every k, one more round for the negative-cycle flag; signed
floats need a CAS loop.

Closeness (A5): batched multi-source BFS (32 sources per `u32` word as a
bit-parallel frontier for unweighted graphs; repeated near-far for
weighted), per-source distance rows reduced on the device (sum, sum of
1/d, max) without materialising `n x n` (note 04 section 7).

Betweenness (A7; McLaughlin-Bader CACM 2018 from the author's mirror
https://davidbader.net/publication/2018-mb/2018-mb.pdf, note 04 section 6):
forward pass = BFS with `sigma` as `array<atomic<u32>>` (`atomicAdd`,
exact until 2^32 paths, with a saturation flag reported in the result) and
per-level `S` / `ends` ranges written by the compaction; backward pass =
per level from the deepest, each `w` PULLS over its successors `v` with
`depth[v] == depth[w] + 1`: `delta[w] += sigma[w] / sigma[v] * (1 +
delta[v])` -- "eliminate the use of atomics by checking successors" -- and
`bc[w] += delta[w]` is a plain add. Sources are batched (cuGraph's tagged
multi-source BFS with `n x k` sigma / depth arrays capped by
`maxBufferSize`, note 04 section 6); `options.sources` / `options.k`
(sampled) give the approximate variant graphty needs at 1M nodes (exact BC
is `O(n m)`); the McLaughlin-Bader online switch to the edge-parallel form
(median BFS depth `< gamma log2 n`) is implemented as a per-batch choice.
Edge betweenness writes per arc and folds with `foldArcs(s, vec, "first")`
(design 10.7), halved on undirected snapshots as both papers do.

### 8.5 Structure: k-core (A10), triangle counting / k-truss (A13)

k-core (cuGraph `core_number_impl.cuh`, note 04 section 9): counts start
at degree; rounds of "frontier of vertices with count < k" -> `atomicSub`
neighbour counts -> compaction; `k` increases when the frontier empties;
`O(max core)` host-visible rounds, batched 32 per submit. Triangle
counting: orient edges low-to-high degree (tie by id) as a compaction of
arcs, intersect sorted rows by merge (`flags.sortedRows` makes it a merge,
binary search into the longer list when degrees differ by > 32x), `u32`
atomic per-node counts, workgroup-per-arc tier for hub pairs. k-truss
peels edges with support `< k - 2` over an `EdgeMask`.

### 8.6 Community: label propagation (A8), Louvain (A14)

Label propagation: per node the (weighted) mode of neighbour labels via a
per-row group-by-key -- rows <= 256 arcs sort keys in workgroup memory,
larger rows use a global open-addressing hash region sized `2 x degree`
(nu-Louvain's layout, note 04 section 10) -- synchronous updates with
cuGraph's `up_down` swap-avoidance rule, changed-count reduce every k.

Louvain (cuGraph `louvain_impl.cuh` lines 172-215, `detail/common_methods.cuh`
lines 70-152 and 402-446, note 04 section 10): synchronous best-move with
the `delta_Q` formula, cluster weights recomputed by reduce-by-key (never
adjusted atomically), the `up_down` direction flag, modularity by an edge
reduce; contraction ON THE DEVICE by `radixSort` of arcs by `(community
src, community dst)` + `segmentedReduce` + `cooToCsr` (the format's
`contract()` is the CPU alternative the accelerator does not use, so the
package never depends on the CPU for a level). Expectation management:
2-10x over the CPU at 1M edges, not 100x (nu-Louvain finds GPU Louvain only
1.03x faster than a 64-thread CPU because later passes lose parallelism,
https://arxiv.org/html/2501.19004); partitions are not identical to the
CPU's -- parity is a modularity band. Leiden's refinement (maximal
independent moves) follows once Louvain is stable.

### 8.7 Dense: APSP / Floyd-Warshall (A9)

Blocked Floyd-Warshall with 32 x 32 tiles for weighted graphs, `n` batched
BFS rows for unweighted; result `Float32Array(n * n)` bounded by
`maxBufferSize` (`n <= 8,192` at the 256 MiB default, ~32k at 4 GiB);
feeds Kamada-Kawai's `dist` (design 14.3 line 4013-4014) and
`allPairsShortestPath` / `floydWarshall` adapters.

### 8.8 Priority order and rationale

| Order | Algorithm | Score (note 02) | Why here |
| --- | --- | --- | --- |
| 1 | PageRank + personalized | 25 | one pull kernel, two reductions, no atomics; first parity test against `indexed.pageRank`; the SpMV primitive it builds serves HITS / eigenvector / Katz for free |
| 2 | HITS, eigenvector, Katz | 15 | same kernel, different normalisation |
| 3 | WCC (Afforest) | 8 | edge-parallel + u32 CAS + `renumberPartition`: proves the edge-list path and the CPU relabel |
| 4 | BFS (+ direction-optimizing) | 5.3 | builds the frontier machinery every later traversal reuses; the indirect-dispatch host loop |
| 5 | Closeness | 5.3 | batched multi-source BFS on the same machinery |
| 6 | SSSP near-far, Bellman-Ford | 4 / 6 | `atomicMin` on f32 bits; predecessor pass |
| 7 | Betweenness (sampled and exact) | 6.3 | the most expensive thing graphty users run; needs everything above |
| 8 | k-core, triangles, label propagation | 4.5 / 4 / 4 | as demand appears |
| 9 | APSP (then Kamada-Kawai) | 5 | dense; bounded n |
| 10 | Louvain, Leiden | 3 | highest value, highest risk; after the sort primitive is proven by the layout's grid build |

Not GPU targets (section 1.2) keep their CPU adapters; `degree()` stays as
the walking-skeleton diagnostic only (cheaper on the CPU than the upload).

---------------------------------------------------------------------------

## 9. Integration with @graphty/algorithms, @graphty/layout and @graphty/graphty-element

### 9.1 Dependency direction

```
@graphty/graph-format  <---- runtime dependency ----  @graphty/webgpu-graph-algorithms   (types only, dev: @graphty/algorithms, @graphty/layout)
        ^                                                        ^
        |                                                        | injected object (no import)
@graphty/algorithms, @graphty/layout  <---- runtime ----  @graphty/graphty-element  <---- runtime ----  @graphty/graphty (the app)
   (own the accelerator interfaces)                          (owns the `accelerator` property, the bridges)      (imports the GPU package, probes, injects)
```

Acyclic: the GPU package imports the CPU packages ONLY as devDependencies
for type conformance (`expectTypeOf(gpu.accelerator()).toMatchTypeOf<
AlgorithmAccelerator & LayoutAccelerator>()`); the CPU packages import
nothing new; graphty-element imports nothing new (it types the property
against interfaces exported by packages it already depends on); the app
is the only runtime importer of the GPU package (note 02 section 4.5,
mechanism (a) + (d)). The registry mechanism (self-registration through a
side-effect module) is rejected: inverted dependency, global state that
breaks with duplicate package copies, tree-shaking defeated, and it puts
the "GPU threw, what now?" decision in the CPU package (note 02 section
4.2).

### 9.2 @graphty/algorithms (lands with A2; can land as the FIRST A2 commit)

New file `algorithms/src/indexed/accelerator.ts`, exported from the barrel
next to the `indexed` namespace of design 14.2. It contains no WebGPU
types.

```ts
import type { GraphSnapshot, F32, F64, U32, NumericVector } from "@graphty/graph-format";

/** Result shapes the accelerator may return: scores may be f32 (GPU) or f64 (CPU). */
export interface ScoresResultLike       { readonly scores: NumericVector; readonly iterations: number; readonly converged: boolean; }
export interface PageRankResultLike     extends ScoresResultLike { readonly danglingMass?: number | undefined; }
export interface HitsResultLike         { readonly hubs: NumericVector; readonly authorities: NumericVector; readonly iterations: number; readonly converged: boolean; }
export interface LabelResultLike        { readonly labels: U32; readonly count: number; groups(): U32[]; }
export interface BfsResultLike          { readonly depth: U32; readonly parent: U32; readonly order: U32; readonly visitedCount: number; }
export interface SsspResultLike         { readonly dist: NumericVector; readonly predArc: U32; }
export interface BellmanFordResultLike  extends SsspResultLike { readonly hasNegativeCycle: boolean; }
export interface EdgeScoresResultLike   { readonly scores: NumericVector; }
export interface ApspResultLike         { readonly dist: NumericVector; readonly n: number; }
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
    kCoreDecomposition?(s: GraphSnapshot): Promise<{ readonly coreness: U32 }>;
    labelPropagation?(s: GraphSnapshot, options?: LabelPropagationOptions): Promise<LabelResultLike>;
    louvain?(s: GraphSnapshot, options?: LouvainOptions): Promise<CommunityResultLike>;
    release?(s: GraphSnapshot): void;
    dispose?(): void;
}

/** The async dispatcher: one method per accelerable indexed function; delegates to the accelerator when it has the method, else runs indexed.* on the CPU. */
export interface AcceleratedAlgorithms {
    readonly accelerator: AlgorithmAccelerator | null;
    pageRank(s: GraphSnapshot, options?: PageRankOptions): Promise<PageRankResultLike>;
    // ... the same list, non-optional
}
export function accelerated(acc: AlgorithmAccelerator | null | undefined): AcceleratedAlgorithms;
```

Implementation of `accelerated()` is ~60 lines: for each method, `acc?.x !==
undefined ? acc.x(s, ...) : Promise.resolve(indexed.x(s, ...))`. The
`Promise.resolve` wrapper makes the CPU path async too, which is what lets
graphty-element's `async run()` adapters treat both alike (note 02 finding
1). The sync `indexed.*` functions and the legacy facades never change.
Options types: the accelerator methods reuse the `indexed.*` option types
(`PageRankOptions` etc.) so an app cannot pass a GPU option the CPU does not
understand; GPU-only tuning goes through the GPU package's own factory
options (section 3.3), never through the dispatcher.

Tests in `@graphty/algorithms`: a fake accelerator `{ kind: "fake", pageRank:
async () => fixture }` proves delegation; `{ kind: "fake" }` (no methods)
proves the CPU path; a throwing method proves the throw propagates
unchanged (no fallback in the dispatcher).

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
export interface ForceAtlas2Options extends CommonLayoutOptions {
    readonly maxIter?: number; readonly jitterTolerance?: number; readonly scalingRatio?: number; readonly gravity?: number;
    readonly strongGravity?: boolean; readonly distributedAction?: boolean; readonly linlog?: boolean;
    readonly nodeMass?: F32 | string | Readonly<Record<NodeId, number>> | null; readonly nodeSize?: F32 | string | Readonly<Record<NodeId, number>> | null;
    readonly weight?: boolean | string | null; readonly dissuadeHubs?: boolean;
    readonly settleThreshold?: number; readonly settleWindow?: number; readonly iterationsPerStep?: number;
}
export interface FruchtermanReingoldOptions extends CommonLayoutOptions { readonly k?: number | null; readonly iterations?: number; readonly fixed?: NodeMask | string | null; readonly settleThreshold?: number; readonly settleWindow?: number; }

export interface LayoutAccelerator {
    readonly kind: string;
    forceAtlas2?(options?: ForceAtlas2Options): LayoutSimulation;
    fruchtermanReingold?(options?: FruchtermanReingoldOptions): LayoutSimulation;
    release?(s: GraphSnapshot): void;
    dispose?(): void;
}
export type SimulationType = "forceatlas2" | "fruchtermanReingold" | "spring";
export function createSimulation(type: SimulationType, options?: ForceAtlas2Options | FruchtermanReingoldOptions, accelerator?: LayoutAccelerator | null): LayoutSimulation;
export declare class ForceAtlas2Simulation implements LayoutSimulation { constructor(options?: ForceAtlas2Options); step(iterations?: number): void; /* sync, CPU */ }
export declare class FruchtermanReingoldSimulation implements LayoutSimulation { /* sync, CPU */ }
export function resolveNodeVector(spec: F32 | string | Readonly<Record<NodeId, number>> | null | undefined, s: GraphSnapshot, fallback: (i: number) => number): F32;   // design 14.3 line 4029
export function resolveWeights(spec: boolean | string | null | undefined, s: GraphSnapshot): F32 | null;                                                      // snapshot.weights | expandEdges(column) | null
```

`createSimulation` is the layout-side dispatcher: `accelerator?.forceAtlas2
!== undefined ? accelerator.forceAtlas2(options) : new
ForceAtlas2Simulation(options)`. The CPU FA2 becomes STEPPABLE in the L1
rewrite (one code path in graphty-element; the GPU is an implementation
swap, note 01 section 8.8 item 3): the one-shot `indexed.forceAtlas2(s,
options): LayoutResult` is `sim.load(s, seed); for (maxIter) sim.step();
return { positions, dim, n }` over the CSR row loop and the all-pairs loop
with no per-pair allocation (design 14.3 port 2). The L1 rewrite adopts the
7.2 reference formulas; `layout/test/forceatlas2-layout.test.ts` asserts
no exact coordinate (note 01 section 2.1.8) so it keeps passing, and the
Chromatic re-baseline commit documents the change.

### 9.4 @graphty/graphty-element (lands with E1)

Exact changes:

1. `Graph` gains `accelerator: GraphAccelerator | null` (default `null`)
   with `setAccelerator(acc)` and an `accelerator-changed` event, where

   ```ts
   export type GraphAccelerator = AlgorithmAccelerator & LayoutAccelerator & { release(s: GraphSnapshot): void; dispose?(): void };
   ```

   (both interface types come from packages the element already depends
   on: `graphty-element/package.json` lists `@graphty/algorithms` and
   `@graphty/layout`). `setAccelerator(null)` after a GPU failure is the
   user's "disable acceleration" action; the element never does it by
   itself.
2. `DataManager`'s `snapshot-replaced` listener list (design 14.4 line
   4155) gets `graph.accelerator?.release(previous)`. `Graph.dispose()`
   calls `accelerator?.release(current)` but NOT `dispose()` (the app owns
   the accelerator's lifetime; two elements may share one).
3. Algorithm adapters (`graphty-element/src/algorithms/*Algorithm.ts`,
   `async run()` at `Algorithm.ts:217`): the body of design 14.4 M7
   becomes `const s = dm.getSnapshot(); const r = await
   indexed.accelerated(this.graph.accelerator).pageRank(s, opts); for (i <
   n) addNodeResult(s.ids.idOf(i), "rank", r.scores[i]);` -- ONE
   result-writing loop for CPU and GPU (the `*Pct` normalisation is an
   O(n) pass over the readback, CPU side, unchanged). Adapters whose
   algorithm is not a GPU target call `indexed.x(s)` directly. Undirected
   adapters pass `dm.undirected(s).snapshot` exactly as for the CPU
   (design 14.4 line 4189-4197), so the GPU never sees a graph the CPU
   would not.
4. `LayoutManager._setLayoutInternal` (note 01 section 4.2): for
   `SimulationType` layouts it creates `new SimulationLayoutEngine(type,
   opts, createSimulation(type, opts, graph.accelerator))`, a bridge
   implementing today's abstract `LayoutEngine`
   (`graphty-element/src/layout/LayoutEngine.ts` lines 36-63) over a
   `LayoutSimulation`:

   ```ts
   class SimulationLayoutEngine extends LayoutEngine {
       async init(): Promise<void> { /* nothing: load happens in setLayout after getSnapshot() */ }
       load(snapshot: GraphSnapshot, positions: F32): void { this.sim.load(snapshot, positions); this.settledSeen = false; }
       reload(snapshot: GraphSnapshot, report: FreezeReport, positions: F32): void { this.sim.load(snapshot, positions); }
       step(): void {                                             // called stepMultiplier times per frame by UpdateManager.updateLayout()
           const r = this.sim.step(this.iterationsPerStep);
           if (r !== undefined) { r.catch((err) => this.onError(err)); }   // fire-and-forget (D6); sync CPU sims return void
       }
       get isSettled(): boolean { return this.sim.settled; }
       pin(n: Node): void { maskSet(this.mask, n.index, true); this.sim.setFixed(this.mask); }
       unpin(n: Node): void { maskSet(this.mask, n.index, false); this.sim.setFixed(this.mask); }
       setNodePosition(n: Node, p: Position): void { this.sim.setPosition(n.index, p.x, p.y, p.z ?? 0); }
       beginDrag(n: Node): void { this.dragBit(n.index, true); }     // temporary fixed bit (7.11); called from NodeBehavior.onDragStart
       endDrag(n: Node, pin: boolean): void { this.dragBit(n.index, pin); }
       getNodePositionInto(index: number, out: F32): void { out[0] = this.positions[3 * index]; /* ... */ }
       dispose(): void { this.sim.dispose(); }
   }
   ```

   `UpdateManager.updateLayout()` (note 01 section 4.1) additionally calls
   `positionColumn.markDirty()` once per frame after the steps (design
   14.4 M12). The GPU simulation's `step()` returning the same in-flight
   promise when saturated means `stepMultiplier` calls per frame cost one
   submission (7.18 item 3); the bridge therefore calls `sim.step(
   stepMultiplier)` ONCE per frame and `LayoutManager.step()` is called
   once (a one-line change in `UpdateManager.updateLayout`, lines 203-214,
   keeping the loop for engines that are not simulations).
5. `NodeBehavior` (`onDragStart` / `onDragUpdate` / `onDragEnd`, note 01
   section 4.5): calls `engine.beginDrag(node)` when the engine has it,
   keeps calling `setNodePosition` per pointer move, and `endDrag(node,
   pinOnDrag)`; `context.setRunning(true)` on drag end is unchanged (the
   simulation reheated itself on `setPosition`).
6. `ForceAtlas2LayoutEngine` / `SpringLayoutEngine` (the one-shot
   `SimpleLayoutEngine` subclasses) are REPLACED by `SimulationLayoutEngine`
   registrations under the same type names with the same zod schemas
   (Storybook controls keep working, `stories/Layout.stories.ts` lines
   87-137); their `scalingFactor` becomes the simulation's `scale` option.
7. Config: `behavior.layout.iterationsPerStep` (default = `stepMultiplier`)
   and `behavior.layout.gpuMinNodes` (default 0: whenever an accelerator
   is injected) are the two product knobs; no `"auto"` acquisition in v1.
   A later `accelerator: "auto"` element option would use the
   `@mlc-ai/web-llm` isolation pattern (`vite.config.ts:39` external, a
   loader module never imported from the barrel, `peerDependenciesMeta`
   optional; note 02 section 4.3) -- not on the critical path.

Settlement, screenshots and label animation (note 01 section 4.3) work
unchanged because `isSettled` is truthful and bounded (7.16).

### 9.5 The graphty app: detection

```ts
// graphty/src/gpu/accelerator.ts (sketch)
import { probeBrowserWebGpu, requestGpuContext } from "@graphty/webgpu-graph-algorithms/browser";   // static import: the app owns its bundle; import() for code splitting is the app's choice
export async function attachAccelerator(element: GraphtyElement, prefs: { gpu: "auto" | "off" | "required" }): Promise<void> {
    if (prefs.gpu === "off") { return; }
    const probe = await probeBrowserWebGpu({ rejectSoftware: prefs.gpu === "auto" });
    if (!probe.ok) { if (prefs.gpu === "required") { throw new Error(probe.reason ?? probe.code); } return; }   // "auto": stay on the CPU path; nothing was created
    const ctx = await requestGpuContext({ limits: "raise" });
    element.setAccelerator(ctx.accelerator());
    ctx.lost.then((info) => { element.setAccelerator(null); showToast(`GPU device lost: ${info.message}`); });   // the user's element keeps working on the CPU path for NEW runs; nothing in flight is retried
}
```

The app surfaces "GPU acceleration: on (NVIDIA lovelace) / off" from
`ctx.caps`. Node consumers (a CLI, a benchmark, a test of the element)
do the same with `createNodeGpuContext()` from `./node`.

### 9.6 Detection in Node

`@graphty/algorithms` users in Node inject exactly the same way: `const
ctx = await createNodeGpuContext(); const r = await indexed.accelerated(
ctx.accelerator()).pageRank(s);`. The element never obtains a Dawn `GPU`
itself (it must not depend on the native module, note 02 section 4.3).

### 9.7 Result-shape parity

| GPU method | Returns | Parity check against `indexed.*` (design 16.2, note 02 section 5) | Adapter writes |
| --- | --- | --- | --- |
| `pageRank` | `{ scores: F32, iterations, converged, danglingMass }` | relative error `<= 1e-5` per node after equal iterations; identical `converged`; top-k rank order | `rank`, `rankPct = rank / maxRank`, graph `iterations` / `converged` / `maxRank` (`PageRankAlgorithm.ts:224-240`) |
| `hits`, `eigenvectorCentrality`, `katzCentrality` | `F32` vectors | same | as today |
| `connectedComponents` | `{ labels: U32, count, groups() }` | partition equality; IDENTICAL labels after first-seen renumbering | `component` per node |
| `breadthFirstSearch` | `{ depth, parent, order, visitedCount }` | `depth` exact; `parent[v]` any vertex with `depth[parent] === depth[v] - 1` and an arc to `v`; `order` grouped by level | as today (visited set, order) |
| `sssp` / `bellmanFord` | `{ dist: F32 (+Inf unreached), predArc: U32, hasNegativeCycle? }` | `dist` within `1e-5` relative; `predArc` any arc attaining `dist`; negative-cycle flag exact | `distance`, `isInPath` via `pathTo()` reconstruction on the CPU from `predArc` |
| `betweennessCentrality` | `F32(n)` raw, same normalisation convention as `indexed` | `1e-4` relative (f32 accumulation over many sources); top-k order; exact sources only (sampled BC is compared against sampled CPU BC with the same source list) | `score`, min-max `scorePct` (`BetweennessCentralityAlgorithm.ts:59-78`) |
| `edgeBetweennessCentrality` | `F32(edgeCount)` via `foldArcs` | same | per edge through `edgeRemap` (design 14.4 line 4110-4113) |
| `closenessCentrality` | `F32(n)` | `1e-5` (integer distances before division) | as today |
| `allPairsShortestPath` | `{ dist: F32(n*n), n }` | exact unweighted, `1e-5` weighted | KK `dist` input |
| `labelPropagation` | `{ labels: U32, iterations, converged }` | planted-partition recovery (LPA is tie-nondeterministic on the CPU too) | `communityId` |
| `louvain` | `{ labels, count, groups(), modularity, levels }` | modularity within a band (`>= cpu - 0.02`) | `communityId`, `groupCount`, `modularity` (`LouvainAlgorithm.ts:166-184`) |
| `forceAtlas2` / `fruchtermanReingold` | a `LayoutSimulation` writing the owner's array | no coordinate parity (chaotic); swing / traction / speed trace within tolerance for the first iterations from an identical seed; distributional metrics (edge-length histogram, stress) after `maxIter` | positions read per frame |

Rules: results are index-aligned typed arrays attached by reference
(`nodes.set(name, vec)`) when the element adopts Option B of design 14.4
M7; nothing keyed by id leaves the GPU package; `Float32Array` scores
satisfy `NumericVector` so no conversion happens in the adapters.

### 9.8 Timeline against the landing order (design 14.6)

| Step | Package | Integration content | Precondition |
| --- | --- | --- | --- |
| W0 (now, this repo) | webgpu-graph-algorithms | standalone development against `@graphty/graph-format` only; structural copies of `AlgorithmAccelerator` / `LayoutAccelerator` / `LayoutSimulation` in `src/types/accelerator.ts` (marked "mirror of A2 / L1; verified at W1"); CPU reference implementations in `test/helpers/oracle.ts` written from design Ports 1-6; two CI lanes (section 12) | F1 done (it is) |
| A2 (first commit) | algorithms | `indexed/accelerator.ts` (9.2) + `accelerated()` dispatcher + fake-accelerator tests; ADDITIVE, so it can be the first A2 PR and does not wait for all 95 ports | A1 merged, F2 cut |
| L1 | layout | `LayoutSimulation`, `LayoutAccelerator`, `createSimulation`, steppable CPU FA2 / FR with the 7.2 formulas, `resolveNodeVector` / `resolveWeights`, Chromatic re-baseline | A1 merged |
| E1 | graphty-element | `accelerator` property, `snapshot-replaced -> release`, adapter dispatch through `accelerated()`, `SimulationLayoutEngine` bridge, drag hooks, config knobs (9.4) | A2 first commit + L1 |
| W1 | webgpu-graph-algorithms | move-in; replace the structural mirrors with `import type` from the real packages (devDependencies) and the type conformance test; differential tests switch from `test/helpers/oracle.ts` to `indexed.*`; two CI shards + the GPU job join `ci.yml` (section 12.5) | A2 complete, L1, E1 |
| W2 (new) | graphty (app), graphty-element stories | `attachAccelerator` (9.5), the "GPU: on / off" indicator, one Storybook story per GPU layout and algorithm under a `gpu` tag (skipped on Chromatic's software renderer), the `gpuMinNodes` default measured from 7.20 | W1 |
| D1 / 2.0 | -- | no GPU-specific content; the dispatcher's CPU branch calls the promoted top-level functions | |

Versioning: the GPU package is an independent nx project (`projectsRelationship:
independent`, `graphty-monorepo/nx.json`), conventional commits with scope
`webgpu-graph-algorithms`, OIDC trusted publishing like its siblings
(`release.yml` lines 36-52, note 02 section 3). Its peer range on
graph-format is `^1.0.0` from F2; it declares NO peer on algorithms /
layout (types only, dev), so an app can combine any versions whose
structural interfaces match -- the type conformance test in W1 is what
guards drift, and a breaking change to an accelerator interface is a
`feat!:` in the OWNING CPU package.

---------------------------------------------------------------------------

## 10. Performance targets and memory model

Tiers from design 15.3; undirected weighted snapshots; edges = 10n except
where the design's benchmark graph fixes them. Basis codes: [M] measured
on the dev box (probes cited in 7.20 and note 06 section 3.5), [D] design
15.1 arithmetic, [X] extrapolation from [M] scaled linearly in bytes / arcs
/ pairs, [P] published numbers (cosmos, GraphWaGu, Brinkmann; note 03).
Every [X] is replaced by a measurement in P1-P3 (section 13) and recorded in
`benchmarks/results/`.

### 10.1 Device memory (resident, excluding the staging ring)

| Nodes / edges | Core hot prefix [D] | + views typical (`edgeList` or `coo`, `degreeOrder`) [D] | FA2 exact scratch [D] | FA2 grid scratch + pyramid (2D / 3D) [D] | PageRank scratch [D] | BFS scratch [D] | Betweenness batch (k = 64 sources) [D] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10k / 100k | 1.6 MB | +0.8 MB | 0.5 MB | 0.7 MB + 0.3 MB / 1 MB | 0.1 MB | 0.2 MB | 7.7 MB |
| 100k / 1M | 16.4 MB | +8.4 MB | 5.3 MB | 7 MB + 5.6 MB / 38 MB | 1.2 MB | 1.6 MB | 77 MB |
| 1M / 10M | 164 MB | +84 MB | 53 MB | 70 MB + 5.6 MB / 38 MB | 12 MB | 16 MB (+ 40 MB edge queue) | 768 MB (planner caps k by `maxBufferSize`) |
| 10M / 100M | 1.64 GB (per-array; raised limits) | +840 MB | 530 MB | 700 MB + 5.6 MB / 38 MB | 120 MB | 160 MB | k capped to ~6 on a 12 GB card |

The 12 GB RTX 4070 SUPER holds every tier; lavapipe's 128 MiB binding cap
excludes the 1M / 10M core from the default CI lane (`colIdx` = 80 MB fits,
but `colIdx + weights` bound together as one arena exceeds it only if the
arena path is used -- the planner falls back to per-array; the 10M tier is
GPU-lane only).

### 10.2 Upload and readback

| Nodes / edges | Upload hot prefix (5-12 GB/s `writeBuffer`) [X] | Freeze on the CPU for comparison [M, design 15.4] | Positions readback per batch (Chromium 2.65 ms/MiB [M]; Node ~10x less [X]) | Result readback (one `F32(n)`) |
| --- | --- | --- | --- | --- |
| 10k / 100k | ~0.3 ms | ~2 ms | 0.3 / 0.05 ms | < 0.2 ms |
| 100k / 1M | 2-4 ms | 22 ms directed / 48 ms undirected | 3 / 0.4 ms | 1 / 0.1 ms |
| 1M / 10M | 20-40 ms | ~250-500 ms [X] | 30 / 3 ms | 10 / 1 ms |
| 10M / 100M | 200-400 ms | seconds | 300 / 30 ms | 100 / 10 ms |

### 10.3 Per-iteration / per-run time on the RTX 4070 SUPER class

| Nodes / edges | FA2 exact / iter | FA2 grid 2D / iter | PageRank / iter (bandwidth-bound: ~24 B/arc) [X from A] | BFS (whole run, 32 levels/submit) [X] | WCC (Afforest, ~6 rounds) [X] | Betweenness exact / 1k sampled sources [X] |
| --- | --- | --- | --- | --- | --- | --- |
| 10k / 100k | 0.3 ms [X from M] | 1-2 ms (build-dominated) | 0.05 ms + dispatch latency (~0.02 ms/dispatch) | ~1 ms | ~1 ms | 0.5 s / 0.05 s |
| 100k / 1M | 28 ms [X] | 3-8 ms [P-bracketed] | 0.1-0.7 ms [M] | 2-5 ms (diameter ~10-20) | ~5 ms | 60 s / 0.6 s |
| 1M / 10M | 2.8 s [X] | 30-80 ms [P-bracketed] | 1-7 ms | 30-80 ms | ~50 ms | hours / 8 s |
| 10M / 100M | n/a | 0.3-0.8 s | 10-70 ms | 0.5-1 s | ~0.5 s | n/a / 100 s |

Integrated GPU (Iris Xe / Apple M-class base) expectations: 5-10x slower on
compute [P: GraphWaGu Iris Xe ratios; note 03 section 8.2], 2x on
readback; the exact-tier crossover halves to ~8k nodes; the 100k FA2 grid
tier lands at 30-70 ms/iter (10-15 fps) and is still usable interactively
with `iterationsPerStep = 1`.

Targets (gates for P3 / P6): FA2 grid 2D at 100k / 1M `<= 10 ms/iter`
and at 1M / 10M `<= 100 ms/iter` on the 4070; FA2 exact at 16k `<= 2
ms/iter`; PageRank 100k / 1M 50 iterations `<= 50 ms` end-to-end
including upload; BFS 1M / 10M `<= 100 ms`; a `3x` regression gate against
the checked-in baseline per runner class (the design 15.5 convention).

---------------------------------------------------------------------------

## 11. Testing strategy

### 11.1 Projects (vitest 3.2.x, one config, four projects)

| Project | Environment | Contents | Default lane | GPU lane |
| --- | --- | --- | --- | --- |
| `node` (PRIMARY) | Node 22, `pool: "forks"` (native addon; verified by graph-format's 1309-test suite, `packages/graph-format/vitest.config.ts` line 7), Dawn via `test/setup/gpu.ts` | every unit, kernel, primitive, algorithm, layout and planner test; differential tests; property tests; device-loss; carries the 80/80/75/80 thresholds when run whole | yes (lavapipe, `GRAPHTY_GPU_ADAPTER=llvmpipe`) | yes (NVIDIA, `GRAPHTY_GPU_REQUIRE=nvidia`) |
| `node-limits` | same | tests that need limits above lavapipe's (bindings > 128 MiB, `maxBufferSize` near 2 GiB, a real 2D dispatch above 16,776,960 items, vendor feature assertions) | no (selected out by `--project`) | yes |
| `bench` | same, `singleFork` | `vitest bench`: primitives, FA2 exact / grid at 4k / 16k / 65k / 262k / 1M, PageRank, BFS; JSON output compared with `benchmarks/results/<runner-class>.json` | no | yes |
| `browser` (LIGHT) | Playwright Chromium, `fileParallelism: false`, flags by `GRAPHTY_BROWSER_GPU` (`swiftshader` / `nvidia`) | 11.5 | yes (SwiftShader) | yes (NVIDIA, four flags + `libEGL.so.1`) |

The config is the note 07 section 4.5 sketch plus the two extra projects
and the flag switch of note 06 section 5. The per-instance Playwright
`launch` spelling must be confirmed against the installed `@vitest/browser`
when written (note 07 unverified item 1) and recorded in the package
`CLAUDE.md`.

### 11.2 Device policy in tests (`test/setup/gpu.ts`)

Generalises `packages/graph-format/test/audit/gpu-upload.test.ts`
`acquire()` (lines 38-63): dynamic `import("webgpu")`, install globals,
`create([...])` with `adapter=${GRAPHTY_GPU_ADAPTER}` when set,
`requestAdapter`, `adapter.info`, `requestDevice` with raised limits; each
failure is an `E_NO_ADAPTER: ...` reason. Policy: no adapter -> `t.skip(
reason)` locally, HARD FAIL when `GRAPHTY_GPU_REQUIRE` is set; `GRAPHTY_
GPU_REQUIRE=nvidia` also asserts `adapter.info.vendor === "nvidia"` and
`!software` so a silent lavapipe / SwiftShader run on the GPU lane is red
(the failure mode `HEADLESS_GPU_REPORT.md` found locally). "A wrong result
is never a skip." An `uncapturederror` listener fails the current test.
`gpuScale()` returns 1 on hardware and 1/50 on a software adapter and
scales fixture sizes and iteration counts (lavapipe is ~350x slower on the
exact tile, note 05 section 2.5). `XDG_RUNTIME_DIR` is set to silence
Mesa's stderr lines. The Node setup drops the `GPU` reference in `afterAll`
so forks exit.

### 11.3 Test kinds

| Kind | What | Oracle |
| --- | --- | --- |
| Planner unit tests (no device) | `planUpload`, `plan1d` / `planGridStride`, window boundaries (`start = rowPtr[v0] - (rowPtr[v0] % 64)`, a row longer than a window, arc indices above 2^31 with no bitwise ops), uniform packing offsets, pipeline cache keys | hand-computed expectations under faked caps: spec defaults, SwiftShader-like, lavapipe-like, NVIDIA-like (note 05 section 4) |
| Upload contract | arena hot prefix bindings equal CPU views; per-array path on a `fromCsr` snapshot (`arena === null`); windowed path on a 64-arc boundary; `arena.byteOffset !== 0` from `fromBytes`; packed `u8` / `bool` columns; identity permutations never materialised (`byteLength({ views: true })` unchanged); `release` destroys every buffer (`stats().buffers === 0`, no uncaptured error) | copied from `gpu-upload.test.ts` lines 254-666 (note 07 section 6) |
| Primitive differential | each primitive vs `test/helpers/oracle.ts` on random and adversarial inputs (all-equal keys, one giant row, empty, exactly 16,776,960 and +1 items in `node-limits`) | f64 CPU references; exact for u32 |
| Algorithm differential | vs `test/helpers/oracle.ts` (index-based ports of design Ports 1-6) until W1, then vs `indexed.*`; on karate, grid, random G(n, m) with self-loops and parallels, a 10k-degree hub graph, directed and undirected, weighted and not; `validate({ checksum: true })` after every call (no write into a view) | tolerances of 9.7 |
| Layout parity | seeded start from the CPU LCG; per-iteration swing / traction / speed trace vs the CPU `ForceAtlas2Simulation` (post-L1; until then an oracle transcribed from 7.2) within `1e-3` relative for the first 20 iterations on 10-1,000 nodes; final distributional metrics (edge-length histogram, stress, nearest-neighbour distance) within bands | CPU simulation |
| Exact-vs-approximate | grid-tier forces vs the exact tile on the same positions: relative force error `<= 5%` per node on uniform and clumpy inputs (cosmos's 163-node country graph shape; 1,024 points in one cell), and the layouts after 100 iterations agree distributionally | the exact kernel (7.6) |
| Property / invariant | fixed nodes never move; `dim === 2` leaves z exactly 0; `setPosition` visible in the next readback and not clobbered by an older batch; `settled` becomes true within `maxIter`; reheat on unpin; disconnected components separate; gravity pulls the centroid to 0; energy (stress) decreases over the first iterations; `dispose` frees everything; `arcCount === 0` binds nothing; results are `<ArrayBuffer>`-typed | fast-check generators sized by `gpuScale()` |
| Subgroup variants | every subgroup kernel at sizes 4 / 8 / 32 across the lanes; `GRAPHTY_GPU_NO_SUBGROUPS=1` forces the fallback path on any adapter | the non-subgroup variant |
| Device loss / errors | `device.destroy()` mid-batch rejects pending promises with `E_DEVICE_LOST` and leaves the context in `lost`; an OOM scope on a deliberately oversized buffer yields `E_OUT_OF_MEMORY`; a bad binding yields `E_VALIDATION` with the label | both runtimes |
| Type-level | `expectTypeOf` for the public surface; `tsconfig.strict-consumer.json` compiles a consumer sample with `noUncheckedIndexedAccess` and `exactOptionalPropertyTypes` ON; at W1 mutual assignability with `AlgorithmAccelerator` / `LayoutAccelerator` / `LayoutSimulation` | tsc |
| Build output | `dist/webgpu-graph-algorithms.js` and `dist/browser.js` contain no `"webgpu"` specifier; `package.json` exports map; `sideEffects: false` | file reads |

### 11.4 What the walking skeleton proves (P0 gate)

One Node test and one browser test that: create a context through
`./node` / `./browser`; upload the hot prefix of a `fromEdgeArrays`
snapshot (arena path) AND a `fromCsr` snapshot (per-array path); run the
`degree` kernel with the `USE_PERM` dummy-binding pattern; read back a
`Uint32Array` equal to `outDegree()`; run one exact-tile FA2 iteration on
karate and read the swing / traction trace; `release(snapshot)` then
`stats().buffers === 0`; no uncaptured error; the `uncapturederror` hook
fails a deliberately broken bind group; `plan1d(16_776_960)` is 1D and
`+1` is 2D; the whole thing on lavapipe, SwiftShader and NVIDIA.

### 11.5 Light browser testing

`test/browser/*.test.ts`: (1) `requestGpuContext()` and the typed
`E_NO_WEBGPU` when `navigator.gpu` is deleted; (2) the skeleton of 11.4;
(3) `createForceAtlas2` on a 500-node graph: `load`, `step(10)` five
times with `maxInFlight = 2`, positions written back, `setPosition` /
`setFixed` honoured, `dispose`; (4) one PageRank and one CC on karate vs
the oracle; (5) a subgroup-variant kernel when `features.has("subgroups")`
(SwiftShader size 4); (6) `release()` leaves no buffers. Passes on
SwiftShader (default lane) and NVIDIA (GPU lane); a hard job timeout
backstops the `browser.close()` hang seen after GPU work on the NVIDIA
path (note 06 section 7).

### 11.6 Benchmarks and baselines

`benchmarks/` copies graph-format's harness (`bench()`, `printTable`,
`appendSession` with a `gpu` field; note 07 section 6) and `datasets.ts`
(seeded `randomEdges`, R-MAT-like hub graph). Groups: `upload`,
`primitives`, `layout-exact`, `layout-grid`, `pagerank`, `bfs`, `cc`.
Sessions record adapter vendor / architecture / device and requested
limits so 4070, CI-runner and browser numbers never mix. The GPU lane
uploads the JSON (90-day retention) and fails on `> 3x` the checked-in
baseline for its runner class. Software adapters never time anything.

### 11.7 Coverage

Thresholds 80 / 80 / 75 / 80 on the `node` project when run whole (the
graph-format convention); `src/wgsl/**` excluded (template strings);
device-limit branches (windowed uploads, 2D dispatch, OOM) are covered by
the faked-caps unit tests so the default lane reaches the thresholds
without the GPU lane (note 06 section 9 item 6). Coverage is produced by
the default lane only (section 12.4).

---------------------------------------------------------------------------

## 12. CI/CD

### 12.1 Lanes and runners

| Lane | Runner | Adapter | Runs | Required check | Contents |
| --- | --- | --- | --- | --- | --- |
| default | `ubuntu-latest` (GitHub-hosted, free for public repos) | Dawn-in-Node on Mesa lavapipe (`apt-get install mesa-vulkan-drivers libvulkan1`; `create(["adapter=llvmpipe"])` -- the `webgpu` npm package's own CI pattern, also wgpu and three.js, note 06 section 3.5) + Chromium on bundled SwiftShader (`--enable-unsafe-webgpu --use-angle=swiftshader --enable-unsafe-swiftshader`; verified bit-identical results to NVIDIA on the probe kernel) | every push and PR | YES | build, lint, typecheck, strict-consumer typecheck, the whole `node` project with coverage, the `browser` smoke project |
| GPU | self-hosted, ephemeral, on the owner's dev box (RTX 4070 SUPER, driver 580.173.02), labels `[self-hosted, linux, x64, gpu, nvidia]` | NVIDIA via Vulkan, `GRAPHTY_GPU_REQUIRE=nvidia` (software fallback = failure) | push to `master`, nightly, `workflow_dispatch`, and same-repo PRs labelled `gpu` | NO (never blocks a merge; a powered-off box must not block the team) | `node` + `node-limits` on NVIDIA, `bench` with baseline comparison, `browser` smoke on the real GPU, `gpu-report.json` + bench JSON artifacts |

Why not GitHub-hosted GPU runners: the T4 larger runners require GitHub
Team or Enterprise Cloud and `gh api /orgs/graphty-org` reports `plan:
free`; they are also not free for public repos ($0.052/min Linux) (note 06
sections 0 and 3.2). Revisit if the org plan changes ($4 / user / month
for Team; a 10-minute GPU job is $0.52). Third-party runners in the
project's own cloud (Cirun -- free platform for public repos; RunsOn;
machine.dev) need a cloud account the project does not have and are the
documented escape hatch if the dev box proves unreliable (note 06 section
3.4). This is the same shape as `atoms-org/cuda-ffi` (`runs-on:
cudaffi-gpu-runner` + `container: { options: --gpus all }`,
`tmp/webgpu-plan/repos/cuda-ffi/.github/workflows/build.yml`), improved in
two ways: cuda-ffi has no GPU-free correctness lane (only lint runs
without a GPU) and triggers only on push (no PR path); this plan runs the
whole correctness suite on the default lane and gates the GPU lane by
event + label.

### 12.2 Security for a self-hosted runner on public repositories

GitHub's guidance is that self-hosted runners "should almost never be used
for public repositories" because fork PRs can run code on them (note 06
section 3.3). Mitigations, all applied: the runner is EPHEMERAL (`config.sh
--ephemeral --disableupdate`, or JIT `generate-jitconfig`), one job per
registration, in its own sibling container (not the dev workspace), with
`--gpus all` and `NVIDIA_DRIVER_CAPABILITIES=all` (needed for the Vulkan
ICD, `HEADLESS_GPU_REPORT.md` line 32); the GPU job's `if` requires
`github.event.pull_request.head.repo.full_name == github.repository` AND
the `gpu` label (fork PRs can never satisfy the first clause; only
triage / write users can apply the label); repository setting "Require
approval for all external contributors"; workflow permissions read-only;
NO secrets in the GPU job (artifacts only); `concurrency: gpu-runner` so
one job at a time; the registration token (fine-grained PAT with
`Administration: write`, or a GitHub App) lives only in the host-side
loop. Org level (later): a runner group scoped to the two repos.

### 12.3 Workflow for this repository now

Replaces the stale scaffold `.github/workflows/test.yml` (Node 18/20,
`npm ci`, Xvfb, SwiftShader-forcing flags; note 06 section 1). Package
manager follows `packages/` (pnpm workspace root `packages/package.json`).

```yaml
name: CI
on:
    push: { branches: [master] }
    pull_request: { types: [opened, synchronize, reopened, labeled] }
    schedule: [{ cron: "17 6 * * *" }]          # nightly GPU lane
    workflow_dispatch:
permissions: { contents: read }
concurrency:
    group: ${{ github.workflow }}-${{ github.ref }}
    cancel-in-progress: ${{ github.event_name == 'pull_request' }}
env:
    WEBGPU_NPM_PIN_NOTE: "webgpu@0.4.0: 0.5+ needs glibc 2.38; dev box is Ubuntu 22.04 / 2.35"

jobs:
    test:                                        # ---------------- default lane (required check)
        name: Test (software adapters)
        if: github.event_name != 'schedule'
        runs-on: ubuntu-latest
        timeout-minutes: 30
        defaults: { run: { working-directory: packages } }
        steps:
            - uses: actions/checkout@v4
            - uses: pnpm/action-setup@v4
            - uses: actions/setup-node@v4
              with: { node-version: 22.x, cache: pnpm, cache-dependency-path: packages/pnpm-lock.yaml }
            - run: pnpm install --frozen-lockfile
            - name: Install Mesa lavapipe (software Vulkan ICD for Dawn-in-Node; not preinstalled on ubuntu-24.04)
              run: sudo apt-get update && sudo apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1
            - run: pnpm -r run build
            - run: pnpm -r run lint
            - run: pnpm --filter @graphty/webgpu-graph-algorithms run typecheck:strict-consumer
            - name: Node suite on lavapipe (coverage)
              working-directory: packages/webgpu-graph-algorithms
              env:
                  GRAPHTY_GPU_ADAPTER: llvmpipe
                  VK_DRIVER_FILES: /usr/share/vulkan/icd.d/lvp_icd.x86_64.json
                  XDG_RUNTIME_DIR: /tmp
              run: pnpm exec vitest run --project=node --coverage
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
              env: { GRAPHTY_BROWSER_GPU: swiftshader }
              run: timeout 600 pnpm exec vitest run --project=browser     # hard kill backstop: browser.close() can hang after GPU work
            - uses: actions/upload-artifact@v4
              if: ${{ !cancelled() }}
              with: { name: coverage-webgpu-graph-algorithms, path: packages/webgpu-graph-algorithms/coverage/lcov.info, retention-days: 1, if-no-files-found: error }

    test-gpu:                                    # ---------------- GPU lane (never required)
        name: Test (NVIDIA, self-hosted)
        if: >-
            github.event_name == 'push' || github.event_name == 'schedule' || github.event_name == 'workflow_dispatch' ||
            (github.event_name == 'pull_request' &&
             github.event.pull_request.head.repo.full_name == github.repository &&
             contains(github.event.pull_request.labels.*.name, 'gpu'))
        runs-on: [self-hosted, linux, x64, gpu, nvidia]
        timeout-minutes: 45
        concurrency: { group: gpu-runner, cancel-in-progress: false }
        env:
            GRAPHTY_GPU_REQUIRE: nvidia                 # a software adapter fails the job
            GRAPHTY_BROWSER_GPU: nvidia                 # --enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan --disable-vulkan-surface
            XDG_RUNTIME_DIR: /tmp
            # Only while the runner image lacks libegl1 (HEADLESS_GPU_REPORT.md appendix D):
            # LD_LIBRARY_PATH: /opt/egl/usr/lib/x86_64-linux-gnu
        defaults: { run: { working-directory: packages/webgpu-graph-algorithms } }
        steps:
            - uses: actions/checkout@v4
            - uses: pnpm/action-setup@v4
            - uses: actions/setup-node@v4
              with: { node-version: 22.x }
            - run: pnpm install --frozen-lockfile
              working-directory: packages
            - run: pnpm -r run build
              working-directory: packages
            - name: Adapter report (fails loudly on a software adapter)
              run: node scripts/gpu-report.mjs | tee gpu-report.json
            - run: pnpm exec vitest run --project=node --project=node-limits
            - run: pnpm exec vitest bench --project=bench --outputJson bench/results.json
            - run: timeout 900 pnpm exec vitest run --project=browser
            - uses: actions/upload-artifact@v4
              if: ${{ !cancelled() }}
              with: { name: "gpu-results-${{ github.run_id }}", path: "packages/webgpu-graph-algorithms/gpu-report.json\npackages/webgpu-graph-algorithms/bench/results.json", retention-days: 90 }
```

`scripts/gpu-report.mjs` prints adapter info, features, the four limits,
subgroup size and the 4-byte round-trip latency (the probe scripts of
`tmp/webgpu-plan/probe/` productised) and exits non-zero when
`GRAPHTY_GPU_REQUIRE` is set and the vendor does not match.

### 12.4 Runner recipe for the dev box (host side, not committed to the package)

The dev container has no Docker socket, so the runner runs as a SIBLING
container started on the host with the NVIDIA Container Toolkit (note 06
section 5.1):

```
FROM ghcr.io/actions/actions-runner:latest
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libvulkan1 mesa-vulkan-drivers vulkan-tools \
    && rm -rf /var/lib/apt/lists/*            # + `npx playwright install-deps chromium` at build time
USER runner
```

```
docker run -d --restart unless-stopped --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e RUNNER_REPO=graphty-org/webgpu-graph-algorithms -e RUNNER_LABELS=gpu,nvidia,rtx4070 \
  -v /srv/gha-runner/token:/run/secrets/gh-token:ro graphty-gpu-runner
# entry loop: TOKEN=$(gh api -X POST /repos/$RUNNER_REPO/actions/runners/registration-token --jq .token);
#             ./config.sh --unattended --ephemeral --disableupdate --url https://github.com/$RUNNER_REPO --token "$TOKEN" --labels "$RUNNER_LABELS" --name "devbox-$RANDOM"; ./run.sh; rm -rf _work; loop
```

`libegl1` in the image removes the `LD_LIBRARY_PATH` workaround (the root
cause in `HEADLESS_GPU_REPORT.md`: the NVIDIA Vulkan ICD `dlopen()`s
`libEGL.so.1`). Keep `webgpu@0.4.0` until this image AND the dev container
move to Ubuntu 24.04 (glibc 2.39), then bump once everywhere (D11).

Coverage comes only from the default lane: `tools/merge-coverage.sh --ci`
fails on any missing package artifact and `coverage.yml` runs only on a
successful CI (note 06 section 4.4), so a sometimes-skipped GPU lane must
never upload `coverage-*`. It uploads `gpu-results-*` (90-day retention,
like the monorepo's `performance-baseline`).

### 12.5 Slotting into the monorepo at W1

`graphty-monorepo/.github/workflows/ci.yml` builds once, then fans out a
`Test (${{ matrix.shard }})` matrix on `ubuntu-latest` (lines 236-351:
`algorithms-default`, `algorithms-browser`, `layout`, five
`graphty-element-browser-*` shards, ...) with `needs-browser` /
`needs-storybook` flags and a cached `playwright install chromium
--with-deps` step (lines 413-427); `all-checks` (lines 706-738) needs only
`test` and the Chromatic jobs. Following `packages/move/root-touch-points.diff`
(note 06 section 6):

```diff
@@ jobs.build
+            - name: Build webgpu-graph-algorithms (PR)
+              if: github.event_name == 'pull_request'
+              run: pnpm exec nx run webgpu-graph-algorithms:build
+            - uses: actions/upload-artifact@v4
+              with: { name: build-webgpu-graph-algorithms, path: webgpu-graph-algorithms/dist/, retention-days: 1 }
@@ jobs.test.strategy.matrix.shard
+                    - webgpu-graph-algorithms-node
+                    - webgpu-graph-algorithms-browser
@@ jobs.test.strategy.matrix.include
+                    - shard: webgpu-graph-algorithms-node        # Dawn on Mesa lavapipe, no GPU
+                      package: webgpu-graph-algorithms
+                      test-command: cd webgpu-graph-algorithms && pnpm exec vitest run --project=node --coverage
+                      needs-browser: false
+                      needs-storybook: false
+                      needs-vulkan: true
+                    - shard: webgpu-graph-algorithms-browser     # Chromium SwiftShader smoke
+                      package: webgpu-graph-algorithms
+                      test-command: cd webgpu-graph-algorithms && timeout 600 pnpm exec vitest run --project=browser
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
+                  GRAPHTY_BROWSER_GPU: swiftshader
+                  VK_DRIVER_FILES: /usr/share/vulkan/icd.d/lvp_icd.x86_64.json
@@ coverage upload condition
+  || startsWith(matrix.shard, 'webgpu-graph-algorithms-node')      # the browser shard uploads no coverage
@@ new job (after "performance"); NOT added to all-checks.needs
+    test-gpu:  (the 12.3 job body, runs-on: [self-hosted, linux, x64, gpu, nvidia], needs: build, same `if`, downloads build-graph-format and build-webgpu-graph-algorithms)
```

Plus the root touch points: `pnpm-workspace.yaml`, `commitlint.config.js`
scope, `knip.config.ts` workspace (`src/node/index.ts` and
`src/browser/index.ts` as extra entries), `tools/merge-coverage.sh`
`PACKAGES`, `tools/prepush.sh` (`test:node` only; the pre-push hook on the
dev box uses NVIDIA or `GRAPHTY_GPU_ADAPTER=llvmpipe` to match CI),
`release.yml` download step (OIDC trusted publishing, npm 11, gated on CI
success -- unchanged for this package). The shard matrix has no
`nx affected` gating (note 06 section 6), so the two software shards
simply join it; the GPU job is gated by event + label, not by paths ("a
change to graph-format is exactly when the GPU lane should run").

This is the owner's "one runner for GPU and a default runner for other
tests": the package-level split into vitest projects is what makes the
routing a `runs-on` per job.

### 12.6 Budgets and drift

| Job | timeout | Expected duration | Artifacts | Retention |
| --- | --- | --- | --- | --- |
| default `test` | 30 min | lavapipe suite 3-10 min (wgpu budgets 5-15 min for its whole lavapipe job; the 100k / 1M gather probe took ~190 ms per 50 iterations at 4 threads, note 06 section 3.5) + 1-2 min browser smoke | `coverage-webgpu-graph-algorithms` | 1 day |
| `test-gpu` | 45 min | 2-6 min tests + 3-10 min bench | `gpu-results-*` | 90 days |
| nightly `test-gpu` | 45 min | same; opens / refreshes a tracking issue on failure (`actions/github-script`) | same | 90 days |

Drift risks handled: lavapipe version skew (dev box Mesa 23.2 vs runner
Mesa 25.2) is diagnosable because every job prints the adapter
description first; Playwright bumps change the bundled Chromium (139
today) -- the nightly GPU lane catches a broken flag set; the `webgpu`
pin is re-evaluated when the images move to 24.04.

---------------------------------------------------------------------------

## 13. Phased implementation plan

Sizes are working days for one engineer familiar with the code base
(rough; the WGSL phases carry the most uncertainty). Each phase's gate must
be green on BOTH CI lanes before the next phase starts. The owner's
ordering constraint -- force-directed layout right after the skeleton -- is
P2 / P3.

| Phase | Scope | Deliverables | Verification gate | Size |
| --- | --- | --- | --- | --- |
| P0 Reset + walking skeleton | scaffold triage per note 07 section 5 (delete `src/types/index.ts` `CSRGraph`, the Vitest 2.1 config, `vite.config.ts`, eslint / knip / husky / `.env` / `test.yml`, `STRATEGY.md`, `IMPLEMENTATION_CHECKLIST.md`; keep `HEADLESS_GPU_REPORT.md`, `CLAUDE.md`, `packages/**`); create `packages/webgpu-graph-algorithms/` with the 3.1 skeleton; `GpuContext` (2.2), `./browser`, `./node`, `WebGpuGraphError`, `webgpu-constants`; minimal `GraphResidency` (arena + per-array), `Readback`, `PipelineCache`, `DispatchPlanner`, `UniformRing`; the `degree` kernel and ONE exact-tile FA2 iteration kernel; `test/setup/gpu.ts`; the default CI lane (12.3 `test` job) | package builds with `tsc` and the vite bundle; `./node` and `./browser` work; 11.4 passes on lavapipe, SwiftShader and NVIDIA locally | 11.4 green on the default lane; `build-output.test.ts` proves no `"webgpu"` in the root / browser bundles; adapter report printed | 6-8 d |
| P1 Memory + dispatch infrastructure | full `planUpload` (windowed path), `BufferPool` + `Lease`, staging ring, `CommandBatch`, error scopes, device-loss state, `Profiler`; primitives `reduce`, `scan`, `segmentedReduce` (tiers), `bbox`; the GPU CI lane (12.3 `test-gpu`, the 12.4 runner) | faked-caps unit tests for every planner branch; primitive differential tests; `node-limits` project with the real 2D dispatch and > 128 MiB binding tests | thresholds 80/80/75/80 on the default lane; GPU lane green with `GRAPHTY_GPU_REQUIRE=nvidia`; `gpu-report.json` uploaded | 6-8 d |
| P2 Force simulation core: FA2 exact + FR (FIRST DELIVERABLE) | `ForceSimulation` state machine (7.18), `createForceAtlas2` with the exact tier (7.4-7.10, 7.16-7.17), `setFixed` / `setPosition` semantics (7.11), `run()` batch driver, `createFruchtermanReingold` (7.19), `GpuAccelerator.forceAtlas2 / fruchtermanReingold`; the CPU oracle `ForceAtlas2Oracle` in `test/helpers/oracle.ts` transcribed from 7.2 (becomes the L1 `ForceAtlas2Simulation`'s spec); browser smoke test (3) of 11.5; `bench/layout-exact` | layout parity tests (trace + distributional), property tests, exact tile measured at 1k / 4k / 16k / 65k on the 4070 and recorded | parity within 11.3 tolerances; 16k nodes `<= 2 ms/iter` on the 4070; browser smoke green on SwiftShader and NVIDIA; a Node CLI (`benchmarks/layout-run.ts`) lays out the 100k / 1M graph exact tier end to end (slow but correct) | 10-12 d |
| P3 Scale: grid pyramid + degree tiers | `radixSort`, `histogram` / counting sort, `compact`; `RepulsionGrid` (7.7) 2D and 3D; attraction tiers over `degreeOrder()`; `repulsion: "auto"` crossover; `bench/layout-grid` at 65k / 262k / 1M; hub-heavy fixtures; the exact-vs-approximate tests | 100k / 1M grid `<= 10 ms/iter`, 1M / 10M `<= 100 ms/iter` (2D) on the 4070, or a written analysis of why not and whether option B (cluster tree) is needed; `exactMaxNodes` default re-fixed from measurements; determinism tests for `deterministic: true` | gates above; near-field error `<= 5%` vs exact on clumpy fixtures | 10-14 d |
| P4 Integration PRs (monorepo, parallel with P3 once A1 has merged) | `algorithms`: 9.2 interfaces + `accelerated()` + fake tests (first A2 commit); `layout`: 9.3 `LayoutSimulation`, `LayoutAccelerator`, `createSimulation`, steppable CPU FA2 / FR with the 7.2 formulas; `graphty-element`: 9.4 property, `snapshot-replaced -> release`, `SimulationLayoutEngine`, drag hooks; app: 9.5 `attachAccelerator` | monorepo tests green; Chromatic re-baseline commit for FA2; a story that runs the GPU FA2 when an accelerator is injected (skipped on Chromatic) | element tests + stories green; the GPU package's structural mirrors match the real interfaces (type test run manually against the monorepo checkout until W1) | 8-10 d (split across three packages) |
| P5 SpMV family + WCC | `spmvPull`, PageRank (+ personalized), HITS, eigenvector, Katz, Afforest WCC, `renumberPartition` on readback, `GpuAccelerator` methods; differential tests vs the oracle (vs `indexed.*` after W1) | PageRank 100k / 1M 50 iterations `<= 50 ms` end to end; WCC parity identical after renumbering; bindings fit 8 storage buffers | gates above; browser smoke (4) green | 8-10 d |
| P6 Frontier family | `Frontier`, `advance` (block_mapped + workgroup tier + subgroup variant), `dedupe`, indirect args; BFS (+ direction-optimizing), closeness, SSSP near-far, Bellman-Ford; then betweenness (batched, sampled) | BFS 1M / 10M `<= 100 ms`; road-network fixture (grid 1000 x 1000, diameter ~2000) faster than the CPU BFS oracle in Node; BC parity `1e-4` on karate / random; sigma overflow flag test | gates above | 14-18 d |
| P7 Move-in (W1) | move to `graphty-monorepo/webgpu-graph-algorithms/`; replace structural mirrors with `import type` from `@graphty/algorithms` / `@graphty/layout`; conformance type test; differential tests switch to `indexed.*`; 12.5 `ci.yml` shards + `test-gpu` job; root touch points; `release.yml` | the packages/README.md move checklist | monorepo CI green including the two new shards; GPU job green on the labelled PR; first release `0.1.0` with provenance | 3-4 d |
| P8 Structure + community | k-core, triangle counting / k-truss, label propagation, Louvain (device-side contraction via `cooToCsr`), APSP (+ Kamada-Kawai `dist`) | differential tests; Louvain modularity band; APSP bounded-n tests in `node-limits` | gates above; Louvain 2-10x vs the CPU at 1M edges documented in the bench JSON | 14-18 d |
| P9 Element polish (W2) | `gpuMinNodes` default from measurements, `iterationsPerStep` auto-raise above 250k nodes, GPU stories, device-loss UX (toast + CPU path for new runs), docs | Storybook stories under a `gpu` tag; README with the Node and browser recipes | stories green; nightly GPU lane green for a week | 4-6 d |

Critical path to the owner's first need: P0 -> P1 -> P2 (about 4-5 weeks)
gives an interactive GPU ForceAtlas2 usable from Node and from a
graphty-element story with an injected accelerator; P3 (another 2-3 weeks)
takes it to 10^5-10^6 nodes; P4 makes it "detected" in the app. Everything
after P4 is additive.

---------------------------------------------------------------------------

## 14. Risks and open questions for the owner

Each item names a recommended default; silence means the default stands.

1. FA2 reference formulas (7.2 / D5): the CPU port's `1/d^2` repulsion and
   position-based swing / traction differ from the paper, Gephi and
   cuGraph. Default: adopt the published formulas in BOTH implementations
   at L1 (a documented behaviour change with a Chromatic re-baseline).
   Alternative: keep the port's formulas as the reference (the GPU kernel
   takes them as constants either way; pick before P2 starts).
2. Node-first testing departs from design 14.5's "browser-only vitest
   project". Default: amend 14.5 and 16.7 when this plan is accepted.
3. Self-hosted runner on the dev box for public repos. Default: proceed
   with the 12.2 mitigations; the GPU job is never a required check. If
   the box is unreliable or the security posture is unwanted, Cirun (free
   platform for public repos, runners in a cloud account) is the escape
   hatch; GitHub's T4 runners need the Team plan.
4. `webgpu@0.4.0` pin versus upgrading the dev container / runner image to
   Ubuntu 24.04 (glibc 2.39) to use 0.6.x (which also adds the
   unmap-on-destroy shim). Default: stay on 0.4.0 through P3; upgrade the
   images in one change afterwards.
5. Grid pyramid (option A) as the primary approximation versus a
   GraphWaGu-style cluster tree (option B). Default: A first (no locks, no
   float atomics, fixed loops, 3D, production reference in cosmos); B only
   if P3's hub-heavy fixtures show the near field degrading. Owner may
   prefer B for parity with GraphWaGu's published numbers.
6. `exactMaxNodes` default 16,384 is an estimate; the crossover on
   integrated GPUs is ~8k. Default: fix from P3 measurements on the 4070
   and one integrated GPU; expose the option.
7. Readback at 1M nodes in the browser (~30 ms per batch) is frame-limiting.
   Default: `iterationsPerStep` auto-raise above 250k nodes (fewer
   readbacks); the structural fix is sharing a device with a Babylon
   `WebGPUEngine`, which is not scheduled (the element renders on WebGL).
8. Settlement threshold for FA2 (7.16): relative `1e-3` over 10
   iterations is a guess; FA2's adaptive speed may never reach it, in
   which case `maxIter` (100) stops it. Default: ship with these values,
   tune from stories; `maxIter` remains the hard stop.
9. Element default engine: should `ngraph` route to the GPU simulation
   above `gpuMinNodes` when an accelerator is present (the `"coulomb"`
   preset of 7.19)? Default: no in v1 -- the GPU layout is selected by
   type (`forceatlas2` / `spring`); the preset is reserved.
10. `weight` becomes LIVE for FA2 through the snapshot (it is inert in the
    element today, note 01 section 7.1). Default: accept the documented
    behaviour change at L1 / E1.
11. Directed snapshots passed directly to a GPU layout in Node. Default:
    `E_SNAPSHOT` ("pass `toUndirected().snapshot`"), matching
    `toLayoutSnapshot` semantics; graphty-element always passes the
    undirected copy.
12. Labels renumbered on the CPU in first-seen order (identical `groups()`)
    costs O(n) per CC readback. Default: do it (parity is worth 1 ms at
    1M nodes); a `renumber: false` option returns raw roots.
13. Betweenness at 1M nodes is minutes even sampled with 1k sources.
    Default: the accelerator exposes `sources` / `k`, `onProgress` and
    `signal`; graphty-element's adapter defaults to sampled BC above 50k
    nodes (a product decision for E1 / W2).
14. Louvain partitions will not match the CPU's; parity is a modularity
    band. Default: accept; the adapter labels the result "GPU Louvain".
15. Cancellation semantics: `E_ABORTED` on `signal.abort()`; a submitted
    batch still completes on the device. Default: accept.
16. Unverified platform facts carried from the notes: Dawn-node's default
    device limits without `requiredLimits` (assumed spec defaults);
    whether Dawn-node quantises timestamps; Firefox / Safari exposure of
    `subgroups` / `timestamp-query`; lavapipe on an actual GitHub runner
    (only the local container was tested); the exact Vitest 3 spelling of
    per-instance Playwright launch args; the spec's "does nothing" for
    over-limit indirect dispatches (moot: the finalize kernel clamps).
    Default: P0 / P1 verify each and record the answer in the package
    `CLAUDE.md`.
17. Performance numbers in 7.20 and 10 are extrapolations except the cited
    probes. Default: treat the P2 / P3 / P5 / P6 gates as the real
    targets; revise the tables from `benchmarks/results/`.
18. Where does the plan directory (`tmp/webgpu-plan`) go? Default: the
    accepted plan is committed as `docs/webgpu-plan.md` in the package;
    the research notes stay gitignored under `tmp/`.

---------------------------------------------------------------------------

## 15. References

Local (read-only unless stated):

- `/home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md` -- sections 10 (GPU contract), 14.3 (`LayoutSimulation`, layout ports), 14.4 (graphty-element ownership, `snapshot-replaced`), 14.5 (WebGPU package move-in), 14.6 (landing order), 15 (performance model), 16 (testing).
- `/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/01-layout-needs.md` .. `07-format-api-and-conventions.md` -- the seven research notes this plan is built on (each carries its own URL list).
- `/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/src/index.ts`, `src/types/columns.ts`, `src/types/snapshot.ts`, `src/snapshot/views.ts`, `src/snapshot/derived.ts`, `src/builder/arena.ts`, `src/util/mask.ts`, `src/columns/column.ts` -- the implemented format API (names used throughout).
- `/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/test/audit/gpu-upload.test.ts`, `gpu-contract.test.ts` -- Dawn-in-Node device acquisition, the upload contract audit, the 65,535 dispatch assertion.
- `/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-io/package.json`, `scripts/*.js`, `tsconfig*.json`, `vitest.config.ts` -- the package skeleton mirrored.
- `/home/apowers/Projects/webgpu-graph-algorithms/HEADLESS_GPU_REPORT.md` -- the four Chromium flags, the `libEGL.so.1` root cause, Chrome 145 `powerPreference`.
- `/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/probe/*.mjs` -- the measurements cited (`dawn-perf.mjs` exact tile 1.11 ms at 20k; `dawn-latency.mjs`, `chromium-latency.mjs` round trips; `bench-node.mjs` gather kernel).
- `/home/apowers/Projects/graphty-monorepo/layout/src/layouts/force-directed/forceatlas2.ts`, `fruchterman-reingold.ts`, `layout/src/utils/random.ts` -- the CPU layouts and LCG reproduced.
- `/home/apowers/Projects/graphty-monorepo/graphty-element/src/layout/LayoutEngine.ts`, `NGraphLayoutEngine.ts`, `ForceAtlas2LayoutEngine.ts`, `src/managers/LayoutManager.ts`, `UpdateManager.ts`, `src/algorithms/Algorithm.ts`, `src/config/GraphBehavior.ts`, `src/ai/providers/index.ts` -- the element seams changed in 9.4.
- `/home/apowers/Projects/graphty-monorepo/.github/workflows/ci.yml`, `release.yml`, `nx.json`, `tools/merge-coverage.sh`, `packages/move/root-touch-points.diff` -- the CI shard matrix and release process slotted into in 12.5.
- `/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/repos/cuda-ffi/.github/workflows/build.yml` -- the owner's self-hosted GPU runner precedent (`runs-on: cudaffi-gpu-runner`, `--gpus all`).

External (URLs; what each was used for):

- https://github.com/atoms-org/cuda-ffi -- self-hosted GPU runner pattern (owner-supplied).
- https://github.com/cosmosgl/cosmos -- grid-pyramid many-body force, Monte-Carlo near field, measured ms/step, failure modes (owner-supplied; source read, MIT). https://cosmograph.app/examples , https://pypi.org/project/cosmograph/ -- the product over cosmos.gl (owner-supplied; not fetched by the notes, no algorithmic content used).
- https://github.com/harp-lab/GraphWaGu -- WebGPU FR + Barnes-Hut, WGSL radix sort, level-wise tree build, i32 fixed-point bbox atomics, published ms/iteration (owner-supplied; source and PacificVis / EGPGV papers read, MIT).
- https://github.com/jaredmcqueen/analytics -- WebGL1 O(n^2) FR; excluded (GPL, not credible at 1M) (owner-supplied).
- https://research.nvidia.com/publication/2011-08_high-performance-and-scalable-gpu-graph-traversal -- Merrill, Garland, Grimshaw 2011: scan-based frontier expansion, gather tiers, expand / contract couplings (owner-supplied).
- https://dl.acm.org/doi/10.1145/3230485 (403) / https://davidbader.net/publication/2018-mb/2018-mb.pdf -- McLaughlin, Bader: work-efficient vs edge-parallel BC, atomic-free dependency accumulation, sampling (owner-supplied).
- https://cse.buffalo.edu/tech-reports/2023-06.pdf -- Kumar, dense-matrix BC; excluded as a negative result for sparse graphs (owner-supplied).
- https://developer.nvidia.com/discover/cluster-analysis -- nvGRAPH-era spectral / multilevel overview; background only (owner-supplied).
- https://github.com/rapidsai/cugraph -- FA2 (Burtscher BH port, `fa2_kernels.cuh`), PageRank pull, BFS direction-optimizing constants, SSSP near-far, BC batching, Louvain, k-core, triangle counting (Apache-2.0; read as design input, not copied).
- https://github.com/gunrock/gunrock -- `block_mapped` advance, `neighborreduce`, push PR / BC as float-atomic counter-examples.
- https://raw.githubusercontent.com/sbeamer/gapbs/master/src/cc.cc -- Afforest connected components.
- https://scottbeamer.net/pubs/beamer-sc2012.pdf -- direction-optimizing BFS alpha / beta.
- https://escholarship.org/content/qt8qr166v2/qt8qr166v2.pdf -- Davidson et al. 2014 near-far SSSP and the ownership dedupe.
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679 -- ForceAtlas2 (Jacomy et al. 2014) formulas (CC-BY); Gephi `ForceAtlas2.java` / `ForceFactory.java` (GPL / CDDL: formulas only, no code).
- https://userweb.cs.txstate.edu/~burtscher/papers/gcg11.pdf , https://liacs.leidenuniv.nl/~takesfw/pdf/exploiting-gpus-fast.pdf -- Burtscher-Pingali BH and Brinkmann et al. 2017 timings; why the locked tree is not portable.
- https://arxiv.org/html/2501.19004 , https://arxiv.org/html/2608.01503 -- nu-Louvain and Gilbert-Madduri: GPU Louvain expectations.
- https://gpuweb.github.io/gpuweb/ , https://gpuweb.github.io/gpuweb/wgsl/ -- limits, `requestDevice`, `dispatchWorkgroupsIndirect`, error scopes, device loss; WGSL atomics (6.2.8), recursion (11.4), overrides (7.2.2), uniform layout (14.4.5), subgroups (17.12).
- https://github.com/gpuweb/gpuweb/wiki/Implementation-Status , https://developer.chrome.com/blog/new-in-webgpu-120 / -121 / -128 / -134 , https://mozillagfx.wordpress.com/2025/07/15/shipping-webgpu-on-windows-in-firefox-141/ , https://webkit.org/blog/17333/webkit-features-in-safari-26-0/ -- browser availability and feature timeline (subgroups Chrome 134, timestamp quantisation, `adapter.info`).
- https://registry.npmjs.org/webgpu , https://github.com/dawn-gpu/node-webgpu , https://dawn.googlesource.com/dawn/+/refs/heads/main/src/dawn/node/README.md -- the `webgpu` npm package versions, glibc requirements, `create()` options, `adapter=llvmpipe`.
- https://raw.githubusercontent.com/dawn-gpu/node-webgpu/main/.github/workflows/build.yml , https://raw.githubusercontent.com/gfx-rs/wgpu/trunk/.github/workflows/ci.yml , https://raw.githubusercontent.com/mrdoob/three.js/dev/.github/workflows/ci.yml , https://raw.githubusercontent.com/mrdoob/three.js/dev/test/e2e/puppeteer.js -- lavapipe / SwiftShader CI patterns, the `browser.close()` hang.
- https://github.com/chromium/chromium/blob/main/docs/gpu/swiftshader.md -- SwiftShader flags.
- https://docs.github.com/en/actions/reference/runners/larger-runners , https://docs.github.com/en/billing/reference/actions-runner-pricing , https://docs.github.com/en/enterprise-cloud@latest/actions/concepts/runners/larger-runners , https://github.blog/changelog/2024-07-08-github-actions-gpu-hosted-runners-are-now-generally-available/ -- GPU runner spec, price, plan gate.
- https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions#hardening-for-self-hosted-runners , https://docs.github.com/en/actions/reference/runners/self-hosted-runners , https://docs.github.com/en/rest/actions/self-hosted-runners?apiVersion=2022-11-28#create-configuration-for-a-just-in-time-runner-for-a-repository -- ephemeral / JIT runners, security guidance.
- https://cirun.io/ , https://runs-on.com/runners/gpu/ , https://machine.dev/docs/platform-specifications/gpu-runners/ -- third-party GPU runner options (not evaluated beyond their pages).
- https://betatim.github.io/posts/github-action-with-gpu/ , https://davesnider.com/posts/gputests -- practitioner reports (label-gated GPU jobs; headless Chromium on a T4 needing Vulkan flags).
- https://vite.dev/guide/assets -- `?raw` imports (rejected in favour of `.wgsl.ts`).
- https://docs.npmjs.com/cli/v10/configuring-npm/package-json -- optional peer dependency semantics.
- https://arxiv.org/abs/2303.03964 -- t-FDP (FFT far field), noted as a non-pursued option.
