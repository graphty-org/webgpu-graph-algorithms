# P0-P3 interface contract for @graphty/webgpu-graph-algorithms

Status: normative interface contract for phases P0, P1, P2 and P3 of
`design/webgpu-acceleration-plan.md` (the "spec"). Date: 2026-09-14.
Scope: the package `packages/webgpu-graph-algorithms/`, the repository root
after the P0 triage, the CI workflows, and every test, benchmark and script
file those four phases create. Plain ASCII throughout.

## 0. How to use this document

This document is NORMATIVE for every plan writer and implementer of P0-P3:
a file that is not declared here is not written, a signature that is not
declared here does not exist, and a declaration here is copied, not
re-derived. Where a later phase (P4+) needs a name to exist for API
stability (the windowed `ArcWindow` type, `planGridStride`, `planIndirect`,
the grid fields of `LayoutStatsBase`, the `GpuLayoutTuning` grid knobs) the
name is declared with its P4+ behaviour stubbed as `E_UNSUPPORTED` or `null`,
and the declaration says so; nothing from P4+ is declared as implemented.
The spec sections a declaration comes from are cited inline as `(spec 4.2)`;
the lead's binding decisions are cited as `(lead a)` .. `(lead f)`.

Reading order for an implementer: section 7 (the task you own and the files
it owns), then section 3 (the declarations of those files), then section 4
(the WGSL you write), then section 5 or 6 (the tests or benchmarks you write).
A plan writer reads sections 1, 7 and 9 first. Every declaration uses the
house conventions of spec 3.6: `?: T | undefined` for options, explicit
return types, `.js` suffixes on relative imports, no default exports, a JSDoc
one-liner on every export, results typed `Uint32Array<ArrayBuffer>` /
`Float32Array<ArrayBuffer>` through graph-format's `U32` / `F32` aliases.

Where the spec leaves something open and a choice had to be made, the choice
is marked `CONTRACT DECISION:` with a one-line reason; a plan writer who
disagrees changes THIS document first (and the spec's Review log, per its
"How to read" rule), never the code alone. Two of those decisions correct
factual statements of spec 7.2 about NetworkX (section 9 lists them for the
owner, because the 7.2 table is the G0 sign-off artefact); every other
decision fills a gap rather than contradicting the spec.

## 1. Repository and package tree after P3

### 1.1 The repository root after the P0 triage (spec 13 row P0; note 07 section 5)

```
/ (graphty-org/webgpu-graph-algorithms, the public staging repository)
+-- CLAUDE.md                          kept as is ("Never create fallbacks if WebGPU isn't supported")
+-- README.md                          REWRITTEN at P0: one screen -- what the repository stages, the package list, pointers to packages/README.md, design/ and docs/superpowers/plans/
+-- .gitignore                         REWRITTEN at P0: node_modules, dist, coverage, tmp/, .tmp/, *.tsbuildinfo, benchmarks/out/, browser-results.json, gpu-report.json, *.log, .DS_Store
+-- .github/workflows/ci.yml           P0 (section 2.9; the spec 12.3 text plus the P0 deltas)
+-- .github/workflows/gpu.yml          P0 (section 2.9)
+-- design/webgpu-acceleration-plan.md the spec (unchanged by P0-P3 except Review-log appendices the owner writes)
+-- docs/superpowers/plans/            this contract and the per-phase plans (2026-09-14-webgpu-p0-p3-interfaces.md, <date>-webgpu-p0.md, -p1.md, -p2.md, -p3.md)
+-- packages/                          the pnpm workspace root (unchanged files: package.json*, pnpm-workspace.yaml*, pnpm-lock.yaml*, knip.config.ts*, eslint.config.js, tsconfig.base.json, vite.shared.config.ts, vitest.shared.config.ts, .prettierrc, .prettierignore, .npmrc, .gitignore*, README.md, STATUS.md, CONFORMANCE.md, MIGRATION_PROMPT.md, move/)   * = edited at P0 (section 2.8)
|   +-- graph-format/                  untouched
|   +-- graph-io/                      untouched
|   +-- webgpu-graph-algorithms/       the package (1.2)
+-- tmp/                               gitignored scratch (research clones, probes, the nx venv)
```

Deleted at P0 (every one is untracked-by-intent scaffold, note 07 section 5):
`.env`, `.env.example`, `.husky/`, `.vscode/`, `dist/`, `examples/`,
`eslint.config.js`, `knip.json`, `node_modules/` (root), `package.json`
(root), `package-lock.json`, `src/`, `test/`, `tsconfig.json`,
`vite.config.ts`, `vitest.config.ts`, `STRATEGY.md`,
`IMPLEMENTATION_CHECKLIST.md`, `.github/workflows/test.yml`. Moved at P0:
`HEADLESS_GPU_REPORT.md` -> `packages/webgpu-graph-algorithms/docs/HEADLESS_GPU_REPORT.md`
(`git mv`, so history follows). Copied at P0 (spec Q-18):
`tmp/webgpu-plan/0[1-7]-*.md` and `draft-{A,B,C}.md` ->
`packages/webgpu-graph-algorithms/docs/research/`; `tmp/webgpu-plan/review/`
(`*-review.md`, `*-verdicts.md`, `check-report.md`, `fix-report.md`,
`repair-report.md`, `probes/*.mjs`, `probes/*.log`) ->
`packages/webgpu-graph-algorithms/docs/research/review/`. `tmp/` itself stays
gitignored (cloned repositories, papers, the venv).

### 1.2 The package tree after P3

Every line: path, the phase (and task, section 7) that CREATES it, one-line
responsibility. A later phase may modify a file its task list names; within
one phase exactly one task owns each file (lead d).

```
packages/webgpu-graph-algorithms/
+-- package.json                       P0-T2  manifest (2.1); replaces the 0.0.0 placeholder committed on 2026-09-14 (Q-28)
+-- project.json                       P0-T2  Nx project "webgpu-graph-algorithms" (2.2)
+-- tsconfig.json                      P0-T2  lint / typecheck config (2.3)
+-- tsconfig.build.json                P0-T2  emit config, stripInternal (2.3)
+-- tsconfig.strict-consumer.json      P0-T2  the design-16.6 strict compile of test/types/*.test-d.ts against dist/*.d.ts (2.3)
+-- eslint.config.js                   P0-T2  root config spread + layer zones + globals + entry isolation (2.4)
+-- vitest.config.ts                   P0-T2  three projects node / node-limits / browser, thresholds rule, env forwarding, browser flags, the browser commands bridge (2.5)
+-- README.md                          P0-T3  package README (Node + browser recipes, the degree diagnostic, the performance table placeholder); P1-T7 and P3-T7 extend
+-- CLAUDE.md                          P0-T3  the package guidance (spec 3.1 section list); every phase's last task updates "Verified Platform Facts"
+-- LICENSE                            exists (MIT, committed with the placeholder)
+-- scripts/entries.js                 P0-T2  the three bundle entries (2.6)
+-- scripts/build-bundle.js            P0-T2  graph-io's multi-entry vite lib build, unchanged but for the header (2.6)
+-- scripts/bundle-types.js            P0-T2  graph-io's d.ts shim writer, unchanged but for the header (2.6)
+-- scripts/gpu-policy.js              P0-T2  parseGpuRequire / checkAdapter / isSoftwareInfo -- the ONE copy of the adapter policy (6.5)
+-- scripts/gpu-policy.d.ts            P0-T2  hand-written declarations so test/setup/gpu.ts imports the .js under strict TypeScript (6.5)
+-- scripts/runner-class.js            P0-T2  runnerClass(info, env) -- the ONE copy of the runner-class rule + the GRAPHTY_RUNNER_CLASS override (6.9)
+-- scripts/runner-class.d.ts          P0-T2  its declarations (benchmarks/harness.ts re-exports it; policy.test.ts pins it) (6.9)
+-- scripts/gpu-report.js              P0-T2  adapter report, policy exit code, runner class, nvidia-smi sample (6.6)
+-- scripts/run-browser-project.js     P0-T2  the timeout -k 10 600 wrapper with the exit-124 rule (6.7)
+-- scripts/bench-compare.js           P0-T2  the 3x regression check with the quiet-GPU skip rule (6.8); rule complete at P1-T7
+-- docs/HEADLESS_GPU_REPORT.md        P0-T1  moved from the repository root
+-- docs/research/0[1-7]-*.md          P0-T1  the seven research notes (Q-18)
+-- docs/research/draft-{A,B,C}.md     P0-T1  the three drafts (Q-18)
+-- docs/research/review/              P0-T1  the six review pairs, the three reports, probes/*.mjs and *.log (Q-18)
+-- docs/decisions/G0.md .. G3.md      P0-T4, P1-T7, P2-T3, P3-T7  the gate records: measured numbers, sign-offs, the re-fixed constants with their citations (spec 10.4 rule)
+-- src/index.ts                       P0-T3  the only public barrel; explicit named exports; /// <reference types="@webgpu/types" /> (3.15)
+-- src/errors.ts                      P0-T3  WebGpuGraphError, WebGpuGraphErrorCode, PASSTHROUGH_FORMAT_CODES (3.1)
+-- src/constants.ts                   P0-T3  every numeric constant the package and the WGSL prelude share (3.2)
+-- src/browser/index.ts               P0-T3 (empty module) / P1-T1  the ./browser entry (3.7)
+-- src/node/index.ts                  P0-T3 (createNodeGpu) / P1-T1 (the two context helpers)  the ./node entry; the ONLY importer of "webgpu" (3.7)
+-- src/types/context.ts               P0-T3 (AdapterInfoLike, AdapterSummary) / P1-T1 (the rest)  GpuCaps, GpuContextOptions, LimitPolicy, RaisableLimit, ProbeOptions, ProbeResult, AdapterSummary, AdapterInfoLike, PlanCaps (3.3)
+-- src/types/run.ts                   P1-T1  GpuRunOptions (3.3)
+-- src/types/memory.ts                P1-T1  Binding, ArcWindow (types only, so kernel/ never imports memory/ for a type) (3.3)
+-- src/types/options.ts               P3-T1  CommonLayoutOptions, SimulationOptions, ForceAtlas2Options (+ the FR / spring-electrical option mirrors, types only) (3.3)
+-- src/types/layout.ts                P3-T1  LayoutStatsBase, ForceAtlas2Stats, GpuLayoutSimulation, GpuLayoutTuning, RunOptions, TraceRecord (3.3)
+-- src/types/accelerator.ts           P3-T1  the structural mirrors LayoutSimulation / LayoutAccelerator / AlgorithmAccelerator + result-like types, GpuAccelerator, AcceleratorOptions (3.3)
+-- src/device/webgpu-constants.ts     P0-T3  BufferUsage / MapMode / ShaderStage numeric constants (3.4)
+-- src/device/acquire.ts              P0-T3 (isSoftwareAdapter, summarizeAdapter) / P1-T1 (requestAdapter, requestDevice, buildRequiredLimits, RAISABLE_LIMITS) (3.4)
+-- src/device/caps.ts                 P1-T1  captureCaps, capsFromDevice, assertPlanLimits (3.4)
+-- src/device/error-scope.ts          P1-T1  withValidationScope, AllocationTracker (OOM scopes), formatCompilationInfo (3.4)
+-- src/device/lost.ts                 P1-T1  PendingErrorSlot, installUncapturedErrorSink, watchDeviceLost (3.4)
+-- src/context.ts                     P1-T1  GpuContext: probe / create / from / release / dispose; owns the singletons (3.5)
+-- src/memory/upload-plan.ts          P1-T2  planUpload (pure), UploadPlan union, planArcWindows (3.8; ArcWindow lives in src/types/memory.ts)
+-- src/memory/residency.ts            P1-T2  GraphResidency, CoreBinding / ViewBinding / ColumnBinding / ArrayBinding, ResidencyStats (3.8; Binding lives in src/types/memory.ts)
+-- src/memory/buffer-pool.ts          P1-T2  BufferPool size classes, lease() (3.8)
+-- src/memory/readback.ts             P1-T2  Readback staging ring: read / readU32 / borrowSlot / returnSlot (3.8)
+-- src/memory/lease.ts                P2-T1  Lease scope object (3.8)
+-- src/kernel/wgsl.ts                 P1-T3  BindingDecl, OverrideDecl, WgslModuleSpec, composeWgsl, STANDARD_OVERRIDES, ComposedModule (3.9)
+-- src/kernel/prelude.ts              P1-T3  the prelude text + the two reduction-helper blocks (CONTRACT DECISION 3.9: lives in kernel/, not wgsl/)
+-- src/kernel/struct-block.ts         P1-T3  UniformBlock (uniform + storage layouts), UniformValues, UniformFieldType (3.9)
+-- src/kernel/pipeline-cache.ts       P1-T3  PipelineCache: get / kernel / warm / key / size (3.9)
+-- src/kernel/kernel.ts               P1-T3  Kernel: bind / dispatch; BoundKernel; KernelBindings; the graph-group dummy rules (3.9)
+-- src/kernel/dispatch.ts             P1-T3  DispatchPlan, plan1d, plan2d, planGridStride (stub), planIndirect (stub) (3.9)
+-- src/kernel/profiler.ts             P1-T1 (shell) / P2-T1  Profiler timestamps (3.9)
+-- src/kernel/uniform-ring.ts         P2-T1  UniformRing (3.9)
+-- src/kernel/batch.ts                P2-T1  CommandBatch, BatchHost, SubmittedBatch, ReadbackRequest (3.9)
+-- src/kernels.ts                     P1-T4 (degree, reduce, fill, fa2-repulsion-exact, fa2-speed-finalize + the blocks) / P2-T2 (segmented-reduce) / P3-T2 (fa2-stats-finalize, fa2-attraction, fa2-integrate, fa2-to-scene)  THE registry (3.10)
+-- src/wgsl/degree.wgsl.ts            P1-T4  body (4.5)
+-- src/wgsl/reduce.wgsl.ts            P1-T4  body (4.5)
+-- src/wgsl/fill.wgsl.ts              P1-T4  body (4.5)
+-- src/wgsl/fa2-repulsion-exact.wgsl.ts  P1-T4  K3 body (4.5)
+-- src/wgsl/fa2-speed-finalize.wgsl.ts   P1-T4  K4 body (4.5)
+-- src/wgsl/segmented-reduce.wgsl.ts  P2-T2  thread-per-row body (4.5)
+-- src/wgsl/fa2-stats-finalize.wgsl.ts   P3-T2  K1 body (4.5)
+-- src/wgsl/fa2-attraction.wgsl.ts    P3-T2  K2 body (4.5)
+-- src/wgsl/fa2-integrate.wgsl.ts     P3-T2  K5 body (4.5)
+-- src/wgsl/fa2-to-scene.wgsl.ts      P3-T2  toScene body (4.5)
+-- src/primitives/reduce.ts           P1-T5  reduce(batch-or-pass ...) driver, 2-3 levels (3.11)
+-- src/primitives/segmented-reduce.ts P2-T2  segmentedReduce, thread-per-row tier; DegreeTiers (3.11)
+-- src/algorithms/degree.ts           P1-T5  degree(ctx, s, options?) -- the walking-skeleton algorithm and public diagnostic (3.12)
+-- src/layouts/repulsion-exact.ts     P1-T6 (K3 + K4 stage) / P3-T2 (wired into the model)  RepulsionExact (3.13)
+-- src/layouts/seed.ts                P3-T1  the LCG and seedPositions (3.13)
+-- src/layouts/inputs.ts              P3-T1  resolveNodeMass, resolveWeights (3.13)
+-- src/layouts/force-simulation.ts    P3-T1  ForceSimulation + ForceModel + BufferSpec + StateWriter + ModelResources (3.13)
+-- src/layouts/forceatlas2.ts         P3-T2  ForceAtlas2Model, createForceAtlas2 (3.13)
+-- src/accelerator.ts                 P3-T3  createAccelerator (3.14)
+-- test/setup/gpu.ts                  P0-T3 (policy, acquireRaw, requireGpu, gpuScale) / P1-T1 (acquire -> GpuContext)  the node projects' setup file (5.1)
+-- test/setup/browser.ts              P0-T3  the browser project's setup file (5.1)
+-- test/setup/global.ts               P0-T3 (empty) / P2-T2  vitest globalSetup of the node project: the override-matrix coverage teardown (5.1)
+-- test/setup/browser-commands.d.ts   P0-T3 / P1-T7 (NoiseRow) / P3-T6 (BrowserBenchPayload)  the BrowserCommands augmentation for the three commands of 2.5 and the ImportMetaEnv keys (5.1)
+-- test/helpers/device.ts             P1-T2  raw-buffer helpers around a context (5.2)
+-- test/helpers/kernel.ts             P1-T3  runKernel, the pre-CommandBatch dispatch driver (5.2)
+-- test/helpers/linear-id.ts          P1-T4  the 17M-item test's shared constants and checksum (5.2)
+-- test/helpers/graphs.ts             P1-T2  KARATE_EDGES, gridEdges, pathEdges, starEdges, completeEdges, randomEdges, snapshotOf, fixtures list (5.2)
+-- test/helpers/matchers.ts           P1-T2  expectAllClose, expectBitwiseEqual, flooredRelError, maxRelError (5.2)
+-- test/helpers/leak-counter.ts       P1-T7  LeakCounter around a device (5.2)
+-- test/helpers/caps-tables.ts        P1-T2  the faked PlanCaps tables (5.2; lead b); P2-T2 adds CAPS_INTEL_XE
+-- test/helpers/override-matrix.ts    P2-T2  OVERRIDE_MATRIX and matrixCovers (5.2)
+-- test/helpers/sabotage.ts           P1-T5 (degree, reduce, K3, K4 rows -- the bodies are normative in 4.5, so no other P1 task edits it) / P2-T2 (segmented-reduce) / P3-T5 (K1, K2, K5)  the mutation table, SABOTAGE_PHASES and the splice helper (5.2)
+-- test/helpers/noise-floor.ts        P1-T5  noise fixture writer / reader, recordNoiseRow, noiseFloorFor (5.2; Node only)
+-- test/helpers/noise-floor-browser.ts  P1-T7  the browser twin of the two writers over the commands bridge (5.2)
+-- test/helpers/frame-loop.ts         P3-T6  runFrameLoop, the element bridge logic (5.2)
+-- test/helpers/metrics.ts            P3-T5  stress, edge-length quantiles, nearest-neighbour histogram, component separation (5.2)
+-- test/oracle/degree.ts              P1-T5  outDegreeOracle (5.3)
+-- test/oracle/reduce.ts              P1-T5  reduceOracle in f64 (5.3)
+-- test/oracle/segmented-reduce.ts    P2-T2  segmentedReduceOracle in f64 (5.3)
+-- test/oracle/forceatlas2.ts         P3-T4  ForceAtlas2Oracle: f64 / f32, compat modes, stages, trace (5.3)
+-- test/fixtures/networkx/generate.py P3-T4  the fixture generator (5.4)
+-- test/fixtures/networkx/*.json      P3-T4  the committed NetworkX trajectories (5.4)
+-- test/fixtures/noise/*.json         P1-T5 (degree, reduce) / P1-T6 (fa2 skeleton) / P1-T7 (the SwiftShader files, generated by the browser skeleton run) / P2-T2 (segmented-reduce) / P3-T5 (K1-K5)  per-adapter raw outputs for the noise floor (5.6)
+-- test/fixtures/rich-v1.gsnp         P1-T2  copied from graph-format: the fromBytes container with every dtype (upload contract)
+-- test/device/*.test.ts              P0-T3, P1-T1, P2-T1  (5.5)
+-- test/memory/*.test.ts              P1-T2, P2-T1  (5.5)
+-- test/kernel/*.test.ts              P1-T3, P1-T4, P2-T1, P2-T2  (5.5)
+-- test/primitives/*.test.ts          P1-T5, P2-T2  (5.5)
+-- test/algorithms/degree.test.ts     P1-T5  (5.5)
+-- test/layouts/*.test.ts             P1-T6, P3-T1, P3-T2, P3-T5, P3-T6  (5.5)
+-- test/oracle/*.test.ts              P3-T4  forceatlas2-networkx.test.ts, swing-mode.test.ts (5.5)
+-- test/sabotage/*.test.ts            P1-T5 (degree, reduce, coverage), P1-T6 (fa2-skeleton), P2-T2 (segmented-reduce), P3-T5 (fa2)  (5.5)
+-- test/browser/*.test.ts             P0-T3 (webgpu-check), P1-T7 (entry, skeleton), P2-T1 (batch, lost, state-roundtrip, uniform-layout), P2-T2 (compile-matrix), P3-T6 (forceatlas2, bench)  (5.5)
+-- test/leak.test.ts                  P1-T7  (5.5)
+-- test/accelerator.test.ts           P3-T3  (5.5)
+-- test/limits/                       directory created at P2-T2 with a README.md placeholder; tests land at P4 (the node-limits project needs an include path)
+-- test/types/public-api.test-d.ts    P0-T3 (extended P1-T7, P3-T3)  the strict-consumer sample (5.5)
+-- test/types/accelerator.test-d.ts   P3-T3  (5.5)
+-- test/types/options.test-d.ts       P3-T3  (5.5)
+-- test/index.test.ts                 P0-T3 (pinned list updated by P1-T7, P3-T3)  the barrel export list (5.5)
+-- test/build-output.test.ts          P0-T3 (P1-T7 adds the bundle specifier assertions)  (5.5)
+-- test/layers.test.ts                P0-T3  import graph, cycles, greps (5.5; lead a)
+-- test/errors.test.ts                P0-T3  (5.5)
+-- test/noise-floor.test.ts           P1-T6 (P3-T5 extends)  (5.6)
+-- benchmarks/run.ts                  P0-T2 (stub) / P1-T7 (harness wiring) / P3-T7 (layout-exact group)  (6.1)
+-- benchmarks/harness.ts              P1-T7  bench (async), BenchResult, appendSession with the gpu field, runnerClass (6.1)
+-- benchmarks/datasets.ts             P1-T7  seeded generators (6.2)
+-- benchmarks/upload.bench.ts         P1-T7  T-1 (6.3)
+-- benchmarks/roundtrip.bench.ts      P1-T7  T-2, T-3 (6.3)
+-- benchmarks/layout-exact.bench.ts   P3-T7  T-4 ladder (6.3)
+-- benchmarks/layout-run.ts           P3-T7  the end-to-end Node layout driver (6.3)
+-- benchmarks/results/<runner-class>.json   P1-T7 (nvidia 4070), P0-T4 / P1-T7 (gpu-linux-t4)  checked-in baselines (6.4)
+-- benchmarks/results/noise-floor.json      P1-T6 (P3-T5 extends)  (5.6)
+-- benchmarks/out/                    gitignored run output
```

## 2. Manifest and configuration files

Every file in this section is given in full; the P0-T2 implementer copies
it. Where a later phase edits one of them, the edit is named in section 7.

### 2.1 package.json (spec 3.1, 2.5; note 07 section 4.2)

```json
{
    "name": "@graphty/webgpu-graph-algorithms",
    "version": "0.1.0",
    "description": "WebGPU-accelerated graph algorithms and layouts over the @graphty/graph-format snapshot, for Node (Dawn) and browsers",
    "author": "Adam Powers <apowers@ato.ms>",
    "type": "module",
    "main": "dist/webgpu-graph-algorithms.js",
    "types": "dist/webgpu-graph-algorithms.d.ts",
    "exports": {
        ".": {
            "types": "./dist/webgpu-graph-algorithms.d.ts",
            "import": "./dist/webgpu-graph-algorithms.js",
            "default": "./dist/webgpu-graph-algorithms.js"
        },
        "./browser": {
            "types": "./dist/browser.d.ts",
            "import": "./dist/browser.js",
            "default": "./dist/browser.js"
        },
        "./node": {
            "types": "./dist/node.d.ts",
            "import": "./dist/node.js",
            "default": "./dist/node.js"
        }
    },
    "sideEffects": false,
    "files": ["dist/", "src/", "README.md", "LICENSE"],
    "publishConfig": { "access": "public", "provenance": true },
    "engines": { "node": ">=18.19.0" },
    "scripts": {
        "build": "tsc -p tsconfig.build.json",
        "build:bundle": "node scripts/build-bundle.js",
        "build:all": "npm run build && npm run build:bundle",
        "lint": "eslint && tsc --noEmit -p tsconfig.json && tsc -p tsconfig.strict-consumer.json",
        "lint:fix": "eslint --fix",
        "typecheck": "tsc --noEmit -p tsconfig.json",
        "typecheck:strict-consumer": "tsc -p tsconfig.strict-consumer.json",
        "test": "vitest",
        "test:ui": "vitest --ui",
        "test:run": "vitest run --project=node",
        "test:node": "vitest run --project=node",
        "test:browser": "vitest run --project=browser",
        "test:browser:ci": "node scripts/run-browser-project.js",
        "test:limits": "vitest run --project=node-limits",
        "coverage": "vitest run --project=node --coverage",
        "coverage:preview": "npx serve coverage -p 9058",
        "bench": "tsx benchmarks/run.ts",
        "bench:compare": "node scripts/bench-compare.js",
        "benchmark": "tsx benchmarks/run.ts",
        "gpu:report": "node scripts/gpu-report.js",
        "ready:commit": "npm run build:all && npm run lint && npm run test:node"
    },
    "repository": {
        "type": "git",
        "url": "git+https://github.com/graphty-org/webgpu-graph-algorithms.git",
        "directory": "packages/webgpu-graph-algorithms"
    },
    "keywords": ["graph", "webgpu", "wgsl", "gpu", "graph-algorithms", "force-directed", "forceatlas2", "layout", "graph-format"],
    "license": "MIT",
    "bugs": { "url": "https://github.com/graphty-org/webgpu-graph-algorithms/issues" },
    "homepage": "https://github.com/graphty-org/webgpu-graph-algorithms/tree/master/packages/webgpu-graph-algorithms#readme",
    "dependencies": {
        "@graphty/graph-format": "workspace:^",
        "@webgpu/types": "^0.1.72"
    },
    "peerDependencies": {
        "@graphty/algorithms": "^1.0.0",
        "@graphty/graph-format": "^0.1.0",
        "@graphty/layout": "^1.0.0",
        "webgpu": ">=0.4.0 <1.0.0"
    },
    "peerDependenciesMeta": {
        "@graphty/algorithms": { "optional": true },
        "@graphty/layout": { "optional": true },
        "webgpu": { "optional": true }
    },
    "devDependencies": {
        "@vitest/browser": "^3.2.4",
        "@vitest/coverage-v8": "^3.2.4",
        "@vitest/ui": "^3.2.4",
        "fast-check": "^4.2.0",
        "playwright": "^1.54.1",
        "tsx": "^4.20.3",
        "typescript": "^5.9.3",
        "vite": "^7.0.5",
        "vitest": "^3.2.4",
        "webgpu": "0.4.0"
    }
}
```

Notes: `workspace:^` not `workspace:*` (spec 2.5, Q-31); `webgpu` devDependency
is the EXACT pin `0.4.0` (spec 2.5: bumped once at P-ENV together with the
`E_NO_WEBGPU` install hint) while the peer range admits 0.6.x; the two CPU
packages are optional peers from day one (spec 2.5 literal; D27) but NOT
devDependencies until W1 (the mirrors stand in). `test:run` is the node
project only so `pnpm -r run test:run` from the workspace root never launches
Chromium. CONTRACT DECISION: `repository.url` / `directory` keep the values
the published 0.0.0 placeholder already carries (the staging repository and
`packages/webgpu-graph-algorithms`), because npm provenance checks the
publishing repository; W1 rewrites them to the monorepo (spec 3.1's
"`webgpu-graph-algorithms` in `repository.directory`" is the W1 value).
`@graphty/graph-format`'s peer range bumps to `^1.0.0` at F2 (spec 3.1).

### 2.2 project.json (note 07 section 4.3)

```json
{
    "name": "webgpu-graph-algorithms",
    "$schema": "../node_modules/nx/schemas/project-schema.json",
    "sourceRoot": "webgpu-graph-algorithms/src",
    "projectType": "library",
    "tags": [],
    "targets": {
        "build": {
            "executor": "nx:run-commands",
            "outputs": ["{projectRoot}/dist"],
            "options": { "command": "npm run build:all", "cwd": "webgpu-graph-algorithms" },
            "dependsOn": ["^build"]
        },
        "test": { "executor": "nx:run-commands", "options": { "command": "npm run test:run", "cwd": "webgpu-graph-algorithms" } },
        "test:node": { "executor": "nx:run-commands", "options": { "command": "npm run test:node", "cwd": "webgpu-graph-algorithms" } },
        "test:browser": { "executor": "nx:run-commands", "options": { "command": "npm run test:browser:ci", "cwd": "webgpu-graph-algorithms" } },
        "test:limits": { "executor": "nx:run-commands", "options": { "command": "npm run test:limits", "cwd": "webgpu-graph-algorithms" } },
        "test:ui": { "executor": "nx:run-commands", "options": { "command": "vitest --ui", "cwd": "webgpu-graph-algorithms" } },
        "coverage": { "executor": "nx:run-commands", "options": { "command": "npm run coverage", "cwd": "webgpu-graph-algorithms" } },
        "lint": { "executor": "nx:run-commands", "options": { "command": "npm run lint", "cwd": "webgpu-graph-algorithms" } },
        "typecheck": { "executor": "nx:run-commands", "options": { "command": "npm run typecheck", "cwd": "webgpu-graph-algorithms" } },
        "benchmark": { "executor": "nx:run-commands", "options": { "command": "npm run bench", "cwd": "webgpu-graph-algorithms" } }
    }
}
```

The `cwd` values are the MONOREPO paths (the file is inert in the staging
repository, exactly as graph-io's is; `packages/README.md` move checklist).

### 2.3 The tsconfig trio (spec 3.1; note 07 section 4.4)

`tsconfig.json`:

```json
{
    "extends": "../tsconfig.base.json",
    "compilerOptions": {
        "composite": true,
        "noEmit": true,
        "noImplicitOverride": true,
        "noUncheckedIndexedAccess": false,
        "exactOptionalPropertyTypes": false,
        "lib": ["ES2020", "DOM", "DOM.Iterable"],
        "types": ["node", "vitest/globals", "vite/client", "@webgpu/types"],
        "paths": {
            "@graphty/webgpu-graph-algorithms": ["./src/index.ts"],
            "@graphty/webgpu-graph-algorithms/browser": ["./src/browser/index.ts"],
            "@graphty/webgpu-graph-algorithms/node": ["./src/node/index.ts"],
            "@graphty/graph-format": ["../graph-format/src/index.ts"]
        }
    },
    "include": ["src/**/*.ts", "test/**/*.ts", "benchmarks/**/*.ts", "scripts/**/*.d.ts", "../graph-format/src/**/*.ts"],
    "exclude": ["node_modules", "dist", "coverage", "tmp"]
}
```

`tsconfig.build.json`:

```json
{
    "extends": "./tsconfig.json",
    "compilerOptions": {
        "noEmit": false,
        "rootDir": ".",
        "outDir": "./dist",
        "declaration": true,
        "declarationMap": true,
        "sourceMap": true,
        "stripInternal": true,
        "types": ["node", "@webgpu/types"],
        "paths": {}
    },
    "include": ["src/**/*.ts"],
    "exclude": ["node_modules", "dist", "coverage", "tmp", "test", "benchmarks", "scripts"]
}
```

`tsconfig.strict-consumer.json`:

```json
{
    "extends": "./tsconfig.json",
    "compilerOptions": {
        "composite": false,
        "noEmit": true,
        "noUncheckedIndexedAccess": true,
        "exactOptionalPropertyTypes": true,
        "types": ["node", "@webgpu/types"],
        "paths": {
            "@graphty/webgpu-graph-algorithms": ["./dist/webgpu-graph-algorithms.d.ts"],
            "@graphty/webgpu-graph-algorithms/*": ["./dist/*.d.ts"],
            "@graphty/graph-format": ["../graph-format/dist/graph-format.d.ts"]
        }
    },
    "include": ["test/types/**/*.test-d.ts"],
    "exclude": ["node_modules", "dist", "coverage", "tmp"]
}
```

`stripInternal: true` removes every member tagged `@internal` from the
published d.ts (spec 3.1: `GpuContext.residency` and the other internal
members of section 3). The three entry files each start with
`/// <reference types="@webgpu/types" />` so every d.ts shim resolves the
`GPU*` names for consumers. `scripts/**/*.d.ts` is included so
`scripts/gpu-policy.d.ts` and `scripts/runner-class.d.ts` type the `.js`
modules `test/setup/gpu.ts` and `benchmarks/harness.ts` import.
`vite/client` is in `types` because `test/setup/browser.ts` and
`test/browser/webgpu-check.test.ts` read `import.meta.env.GRAPHTY_*` (2.5):
`vitest/globals` declares only the test globals and `vitest/importMeta`
only `url` / `vitest`, so without `vite/client`'s `ImportMetaEnv` (which
carries an index signature) `tsc --noEmit -p tsconfig.json` fails with
"Property 'env' does not exist on type 'ImportMeta'". The two other
tsconfigs override `types` and stay unaffected.

### 2.4 eslint.config.js (spec 2.4, 3.2; lead a)

```js
/**
 * Package-local ESLint config: the root flat config of packages/ spread first, then the rules only this
 * package needs (spec 2.4 / 3.2): the layer rule as `no-restricted-imports` zones (one block per layer
 * directory forbidding the relative specifiers of every HIGHER layer), the entry isolation (src/browser
 * and src/node are imported by nothing else in src/), the no-navigator / no-process rules, and the ban on
 * runtime imports of the CPU packages (type imports allowed in src/types/accelerator.ts only, D27).
 * test/layers.test.ts enforces the same rules plus cycle detection by walking the import graph.
 */

import tseslint from "typescript-eslint";

import root from "../eslint.config.js";

const LAYER_MESSAGE = "layer rule (spec 3.2): a lower layer never imports a higher one";
const ENTRY_MESSAGE = "src/browser and src/node are imported by nothing else in src/ (spec 2.4)";
const CPU_MESSAGE = "no runtime import of the CPU packages (D3, D27); type imports only in src/types/accelerator.ts";
const TYPES_MESSAGE = "src/types imports values from nothing above errors.ts / constants.ts; type imports are allowed";

// Flat config does NOT merge rule options: a later matching block that gives options REPLACES the earlier ones
// for that rule id. Every file set below therefore gets exactly ONE options object per rule id
// ("@typescript-eslint/no-restricted-imports" carries both `paths` and `patterns` where a file needs both), and
// the generic CPU-ban block excludes src/types/** so it cannot clobber the types zone. test/layers.test.ts
// asserts the computed config (ESLint.calculateConfigForFile) still carries both halves.
const CPU_PATHS = [
    { name: "@graphty/algorithms", message: CPU_MESSAGE },
    { name: "@graphty/layout", message: CPU_MESSAGE },
];
const CPU_PATHS_TYPES_ALLOWED = CPU_PATHS.map((p) => ({ ...p, allowTypeImports: true }));

// Relative specifiers of each layer as seen from one directory below src/ and from src/ itself.
const UP = {
    context: ["../context.js", "./context.js"],
    memory: ["../memory/*", "./memory/*"],
    kernel: ["../kernel/*", "./kernel/*"],
    registry: ["../kernels.js", "./kernels.js"],
    wgsl: ["../wgsl/*", "./wgsl/*"],
    primitives: ["../primitives/*", "./primitives/*"],
    algorithms: ["../algorithms/*", "../algorithms/**", "./algorithms/*", "./algorithms/**"],
    layouts: ["../layouts/*", "./layouts/*"],
    accelerator: ["../accelerator.js", "./accelerator.js"],
    entries: ["../browser/*", "../node/*", "./browser/*", "./node/*"],
    barrel: ["../index.js", "./index.js"],
};

/**
 * One layer zone: the files may not import the specifiers of the named higher layers.
 * @param files - glob(s) of the zone
 * @param higher - keys of UP that are above this zone
 * @returns a flat-config block
 */
function zone(files, higher) {
    const group = higher.flatMap((name) => UP[name]);
    return {
        files,
        rules: {
            "no-restricted-imports": ["error", { patterns: [{ group, message: LAYER_MESSAGE }] }],
        },
    };
}

// The type-only zone of src/types/**: every layer above errors.ts / constants.ts, values forbidden, `import type` allowed.
const TYPES_PATTERNS = [
    {
        group: [...UP.context, ...UP.memory, ...UP.kernel, ...UP.registry, ...UP.wgsl, ...UP.primitives, ...UP.algorithms, ...UP.layouts, ...UP.accelerator, ...UP.entries, ...UP.barrel, "../device/*"],
        message: TYPES_MESSAGE,
        allowTypeImports: true,
    },
];

export default tseslint.config(
    ...root,
    // ---- layer zones (spec 3.2: device < context < memory < kernel < kernels.ts < primitives < algorithms / layouts < accelerator)
    zone(["src/errors.ts", "src/constants.ts"], ["context", "memory", "kernel", "registry", "wgsl", "primitives", "algorithms", "layouts", "accelerator", "entries", "barrel"]),
    zone(["src/device/**/*.ts"], ["context", "memory", "kernel", "registry", "wgsl", "primitives", "algorithms", "layouts", "accelerator", "entries", "barrel"]),
    zone(["src/memory/**/*.ts"], ["context", "kernel", "registry", "wgsl", "primitives", "algorithms", "layouts", "accelerator", "entries", "barrel"]),
    zone(["src/kernel/**/*.ts"], ["context", "registry", "wgsl", "primitives", "algorithms", "layouts", "accelerator", "entries", "barrel"]),
    zone(["src/kernels.ts"], ["context", "primitives", "algorithms", "layouts", "accelerator", "entries", "barrel"]),
    zone(["src/context.ts"], ["registry", "wgsl", "primitives", "algorithms", "layouts", "accelerator", "entries", "barrel"]),
    zone(["src/primitives/**/*.ts"], ["context", "wgsl", "algorithms", "layouts", "accelerator", "entries", "barrel"]),
    zone(["src/algorithms/**/*.ts"], ["wgsl", "layouts", "accelerator", "entries", "barrel"]),
    zone(["src/layouts/**/*.ts"], ["wgsl", "algorithms", "accelerator", "entries", "barrel"]),
    zone(["src/accelerator.ts"], ["wgsl", "entries", "barrel"]),
    zone(["src/browser/**/*.ts", "src/node/**/*.ts"], ["memory", "kernel", "registry", "wgsl", "primitives", "algorithms", "layouts", "accelerator", "barrel"]),
    // src/types/** holds types only: it may `import type` from anywhere below the accelerator (GpuContext, GraphResidency), never a value;
    // the SAME options object also carries the CPU-package ban (one options object per rule id per file set, see the note above)
    {
        files: ["src/types/**/*.ts"],
        ignores: ["src/types/accelerator.ts"],
        rules: {
            "@typescript-eslint/no-restricted-imports": ["error", { paths: CPU_PATHS, patterns: TYPES_PATTERNS }],
        },
    },
    {
        files: ["src/types/accelerator.ts"],
        rules: {
            "@typescript-eslint/no-restricted-imports": ["error", { paths: CPU_PATHS_TYPES_ALLOWED, patterns: TYPES_PATTERNS }],
        },
    },
    // ---- the entries are imported by nothing else in src/ (the zones above already forbid them; the barrel is the remaining file)
    {
        files: ["src/index.ts"],
        rules: {
            "no-restricted-imports": ["error", { patterns: [{ group: UP.entries, message: ENTRY_MESSAGE }] }],
        },
    },
    // ---- no runtime import of the CPU packages anywhere else in src/ (src/types/** is handled by its own blocks above, D27)
    {
        files: ["src/**/*.ts"],
        ignores: ["src/types/**/*.ts"],
        rules: {
            "@typescript-eslint/no-restricted-imports": ["error", { paths: CPU_PATHS }],
        },
    },
    // ---- the core never references the runtime globals (spec 2.1, 2.4)
    {
        files: ["src/**/*.ts"],
        ignores: ["src/browser/**/*.ts"],
        rules: {
            "no-restricted-globals": [
                "error",
                { name: "navigator", message: "only src/browser/** may read navigator.gpu (spec 2.4)" },
                { name: "window", message: "the core never references window (spec 2.1)" },
                { name: "document", message: "the core never references document (spec 2.1)" },
            ],
        },
    },
    {
        files: ["src/**/*.ts"],
        ignores: ["src/node/**/*.ts"],
        rules: {
            "no-restricted-globals": ["error", { name: "process", message: "the core never references process (spec 2.1); env vars are read by test/setup and scripts only (spec 2.3)" }],
        },
    },
    // ---- test relaxations beyond the root's: tests may import any layer and any global; setup files read process.env
    {
        files: ["test/**/*.ts"],
        rules: {
            "no-restricted-imports": "off",
            "@typescript-eslint/no-restricted-imports": "off",
            "no-restricted-globals": "off",
        },
    },
);
```

The root config already ignores `**/scripts/**`, `**/benchmarks/**`,
`**/docs/**` and `**/*.config.*`, and relaxes `**/test/**` (JSDoc, explicit
return types, `no-console`). The zone for `src/node/**` cannot catch the
dynamic `import("webgpu")` (ESLint's rule ignores dynamic imports);
`test/layers.test.ts` greps for the specifier (section 5.5). `src/kernel/**`
may `import type` from `src/memory/**` (CommandBatch reaches the Readback
slot through its host; the shared `Binding` / `ArcWindow` types live in
`src/types/memory.ts` so no value edge exists, 3.3) and `src/context.ts` may
import memory and kernel (spec 3.2 names it the one exception);
`src/kernels.ts` may import `src/kernel/**` and `src/wgsl/**` (it is the
only importer of the bodies).

### 2.5 vitest.config.ts (spec 11.1, 11.8, 2.3, 12.2; note 06 section 5)

```ts
/// <reference types="@vitest/browser/providers/playwright" />
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

/**
 * The two Chromium flag sets of spec 12.2, selected by GRAPHTY_BROWSER_GPU ("nvidia" | "swiftshader", default
 * swiftshader). Kept in one exported constant so the Vitest 4 provider change is a one-line move (spec 11.1).
 * @public exported for scripts/run-browser-project.js and the CLAUDE.md "Verified Platform Facts" table
 */
export const BROWSER_FLAGS = Object.freeze({
    nvidia: Object.freeze(["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan", "--disable-vulkan-surface"]),
    swiftshader: Object.freeze(["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]),
});

const here = dirname(fileURLToPath(import.meta.url));
const browserGpu = process.env.GRAPHTY_BROWSER_GPU === "nvidia" ? "nvidia" : "swiftshader";
const gpuRequire = process.env.GRAPHTY_GPU_REQUIRE ?? "";
const noiseFloorWrite = process.env.GRAPHTY_NOISE_FLOOR_WRITE ?? "";

/**
 * The environment of the Chromium child (spec 12.2 GRAPHTY_EGL_LIB_DIR): on the dev box headless Chromium finds
 * the NVIDIA GPU only with the extracted libEGL tree on LD_LIBRARY_PATH; the variable is prepended when set,
 * otherwise Playwright inherits process.env unchanged (undefined). Only defined values are copied (LaunchOptions.env
 * is a string map).
 * @returns the env map for `launch.env`, or undefined
 */
function browserLaunchEnv(): Record<string, string> | undefined {
    const eglDir = process.env.GRAPHTY_EGL_LIB_DIR;
    if (eglDir === undefined || eglDir === "") {
        return undefined;
    }
    const env: Record<string, string> = {};
    for (const [key, value] of Object.entries(process.env)) {
        if (value !== undefined) {
            env[key] = value;
        }
    }
    env.LD_LIBRARY_PATH = [eglDir, process.env.LD_LIBRARY_PATH].filter((v) => v !== undefined && v !== "").join(":");
    return env;
}

/**
 * The project names selected on the command line, from both `--project=name` and `--project name`.
 * @returns the names in command-line order
 */
function selectedProjects(): string[] {
    const argv = process.argv;
    const names: string[] = [];
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a.startsWith("--project=")) {
            names.push(a.slice("--project=".length));
        } else if (a === "--project" && i + 1 < argv.length) {
            names.push(argv[i + 1]);
        }
    }
    return names;
}

// Thresholds apply when the selected project set is EXACTLY `node` and no merge run is in progress (spec 11.8).
const projects = selectedProjects();
const thresholdsActive = projects.length === 1 && projects[0] === "node" && process.env.COVERAGE_DIR === undefined;

/**
 * The browser-side benchmark bridge (spec 11.6 item 8, 11.7): a browser test cannot write files, so it calls
 * `commands.appendBenchRecord(payload)` and this server-side command appends a session to
 * benchmarks/out/<runner-class>.json in the harness's file shape (section 6.4). Returns the path written.
 * @param _context - the Vitest command context (unused)
 * @param payload - the browser session: gpu summary, runner class, results
 * @returns the path of the file written
 */
async function appendBenchRecord(_context: unknown, payload: { runnerClass: string; session: unknown }): Promise<string> {
    const dir = resolve(here, "benchmarks/out");
    mkdirSync(dir, { recursive: true });
    const file = resolve(dir, `${payload.runnerClass.replace(/[^A-Za-z0-9_.-]/g, "_")}.json`);
    const sessions: unknown[] = existsSync(file) ? (JSON.parse(readFileSync(file, "utf8")) as unknown[]) : [];
    sessions.push(payload.session);
    writeFileSync(file, `${JSON.stringify(sessions, null, 4)}\n`);
    appendFileSync(file, "");
    return file;
}

/**
 * The browser side of test/helpers/noise-floor.ts writeNoiseFixture (spec 11.9 item 3, 11.5: the SwiftShader
 * leg of the cross-adapter fixtures). Writes test/fixtures/noise/<kernel>-<fixture>-<adapterClass>.json in the
 * 5.6 fixture shape ONLY when GRAPHTY_NOISE_FLOOR_WRITE=1 in the server's environment; returns the path, or ""
 * when writing is off.
 * @param _context - the Vitest command context (unused)
 * @param payload - the raw output of one kernel on one adapter
 * @returns the path written, or ""
 */
async function writeNoiseFixture(_context: unknown, payload: { kernel: string; fixture: string; adapterClass: string; values: readonly number[]; dtype: "f32" | "u32" }): Promise<string> {
    if (noiseFloorWrite !== "1") {
        return "";
    }
    const dir = resolve(here, "test/fixtures/noise");
    mkdirSync(dir, { recursive: true });
    const safe = (s: string): string => s.replace(/[^A-Za-z0-9_.-]/g, "_");
    const file = resolve(dir, `${safe(payload.kernel)}-${safe(payload.fixture)}-${safe(payload.adapterClass)}.json`);
    writeFileSync(file, `${JSON.stringify({ kernel: payload.kernel, fixture: payload.fixture, adapterClass: payload.adapterClass, dtype: payload.dtype, values: payload.values }, null, 4)}\n`);
    return file;
}

/**
 * The browser side of test/helpers/noise-floor.ts recordNoiseRow: appends one 5.6 row to
 * benchmarks/results/noise-floor.json (replacing a row with the same id) ONLY when GRAPHTY_NOISE_FLOOR_WRITE=1.
 * @param _context - the Vitest command context (unused)
 * @param row - the noise row (5.6 schema)
 * @returns the path written, or ""
 */
async function recordNoiseRow(_context: unknown, row: Record<string, unknown>): Promise<string> {
    if (noiseFloorWrite !== "1") {
        return "";
    }
    const file = resolve(here, "benchmarks/results/noise-floor.json");
    const doc = existsSync(file)
        ? (JSON.parse(readFileSync(file, "utf8")) as { rows: Record<string, unknown>[]; [k: string]: unknown })
        : { recordedAt: new Date().toISOString(), adapters: [], rows: [], tolerances: {} };
    doc.rows = [...doc.rows.filter((r) => r.id !== row.id), row];
    writeFileSync(file, `${JSON.stringify(doc, null, 4)}\n`);
    return file;
}

export default defineConfig({
    test: {
        reporters: ["verbose"],
        coverage: {
            all: true,
            provider: "v8",
            reporter: ["text", "json-summary", "json", "lcov", "html"],
            reportsDirectory: process.env.COVERAGE_DIR ?? "coverage",
            include: ["src/**/*.ts"],
            exclude: ["**/*.d.ts", "**/*.test.ts", "src/index.ts", "src/wgsl/**"],   // only the root barrel and the template strings (spec 11.8): src/node/index.ts and src/browser/index.ts carry logic and count
            thresholds: thresholdsActive ? { lines: 80, functions: 80, branches: 75, statements: 80 } : undefined,
        },
        projects: [
            {
                test: {
                    name: "node",
                    globals: true,
                    environment: "node",
                    pool: "forks",
                    testTimeout: 30_000,
                    hookTimeout: 60_000,
                    include: [
                        "test/*.test.ts",
                        "test/{device,node,memory,kernel,primitives,algorithms,layouts,oracle,sabotage,types}/**/*.test.ts",
                    ],
                    exclude: ["test/limits/**", "test/browser/**"],
                    setupFiles: ["test/setup/gpu.ts"],
                    globalSetup: ["test/setup/global.ts"],
                },
            },
            {
                test: {
                    name: "node-limits",
                    globals: true,
                    environment: "node",
                    pool: "forks",
                    testTimeout: 600_000,
                    hookTimeout: 120_000,
                    include: ["test/limits/**/*.test.ts"],
                    setupFiles: ["test/setup/gpu.ts"],
                },
            },
            {
                define: {
                    "import.meta.env.GRAPHTY_GPU_REQUIRE": JSON.stringify(gpuRequire),
                    "import.meta.env.GRAPHTY_BROWSER_GPU": JSON.stringify(browserGpu),
                    "import.meta.env.GRAPHTY_NOISE_FLOOR_WRITE": JSON.stringify(noiseFloorWrite),
                },
                test: {
                    name: "browser",
                    globals: true,
                    include: ["test/browser/**/*.test.ts"],
                    testTimeout: 120_000,
                    hookTimeout: 120_000,
                    env: { GRAPHTY_GPU_REQUIRE: gpuRequire, GRAPHTY_BROWSER_GPU: browserGpu, GRAPHTY_NOISE_FLOOR_WRITE: noiseFloorWrite },
                    setupFiles: ["test/setup/browser.ts"],
                    browser: {
                        enabled: true,
                        headless: true,
                        provider: "playwright",
                        fileParallelism: false,
                        commands: { appendBenchRecord, writeNoiseFixture, recordNoiseRow },
                        instances: [{ browser: "chromium", launch: { args: [...BROWSER_FLAGS[browserGpu]], env: browserLaunchEnv() } }],
                    },
                },
            },
        ],
    },
});
```

Verified against `vitest@3.2.7`'s types (`packages/node_modules/vitest`)
and `@vitest/browser@3.2.4` (the nearest installed copy, under
`~/Projects/pupt-monorepo`; 3.2.7 is re-read after P0-T2's `pnpm install`,
section 8): `fileParallelism` is a root-only option (`NonProjectOptions`),
so the per-project spelling is `browser.fileParallelism: false` (spec 11.1),
which the config uses; `browser.instances[]` entries are
`BrowserInstanceOption extends BrowserProviderOptions`, and the Playwright
provider's `providers/playwright.d.ts` augments `BrowserProviderOptions`
with `launch?: LaunchOptions` -- `BrowserProviderOptions` is `{}` unless
that declaration file is referenced, hence the `/// <reference
types="@vitest/browser/providers/playwright" />` at the top so `launch`
(`args`, `env`) is typed if the config is ever type-checked -- so the
per-instance spelling is `instances: [{ browser: "chromium", launch: {
args, env } }]` (record in CLAUDE.md at P0, spec 13 row P0).
`browser.commands` is `Record<string, BrowserCommand>` and the browser side
reaches it as `commands` from `@vitest/browser/context` ("a shortcut to
`server.commands`"); the three commands (`appendBenchRecord`,
`writeNoiseFixture`, `recordNoiseRow`) are declared for the browser side in
`test/setup/browser-commands.d.ts` (5.1, P0-T3) because a browser test
cannot write files itself (spec 11.7) and the SwiftShader legs of the
noise-floor fixtures (spec 11.5, 11.9 item 3) need a path to disk. CONTRACT
DECISION: the browser project forwards the three variables
(`GRAPHTY_GPU_REQUIRE`, `GRAPHTY_BROWSER_GPU`, `GRAPHTY_NOISE_FLOOR_WRITE`)
through vite `define` of `import.meta.env.*` IN ADDITION to `test.env`,
because `test.env` populates `process.env` of Node workers while the
browser tester sees only what Vite inlines; `test/setup/browser.ts` reads
`import.meta.env` (spec 2.3) and P0-T3's browser test asserts the values
arrive (a G0 item). `GRAPHTY_EGL_LIB_DIR` (spec 12.2, local only) is read
here and prepended to the Chromium child's `LD_LIBRARY_PATH` through
`launch.env`; the developer no longer exports it in the shell.
`test/setup/global.ts` reaches its P2-T2 form at P2; P0-T2 writes the
config line and P0-T3 ships an empty `global.ts` (`export function
setup(): void {}` / `teardown`) so the config resolves.

### 2.6 scripts/entries.js, build-bundle.js, bundle-types.js (spec 2.5; note 07 section 4.6)

`scripts/entries.js`:

```js
/**
 * The bundle entries of @graphty/webgpu-graph-algorithms (spec 2.5): the root barrel and the two acquisition
 * subpaths. Shared by scripts/build-bundle.js (dist/<name>.js) and scripts/bundle-types.js (dist/<name>.d.ts);
 * package.json "exports" lists the same names, which test/build-output.test.ts checks. There is NO root
 * webgpu-graph-algorithms.ts shim (graph-io's convention, not graph-format's).
 *
 * Keys are the output names under dist/, values the source entry relative to the package root.
 */

export const ENTRIES = Object.freeze({
    "webgpu-graph-algorithms": "src/index.ts",
    browser: "src/browser/index.ts",
    node: "src/node/index.ts",
});

/**
 * The declaration file a bundle entry re-exports: the tsc output of its source entry under dist/src/.
 * @param source - the source entry relative to the package root (`src/node/index.ts`)
 * @returns the relative import specifier for the shim (`./src/node/index.js`)
 */
export function declarationSpecifier(source) {
    return `./${source.replace(/\.ts$/, ".js")}`;
}
```

`scripts/build-bundle.js` and `scripts/bundle-types.js`: byte-for-byte copies
of `packages/graph-io/scripts/{build-bundle,bundle-types}.js` with the header
comments rewritten for this package (the root entry is
`dist/webgpu-graph-algorithms.js`; the subpaths `dist/browser.js` and
`dist/node.js`; the externals are every `dependencies` + `peerDependencies`
name, which already includes `webgpu`, `@graphty/graph-format`,
`@graphty/algorithms` and `@graphty/layout`, so no code change is needed).
Deltas from graph-io: none in code. The vite build emits `dist/browser.js` as
an empty chunk at P0 (the browser entry is `export {};` until P1-T1; vite
logs "Generated an empty chunk", which is not an error).

### 2.7 scripts/gpu-policy.js, gpu-report.js, run-browser-project.js, bench-compare.js

Declared in section 6.5-6.8 (signatures, JSON shapes, exit codes). P0-T2
writes all four; `bench-compare.js` reaches its full rule at P1-T7.

### 2.8 Workspace-root edits at P0 (spec 12.3; note 07 section 4.7)

`packages/pnpm-workspace.yaml`:

```yaml
packages:
    - "graph-format"
    - "graph-io"
    - "webgpu-graph-algorithms"
```

`packages/knip.config.ts` gains the fourth workspace (after `graph-io`):

```ts
        // webgpu-graph-algorithms package: two extra entries because the root barrel re-exports neither subpath
        "webgpu-graph-algorithms": {
            entry: [
                "src/index.ts",
                "src/browser/index.ts",
                "src/node/index.ts",
                "test/**/*.test.ts",
                "test/types/**/*.test-d.ts",
                "test/setup/*.ts",
                "test/fixtures/**/*.ts",
                "benchmarks/run.ts",
                "benchmarks/layout-run.ts",
                "scripts/**/*.{ts,js}",
            ],
            project: ["src/**/*.ts", "test/**/*.ts", "benchmarks/**/*.ts", "scripts/**/*.{ts,js}"],
            ignoreDependencies: ["@vitest/browser", "playwright"],
        },
```

(`@vitest/browser` and `playwright` are reached through vitest's config,
not by an import knip can see; `benchmarks/*.bench.ts` are reached from
`run.ts`.) `packages/package.json`: the `description` string gains
`, webgpu-graph-algorithms` and nothing else changes (its devDependencies
already carry eslint 9, typescript-eslint, jsdoc, simple-import-sort, knip,
prettier, tsx, typescript 5.9.3, vite 7, vitest 3.2.7, fast-check 4).
`packages/.gitignore` gains `benchmarks/out/`, `browser-results.json`,
`gpu-report.json`.

### 2.9 .github/workflows/ci.yml and gpu.yml (spec 12.3, verbatim, then the P0 deltas)

`ci.yml` as spec 12.3 gives it:

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

`gpu.yml` as spec 12.3 gives it:

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

P0 deltas applied to the verbatim text (each is a mechanical necessity of
an empty package; none changes the lane design):

1. ci.yml, the no-subgroups pass: `GRAPHTY_GPU_NO_SUBGROUPS=1 pnpm exec vitest run --project=node test/primitives test/layouts --passWithNoTests`
   -- until P1 (primitives) and P1-T6 (layouts) exist the filter matches no
   file and vitest would exit 1 with "No test files found".
2. gpu.yml, `--project=node --project=node-limits`: `test/limits/` holds
   only a README until P4; a project with no files inside a multi-project
   run is not an error, so no change; the `node-limits` project definition
   exists from P0 (spec P2 defines it; the contract declares it in 2.5 so the
   config never changes shape).
3. gpu.yml comment on the `bench-compare` step: `benchmarks/baselines/gpu-linux-t4.json`
   is corrected to `benchmarks/results/gpu-linux-t4.json`. CONTRACT DECISION:
   there is ONE baseline directory, `benchmarks/results/<runner-class>.json`
   (spec 10.4 and 11.7 define it; spec 12.1 / 12.4 / 12.3's comment name a
   `baselines/` directory that nothing else defines).
4. gpu.yml `pnpm run bench` at P0 runs the P0 stub of `benchmarks/run.ts`
   (prints "no benchmark groups until P1", exits 0, writes nothing) and
   `bench-compare.js` exits 0 with "nothing to compare" when
   `benchmarks/out/<class>.json` is absent (6.8).
5. Both workflows: `working-directory: packages` for the install / build /
   lint / knip steps is correct as written (the root scaffold is gone at P0).
6. The nightly `github-script` placeholder stays a comment-only script at
   P0 (a valid no-op); the two-consecutive-failures logic is P12 / owner
   work and not part of P0-P3.
7. gpu.yml `test-gpu` job `env:` gains `GRAPHTY_RUNNER_CLASS: gpu-linux-t4`
   so the benchmark runner class of the T4 lane is the runner name, not a
   driver-derived string that drifts with the partner image (6.4).

## 3. src/ module contracts

Conventions of this section: each file gives (a) its imports from
`@graphty/graph-format` and from lower layers, (b) its exported declarations
as TypeScript (complete signatures; JSDoc one-liners; no `...` inside a
declaration), (c) its "throws" list of `WebGpuGraphError` codes and the
graph-format codes that pass through. A member marked `@internal` is present
in `src/` and stripped from the published d.ts (2.3); tests and higher layers
may use it. `F32`, `U32`, `NodeMask`, `TypedArrayData`, `NumericVector`,
`GraphSnapshot`, `AttributeTable`, `Column`, `ViewName`, `CoreArrayName`,
`ArenaLayout`, `SnapshotFlags`, `KnownColumnRole`, `NodeId` are always the
graph-format exports (its barrel `packages/graph-format/src/index.ts`).

### 3.1 src/errors.ts (spec 3.3, 5.7, D12) -- P0-T3

Imports: none.

```ts
/** Every condition the package detects itself carries one of these codes (spec 3.3). */
export type WebGpuGraphErrorCode =
    | "E_NO_WEBGPU"
    | "E_NO_ADAPTER"
    | "E_NO_DEVICE"
    | "E_SOFTWARE_ONLY"
    | "E_DEVICE_LOST"
    | "E_DISPOSED"
    | "E_VALIDATION"
    | "E_SHADER_COMPILE"
    | "E_OUT_OF_MEMORY"
    | "E_TOO_LARGE"
    | "E_UNSUPPORTED"
    | "E_INVALID_ARGUMENT"
    | "E_SNAPSHOT"
    | "E_RELEASED"
    | "E_NOT_LOADED"
    | "E_ABORTED";

/** The graph-format error codes a public call lets propagate unchanged (D12, spec 5.7): raised by accessors the package calls on the caller's behalf. */
export const PASSTHROUGH_FORMAT_CODES: readonly ["E_GPU_INELIGIBLE", "E_UNKNOWN_NODE", "E_UNKNOWN_COLUMN", "E_COLUMN_LENGTH"];

/** The one error class the package throws for conditions it detects itself: a stable `code` and frozen `details` (spec 3.3). */
export class WebGpuGraphError extends Error {
    /** The stable code. */
    readonly code: WebGpuGraphErrorCode;
    /** Frozen shallow copy of the details given to the constructor (`{}` when none). */
    readonly details: Readonly<Record<string, unknown>>;
    /** `name` is always "WebGpuGraphError". */
    override readonly name: "WebGpuGraphError";
    constructor(code: WebGpuGraphErrorCode, message: string, details?: Record<string, unknown> | undefined);
}

/** Brand check that survives two package copies: `name === "WebGpuGraphError"` and a string `code`. */
export function isWebGpuGraphError(x: unknown): x is WebGpuGraphError;

/** Brand check that survives two package copies: `code === x.code` for an `isWebGpuGraphError(x)`. */
export function hasErrorCode(x: unknown, code: WebGpuGraphErrorCode): boolean;
```

`details` keys used by this contract (each code's documented shape, so tests
can assert them): `E_NO_WEBGPU { reason, hint }` (hint = "install the
optional peer dependency webgpu@0.4.0" in Node); `E_NO_ADAPTER { reason }`;
`E_NO_DEVICE { reason, adapter: AdapterSummary | null, limit?, requested?,
available? }` (reason `"consumed"` | `"requestDevice"` | `"limit"` |
`"feature"` | `"maxComputeWorkgroupsPerDimension"`); `E_SOFTWARE_ONLY {
adapter }`; `E_DEVICE_LOST { reason, message }`; `E_DISPOSED { label }`;
`E_VALIDATION { label, message, batchId? }`; `E_SHADER_COMPILE { id, stage:
"compose" | "compile", slot?, messages? }`; `E_OUT_OF_MEMORY { requested,
resident, label }`; `E_TOO_LARGE { needed, limit, path, algorithm }`;
`E_UNSUPPORTED { feature?, option?, hint? }` (exactly one of `feature` /
`option` is present: `feature` for `planGridStride`, `planIndirect`,
`repulsion.grid`, `segmentedReduce.tiers` and the P7+ view names; `option`
for `nodeSize`, `nodeMass`, `packViews`); `E_INVALID_ARGUMENT { argument,
value, expected? }`; `E_SNAPSHOT { reason, serial }`; `E_RELEASED { serial
}`; `E_NOT_LOADED { state }`; `E_ABORTED { batchId? }`.

Throws: none (constructors never throw).

### 3.2 src/constants.ts (spec 3.1, 3.5, 5.2, 7.8, 7.14, 7.17) -- P0-T3

Imports: none (graph-format's `INVALID_INDEX` is imported by the prelude
directly, never copied here).

```ts
/** Workgroup size of every 1D kernel; `WG = min(WORKGROUP_SIZE, caps.limits.maxComputeInvocationsPerWorkgroup)` at runtime (spec 5.1). */
export const WORKGROUP_SIZE = 256;
/** The spec minimum of maxComputeWorkgroupsPerDimension; asserted equal to the device limit at create() (spec 2.2, 5.2). */
export const MAX_WORKGROUPS_PER_DIM = 65535;
/** Items a 1D dispatch of WORKGROUP_SIZE covers: 65535 x 256 = 16,776,960, NOT 2^24 (design 10.6). */
export const MAX_1D_ITEMS = MAX_WORKGROUPS_PER_DIM * WORKGROUP_SIZE;
/** The largest u32 (the `min` identity of the u32 reduce); interpolated into the prelude as `U32_MAX` so no body types the literal (4.1). */
export const U32_MAX = 0xffffffff;
/** Arc-window boundaries are multiples of 64 arcs = 256 bytes (design 10.6). */
export const ARC_WINDOW_ALIGN = 64;
/** Storage-binding offset alignment the package always honours (spec 2.6): the graph-format arena is 256-aligned. */
export const STORAGE_ALIGN = 256;
/** Stride of one UniformRing slot: minUniformBufferOffsetAlignment is 256 on every runtime the package targets (spec 5.3). */
export const UNIFORM_SLOT_BYTES = 256;
/** Exact-tier crossover default: CONSERVATIVE until G3 re-fixes it by the 7.8 rule (spec Q-6; the measured 7.6 curve predicts 32,768). */
export const EXACT_MAX_NODES = 16384;
/** Default number of MAP_READ staging buffers in the Readback ring (spec 4.4). */
export const DEFAULT_STAGING_SLOTS = 3;
/** Timestamp query-set size of the Profiler (spec 5.5: "a query set of 256 slots"; 2 slots per pass). */
export const PROFILER_QUERY_SLOTS = 256;
/** Byte size above which createBuffer runs inside an "out-of-memory" error scope (spec 5.7). */
export const OOM_SCOPE_THRESHOLD_BYTES = 16 * 1024 * 1024;
/** Idle buffers kept per (size class, usage) by the BufferPool (spec 4.4). */
export const POOL_MAX_IDLE_PER_CLASS = 4;
/** Smallest and largest power-of-two pool classes and the linear step above the largest (spec 4.4). */
export const POOL_MIN_CLASS_BYTES = 4 * 1024;
export const POOL_MAX_POW2_CLASS_BYTES = 64 * 1024 * 1024;
export const POOL_LINEAR_STEP_BYTES = 16 * 1024 * 1024;
/** Default `warnUnreleasedSnapshots` (spec 2.2, 4.1). */
export const DEFAULT_WARN_UNRELEASED_SNAPSHOTS = 2;
/** CONTRACT DECISION: the largest `iterations` a single step() records (the trace region and the uniform ring are sized by it); larger values are E_INVALID_ARGUMENT. */
export const MAX_ITERATIONS_PER_STEP = 256;
/** Bytes of one Fa2Trace record (spec 7.3: 32-byte records). */
export const TRACE_RECORD_BYTES = 32;
/** CONTRACT DECISION: the state header is padded to 256 bytes so the trace region that follows it in the same buffer starts at a legal 256-aligned binding offset (spec 7.3 says "128 + k x 32"; 128 is not a legal storage offset). */
export const STATE_HEADER_BYTES = 256;
/** Bytes of one per-workgroup partials record (spec 7.3: exactly 64 B). */
export const PARTIAL_BYTES = 64;
/** ForceAtlas2 defaults (spec 7.14, 7.17, 7.19). */
export const FA2_DEFAULTS: Readonly<{
    maxIter: 100;
    jitterTolerance: 1;
    scalingRatio: 2;
    gravity: 1;
    strongGravity: false;
    distributedAction: false;
    linlog: false;
    dissuadeHubs: false;
    dim: 2;
    scale: 1;
    settleThreshold: 0.001;
    settleWindow: 10;
    iterationsPerStep: 1;
    maxInFlight: 2;
}>;
/** GPU-only layout tuning defaults (spec 7.14): repulsion "auto", exactMaxNodes EXACT_MAX_NODES, nearMax 64, deterministic true, gridMax2D 512, gridMax3D 128, extentFactor 6, compat "paper". */
export const LAYOUT_TUNING_DEFAULTS: Readonly<{
    repulsion: "auto";
    exactMaxNodes: number;
    nearMax: 64;
    deterministic: true;
    gridMax2D: 512;
    gridMax3D: 128;
    extentFactor: 6;
    compat: "paper";
}>;
/** The distance floor `max(d, 0.01)` of spec 7.2, its square, and the coincident threshold `d^2 < 1e-8`; interpolated into the prelude (4.1). */
export const FA2_DISTANCE_FLOOR = 0.01;
export const FA2_DISTANCE_FLOOR_SQ = 0.0001;
export const FA2_COINCIDENT_SQ = 1e-8;
/** Bits of Fa2Params.flags (4.4). */
export const FA2_FLAG_FIRST = 1;
```

Throws: none.

### 3.3 src/types/*.ts (spec 2.2, 3.3, 7.14, 7.19, 9.2, 9.3, D27)

#### src/types/context.ts -- P1-T1

Imports: `import type { WebGpuGraphError } from "../errors.js";`.

```ts
/** The limits `"raise"` takes from the adapter (spec 2.2); maxComputeWorkgroupsPerDimension is deliberately absent. */
export type RaisableLimit =
    | "maxBufferSize"
    | "maxStorageBufferBindingSize"
    | "maxStorageBuffersPerShaderStage"
    | "maxComputeWorkgroupStorageSize"
    | "maxComputeInvocationsPerWorkgroup"
    | "maxComputeWorkgroupSizeX";
/** How `create()` builds `requiredLimits` (spec 2.2 step 3). */
export type LimitPolicy = "default" | "raise" | Readonly<Partial<Record<RaisableLimit, number>>>;
/** Options of GpuContext.create (spec 2.2). */
export interface GpuContextOptions {
    readonly gpu?: GPU | undefined;
    readonly adapter?: GPUAdapter | undefined;
    readonly device?: GPUDevice | undefined;
    readonly powerPreference?: GPUPowerPreference | undefined;
    readonly rejectSoftware?: boolean | undefined;
    readonly limits?: LimitPolicy | undefined;
    readonly optionalFeatures?: readonly GPUFeatureName[] | undefined;
    readonly requiredFeatures?: readonly GPUFeatureName[] | undefined;
    readonly label?: string | undefined;
    readonly onError?: ((error: WebGpuGraphError) => void) | undefined;
    readonly warnUnreleasedSnapshots?: number | undefined;
    /** CONTRACT DECISION: set by the ./browser and ./node entries ("browser" / "node"), never by sniffing globals (spec 2.2 step 4); consumers leave it unset ("unknown"). */
    readonly runtime?: "browser" | "node" | "unknown" | undefined;
}
/** The structural subset of GPUAdapterInfo the package reads (CONTRACT DECISION: structural so tests can fake it without the __brand of @webgpu/types). */
export interface AdapterInfoLike {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
    readonly isFallbackAdapter?: boolean | undefined;
    readonly subgroupMinSize?: number | undefined;
    readonly subgroupMaxSize?: number | undefined;
}
/** What the app displays or logs about an adapter (spec 2.2). */
export interface AdapterSummary {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
    readonly software: boolean;
    readonly subgroupMinSize: number;
    readonly subgroupMaxSize: number;
    readonly features: readonly string[];
    readonly limits: Readonly<Record<string, number>>;
}
/** Result of GpuContext.probe (spec 2.2); `adapter` is the UNUSED adapter to pass to create({ adapter }). */
export interface ProbeResult {
    readonly ok: boolean;
    readonly code: "OK" | "E_NO_WEBGPU" | "E_NO_ADAPTER" | "E_SOFTWARE_ONLY";
    readonly reason: string | null;
    readonly adapter: GPUAdapter | null;
    readonly summary: AdapterSummary | null;
}
/** Options of GpuContext.probe (spec 2.2). `gpu` may be undefined so an app can pass `navigator.gpu` without a guard: undefined -> E_NO_WEBGPU. */
export interface ProbeOptions {
    readonly gpu: GPU | undefined;
    readonly powerPreference?: GPUPowerPreference | undefined;
    readonly rejectSoftware?: boolean | undefined;
}
/** The limits every planner reads; GPUSupportedLimits is structurally assignable to it (CONTRACT DECISION: planners take PlanCaps so faked tables need no cast). */
export interface PlanLimits {
    readonly maxBufferSize: number;
    readonly maxStorageBufferBindingSize: number;
    readonly maxStorageBuffersPerShaderStage: number;
    readonly minStorageBufferOffsetAlignment: number;
    readonly minUniformBufferOffsetAlignment: number;
    readonly maxComputeWorkgroupsPerDimension: number;
    readonly maxComputeInvocationsPerWorkgroup: number;
    readonly maxComputeWorkgroupSizeX: number;
    readonly maxComputeWorkgroupStorageSize: number;
    readonly maxUniformBufferBindingSize: number;
}
/** The capability record every planner and kernel reads (spec 3.3). */
export interface GpuCaps {
    readonly limits: GPUSupportedLimits;
    readonly features: ReadonlySet<string>;
    readonly wgslFeatures: ReadonlySet<string>;
    readonly subgroupMinSize: number;
    readonly subgroupMaxSize: number;
    readonly software: boolean;
    readonly runtime: "browser" | "node" | "unknown";
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
}
/** What the pure planners need of GpuCaps: `limits` narrowed to PlanLimits plus the four scalar facts. A GpuCaps IS a PlanCaps. */
export interface PlanCaps {
    readonly limits: PlanLimits;
    readonly features: ReadonlySet<string>;
    readonly subgroupMinSize: number;
    readonly subgroupMaxSize: number;
    readonly software: boolean;
}
/** Mutable test-build flags on a context (@internal; set by test/setup/gpu.ts from GRAPHTY_GPU_INSPECT, spec 11.9 item 2). */
export interface GpuDebugFlags {
    inspect: boolean;
}
```

#### src/types/run.ts -- P1-T1

```ts
/** Options every algorithm function accepts in addition to its own (spec 3.3, design 10.7). */
export interface GpuRunOptions {
    readonly dest?: Float32Array | Uint32Array | undefined;
    readonly signal?: AbortSignal | undefined;
    readonly onProgress?: ((done: number, total: number) => void) | undefined;
}
```

#### src/types/memory.ts -- P1-T1 (spec 4.1, 4.2)

Imports: none (`GPUBuffer` is the ambient @webgpu/types global). CONTRACT
DECISION: the two types the memory AND kernel layers share live here so
`src/kernel/kernel.ts` (P1-T3) never imports `src/memory/**` (P1-T2) for a
type and the layer graph has no P1 cycle (5.5 layers.test.ts).

```ts
/** One 64-arc-aligned window of colIdx / weights (spec 4.2). P1-P3 PLAN windows; executing them is P4. */
export interface ArcWindow {
    readonly start: number;
    readonly end: number;
    readonly rowFirst: number;
    readonly rowLast: number;
    readonly bufferIndex: number;
    readonly offset: number;
}
/** A storage-buffer range (spec 4.1); `window` is non-null only for a binding of one arc window. */
export interface Binding {
    readonly buffer: GPUBuffer;
    readonly offset: number;
    readonly size: number;
    readonly window: ArcWindow | null;
}
```

#### src/types/options.ts -- P3-T1 (spec 9.3, 7.14; structural mirrors of @graphty/layout, D27)

Imports: `import type { F32, NodeId, NodeMask } from "@graphty/graph-format";`.

```ts
/** Design 14.3 CommonLayoutOptions, mirrored verbatim. */
export interface CommonLayoutOptions {
    readonly dim?: 2 | 3 | undefined;
    readonly scale?: number | undefined;
    readonly center?: ArrayLike<number> | undefined;
    readonly seed?: number | null | undefined;
}
/** Spec 9.3 SimulationOptions, mirrored verbatim (layout-owned; the CPU simulations ignore maxInFlight). */
export interface SimulationOptions {
    readonly settleThreshold?: number | undefined;
    readonly settleWindow?: number | undefined;
    readonly iterationsPerStep?: number | undefined;
    readonly maxInFlight?: number | undefined;
}
/** Spec 9.3 ForceAtlas2Options, mirrored verbatim (same names and defaults as layout/src/layouts/force-directed/forceatlas2.ts lines 26-42). */
export interface ForceAtlas2Options extends CommonLayoutOptions, SimulationOptions {
    readonly maxIter?: number | undefined;
    readonly jitterTolerance?: number | undefined;
    readonly scalingRatio?: number | undefined;
    readonly gravity?: number | undefined;
    readonly strongGravity?: boolean | undefined;
    readonly distributedAction?: boolean | undefined;
    readonly linlog?: boolean | undefined;
    readonly nodeMass?: F32 | string | Readonly<Record<NodeId, number>> | null | undefined;
    readonly nodeSize?: F32 | string | Readonly<Record<NodeId, number>> | null | undefined;
    readonly weight?: boolean | string | null | undefined;
    readonly dissuadeHubs?: boolean | undefined;
}
/** Spec 9.3 FruchtermanReingoldOptions, mirrored for the LayoutAccelerator mirror's method signature (P5 implements it). */
export interface FruchtermanReingoldOptions extends CommonLayoutOptions, SimulationOptions {
    readonly k?: number | null | undefined;
    readonly iterations?: number | undefined;
    readonly fixed?: NodeMask | string | null | undefined;
}
/** Spec 9.3 SpringElectricalOptions, mirrored for the LayoutAccelerator mirror's method signature (P5 implements it). */
export interface SpringElectricalOptions extends CommonLayoutOptions, SimulationOptions {
    readonly springLength?: number | undefined;
    readonly springCoefficient?: number | undefined;
    readonly gravity?: number | undefined;
    readonly dragCoefficient?: number | undefined;
    readonly timeStep?: number | undefined;
}
/** The resolved (defaults applied) ForceAtlas2 option record the simulation keeps; every field present. */
export interface ResolvedForceAtlas2Options {
    readonly maxIter: number;
    readonly jitterTolerance: number;
    readonly scalingRatio: number;
    readonly gravity: number;
    readonly strongGravity: boolean;
    readonly distributedAction: boolean;
    readonly linlog: boolean;
    readonly nodeMass: F32 | string | Readonly<Record<NodeId, number>> | null;
    readonly nodeSize: F32 | string | Readonly<Record<NodeId, number>> | null;
    readonly weight: boolean | string | null;
    readonly dissuadeHubs: boolean;
    readonly dim: 2 | 3;
    readonly scale: number;
    readonly center: readonly [number, number, number];
    readonly seed: number | null;
    readonly settleThreshold: number;
    readonly settleWindow: number;
    readonly iterationsPerStep: number;
    readonly maxInFlight: number;
}
```

#### src/types/layout.ts -- P3-T1 (spec 3.3, 7.19)

Imports: `import type { F32, GraphSnapshot, NodeMask } from "@graphty/graph-format";` and `import type { LayoutSimulation } from "./accelerator.js";`.

```ts
/** Spec 3.3 LayoutStatsBase, verbatim; the three grid fields are null on the exact tier (P3 always). */
export interface LayoutStatsBase {
    readonly iteration: number;
    readonly meanDisplacement: number;
    readonly rmsRadius: number;
    readonly layoutRadius: number;
    readonly centroid: readonly [number, number, number];
    readonly repulsionTier: "exact" | "grid";
    readonly maxCellOccupancy: number | null;
    readonly outsideGrid: number | null;
    readonly msPerIteration: number | null;
}
/** One per-iteration trace record of the last completed batch (spec 3.3 ForceAtlas2Stats.trace element). */
export interface ForceAtlas2TraceRecord {
    readonly swing: number;
    readonly traction: number;
    readonly speed: number;
    readonly speedEfficiency: number;
    readonly meanDisplacement: number;
    readonly settledCount: number;
}
/** Spec 3.3 ForceAtlas2Stats, verbatim. */
export interface ForceAtlas2Stats extends LayoutStatsBase {
    readonly swing: number;
    readonly traction: number;
    readonly speed: number;
    readonly speedEfficiency: number;
    readonly trace: ReadonlyArray<ForceAtlas2TraceRecord>;
}
/** Options of GpuLayoutSimulation.run (spec 3.3). */
export interface RunOptions {
    readonly maxIter?: number | undefined;
    readonly batch?: number | undefined;
    readonly signal?: AbortSignal | undefined;
}
/** Spec 3.3 GpuLayoutSimulation, verbatim (LayoutSimulation is the design-14.3 mirror of accelerator.ts). */
export interface GpuLayoutSimulation<Options, Stats extends LayoutStatsBase> extends LayoutSimulation {
    load(snapshot: GraphSnapshot, positions: F32): void;
    /** Submits k iterations; resolves when their batch has been read back into `positions`. With `maxInFlight` batches in flight the call COALESCES: nothing is queued and the OLDEST pending batch's promise is returned (spec 7.19 item 3). */
    step(iterations?: number | undefined): Promise<void>;
    readonly settled: boolean;
    setFixed(mask: NodeMask): void;
    setPosition(index: number, x: number, y: number, z: number): void;
    dispose(): void;
    readonly inFlight: number;
    readonly iterationsDone: number;
    readonly stats: Stats;
    flush(): Promise<void>;
    reheat(): void;
    setParams(patch: Partial<Options>): void;
    run(options?: RunOptions | undefined): Promise<Stats>;
    inspect?(name: string): Promise<Float32Array | Uint32Array>;
}
/** Spec 3.3 GpuLayoutTuning, verbatim; the grid knobs are accepted and stored in P3 but only `repulsion`, `exactMaxNodes`, `deterministic` and `compat` have an effect (grid tier = P4). */
export interface GpuLayoutTuning {
    readonly repulsion?: "exact" | "grid" | "auto" | undefined;
    readonly exactMaxNodes?: number | undefined;
    readonly nearMax?: number | undefined;
    readonly deterministic?: boolean | undefined;
    readonly gridMax2D?: number | undefined;
    readonly gridMax3D?: number | undefined;
    readonly extentFactor?: number | undefined;
    readonly compat?: "paper" | "networkx" | undefined;
}
/** The resolved tuning record (defaults from constants.ts LAYOUT_TUNING_DEFAULTS applied). */
export interface ResolvedLayoutTuning {
    readonly repulsion: "exact" | "grid" | "auto";
    readonly exactMaxNodes: number;
    readonly nearMax: number;
    readonly deterministic: boolean;
    readonly gridMax2D: number;
    readonly gridMax3D: number;
    readonly extentFactor: number;
    readonly compat: "paper" | "networkx";
}
```

#### src/types/accelerator.ts -- P3-T1 (spec 9.2, 9.3, 3.3; D27)

Imports: `import type { F32, F64, GraphSnapshot, NodeMask, NumericVector, U32 } from "@graphty/graph-format";`, `import type { GpuContext } from "../context.js";`, the option types from `./options.js`, `ForceAtlas2Stats`, `GpuLayoutSimulation`, `GpuLayoutTuning` from `./layout.js`. Until W1 this file holds the STRUCTURAL MIRRORS (spec 9.2 / 9.3 verbatim); at W1 the mirrors become `import type` of the real packages and the `implements` clauses check them.

```ts
// ---- mirrors of @graphty/layout (spec 9.3)
/** Design 14.3 LayoutSimulation, verbatim. */
export interface LayoutSimulation {
    load(snapshot: GraphSnapshot, positions: F32): void;
    step(iterations?: number): void | Promise<void>;
    readonly settled: boolean;
    setFixed(mask: NodeMask): void;
    setPosition(index: number, x: number, y: number, z: number): void;
    dispose(): void;
}
/** Spec 9.3 LayoutAccelerator, verbatim. */
export interface LayoutAccelerator {
    readonly kind: string;
    forceAtlas2?(options?: ForceAtlas2Options): LayoutSimulation;
    fruchtermanReingold?(options?: FruchtermanReingoldOptions): LayoutSimulation;
    springElectrical?(options?: SpringElectricalOptions): LayoutSimulation;
    release?(s: GraphSnapshot): void;
    dispose?(): void;
}
// ---- mirrors of @graphty/algorithms (spec 9.2); the option types named there do not exist before A2, so they are mirrored as empty-extensible records
/** Placeholder for the CPU option types the AlgorithmAccelerator methods take before A2 lands (spec 9.2: "the accelerator methods reuse the indexed.* option types"). */
export type CpuAlgorithmOptions = Readonly<Record<string, unknown>>;
export interface ScoresResultLike { readonly scores: NumericVector; readonly iterations: number; readonly converged: boolean; }
export interface PageRankResultLike extends ScoresResultLike { readonly danglingMass?: number | undefined; }
export interface HitsResultLike { readonly hubs: NumericVector; readonly authorities: NumericVector; readonly iterations: number; readonly converged: boolean; }
export interface LabelResultLike { readonly labels: U32; readonly count: number; groups(): U32[]; }
export interface BfsResultLike { readonly depth: U32; readonly parent: U32; readonly order: U32; readonly visitedCount: number; }
export interface SsspResultLike { readonly dist: NumericVector; readonly predArc: U32; }
export interface BellmanFordResultLike extends SsspResultLike { readonly hasNegativeCycle: boolean; }
export interface EdgeScoresResultLike { readonly scores: NumericVector; }
export interface ApspResultLike { readonly dist: NumericVector; readonly n: number; }
export interface CorenessResultLike { readonly coreness: U32; }
export interface MstResultLike { readonly edges: U32; readonly totalWeight: number; }
export interface CommunityResultLike extends LabelResultLike { readonly modularity: number; }
/** Spec 9.2 AlgorithmAccelerator, verbatim (every member optional; option types are CpuAlgorithmOptions until A2). */
export interface AlgorithmAccelerator {
    readonly kind: string;
    pageRank?(s: GraphSnapshot, options?: CpuAlgorithmOptions): Promise<PageRankResultLike>;
    personalizedPageRank?(s: GraphSnapshot, personalization: F32 | F64, options?: CpuAlgorithmOptions): Promise<PageRankResultLike>;
    hits?(s: GraphSnapshot, options?: CpuAlgorithmOptions): Promise<HitsResultLike>;
    eigenvectorCentrality?(s: GraphSnapshot, options?: CpuAlgorithmOptions): Promise<ScoresResultLike>;
    katzCentrality?(s: GraphSnapshot, options?: CpuAlgorithmOptions): Promise<ScoresResultLike>;
    connectedComponents?(s: GraphSnapshot): Promise<LabelResultLike>;
    weaklyConnectedComponents?(s: GraphSnapshot): Promise<LabelResultLike>;
    breadthFirstSearch?(s: GraphSnapshot, source: number, options?: CpuAlgorithmOptions): Promise<BfsResultLike>;
    sssp?(s: GraphSnapshot, source: number, options?: CpuAlgorithmOptions): Promise<SsspResultLike>;
    bellmanFord?(s: GraphSnapshot, source: number, options?: CpuAlgorithmOptions): Promise<BellmanFordResultLike>;
    closenessCentrality?(s: GraphSnapshot, options?: CpuAlgorithmOptions): Promise<ScoresResultLike>;
    betweennessCentrality?(s: GraphSnapshot, options?: CpuAlgorithmOptions): Promise<ScoresResultLike>;
    edgeBetweennessCentrality?(s: GraphSnapshot, options?: CpuAlgorithmOptions): Promise<EdgeScoresResultLike>;
    allPairsShortestPath?(s: GraphSnapshot, options?: CpuAlgorithmOptions): Promise<ApspResultLike>;
    kCoreDecomposition?(s: GraphSnapshot): Promise<CorenessResultLike>;
    triangleCount?(s: GraphSnapshot): Promise<{ readonly perNode: U32; readonly total: number }>;
    labelPropagation?(s: GraphSnapshot, options?: CpuAlgorithmOptions): Promise<LabelResultLike>;
    minimumSpanningTree?(s: GraphSnapshot): Promise<MstResultLike>;
    louvain?(s: GraphSnapshot, options?: CpuAlgorithmOptions): Promise<CommunityResultLike>;
    release?(s: GraphSnapshot): void;
    dispose?(): void;
}
// ---- the package's own accelerator surface (spec 3.3); grows one method per shipped algorithm from P7
/** Spec 3.3 AcceleratorOptions, verbatim. */
export interface AcceleratorOptions {
    readonly layout?: GpuLayoutTuning | undefined;
    readonly algorithms?: { readonly betweenness?: { readonly k?: number | undefined; readonly sources?: readonly number[] | undefined } | undefined } | undefined;
}
/** The injectable object (spec 3.3); at P3 it carries forceAtlas2, release and dispose -- the algorithm members arrive with P7+. */
export interface GpuAccelerator extends AlgorithmAccelerator, LayoutAccelerator {
    readonly kind: "webgpu";
    readonly ctx: GpuContext;
    readonly options: Readonly<AcceleratorOptions>;
    forceAtlas2(options?: ForceAtlas2Options | undefined): GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>;
    release(s: GraphSnapshot): void;
    dispose(): void;
}
```

CONTRACT DECISION: `CpuAlgorithmOptions` stands in for the CPU option
types the 9.2 mirror names (`PageRankOptions` etc.), which do not exist in
this package and are not needed before P7; at W1 they become
`import type` from `@graphty/algorithms`. The option parameter is optional
and structural, so `GpuAccelerator` stays assignable to the real interface
in both directions once the real option types are substituted.

### 3.4 src/device/*.ts (spec 2.1, 2.2, 2.6, 5.7, D26)

#### src/device/webgpu-constants.ts -- P0-T3

Imports: none.

```ts
/** GPUBufferUsage bits as numbers so the core never reads the global at module top level (spec 2.1 rule 1). */
export const BufferUsage: Readonly<{ MAP_READ: 0x0001; MAP_WRITE: 0x0002; COPY_SRC: 0x0004; COPY_DST: 0x0008; INDEX: 0x0010; VERTEX: 0x0020; UNIFORM: 0x0040; STORAGE: 0x0080; INDIRECT: 0x0100; QUERY_RESOLVE: 0x0200 }>;
/** GPUMapMode bits. */
export const MapMode: Readonly<{ READ: 0x0001; WRITE: 0x0002 }>;
/** GPUShaderStage bits. */
export const ShaderStage: Readonly<{ VERTEX: 0x1; FRAGMENT: 0x2; COMPUTE: 0x4 }>;
```

Each field carries a comment naming the WebGPU spec constant;
`test/device/constants.test.ts` compares them with the runtime globals.

#### src/device/acquire.ts -- P0-T3 (isSoftwareAdapter, summarizeAdapter) / P1-T1 (the rest)

Imports: `WebGpuGraphError` from `../errors.js`; types from `../types/context.js`.

```ts
/** The six raisable limits in the order create() requests them (spec 2.2). */
export const RAISABLE_LIMITS: readonly RaisableLimit[];
/** Software adapter test (spec 2.2 step 2): architecture "software" (Dawn llvmpipe) or "swiftshader", or isFallbackAdapter === true (Chromium); the ONE reader of these fields. */
export function isSoftwareAdapter(info: AdapterInfoLike): boolean;
/** The AdapterSummary of an adapter (info + features + limits); never requests a device. */
export function summarizeAdapter(adapter: GPUAdapter): AdapterSummary;
/** Step 1 of create(): requestAdapter with the power preference; null -> E_NO_ADAPTER. */
export function requestAdapter(gpu: GPU, powerPreference: GPUPowerPreference): Promise<GPUAdapter>;
/** Step 3 of create(): the requiredLimits record for a policy, clamped to the adapter; an explicit value above the adapter -> E_NO_DEVICE { reason: "limit" }. */
export function buildRequiredLimits(adapter: GPUAdapter, policy: LimitPolicy): Record<string, number>;
/** Step 3 of create(): the requiredFeatures list = required + (optional intersect adapter.features); a missing required feature -> E_NO_DEVICE { reason: "feature" }. */
export function buildRequiredFeatures(adapter: GPUAdapter, required: readonly GPUFeatureName[], optional: readonly GPUFeatureName[]): GPUFeatureName[];
/** Step 3 of create(): requestDevice; a rejection -> E_NO_DEVICE with the adapter summary ("consumed" when the adapter already created a device, detected by the OperationError message or a prior-use record). */
export function requestDevice(adapter: GPUAdapter, descriptor: GPUDeviceDescriptor): Promise<GPUDevice>;
```

Throws: `E_NO_ADAPTER`, `E_NO_DEVICE`.

#### src/device/caps.ts -- P1-T1

```ts
/** Step 4 of create(): the GpuCaps of a DEVICE (device.limits, never adapter.limits) with the adapter info and the runtime tag. */
export function captureCaps(device: GPUDevice, info: AdapterInfoLike, runtime: "browser" | "node" | "unknown", wgslFeatures: Iterable<string>): GpuCaps;
/** GpuContext.from(device, info?): caps from device.limits / device.features and the partial info given; software false and runtime "unknown" unless given. */
export function capsFromDevice(device: GPUDevice, info: Partial<GpuCaps> | undefined): GpuCaps;
/** Asserts the two facts the planners bake in: maxComputeWorkgroupsPerDimension === MAX_WORKGROUPS_PER_DIM and WG is a power of two >= 64 (spec 2.2, 5.2). */
export function assertPlanLimits(caps: GpuCaps): void;
/** The workgroup size a device runs: min(WORKGROUP_SIZE, caps.limits.maxComputeInvocationsPerWorkgroup) rounded down to a power of two (spec 5.1). */
export function workgroupSizeFor(caps: PlanCaps): number;
```

Throws: `E_NO_DEVICE { reason: "maxComputeWorkgroupsPerDimension" }` from `assertPlanLimits`.

#### src/device/error-scope.ts -- P1-T1

```ts
/** Runs `fn` inside pushErrorScope("validation"); a captured error becomes E_VALIDATION { label, message } thrown from the returned promise (spec 5.7). */
export function withValidationScope<T>(device: GPUDevice, label: string, fn: () => Promise<T> | T): Promise<T>;
/** Formats GPUCompilationInfo messages with line numbers relative to the kernel BODY (the prelude line count subtracted) (spec 5.1). */
export function formatCompilationInfo(info: GPUCompilationInfo, preludeLines: number): string[];
/** Creates buffers, wrapping sizes >= OOM_SCOPE_THRESHOLD_BYTES in an "out-of-memory" scope whose pop is collected; `check()` surfaces the first OOM (spec 4.4, 5.7). Owned by the context, shared by the residency and the pool. */
export class AllocationTracker {
    constructor(device: GPUDevice, thresholdBytes?: number | undefined);
    /** createBuffer with the scope rule; every buffer is labelled. */
    createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
    /** Destroys a buffer created here and decrements `resident`. */
    destroy(buffer: GPUBuffer): void;
    /** Awaits every outstanding scope pop; the first OOM -> E_OUT_OF_MEMORY { requested, resident, label }; later calls after a failure throw the same error again until `reset()`. */
    check(): Promise<void>;
    /** Forgets a recorded OOM after the caller released what it allocated. */
    reset(): void;
    /** Bytes of live buffers created through this tracker. */
    readonly resident: number;
    /** Number of live buffers created through this tracker. */
    readonly liveBuffers: number;
}
```

Throws: `E_VALIDATION`, `E_OUT_OF_MEMORY`.

#### src/device/lost.ts -- P1-T1

```ts
/** The context's pending-error slot (spec 5.7): the last uncaptured error not yet delivered; `take()` clears it. */
export class PendingErrorSlot {
    /** Stores an error; a second error before `take()` is kept as `details.next` of the first. */
    set(error: WebGpuGraphError): void;
    /** Returns and clears the pending error, or null. */
    take(): WebGpuGraphError | null;
    /** True when an error is pending. */
    readonly pending: boolean;
}
/** Installs the uncapturederror listener: each event becomes E_VALIDATION { label, message } (or E_OUT_OF_MEMORY for GPUOutOfMemoryError) routed to `onError`, else stored in `slot`. Returns the uninstaller. */
export function installUncapturedErrorSink(device: GPUDevice, slot: PendingErrorSlot, onError: ((error: WebGpuGraphError) => void) | null): () => void;
/** Chains device.lost into `onLost` exactly once; returns the promise the context exposes as `ctx.lost`. */
export function watchDeviceLost(device: GPUDevice, onLost: (info: GPUDeviceLostInfo) => void): Promise<GPUDeviceLostInfo>;
/** The E_DEVICE_LOST error for a lost-info record. */
export function deviceLostError(info: GPUDeviceLostInfo): WebGpuGraphError;
```

Throws: none directly (errors are constructed, not thrown).

### 3.5 src/context.ts -- P1-T1 (spec 2.2, 2.8, 3.3, 5.7; D26)

Imports: graph-format `GraphSnapshot` (type); `../errors.js`; `../constants.js`; `../types/context.js`; `./device/*.js`; `./memory/residency.js`, `./memory/buffer-pool.js`, `./memory/readback.js`; `./kernel/pipeline-cache.js`, `./kernel/profiler.js` (never `./kernels.js` or `./wgsl/**`: the cache composes with the prelude of `./kernel/prelude.js` itself).

```ts
/** One device, its caps and the singletons built on it (spec 3.3; the composition root of D26). */
export class GpuContext {
    /** Probe without creating a device (spec 2.2); never throws. */
    static probe(options: ProbeOptions): Promise<ProbeResult>;
    /** The six-step acquisition of spec 2.2. */
    static create(options: GpuContextOptions): Promise<GpuContext>;
    /** Zero-cost adoption of a device the caller owns (ownsDevice false, runtime "unknown" unless info says otherwise). */
    static from(device: GPUDevice, info?: Partial<GpuCaps> | undefined): GpuContext;
    readonly device: GPUDevice;
    readonly caps: GpuCaps;
    /** "ready" | "lost" | "disposed" (a getter). */
    readonly state: "ready" | "lost" | "disposed";
    readonly lost: Promise<GPUDeviceLostInfo>;
    /** @internal the upload cache (spec 3.3). */
    readonly residency: GraphResidency;
    /** Non-null when "timestamp-query" was granted (P2-T1 makes it functional; P1 always null). */
    readonly profiler: Profiler | null;
    /** @internal the compile-once pipeline cache. */
    readonly pipelines: PipelineCache;
    /** @internal the scratch pool. */
    readonly pool: BufferPool;
    /** @internal the staging ring. */
    readonly readback: Readback;
    /** @internal the OOM-scoped allocator shared by residency and pool. */
    readonly allocator: AllocationTracker;
    /** @internal the workgroup size of this device (spec 5.1). */
    readonly workgroupSize: number;
    /** @internal true when create() made the device (dispose() destroys it). */
    readonly ownsDevice: boolean;
    /** @internal the label given to create(). */
    readonly label: string;
    /** @internal mutable test-build flags (GpuDebugFlags). */
    readonly debug: GpuDebugFlags;
    /** @internal throws E_DEVICE_LOST / E_DISPOSED by state and rethrows a pending uncaptured error (spec 5.7: "thrown from the next public call"). */
    assertReady(): void;
    /** @internal takes the pending uncaptured error (used by CommandBatch right after submit). */
    takePendingError(): WebGpuGraphError | null;
    /** @internal monotonically increasing batch ids, starting at 1. */
    nextBatchId(): number;
    /** @internal registers a device-loss listener (simulations enter "disposed"); returns the unregister function. */
    onLost(listener: (info: GPUDeviceLostInfo) => void): () => void;
    /** @internal a disposer run by dispose() (the Node entry attaches the GPU handle's dispose). */
    attachDisposer(dispose: () => void): void;
    /** Destroys every buffer recorded for the snapshot (spec 4.5), then `pool.trim()` (spec 4.4: idle scratch goes with the graph); idempotent; safe on a snapshot never uploaded. */
    release(snapshot: GraphSnapshot): void;
    /** Rejects pending work with E_DISPOSED, destroys residency, pool and staging ring, destroys the device when owned, runs attached disposers; idempotent (spec 2.8). */
    dispose(): void;
}
```

Method contracts: `create()` follows spec 2.2 steps 1-6 in order; step 4
also calls `assertPlanLimits`; `optionalFeatures` default
`["subgroups", "timestamp-query"]`, `limits` default `"raise"`,
`powerPreference` default `"high-performance"`, `warnUnreleasedSnapshots`
default 2, `runtime` default `"unknown"`. Device loss: `state` becomes
`"lost"`, the residency is cleared without destroying (the buffers are gone),
every registered `onLost` listener runs, `assertReady()` throws
`E_DEVICE_LOST` afterwards. `probe()` maps `gpu === undefined` ->
`E_NO_WEBGPU`, `requestAdapter() === null` (or a throw) -> `E_NO_ADAPTER`,
`rejectSoftware && software` -> `E_SOFTWARE_ONLY`, else `OK` with the
UNUSED adapter and its summary.

Throws: `E_NO_WEBGPU` (create with neither gpu, adapter nor device),
`E_NO_ADAPTER`, `E_NO_DEVICE`, `E_SOFTWARE_ONLY`, `E_DEVICE_LOST`,
`E_DISPOSED`, `E_VALIDATION` (rethrown pending error), `E_OUT_OF_MEMORY`
(rethrown pending error).

### 3.6 src/browser/index.ts -- P0-T3 (empty) / P1-T1 (spec 2.3, 3.4)

Imports: `GpuContext`, `WebGpuGraphError`, types. P0 content:
`/// <reference types="@webgpu/types" />` plus `export {};` and a header
comment (CONTRACT DECISION: the bundle entry must exist at P0 for
`build:bundle`; an empty module keeps the barrel honest).

```ts
/** Options of the browser helpers (spec 3.4); a type alias, not an empty `extends` interface, which strictTypeChecked's no-empty-object-type (allowInterfaces "never") reports. */
export type BrowserGpuOptions = Omit<GpuContextOptions, "gpu" | "device" | "runtime">;
/** navigator.gpu absent -> { ok: false, code: "E_NO_WEBGPU" }; else GpuContext.probe. Never throws. */
export function probeBrowserWebGpu(options?: BrowserGpuOptions | undefined): Promise<ProbeResult>;
/** GpuContext.create({ gpu: navigator.gpu, powerPreference: "high-performance", runtime: "browser", ...options }). */
export function requestGpuContext(options?: BrowserGpuOptions | undefined): Promise<GpuContext>;
```

Throws (requestGpuContext): `E_NO_WEBGPU`, `E_NO_ADAPTER`, `E_NO_DEVICE`, `E_SOFTWARE_ONLY`.

### 3.7 src/node/index.ts -- P0-T3 (createNodeGpu) / P1-T1 (spec 2.3, 3.4)

Imports: `GpuContext`, `WebGpuGraphError`, types. The only file that names
the `"webgpu"` module, inside `await import("webgpu")` in a function body.

```ts
/** Options of the Node helpers (spec 3.4). */
export interface NodeGpuOptions extends Omit<GpuContextOptions, "gpu" | "adapter" | "device" | "runtime"> {
    readonly adapter?: string | undefined;
    readonly backend?: "vulkan" | "d3d12" | "d3d11" | "metal" | "opengl" | "opengles" | "null" | undefined;
    readonly dawnFeatures?: readonly string[] | undefined;
    readonly software?: boolean | undefined;
    readonly installGlobals?: boolean | undefined;
    /** @internal test seam: replaces `() => import("webgpu")` so a missing / broken module can be simulated. */
    readonly loadModule?: (() => Promise<unknown>) | undefined;
}
/** The Dawn GPU handle (spec 2.3): dispose() drops the reference so the process can exit. */
export interface NodeGpuHandle {
    readonly gpu: GPU;
    dispose(): void;
}
/** import("webgpu"), install dawn.globals unless installGlobals === false, dawn.create(flags) (spec 2.3). */
export function createNodeGpu(options?: NodeGpuOptions | undefined): Promise<NodeGpuHandle>;
/** createNodeGpu + GpuContext.create({ gpu, runtime: "node", ...options }); ctx.dispose() also disposes the handle. */
export function createNodeGpuContext(options?: NodeGpuOptions | undefined): Promise<GpuContext>;
/** createNodeGpu + GpuContext.probe + dispose; never throws (a load failure is { code: "E_NO_WEBGPU" }). */
export function probeNodeWebGpu(options?: NodeGpuOptions | undefined): Promise<ProbeResult>;
/** The Dawn flag list a NodeGpuOptions maps to (exported for the tests and scripts/gpu-report.js): adapter=<s>, backend=<s>, enable-dawn-features=a,b, and software -> adapter=llvmpipe. */
export function dawnFlags(options: NodeGpuOptions | undefined): string[];
```

Throws: `E_NO_WEBGPU { reason, hint }` (module missing, glibc too old, or
the module lacks `create`), plus everything `GpuContext.create` throws.

### 3.8 src/memory/*.ts (spec 4.1-4.5)

#### src/memory/upload-plan.ts -- P1-T2 (spec 4.2; pure)

Imports: graph-format `GraphSnapshot`, `CoreArrayName`, `ArenaLayout`; `../constants.js`; `../errors.js`; `PlanCaps` / `PlanLimits` from `../types/context.js`; `ArcWindow` (type) from `../types/memory.js` (declared there, 3.3).

```ts
/** Where one core array's bytes go: one or more buffers (split at window boundaries only when the array exceeds maxBufferSize). */
export interface PlannedArray {
    readonly name: CoreArrayName;
    readonly byteLength: number;
    readonly buffers: readonly { readonly byteOffset: number; readonly byteLength: number }[];
}
/** The arena path: ONE buffer of `bytes`, one writeBuffer, per-segment bindings at `segment.byteOffset - arena.byteOffset`. */
export interface ArenaPlan {
    readonly kind: "arena";
    readonly bytes: number;
    readonly includesCold: boolean;
    readonly segments: Readonly<Record<CoreArrayName, { readonly offset: number; readonly size: number } | null>>;
}
/** The per-array path: one buffer per needed array, whole-buffer bindings. */
export interface PerArrayPlan {
    readonly kind: "perArray";
    readonly arrays: readonly PlannedArray[];
}
/** The windowed path: per-array buffers plus the arc windows kernels iterate (P4 executes; P1-P3 only plan). */
export interface WindowedPlan {
    readonly kind: "windowed";
    readonly arrays: readonly PlannedArray[];
    readonly windows: readonly ArcWindow[];
    readonly arcsPerWindow: number;
}
export type UploadPlan = ArenaPlan | PerArrayPlan | WindowedPlan;
/** The pure planner of spec 4.2: arena (hot prefix unless `need` names a cold segment and the full arena fits) -> perArray -> windowed, in that order. */
export function planUpload(s: GraphSnapshot, caps: PlanCaps, need: readonly CoreArrayName[]): UploadPlan;
/** The window list for a rowPtr (pure; unit-tested with a synthetic rowPtr above 2^31 arcs): start = rowPtr[v0] - rowPtr[v0] % ARC_WINDOW_ALIGN, at most `arcsPerWindow` arcs per window, a row longer than a window split across windows. */
export function planArcWindows(rowPtr: Uint32Array, arcCount: number, arcsPerWindow: number): ArcWindow[];
/** The largest 64-aligned arc count one binding holds: floor(maxStorageBufferBindingSize / 4) rounded down to a multiple of ARC_WINDOW_ALIGN. */
export function arcsPerWindowFor(limits: PlanLimits): number;
```

Rules: `need` always contains `"rowPtr"`; `"colIdx"` and `"weights"` are
included by the residency's `core()` unless the caller narrows it; a
`weights` segment is null when unweighted; `edgeToArc` / `arcToEdge` are
cold. A snapshot with `arcCount === 0` plans `rowPtr` only. Arithmetic uses
`%` and `Math.floor`, never a bitwise operator (I3).

Throws: `E_TOO_LARGE { needed, limit, path: "rowPtr" | "binding", algorithm: null }` when even `rowPtr` exceeds `maxStorageBufferBindingSize` (no window can hold it).

#### src/memory/residency.ts -- P1-T2 (spec 4.1, 4.3, 4.5)

Imports: graph-format `GraphSnapshot`, `AttributeTable`, `Column`, `TypedArrayData`, `CoreArrayName`, `ViewName`, `GpuEligibility`, `INVALID_INDEX`; `../constants.js`; `../errors.js`; `./upload-plan.js`; `Binding` / `ArcWindow` (types) from `../types/memory.js`; `AllocationTracker` from `../device/error-scope.js`; `PlanCaps` from `../types/context.js`; `BufferUsage` from `../device/webgpu-constants.js`.

```ts
/** The core arrays of a snapshot on the device (spec 4.1); a null member is absent (zero-length, unweighted or an identity permutation) and is bound as a dummy by the kernel layer (3.9). `Binding` itself is declared in src/types/memory.ts (3.3). */
export interface CoreBinding {
    readonly serial: number;
    readonly plan: "arena" | "perArray" | "windowed";
    readonly rowPtr: Binding;
    readonly colIdx: Binding | null;
    readonly weights: Binding | null;
    readonly arcToEdge: Binding | null;
    readonly edgeToArc: Binding | null;
    readonly windows: readonly ArcWindow[] | null;
    readonly hasWeights: boolean;
}
/** A view's arrays on the device plus the CPU-side scalars (degreeOrder's segmentOffsets) (spec 4.3). */
export interface ViewBinding {
    readonly view: "reverse" | "coo" | "edgeList" | "outDegree" | "inDegree" | "degreeOrder" | "reverseDegreeOrder" | "mate";
    readonly bindings: Readonly<Record<string, Binding>>;
    readonly scalars: Readonly<Record<string, readonly number[]>>;
}
/** A column's gpuView() on the device (spec 4.3). */
export interface ColumnBinding {
    readonly binding: Binding;
    readonly column: Column;
    readonly version: number;
    readonly eligibility: GpuEligibility;
    readonly components: number;
}
/** An ad hoc array on the device (spec 4.1 last row); `destroy()` is the caller's when no owner was given. */
export interface ArrayBinding {
    readonly binding: Binding;
    readonly byteLength: number;
    readonly owner: GraphSnapshot | null;
    destroy(): void;
}
/** stats() shape (spec 4.1). */
export interface ResidencyStats {
    readonly buffers: number;
    readonly bytes: number;
    readonly snapshots: number;
    readonly perSnapshot: readonly { readonly serial: number; readonly label: string | null; readonly bytes: number; readonly buffers: number }[];
}
/** The upload cache (spec 4.1): WeakMap on array objects, WeakMap on snapshots, a strong Map by serial; @internal (reached as ctx.residency). */
export class GraphResidency {
    constructor(device: GPUDevice, caps: PlanCaps, allocator: AllocationTracker, options: { readonly warnUnreleasedSnapshots: number; readonly warn?: ((message: string) => void) | undefined });
    /** Uploads (or finds) the core; `need` defaults to ["rowPtr", "colIdx", "weights"]; cold segments on demand (spec 4.2). Never materialises an identity permutation. */
    core(s: GraphSnapshot, need?: readonly CoreArrayName[] | undefined): CoreBinding;
    /** Uploads (or finds) a view; P1-P3 support outDegree / inDegree / degreeOrder / reverseDegreeOrder; the others -> E_UNSUPPORTED until P7. packViews is accepted and ignored until P7. */
    view(s: GraphSnapshot, name: "reverse" | "coo" | "edgeList" | "outDegree" | "inDegree" | "degreeOrder" | "reverseDegreeOrder" | "mate", options?: { readonly packViews?: boolean | undefined } | undefined): ViewBinding;
    /** gpuView(name) + column.version; re-uploads in place when the version changed and the byte length did not (spec 4.3). CONTRACT DECISION: `owner` is required so release(owner) can find the buffer (a table has no back-reference to its snapshot). */
    column(table: AttributeTable, name: string, owner: GraphSnapshot): ColumnBinding;
    /** Any format array keyed on the object; registered against `owner` when given (spec 4.1). */
    array(key: TypedArrayData, label: string, owner?: GraphSnapshot | undefined): ArrayBinding;
    /** Destroys every buffer recorded for s.serial (siblings included, Q-27) and tombstones the serial; idempotent (spec 4.5). */
    release(s: GraphSnapshot): void;
    /** Drops every record WITHOUT destroying (device lost: the buffers are gone). @internal */
    clearOnLoss(): void;
    /** Destroys everything (ctx.dispose()). @internal */
    destroyAll(): void;
    stats(): ResidencyStats;
    readonly residentBytes: number;
    /** True when the serial was released and not re-uploaded. @internal */
    isReleased(serial: number): boolean;
}
```

Rules: a released serial is tombstoned; `core()` / `view()` / `column()` /
`array()` on a tombstoned snapshot throw `E_RELEASED { serial }` (a new
`load()` of the same snapshot object after `release` is allowed: the
tombstone is lifted by the next `core()` call, which re-uploads; the
tombstone exists to fail a LIVE user, which the simulation checks through
`isReleased(serial)` before every step). The once-only warning fires through
`options.warn` (default `console.warn`) when `stats().snapshots >
warnUnreleasedSnapshots`. `writeBuffer` sizes are multiples of 4 by I10.
Zero-length arrays are never uploaded (spec 5.6).

Throws: `E_RELEASED`, `E_TOO_LARGE` (from the planner), `E_OUT_OF_MEMORY`
(via the allocator's `check()`, surfaced by the caller), `E_UNSUPPORTED`
(view names of P7+, `packViews` requests), `E_SNAPSHOT { reason: "detached"
}` (a detached snapshot), `E_INVALID_ARGUMENT` (an array whose byteLength is
0 or not a multiple of 4, or a SharedArrayBuffer-backed array). Pass-through:
`E_GPU_INELIGIBLE`, `E_UNKNOWN_COLUMN` from `gpuView` / `require`.

#### src/memory/buffer-pool.ts -- P1-T2 (spec 4.4)

Imports: `../constants.js`, `../errors.js`, `AllocationTracker`, `BufferUsage`; `Lease` (P2) from `./lease.js`.

```ts
/** Size-class pool of GPUBuffers by usage (spec 4.4). */
export class BufferPool {
    constructor(device: GPUDevice, allocator: AllocationTracker, maxBufferSize: number, options?: { readonly maxIdlePerClass?: number | undefined } | undefined);
    /** The size class a byte length rounds up to: powers of two from 4 KiB to 64 MiB, then 16 MiB steps (pure, static). */
    static sizeClass(byteLength: number): number;
    /** Acquires a buffer of at least byteLength (an idle one of the class and usage, else a new one through the allocator); E_TOO_LARGE above maxBufferSize. */
    acquire(byteLength: number, usage: number, label: string): GPUBuffer;
    /** Returns a buffer to its class; destroys it when the class already holds maxIdlePerClass idle buffers. */
    release(buffer: GPUBuffer): void;
    /** Destroys every idle buffer. */
    trim(): void;
    /** Destroys everything, idle and live (ctx.dispose()). @internal */
    destroyAll(): void;
    /** A scope object that releases everything acquired through it (P2). */
    lease(): Lease;
    /** Bytes of buffers acquired and not yet released. */
    readonly liveBytes: number;
    /** Bytes of idle buffers. */
    readonly idleBytes: number;
}
```

Throws: `E_TOO_LARGE { needed, limit, path: "pool", algorithm: null }`, `E_INVALID_ARGUMENT` (release of a buffer not from this pool), `E_DISPOSED`.

#### src/memory/readback.ts -- P1-T2 (spec 4.4)

```ts
/** One MAP_READ | COPY_DST staging buffer of the ring. */
export interface StagingSlot {
    readonly index: number;
    readonly buffer: GPUBuffer;
    readonly capacity: number;
}
/** The staging ring (spec 4.4): default 3 slots; grows when every slot is busy; always unmaps and destroys its own buffers. */
export class Readback {
    constructor(device: GPUDevice, allocator: AllocationTracker, options?: { readonly slots?: number | undefined; readonly slotBytes?: number | undefined } | undefined);
    /** Copies `byteLength` bytes from `src` at `srcOffset` (own encoder, own submit), maps, copies out BEFORE unmap; resolves the bytes (a fresh ArrayBuffer, or `dest.buffer` after `dest.set` when given). Requests above the slot size are chunked. */
    read(src: GPUBuffer, byteLength: number, dest?: ArrayBufferView | undefined, srcOffset?: number | undefined): Promise<ArrayBuffer>;
    /** Reads one u32 counter through the same ring. */
    readU32(src: GPUBuffer, byteOffset: number): Promise<number>;
    /** Borrows an unmapped slot of at least byteLength (grows the ring when none is free); the borrower ALWAYS returns it (spec 4.4). */
    borrowSlot(byteLength: number): StagingSlot;
    /** mapAsync(READ) on a borrowed slot, tracked by the ring (CONTRACT DECISION RB-1 below); returns the runtime's promise untranslated. */
    mapSlot(slot: StagingSlot, byteLength: number): Promise<void>;
    /** Returns a borrowed slot (unmapping it if mapped); a slot whose map through the ring is pending stays borrowed until the map settles, when the ring unmaps and frees it. */
    returnSlot(slot: StagingSlot): void;
    /** Number of slots (grows). */
    readonly slots: number;
    /** Slots currently borrowed. */
    readonly borrowed: number;
    /** Destroys every staging buffer (ctx.dispose()). @internal */
    destroyAll(): void;
}
```

Throws: `E_DEVICE_LOST` (mapAsync rejected after loss), `E_DISPOSED`, `E_INVALID_ARGUMENT` (byteLength 0 or not a multiple of 4; a dest too small; `mapSlot` on a slot not borrowed, already mapped or with a map pending).

CONTRACT DECISION RB-1 (2026-09-16, the macOS host lane): a staging buffer whose mapAsync is PENDING is never unmapped and never destroyed. dawn-node 0.4.0 settles the pending promise synchronously on `unmap()` / `destroy()` (`GPUBuffer::DetachMappings`) and again when Dawn's map callback arrives (`AsyncRunner::Reject` on a concluded N-API deferred: SIGSEGV, the crash report of hosts.yml run 35135772420); on the Vulkan backends the callback has already run inside `device.destroy()`, on Dawn's Metal backend it arrives after the device-lost fan-out. So every map on a slot goes through the ring (`read()` and `mapSlot`), which records it as pending and defers a `returnSlot` (the slot stays borrowed) or a `destroyAll` (the slot is destroyed at the settle; a pending `read()` still rejects `E_DISPOSED`) that arrives during it. `CommandBatch.awaitReadback` maps through `mapSlot`, and when device loss wins its race against the map it lets the map settle (bounded by `LOSS_GRACE_MS`, for a runtime that never settles it) before returning the slot -- the readback still rejects `E_DEVICE_LOST`, one macrotask later on dawn-node.

#### src/memory/lease.ts -- P2-T1 (spec 4.4)

Imports: `import type { BufferPool } from "./buffer-pool.js"` (TYPE only: buffer-pool.ts imports `Lease` as a value for `lease()`, so this side of the pair must stay type-only for the 5.5 cycle walk), `../device/webgpu-constants.js`, `../errors.js`.

```ts
/** Scope object algorithms use: every buffer acquired through it is released by `release()` in a finally block (spec 4.4). */
export class Lease {
    constructor(pool: BufferPool);
    /** pool.acquire(byteLength, STORAGE | COPY_SRC | COPY_DST, label). */
    storage(byteLength: number, label: string): GPUBuffer;
    /** pool.acquire(byteLength, UNIFORM | COPY_DST, label). */
    uniform(byteLength: number, label: string): GPUBuffer;
    /** pool.acquire with explicit usage. */
    acquire(byteLength: number, usage: number, label: string): GPUBuffer;
    /** Releases every buffer acquired through this lease; idempotent. */
    release(): void;
    /** Live buffers of this lease. */
    readonly count: number;
}
```

Throws: what `BufferPool.acquire` throws; `E_DISPOSED` after `release()`.

### 3.9 src/kernel/*.ts (spec 3.5, 5.1-5.5, 5.8; D20)

#### src/kernel/wgsl.ts -- P1-T3 (spec 3.5)

Imports: `../errors.js`; `../constants.js`; `./struct-block.js`; `./prelude.js`; `PlanCaps` from `../types/context.js`.

```ts
/** One storage or uniform binding of a module (spec 3.5). `wgslType` is the element / struct type text ("array<u32>", "array<vec4f>", "Fa2State"). */
export interface BindingDecl {
    readonly group: 0 | 1 | 2 | 3;
    readonly binding: number;
    readonly name: string;
    readonly kind: "storage" | "storage-ro" | "uniform";
    readonly wgslType: string;
}
/** One module-specific override (spec 3.5); the five standard ones (WG, USE_PERM, HAS_WEIGHTS, SUBGROUP_MIN, SUBGROUP_MAX) come from the prelude and are never listed here. */
export interface OverrideDecl {
    readonly name: string;
    readonly type: "u32" | "bool" | "f32";
    readonly default: number | boolean;
}
/** Spec 3.5 WgslModuleSpec, verbatim. `needs: ["subgroups"]` means "this body calls a wg_reduce_* helper": the composer splices the subgroup helper block when the device has the feature and the workgroup-memory twin otherwise. */
export interface WgslModuleSpec {
    readonly id: string;
    readonly body: string;
    readonly bindings: readonly BindingDecl[];
    readonly overrideDecls: readonly OverrideDecl[];
    readonly overrides: Readonly<Record<string, number | boolean>>;
    readonly needs: readonly ("subgroups")[];
    readonly uniforms: readonly UniformBlock[];
    readonly snippets?: Readonly<Record<string, string>> | undefined;
}
/** The names the prelude declares as overrides; spec.overrides may set them without an OverrideDecl. */
export const STANDARD_OVERRIDES: readonly ["WG", "USE_PERM", "HAS_WEIGHTS", "SUBGROUP_MIN", "SUBGROUP_MAX"];
/** What composeWgsl returns: the text, the body's first line in it (for compilation-info formatting), the effective overrides (WG, SUBGROUP_MIN, SUBGROUP_MAX filled) and the features to enable. */
export interface ComposedModule {
    readonly id: string;
    readonly code: string;
    readonly bodyLine: number;
    readonly overrides: Readonly<Record<string, number | boolean>>;
    /** CONTRACT DECISION (2026-09-16, demo on an iPad): the subset of `overrides` the code references outside comments and its own `override` lines -- what PipelineCache supplies as the pipeline `constants`. WebKit fails "Compute library failed creation" for a constant naming an unread override (HAS_WEIGHTS in degree, TIER in the thread-per-row tiers); the spec allows it ("not required to be statically used"), Dawn tolerates it. The cache key keeps `overrides`. */
    readonly constants: Readonly<Record<string, number | boolean>>;
    readonly entryPoint: string;
    readonly subgroups: boolean;
}
/** String concatenation of spec 3.5: prelude (constants, standard overrides, helpers), module overrides, struct texts, the bind declarations, then the body with snippets substituted (4.2 gives the exact emitted format). */
export function composeWgsl(spec: WgslModuleSpec, caps: PlanCaps): ComposedModule;
/** The entry-point name of a body: the identifier after `fn` on the line following `@compute` (every body has exactly one entry point). */
export function entryPointOf(body: string): string;
/** The bind-group-layout descriptors derived from spec.bindings: one per group 0..maxGroup (empty groups get an empty entry list); uniform decls carry hasDynamicOffset true (spec 5.1). */
export function bindGroupLayoutDescriptors(spec: WgslModuleSpec): GPUBindGroupLayoutDescriptor[];
/** The names of the bindings of one group in binding order. */
export function bindingNames(spec: WgslModuleSpec, group: number): readonly string[];
```

Compose-time checks (each `E_SHADER_COMPILE { id, stage: "compose", slot }`):
an override key in `spec.overrides` that is neither a `STANDARD_OVERRIDES`
name nor an `overrideDecls` name; a snippet key with no `//@@KEY@@` marker
in the body; a `//@@` marker left after substitution; a body containing
`@group(` or `override ` (the unit test of spec 3.5); two bindings with the
same `(group, binding)` or the same name; `wgslType` naming a struct that no
block in `spec.uniforms` declares; `needs` naming a feature other than
`"subgroups"`; a body or snippet (comments stripped) that declares or reads
an identifier from the WGSL reserved-word list (WGSL spec section 16.2:
`target`, `filter`, `partition`, `layout`, `common`, `ref`, `self`, `type`,
`use`, `with`, ... 147 words; `free`, `valid`, `tile`, `slot`, `key`,
`count` are NOT reserved -- kept as `WGSL_RESERVED_WORDS`, a frozen string
array in `src/kernel/prelude.ts`; CONTRACT DECISION: the
check is textual so a reserved identifier is a compose-time
`E_SHADER_COMPILE { slot: "reserved:<word>" }` on every device instead of a
shader-creation error on the first one). The composer fills `overrides.WG =
workgroupSizeFor(caps)` unless the spec sets `WG`, and, whenever the
subgroup block is spliced, `overrides.SUBGROUP_MAX = caps.subgroupMaxSize`
and `overrides.SUBGROUP_MIN = Math.max(4, caps.subgroupMinSize || 4)` (the
scratch of 4.3 is sized by the MINIMUM size; 4 is the smallest legal
subgroup size).

Throws: `E_SHADER_COMPILE`.

#### src/kernel/prelude.ts -- P1-T3 (spec 3.5; CONTRACT DECISION)

CONTRACT DECISION: the prelude and the two reduction-helper blocks live in
`src/kernel/prelude.ts` instead of `src/wgsl/prelude.wgsl.ts`, because the
composer (kernel layer) must own the text it splices and the rule
"`src/wgsl/**` is imported only by `kernels.ts`" then holds exactly, with
`src/wgsl/` = kernel BODIES only. The literal-grep test of spec 3.5 covers
`src/wgsl/**` AND this file.

```ts
/** The prelude text with the constants interpolated from constants.ts and graph-format's INVALID_INDEX (4.1). */
export const PRELUDE_WGSL: string;
/** The workgroup-memory reduction helpers (the twin) (4.3). */
export const REDUCE_HELPERS_WORKGROUP_WGSL: string;
/** The subgroup reduction helpers, spliced only with `enable subgroups;` (4.3). */
export const REDUCE_HELPERS_SUBGROUP_WGSL: string;
/** Line count of PRELUDE_WGSL (the compilation-info formatter subtracts it plus the emitted declarations). */
export const PRELUDE_LINES: number;
/** The helper function names a body may call when its spec lists needs: ["subgroups"]. */
export const REDUCE_HELPER_NAMES: readonly ["wg_reduce_f32", "wg_reduce_u32", "wg_reduce_vec4"];
/** The WGSL reserved words of spec section 16.2 (the `_reserved` production, 147 words incl. `target`), frozen; composeWgsl rejects a body or snippet that uses one (3.9). */
export const WGSL_RESERVED_WORDS: readonly string[];
```

Throws: none.

#### src/kernel/struct-block.ts -- P1-T3 (spec 5.3; D20)

```ts
/** Field types a block may hold; no vec3 (spec 5.3), no bool (not host-shareable), no arrays. */
export type UniformFieldType = "u32" | "i32" | "f32" | "vec2f" | "vec2u" | "vec4f" | "vec4u";
/** One field. */
export type UniformField = readonly [name: string, type: UniformFieldType];
/** The values written into or read from a block: scalars as numbers, vectors as number arrays of the vector's width. */
export type UniformValues = Readonly<Record<string, number | readonly number[]>>;
/** A generated struct: the padded WGSL text and the byte writer / reader share one field table, so they cannot disagree (D20). */
export class UniformBlock {
    /** Declares a block; fields are laid out in order with 16-byte alignment for vec4 / the struct, 8 for vec2, 4 for scalars; the total is padded to 16 (uniform, storage) or to `padTo` when given. */
    static define(name: string, fields: readonly UniformField[], options?: { readonly layout?: "uniform" | "storage" | undefined; readonly padTo?: number | undefined } | undefined): UniformBlock;
    readonly name: string;
    readonly layout: "uniform" | "storage";
    readonly fields: readonly UniformField[];
    /** The padded byte length (a multiple of 16). */
    readonly byteLength: number;
    /** The `struct <name> { ... }` text with explicit `@size` / `@align` where padding is needed. */
    readonly wgsl: string;
    /** Byte offset of a field; E_INVALID_ARGUMENT for an unknown field. */
    offsetOf(field: string): number;
    /** Writes `values` at `byteOffset` (default 0); a missing field is written as 0; an unknown key is E_INVALID_ARGUMENT; a vector of the wrong width is E_INVALID_ARGUMENT. Always little-endian. */
    write(view: DataView, values: UniformValues, byteOffset?: number | undefined): void;
    /** Reads every field at `byteOffset` (storage mode's reader; also used by tests on uniform blocks). */
    read(view: DataView, byteOffset?: number | undefined): UniformValues;
    /** Reads one field. */
    readField(view: DataView, field: string, byteOffset?: number | undefined): number | readonly number[];
}
```

Throws: `E_INVALID_ARGUMENT` (duplicate field, unknown field, wrong vector
width, `padTo` smaller than the natural size or not a multiple of 16).

#### src/kernel/pipeline-cache.ts -- P1-T3 (spec 5.1)

Imports: `./wgsl.js`, `./kernel.js`, `../device/error-scope.js`, `../errors.js`.

```ts
/** Compile-once pipelines keyed by (id, overrides, needs present on the device, snippets) with explicit bind-group layouts (spec 5.1). */
export class PipelineCache {
    constructor(device: GPUDevice, caps: PlanCaps);
    /** The cache key of a spec on this device: `id + "|" + stableJson(overrides) + "|" + needs.filter(present).join(",") + "|" + hash(snippets)`. */
    key(spec: WgslModuleSpec): string;
    /** createComputePipelineAsync inside a validation scope with the layouts derived from spec.bindings; compilation messages become E_SHADER_COMPILE { id, stage: "compile", messages } with body-relative lines. */
    get(spec: WgslModuleSpec): Promise<GPUComputePipeline>;
    /** The Kernel (pipeline + layouts + binding names) for a spec; cached with the pipeline. */
    kernel(spec: WgslModuleSpec): Promise<Kernel>;
    /** Compiles every spec not yet cached (load() calls it so the first step() does not compile). */
    warm(specs: readonly WgslModuleSpec[]): Promise<void>;
    /** Number of cached pipelines. */
    readonly size: number;
    /** Every key created so far, in creation order (the override-matrix coverage test reads it). @internal */
    keys(): readonly string[];
    /** The bind-group layouts of a cached kernel by key. @internal */
    layoutsOf(key: string): readonly GPUBindGroupLayout[] | null;
}
```

Throws: `E_SHADER_COMPILE`, `E_VALIDATION` (a pipeline error that is not a
compilation message), `E_DISPOSED`.

#### src/kernel/kernel.ts -- P1-T3 (spec 5.1)

Imports: `./wgsl.js`, `./dispatch.js`, `../errors.js`; `Binding` (type) from `../types/memory.js` -- never `../memory/**` (3.3 CONTRACT DECISION).

```ts
/** The resources of one bind(): one Binding per BindingDecl name; a uniform decl takes the ring's whole-buffer binding (the slot is chosen by the dynamic offset at dispatch). */
export type KernelBindings = Readonly<Record<string, Binding>>;
/** A kernel with its bind groups created (spec 5.1: cached per set of buffers and offsets). */
export interface BoundKernel {
    readonly kernel: Kernel;
    readonly bindGroups: readonly GPUBindGroup[];
    /** Group indices whose bind group takes a dynamic offset (the uniform groups), in group order. */
    readonly dynamicGroups: readonly number[];
}
/** A compiled pipeline plus the binding list of its spec (spec 5.1). */
export class Kernel {
    constructor(device: GPUDevice, spec: WgslModuleSpec, composed: ComposedModule, pipeline: GPUComputePipeline, layouts: readonly GPUBindGroupLayout[]);
    readonly spec: WgslModuleSpec;
    readonly pipeline: GPUComputePipeline;
    readonly layouts: readonly GPUBindGroupLayout[];
    readonly workgroupSize: number;
    readonly entryPoint: string;
    /** Creates (or reuses, keyed by every buffer identity + offset + size) the bind groups, each labelled `<spec.id>/<group>`; a missing or extra name is E_INVALID_ARGUMENT; two bindings of one call whose ranges intersect on one buffer while either slot is `storage` (read_write) is E_INVALID_ARGUMENT { argument: "aliasing" } (the host-side mirror of WebGPU's writable buffer-binding-aliasing rule, 3.10.1, so the failure is synchronous and labelled). CONTRACT DECISION: bind() is synchronous, so a createBindGroup validation error (a wrong-size uniform binding, a usage mismatch) is NOT thrown here -- it reaches the pending-error slot as E_VALIDATION { label: "<spec.id>/<group>" } and is thrown by the batch's readback (Dawn-node) or the next assertReady() (browser), spec 5.7. An empty bind group is created for every empty layout index so setBindGroup is called for 0..maxGroup. */
    bind(resources: KernelBindings): BoundKernel;
    /** setPipeline + setBindGroup for every group (dynamic offsets in dynamicGroups order) + dispatchWorkgroups(plan.x, plan.y, 1); a plan with x === 0 records nothing (spec 5.6). */
    dispatch(pass: GPUComputePassEncoder, bound: BoundKernel, plan: DispatchPlan, dynamicOffsets?: readonly number[] | undefined): void;
    /** Drops cached bind groups (a layout's buffers changed). */
    invalidate(): void;
}
```

Throws: `E_INVALID_ARGUMENT`, `E_VALIDATION`.

#### src/kernel/dispatch.ts -- P1-T3 (spec 5.2; pure)

```ts
/** A dispatch shape (spec 5.2). `stride` is the grid-stride step (null for plain 1D / 2D plans). */
export interface DispatchPlan {
    readonly x: number;
    readonly y: number;
    readonly z: 1;
    readonly items: number;
    readonly stride: number | null;
}
/** ceil(items / wg) groups as 1D up to MAX_WORKGROUPS_PER_DIM, else a 2D grid; items 0 -> { x: 0, y: 1 }; y above the limit -> E_TOO_LARGE. */
export function plan1d(items: number, wg: number, caps: PlanCaps): DispatchPlan;
/** The 2D form for a group count (x = MAX_WORKGROUPS_PER_DIM, y = ceil(groups / x)); exported for the indirect finalize kernel's host twin (P4). */
export function plan2d(groups: number, caps: PlanCaps): DispatchPlan;
/** P7 (spec 5.2): groups = min(groups, maxGroups ?? (caps.software ? 64 : 4096)) with the kernel looping by stride. P1-P3: throws E_UNSUPPORTED { feature: "planGridStride" } (lead f). */
export function planGridStride(items: number, wg: number, caps: PlanCaps, maxGroups?: number | undefined): DispatchPlan;
/** P4 (spec 5.4): the (x, y, 1) args a device-side finalize kernel writes for a count. P1-P3: throws E_UNSUPPORTED { feature: "planIndirect" } (lead f). */
export function planIndirect(count: number, wg: number, caps: PlanCaps): DispatchPlan;
/** The workgroup count of a plan (x * y). */
export function groupsOf(plan: DispatchPlan): number;
```

Throws: `E_TOO_LARGE { needed, limit, path: "dispatch", algorithm: null }`, `E_UNSUPPORTED`, `E_INVALID_ARGUMENT` (wg not a power of two, negative items).

#### src/kernel/uniform-ring.ts -- P2-T1 (spec 5.3)

```ts
/** One UNIFORM buffer with UNIFORM_SLOT_BYTES-stride slots for the per-iteration params of a batch (spec 5.3). */
export class UniformRing {
    constructor(device: GPUDevice, allocator: AllocationTracker, slots: number, label: string);
    /** Number of slots. */
    readonly slots: number;
    /** The whole-buffer binding a kernel binds once (size = the block's byteLength; the slot is the dynamic offset). */
    binding(block: UniformBlock): Binding;
    /** Byte offset of a slot. */
    offsetOf(slot: number): number;
    /** Reserves `count` contiguous slots for a batch, wrapping to 0 when the tail is too short; E_INVALID_ARGUMENT when count > slots. Returns the first slot. */
    reserve(count: number): number;
    /** Writes one block's values into a slot of the host shadow; `flush()` sends the dirty range with one writeBuffer. */
    write(slot: number, block: UniformBlock, values: UniformValues): void;
    /** queue.writeBuffer of the dirty slots (called by the batch before submit). */
    flush(): void;
    /** Destroys the buffer. */
    destroy(): void;
}
```

Throws: `E_INVALID_ARGUMENT`, `E_DISPOSED`.

#### src/kernel/batch.ts -- P2-T1 (spec 5.8, 5.7, 4.4)

Imports: `../errors.js`; TYPE-only imports of `Profiler` (`./profiler.js`), `Readback` / `StagingSlot` (`../memory/readback.js`), `AllocationTracker` (`../device/error-scope.js`), `Binding` (`../types/memory.js`) -- every one reached through `host`, so the batch <-> profiler pair is a type-only cycle (5.5 layers.test.ts).

```ts
/** What a CommandBatch needs of its owner (GpuContext satisfies it structurally; kernel/ never imports context.ts). */
export interface BatchHost {
    readonly device: GPUDevice;
    readonly readback: Readback;
    readonly profiler: Profiler | null;
    readonly allocator: AllocationTracker;
    assertReady(): void;
    takePendingError(): WebGpuGraphError | null;
    nextBatchId(): number;
}
/** A readback scheduled into the batch's staging slot: `offset` is the slot-relative byte offset the caller reads at. */
export interface ReadbackRequest {
    readonly src: GPUBuffer;
    readonly srcOffset: number;
    readonly byteLength: number;
    readonly offset: number;
}
/** What submit() returns (spec 5.8). `readback` resolves with the mapped bytes COPIED out (one ArrayBuffer holding every request at its offset) after allocator.check(); it rejects with E_VALIDATION { batchId } when the pending-error slot held an error right after submit, with E_DEVICE_LOST after loss, with E_ABORTED when discarded. */
export interface SubmittedBatch {
    readonly id: number;
    readonly generation: number;
    readonly readback: Promise<ArrayBuffer>;
    /** Marks the batch stale: its readback still awaits mapAsync (so the slot is returned) but resolves with an empty ArrayBuffer and the caller ignores it (spec 7.19 item 6). */
    discard(): void;
}
/** Records dispatches / copies into one encoder and submits once (spec 5.8). */
export class CommandBatch {
    constructor(host: BatchHost, label: string, generation?: number | undefined);
    readonly id: number;
    readonly generation: number;
    readonly label: string;
    /** Begins a compute pass (ending the previous one); with a profiler present the pass carries timestampWrites and `label` names it in Profiler.resolve(). */
    pass(label: string): GPUComputePassEncoder;
    /** Ends the open pass, if any. */
    endPass(): void;
    /** copyBufferToBuffer between two bindings (after endPass). */
    copy(src: Binding, dst: Binding, byteLength: number): void;
    /** Schedules a copy of `byteLength` bytes from `src` into the borrowed staging slot; the slot is borrowed at the first request and sized for the sum of requests. */
    readback(src: GPUBuffer, srcOffset: number, byteLength: number): ReadbackRequest;
    /** Ends passes, records the staging copies, submits, checks the pending-error slot (spec 5.7), returns the handle. A batch can be submitted once. */
    submit(): SubmittedBatch;
    /** Dispatch count recorded so far (tests bound it). */
    readonly dispatches: number;
}
```

Rules: the borrowed staging slot is returned when `readback` settles,
whether it resolved, was discarded, or rejected (spec 4.4); `submit()`
after device loss throws `E_DEVICE_LOST`; a batch never calls
`onSubmittedWorkDone`; `mapAsync` on the slot is the completion signal.

Throws: `E_DEVICE_LOST`, `E_DISPOSED`, `E_VALIDATION`, `E_INVALID_ARGUMENT` (submit twice, pass after submit).

#### src/kernel/profiler.ts -- P1-T1 (shell) / P2-T1 (spec 5.5)

Imports: `../constants.js`; P2-T1 adds `import type { CommandBatch, ReadbackRequest } from "./batch.js"` (TYPE only: batch.ts imports `Profiler` as a type for BatchHost, so the pair is a type-only cycle the 5.5 walk ignores).

P1 shell (P1-T1 writes exactly these members; `batch.ts` does not exist yet,
so nothing batch-typed is declared):

```ts
/** One resolved pass timing. */
export interface PassTiming {
    readonly label: string;
    readonly ns: number;
}
/** Timestamp profiling when "timestamp-query" was granted (spec 5.5). P1 ships the shell: enabled false, beginPass returns undefined, destroy is a no-op. */
export class Profiler {
    /** `slots` defaults to PROFILER_QUERY_SLOTS = 256 (spec 5.5); 2 query slots per pass. */
    constructor(device: GPUDevice, enabled: boolean, quantised: boolean, slots?: number | undefined);
    /** True when the feature was granted and a query set exists. */
    readonly enabled: boolean;
    /** True in browsers (100 us quantisation), false under Dawn-node (spec 2.6). */
    readonly quantised: boolean;
    /** The timestampWrites descriptor for a new pass, or undefined when disabled / out of slots. */
    beginPass(label: string): GPUComputePassTimestampWrites | undefined;
    /** Destroys the query set. */
    destroy(): void;
}
```

P2-T1 adds (the full profiler: the query set is created when enabled,
beginPass hands out slot pairs, the two members below appear):

```ts
    /** Records the resolve + copy into the batch's staging slot; returns the request's byte range, or null when nothing was written. */
    resolveInto(batch: CommandBatch): ReadbackRequest | null;
    /** Decodes the timings of a batch from its readback bytes. */
    timings(bytes: ArrayBuffer, request: ReadbackRequest): readonly PassTiming[];
```

Throws: none (the profiler never throws; a full query set drops timings).

### 3.10 src/kernels.ts -- P1-T4 / P2-T2 / P3-T2 (spec 3.5, 5.1, 5.3, 7.3, 7.4)

Imports: `./kernel/wgsl.js` (types), `./kernel/struct-block.js`, every
`./wgsl/*.wgsl.ts` body, `./constants.js`, `./errors.js`, `CoreBinding`
(type) from `./memory/residency.js`, `Binding` (type) from
`./types/memory.js`.

```ts
/** Every module id of P1-P3 (P4+ ids are appended, never renamed). */
export type KernelId =
    | "degree"
    | "reduce"
    | "fill"
    | "segmented-reduce"
    | "fa2-stats-finalize"
    | "fa2-attraction"
    | "fa2-repulsion-exact"
    | "fa2-speed-finalize"
    | "fa2-integrate"
    | "fa2-to-scene";
/** One registry entry: everything of a WgslModuleSpec except the per-variant overrides and snippets. */
export interface KernelEntry {
    readonly id: KernelId;
    readonly body: string;
    readonly entryPoint: string;
    readonly bindings: readonly BindingDecl[];
    readonly overrideDecls: readonly OverrideDecl[];
    readonly uniforms: readonly UniformBlock[];
    /** ["subgroups"] when the body calls a reduction helper (the twin axis), else []. */
    readonly needs: readonly ("subgroups")[];
    /** The snippet marker names the body carries (segmented-reduce: ["VALUE"]). */
    readonly snippetSlots: readonly string[];
    /** The phase the entry landed in (documentation and the compile-matrix filter). */
    readonly phase: "P1" | "P2" | "P3";
}
/** THE registry (spec 3.5): every entry, keyed by id. */
export const KERNELS: Readonly<Record<KernelId, KernelEntry>>;
/** A WgslModuleSpec for a variant: the entry plus the overrides / snippets given; unknown override names are rejected at compose time. A body override installed by setKernelBodyOverride is used instead of the entry's body. */
export function kernelSpec(id: KernelId, overrides?: Readonly<Record<string, number | boolean>> | undefined, snippets?: Readonly<Record<string, string>> | undefined): WgslModuleSpec;
/** @internal the sabotage seam (spec 11.9 item 1): replaces an entry's body for specs created afterwards (null restores). Tests use a FRESH context per mutation because the pipeline key does not include the body. */
export function setKernelBodyOverride(id: KernelId, body: string | null): void;
/** The group-0 graph bindings of a core with the dummy rules applied (spec 3.5, 4.1): colIdx <- rowPtr when null, weights <- colIdx ?? rowPtr when null, perm <- rowPtr when null. `weights` is the binding to use in the weights slot: omitted -> core.weights (degree, segmented-reduce); a layout passes its RESOLVED weights (ModelResources.weights, 3.13), null meaning "attract with 1.0" even on a weighted snapshot. */
export function graphBindings(core: CoreBinding, perm: Binding | null, weights?: Binding | null | undefined): Readonly<Record<"rowPtr" | "colIdx" | "weights" | "perm", Binding>>;
/** The override values graphBindings implies: USE_PERM = perm !== null, HAS_WEIGHTS = (weights === undefined ? core.weights : weights) !== null. */
export function graphOverrides(core: CoreBinding, perm: Binding | null, weights?: Binding | null | undefined): Readonly<{ USE_PERM: boolean; HAS_WEIGHTS: boolean }>;
// ---- the generated blocks (spec 5.3); field lists in 3.10.2
export const RANGE_PARAMS: UniformBlock;
export const REDUCE_PARAMS: UniformBlock;
export const FILL_PARAMS: UniformBlock;
export const FA2_PARAMS: UniformBlock;
export const FA2_STATE: UniformBlock;
export const FA2_TRACE: UniformBlock;
export const FA2_PARTIAL: UniformBlock;
```

CONTRACT DECISION (resolves an internal spec inconsistency; section 9 item
8): `USE_PERM` and the group-0 slot 3 `perm` are the degreeOrder ROW
permutation (spec 6 row 3, 7.3 and 7.5: `perm` with `rowPtr` as the dummy),
so `graphBindings` never touches `arcToEdge` / `edgeToArc`; the identity
rule of spec 4.1 ("`USE_PERM = false`, `colIdx` in the `arcToEdge` slot and
`rowPtr` in the `edgeToArc` slot") is the ARC permutation of the group-3
cold gathers of P7+ and is guarded by the reserved `USE_ARC_PERM` override
(4.1), not by `USE_PERM`. `CoreBinding` therefore carries `arcToEdge` /
`edgeToArc` bindings (null when not needed or identity) and no `perm`
member; the row permutation is a `ViewBinding` of `degreeOrder` (P4).
Consequence for the weights slot: `HAS_WEIGHTS` is NOT `core.hasWeights`
for a layout -- `weight: false | null` on a weighted snapshot attracts with
1.0 and `weight: "<edge column>"` binds the expanded column, so layouts
pass their resolved weights explicitly (the third argument above), and
only `degree` / `segmented-reduce` take the core's.

#### 3.10.1 Binding tables (the `bindings` of each entry; group 0 = graph, 1 = state, 2 = params, 3 = cold)

| id | group.binding name : kind : wgslType | overrideDecls (type = default) | uniforms | needs | phase |
| --- | --- | --- | --- | --- | --- |
| `degree` | 0.0 rowPtr : storage-ro : array<u32>; 0.1 colIdx : storage-ro : array<u32>; 0.2 weights : storage-ro : array<f32>; 0.3 perm : storage-ro : array<u32>; 1.0 out : storage : array<u32>; 2.0 P : uniform : RangeParams | none (USE_PERM, HAS_WEIGHTS standard) | RANGE_PARAMS | [] | P1 |
| `reduce` | 1.0 src : storage-ro : array<u32>; 1.1 out : storage : array<u32>; 2.0 P : uniform : ReduceParams | OP u32 = 0; DTYPE u32 = 0; FINAL bool = false | REDUCE_PARAMS | ["subgroups"] | P1 |
| `fill` | 1.0 dst : storage : array<u32>; 2.0 P : uniform : FillParams | none | FILL_PARAMS | [] | P1 |
| `segmented-reduce` | 0.0 rowPtr; 0.1 colIdx; 0.2 weights; 0.3 perm (as degree); 1.0 out : storage : array<f32>; 2.0 P : uniform : RangeParams | OP u32 = 0; TIER u32 = 0 | RANGE_PARAMS | [] | P2 |
| `fa2-stats-finalize` (K1) | 1.0 partials : storage-ro : array<Fa2Partial>; 1.1 S : storage : Fa2State; 1.2 T : storage : array<Fa2Trace>; 2.0 P : uniform : Fa2Params | none | FA2_PARAMS, FA2_STATE, FA2_TRACE, FA2_PARTIAL | ["subgroups"] | P3 |
| `fa2-attraction` (K2) | 0.0 rowPtr; 0.1 colIdx; 0.2 weights; 0.3 perm; 1.0 pos : storage-ro : array<vec4f>; 1.1 force : storage : array<f32>; 2.0 P : uniform : Fa2Params | LINLOG bool = false; DISTRIBUTED bool = false; TIER u32 = 0 | FA2_PARAMS | [] | P3 |
| `fa2-repulsion-exact` (K3) | 1.0 pos : storage-ro : array<vec4f>; 1.1 S : storage : Fa2State; 1.2 force : storage : array<f32>; 1.3 oldForce : storage-ro : array<f32>; 1.4 fixedMask : storage-ro : array<u32>; 1.5 partials : storage : array<Fa2Partial>; 2.0 P : uniform : Fa2Params | SWING_MODE u32 = 0; STRONG_GRAVITY bool = false; GRAVITY_CENTER u32 = 0 | FA2_PARAMS, FA2_STATE, FA2_PARTIAL | ["subgroups"] | P1 |
| `fa2-speed-finalize` (K4) | 1.0 partials : storage-ro : array<Fa2Partial>; 1.1 S : storage : Fa2State; 1.2 T : storage : array<Fa2Trace>; 2.0 P : uniform : Fa2Params | SWING_MODE u32 = 0 | FA2_PARAMS, FA2_STATE, FA2_TRACE, FA2_PARTIAL | ["subgroups"] | P1 |
| `fa2-integrate` (K5) | 1.0 force : storage-ro : array<f32>; 1.1 oldForce : storage : array<f32>; 1.2 fixedMask : storage-ro : array<u32>; 1.3 S : storage : Fa2State; 1.4 pos : storage : array<vec4f>; 1.5 partials : storage : array<Fa2Partial>; 2.0 P : uniform : Fa2Params | SWING_MODE u32 = 0 | FA2_PARAMS, FA2_STATE, FA2_PARTIAL | ["subgroups"] | P3 |
| `fa2-to-scene` | 1.0 pos : storage-ro : array<vec4f>; 1.1 scene : storage : array<f32>; 2.0 P : uniform : Fa2Params | none | FA2_PARAMS | [] | P3 |

Storage-buffer counts per stage: degree 5, reduce 2, fill 1,
segmented-reduce 5, K1 3, K2 6, K3 6, K4 3, K5 6, toScene 2 -- all under
the core default of 8 (spec 3.5; the bind-group-budget test of 11.3).
`S` is `storage` (read_write) in every kernel that binds it (spec 5.1's
sharing rule); `partials` is read-only where only read (K1, K4) and
read_write where written (K3, K5); `oldForce` is read-only in K3 (it only
calls `load_old`) and read_write in K5 (the only writer). Dummy bindings:
`degree` / `segmented-reduce` / K2 follow `graphBindings` (read-only slots
only). CONTRACT DECISION (no writable alias): a buffer range is NEVER bound
to a writable slot and to any other slot of the same dispatch -- WebGPU's
"validate encoder bind groups" step rejects a dispatch whose bind groups
hold two buffer-binding-aliasing ranges when either is `storage`
(read_write), so binding `force` in the `oldForce` slot next to the
writable `force` slot would fail every K3 / K5 dispatch of `compat:
"networkx"` (verified on Dawn 0.4.0 / NVIDIA: "Writable storage buffer
binding aliasing found"; Chromium enforces the same step). Therefore
`ForceAtlas2Model.buffers()` allocates `oldForce` (12n bytes) in BOTH swing
modes and K5 simply skips `store_old` in mode 1; the read-only dummies of
group 0 (`colIdx` in `weights`, `rowPtr` in `perm`) are unaffected because
every group-0 slot is read-only. The `state` header and the trace region
are two bindings of ONE buffer with disjoint ranges (`[0, 256)` and `[256,
...)`), which is not aliasing. `Kernel.bind()` enforces the rule host-side
(3.9).

#### 3.10.2 The generated blocks (field order = byte order; offsets in bytes)

`RangeParams` (uniform, 32 B): `start u32 @0`, `end u32 @4`, `arcBase u32
@8`, `arcEnd u32 @12`, `accumulate u32 @16`, `n u32 @20`, `pad0 u32 @24`,
`pad1 u32 @28`. Rows `[start, end)` of the dispatch; arcs of the bound
window `[arcBase, arcEnd)` (`arcBase = 0`, `arcEnd = arcCount` when not
windowed); `accumulate = 1` combines into `out[i]` instead of overwriting
(the P4 windowed loop).

`ReduceParams` (uniform, 16 B): `count u32 @0`, `outOffset u32 @4`, `level
u32 @8`, `pad0 u32 @12`.

`FillParams` (uniform, 16 B): `count u32 @0`, `value u32 @4`, `mode u32 @8`
(0 = constant `value`, 1 = iota `i + value`), `pad0 u32 @12`.

`Fa2Params` (uniform, 96 B; spec 7.3): `n u32 @0`, `dim u32 @4`, `flags u32
@8` (bit 0 = FA2_FLAG_FIRST: the first iteration after load(); K1 keeps the
host-written state and folds nothing), `tierStart u32 @12`, `tierEnd u32
@16`, `iterationIndex u32 @20` (the trace slot of this iteration inside the
batch), `seed u32 @24`, `nearMax u32 @28`, `scalingRatio f32 @32`, `gravity
f32 @36`, `jitterTolerance f32 @40`, `scale f32 @44`, `center vec4f @48`
(xyz, w 0), `settleThreshold f32 @64`, `extentFactor f32 @68`, `gridMax u32
@72`, `levels u32 @76`, `pad vec4f @80` (reserved for the P4 GridSpec).

`Fa2State` (storage, padded to STATE_HEADER_BYTES = 256; spec 7.3): `speed
f32 @0`, `speedEfficiency f32 @4`, `swing f32 @8`, `traction f32 @12`,
`centroid vec4f @16`, `rmsRadius f32 @32`, `radius f32 @36`,
`meanDisplacement f32 @40`, `iteration u32 @44`, `min vec4f @48`, `max vec4f
@64`, `gridMin vec4f @80` (xyz, w = cellSize; P4), `eps f32 @96` (P4),
`settledCount u32 @100`, `outsideGrid u32 @104` (P4), `maxCellOccupancy u32
@108` (P4), then reserved `vec4f` fields `reserved0` .. `reserved8` @112 ..
@240.

`Fa2Trace` (storage record, 32 B; spec 7.3): `swing f32 @0`, `traction f32
@4`, `speed f32 @8`, `speedEfficiency f32 @12` (written by K4),
`meanDisplacement f32 @16`, `settledCount u32 @20`, `iteration u32 @24`
(written by K1), `pad0 u32 @28`. The trace region is `array<Fa2Trace>` at
byte offset STATE_HEADER_BYTES of the state buffer, MAX_ITERATIONS_PER_STEP
records long, bound as a second binding of the same buffer (3.13).

`Fa2Partial` (storage record, 64 B; spec 7.3): `sum vec4f @0` (xyz = sum of
positions, w = sum of |p - centroid|^2), `min vec4f @16` (xyz; w unused,
0), `max vec4f @32` (xyz; w = max of |p - centroid|^2, the exact
`layoutRadius` source) (written by K5), `swingTraction vec2f @48` (B,
written by K3's epilogue),
`dispFree vec2f @56` (C: x = sum |dp| over free rows, y = free count as an
exactly representable f32 <= 256, written by K5).

Throws (kernels.ts): `E_SHADER_COMPILE` (via composeWgsl), `E_INVALID_ARGUMENT` (unknown KernelId).

### 3.11 src/wgsl/*.wgsl.ts and src/primitives/*.ts

#### src/wgsl/*.wgsl.ts (spec 3.5, D9)

Each file exports ONE constant, the kernel BODY (no `@group(`, no `override
`; function declarations and `var<workgroup>` declarations allowed; exactly
one `@compute` entry point). Export name and entry point:

| file | export | entry point | phase / task |
| --- | --- | --- | --- |
| `degree.wgsl.ts` | `degreeWgsl` | `degree` | P1-T4 |
| `reduce.wgsl.ts` | `reduceWgsl` | `reduce` | P1-T4 |
| `fill.wgsl.ts` | `fillWgsl` | `fill` | P1-T4 |
| `fa2-repulsion-exact.wgsl.ts` | `fa2RepulsionExactWgsl` | `repulsion` | P1-T4 |
| `fa2-speed-finalize.wgsl.ts` | `fa2SpeedFinalizeWgsl` | `speed_finalize` | P1-T4 |
| `segmented-reduce.wgsl.ts` | `segmentedReduceWgsl` | `segmented_reduce` | P2-T2 |
| `fa2-stats-finalize.wgsl.ts` | `fa2StatsFinalizeWgsl` | `stats_finalize` | P3-T2 |
| `fa2-attraction.wgsl.ts` | `fa2AttractionWgsl` | `attraction` | P3-T2 |
| `fa2-integrate.wgsl.ts` | `fa2IntegrateWgsl` | `integrate` | P3-T2 |
| `fa2-to-scene.wgsl.ts` | `fa2ToSceneWgsl` | `to_scene` | P3-T2 |

The bodies are normative in section 4.5. Prelude helper functions a body may
call: `linear_id`, `group_id`, `lowbias32`, `mask_bit`, `unpack_u8`,
`pair_hash`, `hash_unit`, `hash_dir`, `kick_dir`, and (with `needs:
["subgroups"]`) `wg_reduce_f32`, `wg_reduce_u32`, `wg_reduce_vec4`.

#### src/primitives/reduce.ts -- P1-T5 (spec 6 row 1)

Imports: `../kernels.js`, `../kernel/*.js`, `../memory/*.js`, `../errors.js`, `../constants.js`.

```ts
/** The reduction operator and element type. */
export type ReduceOp = "sum" | "min" | "max";
export type ReduceDtype = "f32" | "u32" | "vec4f";
/** What reduce() needs of its caller: a pass to record into, scratch, the ring and the cache (P1 has no CommandBatch yet; P2's batch supplies the same record). */
export interface ReduceScope {
    readonly device: GPUDevice;
    readonly caps: PlanCaps;
    readonly pipelines: PipelineCache;
    readonly pool: BufferPool;
    readonly workgroupSize: number;
    /** Acquires scratch released by the caller's scope (a Lease from P2; P1 releases in a finally). */
    scratch(byteLength: number, label: string): GPUBuffer;
    /** The uniform-slot writer: returns the binding and dynamic offset for a params record. */
    params(block: UniformBlock, values: UniformValues): { readonly binding: Binding; readonly offset: number };
}
/** Prepares the reduce pipelines of a scope (compiles once) so record() is synchronous. */
export function prepareReduce(scope: ReduceScope, op: ReduceOp, dtype: ReduceDtype): Promise<ReducePlanner>;
/** A prepared reduce: records the 2-3 dispatches of spec 6 row 1 into a pass. */
export interface ReducePlanner {
    readonly op: ReduceOp;
    readonly dtype: ReduceDtype;
    /** Records: level 1 over `count` elements of `src` into partials; a third level when groups > MAX_WORKGROUPS_PER_DIM; the FINAL one-workgroup level writing one element (4 or 16 bytes) at out[outOffset] (element index). Deterministic order. count 0 writes the identity element. */
    record(pass: GPUComputePassEncoder, src: Binding, count: number, out: Binding, outOffset: number): void;
    /** Dispatches the last record() issued (tests bound it: 2 or 3). */
    readonly lastDispatches: number;
}
```

Throws: `E_TOO_LARGE` (a partials level that still exceeds the 2D limit -- unreachable below 1.1e12 items), `E_INVALID_ARGUMENT` (src too small for count x element size).

#### src/primitives/segmented-reduce.ts -- P2-T2 (spec 6 row 3)

```ts
/** The degree tiers of degreeOrder(): the permutation binding and the CPU-side segmentOffsets [0, hiEnd, midEnd, lowEnd, n]. */
export interface DegreeTiers {
    readonly perm: Binding;
    readonly segmentOffsets: readonly [number, number, number, number, number];
}
/** Options of segmentedReduce. `valueSnippet` is the Gunrock-style functor: WGSL statements assigning `v` from (row, arc, nbr, weight) (4.5; `nbr` because `target` is a WGSL reserved word). */
export interface SegmentedReduceOptions {
    readonly op: ReduceOp;
    readonly valueSnippet: string;
    readonly tiers: DegreeTiers | null;
    readonly accumulate?: boolean | undefined;
}
/** Prepares the thread-per-row pipeline for a snapshot's dummy pattern (USE_PERM, HAS_WEIGHTS) and snippet. */
export function prepareSegmentedReduce(scope: ReduceScope, core: CoreBinding, options: SegmentedReduceOptions): Promise<SegmentedReducePlanner>;
/** A prepared segmented reduce (P2-P3: the thread-per-row tier only; `tiers !== null` -> E_UNSUPPORTED { feature: "segmentedReduce.tiers" } until P4). */
export interface SegmentedReducePlanner {
    /** Records one dispatch over rows [0, n) (tiers null) writing out[i] (f32) per row; a row with no arcs gets the identity element. */
    record(pass: GPUComputePassEncoder, core: CoreBinding, out: Binding): void;
}
```

Throws: `E_UNSUPPORTED` (tiers, windowed cores until P4), `E_SHADER_COMPILE` (a snippet without an assignment to `v`, or naming an identifier outside `(row, arc, nbr, weight, v)` -- checked textually at compose), `E_INVALID_ARGUMENT`.

### 3.12 src/algorithms/degree.ts -- P1-T5 (spec 3.3, 11.5)

Imports: `../context.js`, `../kernels.js`, `../kernel/*.js`, `../memory/*.js`, `../errors.js`, `../types/run.js`, graph-format `GraphSnapshot`, `U32`.

```ts
/** The walking-skeleton kernel, kept public as a diagnostic (spec 3.3): out-degree per node through the row-walking gather with the USE_PERM dummy pattern; equals snapshot.outDegree(). */
export function degree(ctx: GpuContext, s: GraphSnapshot, options?: GpuRunOptions | undefined): Promise<U32>;
```

Contract: `ctx.assertReady()`; `nodeCount === 0` -> `new Uint32Array(0)` (or `dest`) with no GPU work (spec 5.6); `signal.aborted` before any work -> `E_ABORTED`; `dest` given -> must be a `Uint32Array` of length n over an `ArrayBuffer` (else `E_INVALID_ARGUMENT`); `residency.core(s)` (arena / perArray; a windowed plan -> `E_TOO_LARGE { path: "windowed", algorithm: "degree" }` until P4 executes windows); `arcCount === 0` -> the dispatch is skipped and the result is zeros (nothing but rowPtr bound); otherwise ONE `degree` dispatch over rows `[0, n)` with `accumulate = 0` (every row is written, so no `fill` precedes it), `plan1d(n)`; readback into the result; `onProgress(1, 1)`; scratch returned in a `finally`. Two runs are bitwise identical (spec 11.9 item 4).

Throws: `E_DEVICE_LOST`, `E_DISPOSED`, `E_RELEASED`, `E_ABORTED`, `E_INVALID_ARGUMENT`, `E_TOO_LARGE`, `E_OUT_OF_MEMORY`, `E_VALIDATION`, `E_SHADER_COMPILE`, `E_SNAPSHOT`.

### 3.13 src/layouts/*.ts (spec 7.2-7.6, 7.9-7.19; D8, D23, D25, D28)

#### src/layouts/seed.ts -- P3-T1 (spec 7.2, 9.3, 7.19)

```ts
/** The CPU port's LCG constants (layout/src/utils/random.ts): m = 2^35 - 31, a = 185852, c = 1. */
export const LCG_M = 34359738337;
export const LCG_A = 185852;
export const LCG_C = 1;
/** The port's RandomNumberGenerator, bit for bit: `seed || Math.floor(Math.random() * 1000000)` (seed 0 / null = unseeded, the quirk preserved), state = seed % m, next = (a * state + c) % m, value = state / m. */
export class Lcg {
    constructor(seed: number | null);
    /** The seed actually used (a random one when unseeded). */
    readonly seed: number;
    /** Next value in [0, 1). */
    next(): number;
}
/** Seeds the unseeded rows of the owner's stride-3 SCENE array in index order (spec 9.3 seedPositions, 7.14 `pos`, 7.19 topology change): a row is unseeded when any of its `dim` components is NaN; every unseeded component draws one LCG value; when EVERY row is unseeded the draw is uniform in [-1, 1) layout units per axis (`range: "fa2"`; `"fr"` is [0, 1)), otherwise uniform inside the [min, max] box of the finite rows per axis (an axis with no finite value falls back to [-1, 1]); the value is written as `v * scale + center[axis]`; in 2D the third component is written as center[2]. Finite components are never changed. */
export function seedPositions(s: GraphSnapshot, positions: F32, seed: number | null, dim: 2 | 3, scale: number, center: ArrayLike<number> | null, range: "fa2" | "fr"): void;
```

Throws: `E_INVALID_ARGUMENT` (positions.length !== 3 * s.nodeCount, scale <= 0, a non-finite center).

#### src/layouts/inputs.ts -- P3-T1 (spec 7.14, D28)

Imports: graph-format `GraphSnapshot`, `Column`, `F32`, `NodeId`, `NumericVector`, `expandEdges`.

```ts
/** nodeMass resolution by ROLE (spec 7.14): null -> the role-"mass" node column when present (any numeric dtype through gpuView, converted to a fresh F32 when not f32), else outDegree()[i] + 1; an F32 of length n -> as is; a column NAME -> nodes.get(name) (numeric); a Record -> E_UNSUPPORTED with the spec's message. Always returns n values. */
export function resolveNodeMass(s: GraphSnapshot, spec: F32 | string | Readonly<Record<NodeId, number>> | null | undefined): F32;
/** What resolveWeights found. */
export interface ResolvedWeights {
    readonly data: F32 | null;
    readonly source: "arcs" | "column" | "none";
    readonly column: Column | null;
}
/** Weight resolution (spec 7.5, 7.14): true -> { data: s.weights, source: "arcs" } (data null when unweighted -> ones); a string -> a name `s.edges` does not hold is E_INVALID_ARGUMENT (spec 7.14's nodeMass rule applied to weight too), else `column = s.edges.get(name)`, then by `column.dtype`: f32 / f64 / u32 / i32 -> expandEdges(s, s.edges.gpuView(name)) (f64 arrives as gpuView's cached f32 copy; u32 / i32 are converted to a fresh F32; `Column.data` is NOT used because it is U8 for u8 / bool and absent for string / list / json), string / list / json -> the gpuView E_GPU_INELIGIBLE pass-through, u8 / bool / dict -> E_INVALID_ARGUMENT (gpuView returns packed words, not per-edge values); source "column", `column` kept for the version check; false / null / undefined -> none. */
export function resolveWeights(s: GraphSnapshot, spec: boolean | string | null | undefined): ResolvedWeights;
```

Throws: `E_UNSUPPORTED { option: "nodeMass", hint }` (Record form), `E_INVALID_ARGUMENT` (length mismatch, missing column, non-numeric column, NaN / non-positive mass). Pass-through: `E_GPU_INELIGIBLE`, `E_COLUMN_LENGTH`.

#### src/layouts/repulsion-exact.ts -- P1-T6 (K3 + K4 stage) / P3-T2 (wired into the model)

```ts
/** The buffers the exact-tier repulsion stage binds (all in layout units; spec 7.3). */
export interface RepulsionExactResources {
    readonly pos: Binding;
    readonly state: Binding;
    readonly trace: Binding;
    readonly force: Binding;
    readonly oldForce: Binding;
    readonly fixedMask: Binding;
    readonly partials: Binding;
    readonly params: Binding;
}
/** The overrides K3 / K4 compile with. */
export interface RepulsionExactOverrides {
    readonly SWING_MODE: 0 | 1;
    readonly STRONG_GRAVITY: boolean;
    readonly GRAVITY_CENTER: 0 | 1;
}
/** K3 (tiled all-pairs repulsion + gravity + the swing / traction epilogue) followed by K4 (the one-workgroup speed finalize) (spec 7.6, 7.10). */
export class RepulsionExact {
    /** Compiles both kernels through the cache (the twin is selected by caps.features). */
    static create(pipelines: PipelineCache, caps: PlanCaps, overrides: RepulsionExactOverrides): Promise<RepulsionExact>;
    /** The two specs (for warm() and the compile matrix). */
    static specs(overrides: RepulsionExactOverrides): readonly [WgslModuleSpec, WgslModuleSpec];
    readonly overrides: RepulsionExactOverrides;
    /** Creates the bind groups once per load(). */
    bind(resources: RepulsionExactResources): void;
    /** Records K3 (plan1d(n)) then K4 (1 workgroup) with the params slot's dynamic offset. */
    record(pass: GPUComputePassEncoder, n: number, paramsOffset: number): void;
    /** Records K3 only (the inspect() stage split, spec 11.9 item 2). @internal */
    recordRepulsion(pass: GPUComputePassEncoder, n: number, paramsOffset: number): void;
    /** Records K4 only. @internal */
    recordSpeedFinalize(pass: GPUComputePassEncoder, paramsOffset: number): void;
}
```

Throws: `E_SHADER_COMPILE`, `E_VALIDATION`, `E_TOO_LARGE` (n above MAX_1D_ITEMS x 65535, unreachable), `E_NOT_LOADED` (record before bind).

#### src/layouts/force-simulation.ts -- P3-T1 (spec 7.19, 7.12, 7.17, 7.18, D6, D8)

Imports: `../context.js`, `../kernel/*.js`, `../memory/*.js`, `../kernels.js`, `./seed.js`, `ResolvedWeights` (type) from `./inputs.js`, `../errors.js`, `../constants.js`, `../types/*.js`, graph-format `GraphSnapshot`, `F32`, `NodeMask`, `makeMask`, `maskTest`.

```ts
/** A model-owned buffer beyond the shared set (spec 7.19: oldForce, velocity, the grid tier's). */
export interface BufferSpec {
    readonly name: string;
    readonly byteLength: number;
    readonly usage: number;
    readonly zero: boolean;
}
/** Host writes into the state header collected between submits (spec 7.19 onLoad / onReheat / onSetParams). */
export interface StateWriter {
    /** Queues a field write (flushed by one writeBuffer before the next submit; also applied to the host shadow immediately). */
    set(field: string, value: number | readonly number[]): void;
    /** The host shadow of a field (last known value). */
    get(field: string): number | readonly number[];
}
/** What a model gets at bind time (after load() / a resize): the graph, the shared and model buffers, the ring, the cache. */
export interface ModelResources {
    readonly device: GPUDevice;
    readonly caps: PlanCaps;
    readonly pipelines: PipelineCache;
    readonly core: CoreBinding;
    readonly perm: Binding | null;
    /** The RESOLVED weights binding (model.inputs(): source "arcs" -> core.weights (null on an unweighted snapshot), "column" -> the registered ArrayBinding's binding, "none" -> null); group 0 is built as graphBindings(core, perm, weights) (3.10). */
    readonly weights: Binding | null;
    readonly n: number;
    readonly dim: 2 | 3;
    readonly tier: "exact" | "grid";
    readonly ring: UniformRing;
    /** A shared or model-owned buffer by name: "positions", "scenePositions", "fixed", "partials", "state", "trace", plus every BufferSpec name. */
    buffer(name: string): Binding;
}
/** The per-load inputs a model resolves from the snapshot and its options (mass into the `.w` lane; weights into the group-0 slot). */
export interface ModelInputs {
    readonly mass: F32;
    readonly weights: ResolvedWeights;
}
/** Spec 7.19 ForceModel (kind, buffers, overrides, paramsFor, recordIteration, onLoad, onReheat, onSetParams, readStats) with SEVEN additions (CONTRACT DECISION, each with its reason): `stages` (the kernel stage names for debugRunStages / inspect, spec 11.9 item 2); `params` (the model's uniform block, which must declare the shared field names n, dim, flags, iterationIndex, seed, scale, center, settleThreshold, so the simulation can write them); `state` and `trace` (the model's generated Fa2State / Fa2Trace blocks, so the simulation allocates, initialises and decodes the state buffer through the model's own layouts, D20); `specs` (the module specs of an override set, for warm() and the compile matrix); `bind` (the model compiles and binds after load(), when the buffers exist); `inputs` (mass and weight resolution need the model's option names -- nodeMass / weight are ForceAtlas2Options, not CommonLayoutOptions -- so the simulation calls it at load() and derives USE_PERM / HAS_WEIGHTS from ModelResources, 3.10); and the optional `upTo` argument of `recordIteration` (truncates the sequence after a stage, for debugRunStages / inspect). */
export interface ForceModel<Options, Stats extends LayoutStatsBase> {
    readonly kind: "forceatlas2" | "fruchtermanReingold" | "springElectrical";
    readonly stages: readonly string[];
    readonly params: UniformBlock;
    readonly state: UniformBlock;
    readonly trace: UniformBlock;
    buffers(n: number, dim: 2 | 3): readonly BufferSpec[];
    /** Called at load() before the upload: FA2 = { mass: resolveNodeMass(s, nodeMass), weights: resolveWeights(s, weight) }. */
    inputs(s: GraphSnapshot, options: Options): ModelInputs;
    /** The model's OWN override set; the simulation merges USE_PERM / HAS_WEIGHTS from graphOverrides(core, perm, resources.weights) before specs() / bind(). */
    overrides(options: Options): Readonly<Record<string, number | boolean>>;
    specs(overrides: Readonly<Record<string, number | boolean>>, subgroups: boolean): readonly WgslModuleSpec[];
    bind(resources: ModelResources, overrides: Readonly<Record<string, number | boolean>>): Promise<void>;
    paramsFor(iteration: number, options: Options): UniformValues;
    recordIteration(batch: CommandBatch, slot: number, tier: "exact" | "grid", upTo?: string | undefined): void;
    onLoad(state: StateWriter): void;
    onReheat(state: StateWriter): void;
    onSetParams(patch: Partial<Options>, state: StateWriter): void;
    readStats(state: DataView, trace: DataView): Stats;
}
/** The shared layout state machine (spec 7.19): buffers, in-flight batches, readback, settle window, fixed mask, setPosition overrides, trace, batch driver; consumes a ForceModel by composition. */
export class ForceSimulation<Options extends CommonLayoutOptions & SimulationOptions, Stats extends LayoutStatsBase> implements GpuLayoutSimulation<Options, Stats> {
    constructor(ctx: GpuContext, model: ForceModel<Options, Stats>, options: Options, tuning: ResolvedLayoutTuning, resolve: (patch: Partial<Options>, current: Options) => Options);
    readonly ctx: GpuContext;
    readonly model: ForceModel<Options, Stats>;
    /** "created" | "loaded" | "disposed". */
    readonly state: "created" | "loaded" | "disposed";
    /** The current options record (defaults applied). */
    readonly options: Options;
    readonly tuning: ResolvedLayoutTuning;
    readonly tier: "exact" | "grid";
    /** The generation counter bumped by every load() (stale readbacks are discarded). @internal */
    readonly generation: number;
    readonly nodeCount: number;
    readonly dim: 2 | 3;
    load(snapshot: GraphSnapshot, positions: F32): void;
    step(iterations?: number | undefined): Promise<void>;
    readonly settled: boolean;
    readonly inFlight: number;
    readonly iterationsDone: number;
    readonly stats: Stats;
    setFixed(mask: NodeMask): void;
    setPosition(index: number, x: number, y: number, z: number): void;
    reheat(): void;
    setParams(patch: Partial<Options>): void;
    flush(): Promise<void>;
    run(options?: RunOptions | undefined): Promise<Stats>;
    dispose(): void;
    /** Present when ctx.debug.inspect is true: reads back any named buffer (a shared name, a BufferSpec name, "force", "state", "trace") after the last submitted kernel; resolves a Float32Array except for "fixed", "trace" (raw words) which resolve Uint32Array. */
    inspect?: (name: string) => Promise<Float32Array | Uint32Array>;
    /** Present when ctx.debug.inspect is true: records ONE iteration truncated after stage `upTo` (a model stage name), submits, awaits it (no readback into positions, no stats update). @internal */
    debugRunStages?: (upTo: string) => Promise<void>;
    /** The override list (spec 7.12): rows whose readback is skipped while a batch older than the write is in flight. @internal */
    readonly overrides: ReadonlyMap<number, number>;
    /** The last submitted batch id. @internal */
    readonly lastSubmittedBatchId: number;
    /** Number of step() calls that returned an existing batch's promise instead of submitting (spec 7.19 item 3; read by test/helpers/frame-loop.ts). @internal */
    readonly coalesced: number;
    /** The uniform ring, sized (maxInFlight + 1) x MAX_ITERATIONS_PER_STEP slots at construction (CONTRACT DECISION: reserve() wraps to 0 when the tail is short, so maxInFlight batches of up to MAX_ITERATIONS_PER_STEP slots plus one wasted tail always fit without a slot being rewritten while a submitted batch still reads it; hence setParams cannot change maxInFlight, 3.13 forceatlas2.ts). @internal */
    readonly ring: UniformRing;
}
```

Method-level contracts (spec 7.19, 7.12, 7.17, 7.18):

- `load(snapshot, positions)`: `ctx.assertReady()`; `snapshot.directed` ->
  `E_SNAPSHOT { reason: "directed" }` ("pass toUndirected().snapshot",
  Q-11); `positions.length !== 3 * nodeCount` or a SharedArrayBuffer ->
  `E_INVALID_ARGUMENT`; `nodeCount > MAX_1D_ITEMS` -> `E_TOO_LARGE { path:
  "partials" }` (the third partials level is P4); `generation++`; in-flight
  batches are discarded; the core is uploaded through `ctx.residency.core`
  (the previous core's bindings are dropped, never released -- the owner
  releases); `model.inputs(snapshot, options)` resolves mass and weights (a
  `source: "column"` array is registered with `residency.array(expanded,
  "weights", snapshot)` and re-expanded only when `column.version` changed;
  `ModelResources.weights` and the `USE_PERM` / `HAS_WEIGHTS` overrides
  follow 3.10); when `n` or the snapshot serial changed the fixed words and the
  override list are cleared and every buffer is (re)allocated (`positions`
  16n, `scenePositions` 12n, `fixed` 4 ceil(n/32), `partials` 64 x
  ceil(n/WG), `state` STATE_HEADER_BYTES + MAX_ITERATIONS_PER_STEP x
  TRACE_RECORD_BYTES, plus `model.buffers(n, dim)`), else they are kept;
  `seedPositions` seeds the unseeded rows in the array (scene units); the
  array is repacked into `vec4f` (xyz = (scene - center) / scale, w = mass
  from `model.inputs(snapshot, options).mass`; 2D uploads z = 0); the
  initial centroid, min, max, rmsRadius and radius (= max |p - centroid|,
  spec 3.3) are computed on the CPU in f64 and written into `state` with
  `iteration = 0`, `settledCount = 0`, `meanDisplacement = 0`;
  `model.onLoad(writer)`; `model.bind(resources, overrides)` and
  `pipelines.warm(model.specs(...))` are started (a promise the first
  `step()` awaits); `iterationsDone = 0`; state becomes "loaded"; the first
  iteration after load carries `FA2_FLAG_FIRST`.
- `step(k = options.iterationsPerStep)`: state "created" -> reject
  `E_NOT_LOADED`; "disposed" -> `E_DISPOSED`; the residency tombstoned the
  serial -> `E_RELEASED`; `k < 1` or `k > MAX_ITERATIONS_PER_STEP` ->
  `E_INVALID_ARGUMENT`; `settled` -> resolve at once; `inFlight >=
  maxInFlight` -> return the OLDEST pending batch's promise (coalesce;
  `coalesced++`, the @internal counter declared above); await
  the bind / warm promise and `allocator.check()` (an OOM here destroys
  every simulation buffer, returns the state to "created" and rejects
  `E_OUT_OF_MEMORY`); flush the dirty mask (`writeBuffer`) and queued state
  writes; reserve `k` ring slots and write `paramsFor(i)` merged with the
  shared fields (`iterationIndex = i`, `flags`); record `k` iterations
  (`model.recordIteration(batch, slot + i, tier)`), `toScene` and the
  readbacks of `scenePositions` and `state` (header + k trace records) into
  one `CommandBatch` carrying the current generation; submit; `inFlight++`;
  `iterationsSubmitted += k`; the returned promise awaits the batch's
  readback, then: stale generation -> discard; else copy the scene bytes
  into the owner's array row by row skipping rows in the override list
  whose `afterBatch >= batch.id`, clear overrides with `afterBatch <
  batch.id`, decode `stats` through `model.readStats`, `iterationsDone +=
  k`, `settled = iterationsDone >= maxIter || settledCount >= settleWindow`,
  `inFlight--`. Errors (`E_VALIDATION`, `E_DEVICE_LOST`) reject the promise
  and leave `inFlight` consistent.
- `setFixed(mask)`: `mask.length < ceil(n / 32)` -> `E_INVALID_ARGUMENT`;
  copies the words, marks the buffer dirty, `reheat()` iff some bit went 1
  -> 0 (an unpin); state "created" -> `E_NOT_LOADED`.
- `setPosition(i, x, y, z)`: `i >= n` -> `E_INVALID_ARGUMENT`; writes the
  three floats into the owner's array; converts to layout units (z forced to
  0 in 2D) and `queue.writeBuffer(positions, 16 * i, 12 bytes)`; records `{
  i -> lastSubmittedBatchId }` in the override list; `reheat()`.
- `reheat()`: `iterationsDone = 0`, `settledCount = 0` (a state write),
  `model.onReheat(writer)`; nothing else (D8).
- `setParams(patch)`: `dim` in the patch (differing from the current) ->
  `E_INVALID_ARGUMENT`; `nodeSize` non-null -> `E_UNSUPPORTED`; the record
  is replaced through `resolve(patch, current)`; a change of a force LAW
  (`linlog`, `strongGravity`, `distributedAction`, or the tuning's `compat`
  is not patchable here) recompiles (new overrides -> `model.bind` again)
  and `model.onSetParams(patch, writer)` resets the controller (FA2: only
  then); a numeric tweak only changes the next batch's params; then
  `reheat()`.
- `flush()`: resolves when `inFlight === 0`.
- `run({ maxIter, batch = 8, signal })`: loops `await step(batch)` until
  `settled` or `iterationsDone >= (maxIter ?? options.maxIter)` or
  `signal.aborted` (-> `E_ABORTED`; submitted batches complete and are
  discarded); resolves `stats`.
- `dispose()`: discards in-flight batches, destroys every simulation buffer
  (the pool trims), unregisters the loss listener, state "disposed";
  idempotent. Device loss -> "disposed" and every pending promise rejects
  `E_DEVICE_LOST`.

Throws: `E_NOT_LOADED`, `E_DISPOSED`, `E_RELEASED`, `E_DEVICE_LOST`,
`E_INVALID_ARGUMENT`, `E_SNAPSHOT`, `E_TOO_LARGE`, `E_OUT_OF_MEMORY`,
`E_VALIDATION`, `E_SHADER_COMPILE`, `E_UNSUPPORTED`, `E_ABORTED`.

#### src/layouts/forceatlas2.ts -- P3-T2 (spec 7.1-7.18)

Imports: `./force-simulation.js`, `./repulsion-exact.js`, `./inputs.js`, `../kernels.js`, `../kernel/*.js`, `../memory/*.js`, `../types/*.js`, `../constants.js`, `../errors.js`.

```ts
/** The ForceAtlas2 model (spec 7.4: K1 K2 K3 K4 K5 per iteration; toScene once per batch). Stages: ["K1", "K2", "K3", "K4", "K5", "toScene"]. */
export class ForceAtlas2Model implements ForceModel<ForceAtlas2Options, ForceAtlas2Stats> {
    constructor(tuning: ResolvedLayoutTuning, resolved: ResolvedForceAtlas2Options);
    readonly kind: "forceatlas2";
    readonly stages: readonly ["K1", "K2", "K3", "K4", "K5", "toScene"];
    readonly params: UniformBlock;
    readonly state: UniformBlock;
    readonly trace: UniformBlock;
    /** force 12n and oldForce 12n (zeroed) in BOTH swing modes (3.10.1: a writable slot is never aliased; mode 1 leaves oldForce unread and unwritten). */
    buffers(n: number, dim: 2 | 3): readonly BufferSpec[];
    /** { mass: resolveNodeMass(s, resolved.nodeMass), weights: resolveWeights(s, resolved.weight) } (3.13 inputs.ts). */
    inputs(s: GraphSnapshot, options: ForceAtlas2Options): ModelInputs;
    /** { LINLOG, DISTRIBUTED, TIER: 0, SWING_MODE: compat === "networkx" ? 1 : 0, STRONG_GRAVITY, GRAVITY_CENTER: compat === "networkx" ? 1 : 0 }; USE_PERM / HAS_WEIGHTS are merged in by the simulation from ModelResources (3.10, never from core.hasWeights). */
    overrides(options: ForceAtlas2Options): Readonly<Record<string, number | boolean>>;
    specs(overrides: Readonly<Record<string, number | boolean>>, subgroups: boolean): readonly WgslModuleSpec[];
    bind(resources: ModelResources, overrides: Readonly<Record<string, number | boolean>>): Promise<void>;
    paramsFor(iteration: number, options: ForceAtlas2Options): UniformValues;
    recordIteration(batch: CommandBatch, slot: number, tier: "exact" | "grid", upTo?: string | undefined): void;
    /** speed = 1, speedEfficiency = 1, swing = 1, traction = 1 (mode 1 accumulates from 1; mode 0 overwrites them each iteration, the initial value is irrelevant). */
    onLoad(state: StateWriter): void;
    /** Mode 0: nothing (D8). Mode 1 (networkx): swing = traction = 1 (spec 7.2 "load() and reheat() reset them to 1"). */
    onReheat(state: StateWriter): void;
    /** Resets speed / speedEfficiency to 1 only when linlog, strongGravity or distributedAction changed (spec 7.17). */
    onSetParams(patch: Partial<ForceAtlas2Options>, state: StateWriter): void;
    readStats(state: DataView, trace: DataView): ForceAtlas2Stats;
}
/** Applies FA2_DEFAULTS to the option record; validates ranges. */
export function resolveForceAtlas2Options(options: ForceAtlas2Options | undefined, previous?: ResolvedForceAtlas2Options | undefined): ResolvedForceAtlas2Options;
/** Applies LAYOUT_TUNING_DEFAULTS. */
export function resolveLayoutTuning(tuning: GpuLayoutTuning | undefined): ResolvedLayoutTuning;
/** Spec 3.3 createForceAtlas2, verbatim. */
export function createForceAtlas2(ctx: GpuContext, options?: (ForceAtlas2Options & GpuLayoutTuning) | undefined): GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>;
```

Contracts: `repulsion: "grid"` or `"auto"` with `n > exactMaxNodes` ->
`E_UNSUPPORTED { feature: "repulsion.grid" }` at `load()` (P4 lifts it);
`nodeSize` non-null -> `E_UNSUPPORTED { option: "nodeSize" }` at creation
(spec 7.14); `gravity < 0`, `scalingRatio <= 0`, `jitterTolerance <= 0`,
`maxIter < 1`, `settleWindow < 1`, `settleThreshold < 0`, `maxInFlight < 1`,
`iterationsPerStep < 1`, `dim` not 2 or 3, `scale <= 0` ->
`E_INVALID_ARGUMENT`; a `maxInFlight` in a `setParams` patch that differs
from the current value -> `E_INVALID_ARGUMENT` (the uniform ring is sized by
it at construction, 3.13 ForceSimulation); `dissuadeHubs` accepted and
ignored; `arcCount === 0` -> K2 is not recorded and `force` is zeroed by
`fill` (spec 7.5); `weight` resolution happens at `load()` through
`model.inputs()` -> `resolveWeights` (3.13 ForceSimulation load()), so a
weighted snapshot with `weight: false | null` attracts with 1.0
(`HAS_WEIGHTS = false`) and an unweighted snapshot with `weight: "<edge
column>"` binds the expanded column (`HAS_WEIGHTS = true`); the trace is
`ForceAtlas2Stats.trace` of the last completed batch (k records);
`msPerIteration` = wall time of the batch / k (the profiler value when
present, labelled through `stats.msPerIteration` only); `layoutRadius` =
`state.radius` = max |p - centroid| exactly as spec 3.3 defines it, about
the start-of-iteration centroid that `rmsRadius` also uses (K5 carries max
|p - c|^2 in `partials.max.w`, K1 takes the square root; `load()` computes
the initial value on the CPU the same way); `repulsionTier: "exact"`; the
grid fields `null`.

Throws: as ForceSimulation plus `E_UNSUPPORTED`, `E_INVALID_ARGUMENT`; pass-through `E_GPU_INELIGIBLE`, `E_COLUMN_LENGTH`.

### 3.14 src/accelerator.ts -- P3-T3 (spec 3.3, 9)

```ts
/** Spec 3.3 createAccelerator, verbatim: the object implementing AlgorithmAccelerator & LayoutAccelerator structurally; at P3 it carries forceAtlas2, release and dispose. */
export function createAccelerator(ctx: GpuContext, options?: AcceleratorOptions | undefined): GpuAccelerator;
```

Contract: `kind: "webgpu"`; `options` is a frozen deep copy;
`forceAtlas2(o)` = `createForceAtlas2(ctx, { ...o, ...options.layout })`
(the tuning defaults win over nothing the CPU option type can carry, spec
3.3); `release(s)` = `ctx.release(s)`; `dispose()` = `ctx.dispose()`; no
other members exist (a `pageRank` property is `undefined`, which is what
the 9.2 dispatcher tests for).

Throws: what `createForceAtlas2` throws; `E_DISPOSED` / `E_DEVICE_LOST` via `assertReady()`.

### 3.15 src/index.ts -- the export list at the end of each phase (spec 2.5, 3.3)

The barrel starts with `/// <reference types="@webgpu/types" />` and uses
explicit named exports (values and `export type`); `test/index.test.ts`
pins the VALUE list and `test/types/public-api.test-d.ts` the type list.

End of P0 (values): `WebGpuGraphError`, `isWebGpuGraphError`, `hasErrorCode`,
`PASSTHROUGH_FORMAT_CODES`, `WORKGROUP_SIZE`, `MAX_WORKGROUPS_PER_DIM`,
`MAX_1D_ITEMS`, `ARC_WINDOW_ALIGN`, `STORAGE_ALIGN`, `EXACT_MAX_NODES`,
`isSoftwareAdapter`. Types: `WebGpuGraphErrorCode`, `AdapterInfoLike`.

End of P1 adds (values): `GpuContext`, `degree`. Types: `GpuCaps`,
`GpuContextOptions`, `LimitPolicy`, `RaisableLimit`, `ProbeOptions`,
`ProbeResult`, `AdapterSummary`, `PlanCaps`, `PlanLimits`, `GpuRunOptions`,
`Profiler`, `PassTiming`.

End of P2 adds: nothing public (Lease, CommandBatch, UniformRing are
internal); no change to the pinned lists.

End of P3 adds (values): `createForceAtlas2`, `createAccelerator`,
`FA2_DEFAULTS`, `LAYOUT_TUNING_DEFAULTS`, `seedPositions`. Types:
`CommonLayoutOptions`, `SimulationOptions`, `ForceAtlas2Options`,
`FruchtermanReingoldOptions`, `SpringElectricalOptions`, `LayoutStatsBase`,
`ForceAtlas2Stats`, `ForceAtlas2TraceRecord`, `GpuLayoutSimulation`,
`GpuLayoutTuning`, `RunOptions`, `LayoutSimulation`, `LayoutAccelerator`,
`AlgorithmAccelerator`, `GpuAccelerator`, `AcceleratorOptions`,
`ScoresResultLike`, `PageRankResultLike`, `HitsResultLike`,
`LabelResultLike`, `BfsResultLike`, `SsspResultLike`,
`BellmanFordResultLike`, `EdgeScoresResultLike`, `ApspResultLike`,
`CorenessResultLike`, `MstResultLike`, `CommunityResultLike`,
`CpuAlgorithmOptions`.

Never exported from the root: anything of `src/browser/**` or `src/node/**`
(their own entries), `GraphResidency`, `BufferPool`, `Readback`, `Lease`,
`PipelineCache`, `Kernel`, `CommandBatch`, `UniformRing`, `UniformBlock`,
`composeWgsl`, `KERNELS`, `ForceSimulation`, `ForceAtlas2Model` (tests import
them from their files; they are `@internal` surface). `calibrateLayout` and
every algorithm other than `degree` are P4+ / P7+ and absent.

## 4. WGSL contract

### 4.1 The prelude (src/kernel/prelude.ts, `PRELUDE_WGSL`; spec 3.5)

`${...}` are TypeScript interpolations from `constants.ts` and graph-format's
`INVALID_INDEX`; the literal-grep test forbids `65535u`, `256u` and
`0xFFFFFFFFu` in `src/wgsl/**` and in this file's template (bodies take
every u32 sentinel from the prelude: `INVALID_INDEX`, `U32_MAX`). f32
constants are interpolated through a formatter that guarantees a decimal
point or an exponent (`0.01`, `0.0001`, `1e-8`).

```wgsl
// ---- prelude: constants, standard overrides, helpers (every module receives this text first)
const INVALID_INDEX: u32 = ${INVALID_INDEX}u;
const U32_MAX: u32 = ${U32_MAX}u;
const MAX_WORKGROUPS_PER_DIM: u32 = ${MAX_WORKGROUPS_PER_DIM}u;
const FA2_DIST_FLOOR: f32 = ${FA2_DISTANCE_FLOOR};
const FA2_DIST_FLOOR_SQ: f32 = ${FA2_DISTANCE_FLOOR_SQ};
const FA2_COINCIDENT_SQ: f32 = ${FA2_COINCIDENT_SQ};
const FA2_FLAG_FIRST: u32 = ${FA2_FLAG_FIRST}u;
const F32_MAX: f32 = 0x1.fffffep+127;
override WG: u32 = ${WORKGROUP_SIZE}u;
override USE_PERM: bool = false;
override HAS_WEIGHTS: bool = false;
override SUBGROUP_MIN: u32 = 4u;
override SUBGROUP_MAX: u32 = 0u;

fn linear_id(wid: vec3<u32>, lid: u32) -> u32 { return (wid.x + wid.y * MAX_WORKGROUPS_PER_DIM) * WG + lid; }
fn group_id(wid: vec3<u32>) -> u32 { return wid.x + wid.y * MAX_WORKGROUPS_PER_DIM; }
fn lowbias32(x0: u32) -> u32 {
    var x = x0;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}
fn mask_bit(w: u32, i: u32) -> bool { return ((w >> (i & 31u)) & 1u) == 1u; }
fn unpack_u8(w: u32, i: u32) -> u32 { return (w >> (8u * (i & 3u))) & 0xFFu; }
fn pair_hash(i: u32, j: u32) -> u32 { return lowbias32((min(i, j) * 0x9E3779B9u) ^ max(i, j)); }
fn hash_unit(h: u32) -> f32 { return f32(h >> 8u) * (1.0 / 16777216.0); }
fn hash_dir(h: u32, dim: u32) -> vec3f {
    let phi = 6.283185307179586 * hash_unit(h);
    if (dim == 2u) { return vec3f(cos(phi), sin(phi), 0.0); }
    let z = 2.0 * hash_unit(lowbias32(h ^ 0x5bd1e995u)) - 1.0;
    let r = sqrt(max(0.0, 1.0 - z * z));
    return vec3f(r * cos(phi), r * sin(phi), z);
}
fn kick_dir(i: u32, j: u32, dim: u32) -> vec3f {
    let d = hash_dir(pair_hash(i, j), dim);
    return select(d, -d, i > j);
}
```

`kick_dir(i, j) == -kick_dir(j, i)` by construction, so the coincident kick
of spec 7.2 is antisymmetric and the force-sum invariant of 11.4 holds on
the coincident fixture; `hash_unit` uses 24 bits so the value is exact in
f32 on every vendor; `lowbias32` is Wellons' integer hash (no `sin()`, note
03 section 1.3). `USE_ARC_PERM` (a guard for the group-3 `arcToEdge` /
`edgeToArc` gathers of P7+) is reserved and NOT emitted in P0-P3.

### 4.2 The composer's emitted format (composeWgsl; spec 3.5)

The composed text is, in order, each part on its own lines:

```
enable subgroups;                                         <- only when needs has "subgroups" AND caps.features has it
<PRELUDE_WGSL>
override LINLOG: bool = false;                            <- one line per overrideDecl, `override NAME: T = default;`
override TIER: u32 = 0u;                                  <- u32 literals carry the `u` suffix, f32 literals a decimal point, bools true / false
struct Fa2Params { ... }                                  <- every UniformBlock of spec.uniforms, its `wgsl` text, in list order
<REDUCE_HELPERS_SUBGROUP_WGSL or REDUCE_HELPERS_WORKGROUP_WGSL>   <- only when needs has "subgroups"; the subgroup form when the feature is present
@group(0) @binding(0) var<storage, read> rowPtr: array<u32>;         <- one line per BindingDecl in list order:
@group(1) @binding(1) var<storage, read_write> force: array<f32>;      storage-ro -> `var<storage, read>`, storage -> `var<storage, read_write>`,
@group(2) @binding(0) var<uniform> P: Fa2Params;                       uniform -> `var<uniform>`
<body with every //@@NAME@@ line replaced by spec.snippets[NAME]>
```

The override VALUES of `spec.overrides` are not written into the text;
they go to `createComputePipeline({ compute: { constants } })` (a distinct
override set is a distinct pipeline, spec 5.1). `ComposedModule.bodyLine`
is the 1-based line of the body's first line in `code`, so
`formatCompilationInfo` reports body-relative lines.

### 4.3 The reduction helpers (src/kernel/prelude.ts) and the subgroup twin mechanism (D16)

A body whose spec lists `needs: ["subgroups"]` may call `wg_reduce_f32(v,
lid, op)`, `wg_reduce_u32(v, lid, op)`, `wg_reduce_vec4(v, lid, op)` with
`op` 0 = sum, 1 = min, 2 = max; each returns the WORKGROUP total to every
invocation (the caller writes it from `lid == 0u`); each must be called in
UNIFORM control flow by every invocation of the workgroup; results are
bitwise deterministic run to run for both forms (fixed tree order in the
workgroup form; key-sorted slot order in the subgroup form). CONTRACT
DECISION: the helpers RETURN the total instead of writing `partials` (the
spec's K3 sketch says `workgroup_reduce2(sw, tr, lid.x)` "writes
partials[wid].B"): a prelude helper cannot name a kernel's binding, so the
body's lane 0 does the write.

Workgroup-memory form (`REDUCE_HELPERS_WORKGROUP_WGSL`):

```wgsl
var<workgroup> wg_scratch_v: array<vec4f, WG>;
var<workgroup> wg_scratch_u: array<u32, WG>;
fn combine_v(a: vec4f, b: vec4f, op: u32) -> vec4f {
    if (op == 1u) { return min(a, b); }
    if (op == 2u) { return max(a, b); }
    return a + b;
}
fn combine_u(a: u32, b: u32, op: u32) -> u32 {
    if (op == 1u) { return min(a, b); }
    if (op == 2u) { return max(a, b); }
    return a + b;
}
fn wg_reduce_vec4(v: vec4f, lid: u32, op: u32) -> vec4f {
    workgroupBarrier();
    wg_scratch_v[lid] = v;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) { wg_scratch_v[lid] = combine_v(wg_scratch_v[lid], wg_scratch_v[lid + s], op); }
        workgroupBarrier();
    }
    let total = wg_scratch_v[0];
    workgroupBarrier();
    return total;
}
fn wg_reduce_u32(v: u32, lid: u32, op: u32) -> u32 {
    workgroupBarrier();
    wg_scratch_u[lid] = v;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) { wg_scratch_u[lid] = combine_u(wg_scratch_u[lid], wg_scratch_u[lid + s], op); }
        workgroupBarrier();
    }
    let total = wg_scratch_u[0];
    workgroupBarrier();
    return total;
}
fn wg_reduce_f32(v: f32, lid: u32, op: u32) -> f32 { return wg_reduce_vec4(vec4f(v, 0.0, 0.0, 0.0), lid, op).x; }
```

`WG` is a power of two (asserted by `assertPlanLimits`), so the tree is
exact; the leading barrier lets two helper calls follow each other.

Subgroup form (`REDUCE_HELPERS_SUBGROUP_WGSL`; the composer sets
`SUBGROUP_MAX = caps.subgroupMaxSize` and `SUBGROUP_MIN = max(4,
caps.subgroupMinSize || 4)` whenever it splices this block):

```wgsl
override SG_SLOTS: u32 = (WG + SUBGROUP_MIN - 1u) / SUBGROUP_MIN;   // one slot per subgroup; the count is largest when the compiler picks the SMALLEST size
var<workgroup> sg_counter: atomic<u32>;
var<workgroup> sg_val_v: array<vec4f, SG_SLOTS>;
var<workgroup> sg_val_u: array<u32, SG_SLOTS>;
var<workgroup> sg_key: array<u32, SG_SLOTS>;
var<workgroup> sg_sorted_v: array<vec4f, SG_SLOTS>;
var<workgroup> sg_sorted_u: array<u32, SG_SLOTS>;
fn combine_v(a: vec4f, b: vec4f, op: u32) -> vec4f {
    if (op == 1u) { return min(a, b); }
    if (op == 2u) { return max(a, b); }
    return a + b;
}
fn combine_u(a: u32, b: u32, op: u32) -> u32 {
    if (op == 1u) { return min(a, b); }
    if (op == 2u) { return max(a, b); }
    return a + b;
}
fn wg_reduce_vec4(v: vec4f, lid: u32, op: u32) -> vec4f {
    workgroupBarrier();
    if (lid == 0u) { atomicStore(&sg_counter, 0u); }
    workgroupBarrier();
    let s_add = subgroupAdd(v);
    let s_min = subgroupMin(v);
    let s_max = subgroupMax(v);
    var partial = s_add;
    if (op == 1u) { partial = s_min; }
    if (op == 2u) { partial = s_max; }
    let key = subgroupMin(lid);                       // the smallest local id of this subgroup: a stable identity without @builtin(subgroup_id)
    var slot = 0u;
    if (subgroupElect()) { slot = atomicAdd(&sg_counter, 1u); }   // D16: the elected lane takes a slot from the counter
    slot = subgroupBroadcast(slot, 0u);                            // and broadcasts it (lane 0 is the elected lane in uniform control flow)
    if (subgroupElect()) { sg_val_v[slot] = partial; sg_key[slot] = key; }
    workgroupBarrier();
    let count = atomicLoad(&sg_counter);
    if (lid < count) {                                             // rank the slots by key so the final sum has a fixed order (11.9 item 4)
        let mine = sg_key[lid];
        var rank = 0u;
        for (var k = 0u; k < count; k = k + 1u) { if (sg_key[k] < mine) { rank = rank + 1u; } }
        sg_sorted_v[rank] = sg_val_v[lid];
    }
    workgroupBarrier();
    var total = sg_sorted_v[0];
    for (var k = 1u; k < count; k = k + 1u) { total = combine_v(total, sg_sorted_v[k], op); }
    workgroupBarrier();
    return total;
}
fn wg_reduce_u32(v: u32, lid: u32, op: u32) -> u32 {
    workgroupBarrier();
    if (lid == 0u) { atomicStore(&sg_counter, 0u); }
    workgroupBarrier();
    let s_add = subgroupAdd(v);
    let s_min = subgroupMin(v);
    let s_max = subgroupMax(v);
    var partial = s_add;
    if (op == 1u) { partial = s_min; }
    if (op == 2u) { partial = s_max; }
    let key = subgroupMin(lid);
    var slot = 0u;
    if (subgroupElect()) { slot = atomicAdd(&sg_counter, 1u); }
    slot = subgroupBroadcast(slot, 0u);
    if (subgroupElect()) { sg_val_u[slot] = partial; sg_key[slot] = key; }
    workgroupBarrier();
    let count = atomicLoad(&sg_counter);
    if (lid < count) {
        let mine = sg_key[lid];
        var rank = 0u;
        for (var k = 0u; k < count; k = k + 1u) { if (sg_key[k] < mine) { rank = rank + 1u; } }
        sg_sorted_u[rank] = sg_val_u[lid];
    }
    workgroupBarrier();
    var total = sg_sorted_u[0];
    for (var k = 1u; k < count; k = k + 1u) { total = combine_u(total, sg_sorted_u[k], op); }
    workgroupBarrier();
    return total;
}
fn wg_reduce_f32(v: f32, lid: u32, op: u32) -> f32 { return wg_reduce_vec4(vec4f(v, 0.0, 0.0, 0.0), lid, op).x; }
```

The subgroup builtins are called unconditionally (uniform control flow, spec
3.5 rule 1); `subgroup_size` / `subgroup_invocation_id` are never assumed
constant (D16); no `@builtin(subgroup_id)`. CONTRACT DECISION (corrects
spec 3.5 lines 1032-1033 and D16, section 9 item 7): the scratch is sized
`WG / SUBGROUP_MIN` rounded up, NOT `WG / SUBGROUP_MAX` -- the number of
subgroups in a workgroup is `WG / subgroup_size`, which is LARGEST when the
compiler picks the smallest size in `[subgroupMinSize, subgroupMaxSize]`;
on a device with `min != max` (Intel Xe / Arc 8-32, AMD 32-64) a
`SUBGROUP_MAX`-sized scratch would take slots up to `WG / min - 1` and the
elected-lane writes would go out of bounds (clamped silently by WGSL),
which the three CI adapters (4/4, 8/8, 32/32) can never catch; at `WG = 256`
and the floor of 4 the scratch is at most 64 slots (2.8 KiB of workgroup
memory). `SUBGROUP_MAX` stays a standard override (spec 3.5) but sizes
nothing in P0-P3. `test/kernel/wgsl.test.ts` composes the block with
`fakeCaps(CAPS_SPEC_DEFAULT, {}, { subgroupMinSize: 8, subgroupMaxSize: 32
})` and asserts `SG_SLOTS >= WG / 8` (the override values in
`ComposedModule.overrides`) and that `subgroupMinSize` 0 / undefined yields
`SUBGROUP_MIN = 4`. Twins of the same kernel agree to f32 summation-order
noise (11.3: `1e-6` relative on one device) and `u32` results agree
bitwise.

### 4.4 Rules every body obeys, and the bind-group convention per kernel

Rules (spec 3.5): (1) every barrier / subgroup builtin / helper call sits in
UNIFORM control flow -- per-invocation work runs under an `if (valid)` into
locals, reductions after it; an early `return` is legal only in a body
that reaches no barrier after it (degree, segmented-reduce, attraction,
toScene, fill); (2) every mixed `*` / `^` / `&` / `|` expression is fully
parenthesised; (3) no `@group(`, no `override ` in a body; (4) constants
come from the prelude; (5) the `.w` lane of `pos` is the mass and is never
read as `.x` (a sabotage mutation checks the test catches the swap); (6)
`force` / `oldForce` are `array<f32>` of stride 3 read and written through
the per-body `load_force` / `store_force` / `load_old` / `store_old`
helpers; (7) `S` (Fa2State) is `read_write` in every kernel that binds it.

Bind-group convention applied (spec 3.5, 7.4): group 0 = the graph
(`rowPtr`, `colIdx`, `weights` | dummy, `perm` | dummy -- always four slots),
group 1 = state, group 2 = the params uniform with a dynamic offset into
the UniformRing, group 3 = cold arrays (unused in P0-P3).

| Kernel | Dispatch | Reads | Writes |
| --- | --- | --- | --- |
| `degree` | plan1d(rows of [start, end)) | rowPtr, colIdx (bounds-checked targets), perm when USE_PERM, P | out[i] (overwrite, or += when accumulate) |
| `reduce` level 1 | plan1d(count) | src, P | out[outOffset + group] |
| `reduce` FINAL | 1 workgroup | src (the partials), P | out[outOffset] |
| `fill` | plan1d(count) | P | dst[i] |
| `segmented-reduce` | plan1d(rows) | rowPtr, colIdx, weights when HAS_WEIGHTS, perm when USE_PERM, P | out[i] |
| K1 `fa2-stats-finalize` | 1 workgroup | partials (sum / min / max / dispFree of the previous integrate; skipped on FA2_FLAG_FIRST), S.settledCount, S.iteration, P | S.centroid, S.rmsRadius, S.radius, S.min, S.max, S.meanDisplacement, S.settledCount, S.iteration; T[iterationIndex].meanDisplacement / settledCount / iteration |
| K2 `fa2-attraction` | plan1d(tierEnd - tierStart) (P3: [0, n)); skipped when arcCount === 0 (fill zeroes force instead) | rowPtr, colIdx, weights when HAS_WEIGHTS, perm when USE_PERM, pos (xyz, and .w when DISTRIBUTED), P | force (overwrite: the first writer each iteration) |
| K3 `fa2-repulsion-exact` | plan1d(n) | pos (whole array through the tile), S.centroid (GRAVITY_CENTER 0), force, oldForce (SWING_MODE 0), fixedMask (SWING_MODE 0), P | force (+= repulsion + gravity), partials[g].swingTraction |
| K4 `fa2-speed-finalize` | 1 workgroup | partials[*].swingTraction, S.speed, S.speedEfficiency, S.swing / S.traction (SWING_MODE 1 accumulates), P | S.swing, S.traction, S.speed, S.speedEfficiency; T[iterationIndex].swing / traction / speed / speedEfficiency |
| K5 `fa2-integrate` | plan1d(n) | force, oldForce (SWING_MODE 0), fixedMask, S.speed, S.centroid, pos, P | pos, oldForce (SWING_MODE 0), partials[g].sum / min / max / dispFree |
| `fa2-to-scene` | plan1d(n) | pos, P.scale, P.center, P.dim | scene[3i .. 3i+2] |

Per-iteration order (spec 7.4): K1, K2 (or fill), K3, K4, K5; per batch:
k x (that sequence), then toScene, then the staging copies of
`scenePositions` and `state` (header + k trace records). Everything in
ONE compute pass per batch (dispatches in a pass are ordered, spec 5.8);
`toScene` is a second pass so a profiler can time it separately.

### 4.5 The normative bodies

Every body below is the FULL text of its `src/wgsl/<name>.wgsl.ts` export
(the spec's 7.5, 7.6, 7.10 and 7.11 sketches completed; the spec elided
K1, the K5 epilogue, toScene, reduce, degree, segmented-reduce and fill).
A body may be re-formatted but not re-derived; a sabotage mutation is a
textual edit of one of these bodies (5.2).

#### degree.wgsl.ts (`degreeWgsl`, entry `degree`)

```wgsl
@compute @workgroup_size(WG)
fn degree(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = linear_id(wid, lid.x) + P.start;
    if (row >= P.end) { return; }
    let i = select(row, perm[row], USE_PERM);
    let a0 = max(rowPtr[i], P.arcBase);
    let a1 = min(rowPtr[i + 1u], P.arcEnd);
    var d = 0u;
    for (var arc = a0; arc < a1; arc = arc + 1u) {
        let nbr = colIdx[arc - P.arcBase];               // `target` is a WGSL reserved word (spec 16.2); never use it as an identifier
        d = d + select(0u, 1u, nbr < P.n);
    }
    out[i] = select(d, out[i] + d, P.accumulate == 1u);
}
```

#### reduce.wgsl.ts (`reduceWgsl`, entry `reduce`)

```wgsl
// DTYPE 0 = f32, 1 = u32, 2 = vec4f (4 words per element); OP 0 = sum, 1 = min, 2 = max; FINAL = the one-workgroup level
fn identity_f() -> f32 { if (OP == 1u) { return F32_MAX; } if (OP == 2u) { return -F32_MAX; } return 0.0; }
fn identity_u() -> u32 { if (OP == 1u) { return U32_MAX; } return 0u; }        // U32_MAX from the prelude: the literal is forbidden in bodies (4.1)
fn comb_f(a: f32, b: f32) -> f32 { if (OP == 1u) { return min(a, b); } if (OP == 2u) { return max(a, b); } return a + b; }
fn comb_u(a: u32, b: u32) -> u32 { if (OP == 1u) { return min(a, b); } if (OP == 2u) { return max(a, b); } return a + b; }
fn comb_v(a: vec4f, b: vec4f) -> vec4f { if (OP == 1u) { return min(a, b); } if (OP == 2u) { return max(a, b); } return a + b; }
fn load_f(i: u32) -> f32 { return bitcast<f32>(src[i]); }
fn load_v(i: u32) -> vec4f {
    return vec4f(bitcast<f32>(src[4u * i]), bitcast<f32>(src[4u * i + 1u]), bitcast<f32>(src[4u * i + 2u]), bitcast<f32>(src[4u * i + 3u]));
}
fn store_f(i: u32, v: f32) { out[i] = bitcast<u32>(v); }
fn store_v(i: u32, v: vec4f) {
    out[4u * i] = bitcast<u32>(v.x);
    out[4u * i + 1u] = bitcast<u32>(v.y);
    out[4u * i + 2u] = bitcast<u32>(v.z);
    out[4u * i + 3u] = bitcast<u32>(v.w);
}

@compute @workgroup_size(WG)
fn reduce(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    var accF = identity_f();
    var accU = identity_u();
    var accV = vec4f(identity_f());
    if (FINAL) {
        for (var i = lid.x; i < P.count; i = i + WG) {           // sequential per lane in index order: deterministic
            if (DTYPE == 0u) { accF = comb_f(accF, load_f(i)); }
            else if (DTYPE == 1u) { accU = comb_u(accU, src[i]); }
            else { accV = comb_v(accV, load_v(i)); }
        }
    } else {
        let i = linear_id(wid, lid.x);
        if (i < P.count) {
            if (DTYPE == 0u) { accF = load_f(i); }
            else if (DTYPE == 1u) { accU = src[i]; }
            else { accV = load_v(i); }
        }
    }
    // uniform control flow: the workgroup reduction of the selected dtype (DTYPE is a pipeline constant, so the branch is uniform)
    var tF = 0.0;
    var tU = 0u;
    var tV = vec4f(0.0);
    if (DTYPE == 0u) { tF = wg_reduce_f32(accF, lid.x, OP); }
    else if (DTYPE == 1u) { tU = wg_reduce_u32(accU, lid.x, OP); }
    else { tV = wg_reduce_vec4(accV, lid.x, OP); }
    if (lid.x == 0u) {
        let g = select(group_id(wid), 0u, FINAL);
        let o = P.outOffset + g;
        if (DTYPE == 0u) { store_f(o, tF); }
        else if (DTYPE == 1u) { out[o] = tU; }
        else { store_v(o, tV); }
    }
}
```

#### fill.wgsl.ts (`fillWgsl`, entry `fill`)

```wgsl
@compute @workgroup_size(WG)
fn fill(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = linear_id(wid, lid.x);
    if (i >= P.count) { return; }
    dst[i] = select(P.value, i + P.value, P.mode == 1u);
}
```

#### segmented-reduce.wgsl.ts (`segmentedReduceWgsl`, entry `segmented_reduce`; TIER 0 = thread-per-row)

```wgsl
fn identity() -> f32 { if (OP == 1u) { return F32_MAX; } if (OP == 2u) { return -F32_MAX; } return 0.0; }
fn comb(a: f32, b: f32) -> f32 { if (OP == 1u) { return min(a, b); } if (OP == 2u) { return max(a, b); } return a + b; }

@compute @workgroup_size(WG)
fn segmented_reduce(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = linear_id(wid, lid.x) + P.start;
    if (row >= P.end) { return; }
    let i = select(row, perm[row], USE_PERM);
    let a0 = max(rowPtr[i], P.arcBase);
    let a1 = min(rowPtr[i + 1u], P.arcEnd);
    var acc = identity();
    for (var arc = a0; arc < a1; arc = arc + 1u) {
        let nbr = colIdx[arc - P.arcBase];               // the neighbour index (`target` is a WGSL reserved word)
        var weight = 1.0;
        if (HAS_WEIGHTS) { weight = weights[arc - P.arcBase]; }
        var v = 0.0;
        //@@VALUE@@
        acc = comb(acc, v);
    }
    out[i] = select(acc, comb(out[i], acc), P.accumulate == 1u);
}
```

The `VALUE` snippet is one or more WGSL statements assigning `v` and
referencing only `row`, `arc`, `nbr`, `weight`, `v` (e.g. `v = weight;`
for the weighted degree, `v = 1.0;` for the degree); the composer rejects
any other identifier textually (3.11). The TypeScript oracle callback keeps
its own parameter names (5.3).

#### fa2-stats-finalize.wgsl.ts (K1; `fa2StatsFinalizeWgsl`, entry `stats_finalize`)

```wgsl
// K1: folds the previous integrate's partials into the state block (spec 7.4); one workgroup
@compute @workgroup_size(WG)
fn stats_finalize(@builtin(local_invocation_id) lid: vec3<u32>) {
    let groups = (P.n + WG - 1u) / WG;
    let fold = (P.flags & FA2_FLAG_FIRST) == 0u;    // the first iteration after load() keeps the host-written state
    var sum = vec4f(0.0);
    var lo = vec4f(F32_MAX);
    var hi = vec4f(-F32_MAX);
    var disp = 0.0;
    var free = 0u;
    if (fold) {
        for (var g = lid.x; g < groups; g = g + WG) {   // sequential per lane in index order: deterministic
            let q = partials[g];
            sum = sum + q.sum;
            lo = min(lo, q.min);
            hi = max(hi, q.max);
            disp = disp + q.dispFree.x;
            free = free + u32(q.dispFree.y);
        }
    }
    let tSum = wg_reduce_vec4(sum, lid.x, 0u);
    let tLo = wg_reduce_vec4(lo, lid.x, 1u);
    let tHi = wg_reduce_vec4(hi, lid.x, 2u);
    let tDisp = wg_reduce_f32(disp, lid.x, 0u);
    let tFree = wg_reduce_u32(free, lid.x, 0u);
    if (lid.x == 0u) {
        if (fold) {
            let n = f32(P.n);
            let c = tSum.xyz / n;
            S.centroid = vec4f(c, 0.0);
            S.rmsRadius = sqrt(max(tSum.w, 0.0) / n);                  // RMS radius about the previous centroid (7.17)
            S.min = vec4f(tLo.xyz, 0.0);
            S.max = vec4f(tHi.xyz, 0.0);
            S.radius = sqrt(max(tHi.w, 0.0));                          // max |p - centroid| about the same previous centroid as rmsRadius (K5 puts |q|^2 in max.w)
            let meanDisp = select(tDisp / f32(tFree), 0.0, tFree == 0u);  // all-fixed: 0, never NaN (7.4)
            S.meanDisplacement = meanDisp;
            S.settledCount = select(0u, S.settledCount + 1u, meanDisp <= P.settleThreshold * S.rmsRadius);
        }
        S.iteration = S.iteration + 1u;
        T[P.iterationIndex].meanDisplacement = S.meanDisplacement;
        T[P.iterationIndex].settledCount = S.settledCount;
        T[P.iterationIndex].iteration = S.iteration;
    }
}
```

#### fa2-attraction.wgsl.ts (K2; `fa2AttractionWgsl`, entry `attraction`; spec 7.5)

```wgsl
fn store_force(i: u32, f: vec3f) {
    force[3u * i] = f.x;
    force[3u * i + 1u] = f.y;
    force[3u * i + 2u] = f.z;
}

@compute @workgroup_size(WG)
fn attraction(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = linear_id(wid, lid.x) + P.tierStart;
    if (row >= P.tierEnd) { return; }                          // no barrier follows in this tier (3.5 rule 1)
    let i = select(row, perm[row], USE_PERM);
    let pi = pos[i];                                           // xyz + mass in one load (D23)
    var f = vec3f(0.0);
    for (var a = rowPtr[i]; a < rowPtr[i + 1u]; a = a + 1u) {
        let j = colIdx[a];
        if (j == i) { continue; }                              // a self-loop exerts no force
        var w = 1.0;
        if (HAS_WEIGHTS) { w = weights[a]; }
        let d = pos[j].xyz - pi.xyz;                           // toward j
        let len = max(length(d), FA2_DIST_FLOOR);
        let mag = select(w, w * log(1.0 + len) / len, LINLOG); // linear: |F| = w len; linlog: |F| = w log(1 + len)
        f = f + d * mag;
    }
    if (DISTRIBUTED) { f = f / pi.w; }
    store_force(i, f);                                         // overwrites: attraction is the first writer of force each iteration
}
```

#### fa2-repulsion-exact.wgsl.ts (K3; `fa2RepulsionExactWgsl`, entry `repulsion`; spec 7.6, 7.9, 7.10)

```wgsl
var<workgroup> tile: array<vec4f, WG>;                         // xyz + mass, 4 KiB at WG = 256

fn load_force(i: u32) -> vec3f { return vec3f(force[3u * i], force[3u * i + 1u], force[3u * i + 2u]); }
fn store_force(i: u32, f: vec3f) {
    force[3u * i] = f.x;
    force[3u * i + 1u] = f.y;
    force[3u * i + 2u] = f.z;
}
fn load_old(i: u32) -> vec3f { return vec3f(oldForce[3u * i], oldForce[3u * i + 1u], oldForce[3u * i + 2u]); }
fn gravity_force(pi: vec4f) -> vec3f {                         // spec 7.9: centroid (GRAVITY_CENTER 0) or origin (1); regular or strong
    var q = pi.xyz;
    if (GRAVITY_CENTER == 0u) { q = pi.xyz - S.centroid.xyz; }
    if (STRONG_GRAVITY) { return -P.gravity * pi.w * q; }
    let d = length(q);
    if (d > FA2_DIST_FLOOR) { return -P.gravity * pi.w * q / d; }
    return vec3f(0.0);
}

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
        if (j < P.n) { tile[lid.x] = pos[j]; } else { tile[lid.x] = vec4f(0.0); }   // guarded fill; mass 0 marks the pad
        workgroupBarrier();                                                       // uniform: every invocation reaches it
        for (var s = 0u; s < WG; s = s + 1u) {
            let o = tile[s];
            let jj = t * WG + s;
            if (o.w > 0.0 && jj != i) {                                           // mass > 0 for every real node, 0 for the pad
                let d = pi.xyz - o.xyz;
                var d2 = dot(d, d);
                if (d2 < FA2_COINCIDENT_SQ) {                                     // coincident: antisymmetric unit kick of magnitude k m_i m_j / 0.01 (7.2)
                    f = f + kick_dir(i, jj, P.dim) * (P.scalingRatio * pi.w * o.w / FA2_DIST_FLOOR);
                    continue;
                }
                d2 = max(d2, FA2_DIST_FLOOR_SQ);                                  // d >= 0.01
                let k = P.scalingRatio * pi.w * o.w;
                f = f + d * (k / d2);                                             // |F| = k m_i m_j / d along d / d
            }
        }
        workgroupBarrier();
    }
    // epilogue (7.9, 7.10): gravity and force += under the guard, the swing / traction reduction outside it
    var sw = 0.0;
    var tr = 0.0;
    if (valid) {
        f = f + gravity_force(pi);
        let fnew = load_force(i) + f;
        store_force(i, fnew);
        if (SWING_MODE == 1u) {                                                   // NetworkX: positions and forces mixed, every node (7.2)
            sw = pi.w * length(pi.xyz - fnew);
            tr = 0.5 * pi.w * length(pi.xyz + fnew);
        } else if (!mask_bit(fixedMask[i >> 5u], i)) {                            // paper: free nodes only (Gephi ForceAtlas2.java 283-293)
            let fold = load_old(i);
            sw = pi.w * length(fnew - fold);
            tr = 0.5 * pi.w * length(fnew + fold);
        }
    }
    let t = wg_reduce_vec4(vec4f(sw, tr, 0.0, 0.0), lid.x, 0u);                   // uniform control flow: 256 -> 1
    if (lid.x == 0u) { partials[group_id(wid)].swingTraction = t.xy; }
}
```

#### fa2-speed-finalize.wgsl.ts (K4; `fa2SpeedFinalizeWgsl`, entry `speed_finalize`; spec 7.10)

```wgsl
@compute @workgroup_size(WG)
fn speed_finalize(@builtin(local_invocation_id) lid: vec3<u32>) {
    let groups = (P.n + WG - 1u) / WG;
    var st = vec2f(0.0);
    for (var g = lid.x; g < groups; g = g + WG) { st = st + partials[g].swingTraction; }   // sequential per lane: deterministic
    let t = wg_reduce_vec4(vec4f(st, 0.0, 0.0), lid.x, 0u);
    if (lid.x == 0u) {
        var swing = t.x;
        var traction = t.y;
        if (SWING_MODE == 1u) { swing = S.swing + t.x; traction = S.traction + t.y; }   // NetworkX accumulates across iterations from 1
        let n = f32(P.n);
        let optJitter = 0.05 * sqrt(n);
        let minJitter = sqrt(optJitter);
        let maxJitter = 10.0;
        let tr = max(traction, 1.0e-30);                                             // guards the division only (7.10)
        let other = min(maxJitter, optJitter * traction / (n * n));
        var jitter = P.jitterTolerance * max(minJitter, other);
        var eff = S.speedEfficiency;
        if (swing > 2.0 * tr) {                                                      // swing / traction > 2 in the exact form (CONTRACT DECISION K4-1 below: 2 x is exact, a WGSL f32 division is not)
            if (eff > 0.05) { eff = eff * 0.5; }                                     // the CPU's conditional multiply (7.2)
            jitter = max(jitter, P.jitterTolerance);
        }
        let targetSpeed = select(jitter * eff * traction / swing, 1.0e30, swing == 0.0);  // +Inf in the port; 1e30 gives the same min() below (`target` is reserved)
        if (swing > jitter * traction) {
            if (eff > 0.05) { eff = eff * 0.7; }
        } else if (S.speed < 1000.0) {
            eff = eff * 1.3;
        }
        S.speed = S.speed + min(targetSpeed - S.speed, 0.5 * S.speed);
        S.speedEfficiency = eff;
        S.swing = swing;
        S.traction = traction;
        T[P.iterationIndex].swing = swing;
        T[P.iterationIndex].traction = traction;
        T[P.iterationIndex].speed = S.speed;
        T[P.iterationIndex].speedEfficiency = eff;
    }
}
```

CONTRACT DECISION K4-1 (P3-T5 / G3 finding G3-F6; spec 7.10 snippet line
`if (swing / tr > 2.0)`): the halving predicate is written in its exact form
`swing > 2.0 * tr` (the same truth value as the port's `swing / traction >
2` for every `tr > 0`, and `tr = max(traction, 1e-30) > 0`). Reason: every
paper-mode first iteration after `load()` runs with `oldForce = 0`, so
`traction_i = 0.5 m_i |F_i|` is EXACTLY half of `swing_i = m_i |F_i|` per node
and, since a scaling by a power of two commutes with f32 rounding, the folded
`traction` is exactly half the folded `swing`: the predicate sits on its
knife edge at the start of every layout. A multiplication by 2 and the
comparison are exact on every device; WGSL (and Vulkan) grant f32 division
2.5 ULP, and Dawn on the RTX 4070 SUPER returns `2 + 1 ulp` for 2.2% of the
exact-ratio inputs `x / (x / 2)` (`tmp/p3-fix/div-probe.mjs`: 5,028 of 32,768
not exactly 2, 709 above; 27% of random quotients 1 ulp off; lavapipe exact),
which halves `speedEfficiency` at iteration 0 where the CPU reference never
does -- the multi-state twin comparison of `test/layouts/fa2-twins.test.ts`
caught it (the two twins, same device, disagreeing by the branch from the
same positions). The oracle (5.3, `estimateFactor`) uses the same exact form;
the owner amends the 7.10 snippet in the spec's Review log (section 9 item
10).

#### fa2-integrate.wgsl.ts (K5; `fa2IntegrateWgsl`, entry `integrate`; spec 7.11)

```wgsl
fn load_force(i: u32) -> vec3f { return vec3f(force[3u * i], force[3u * i + 1u], force[3u * i + 2u]); }
fn load_old(i: u32) -> vec3f { return vec3f(oldForce[3u * i], oldForce[3u * i + 1u], oldForce[3u * i + 2u]); }
fn store_old(i: u32, f: vec3f) {
    oldForce[3u * i] = f.x;
    oldForce[3u * i + 1u] = f.y;
    oldForce[3u * i + 2u] = f.z;
}

@compute @workgroup_size(WG)
fn integrate(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = linear_id(wid, lid.x);
    var dp = vec3f(0.0);
    var p = vec4f(0.0);
    var free = false;
    var valid = false;
    if (i < P.n) {
        valid = true;
        let f = load_force(i);
        p = pos[i];
        var swing_i = p.w * length(f);                                     // SWING_MODE 1: NetworkX's local swinging m |F| (layout.py line 1497)
        if (SWING_MODE == 0u) { swing_i = p.w * length(f - load_old(i)); }  // paper: m |F(t) - F(t-1)|, recomputed inline (7.2)
        let factor = S.speed / (1.0 + sqrt(S.speed * swing_i));
        let fixed = mask_bit(fixedMask[i >> 5u], i);
        dp = select(f * factor, vec3f(0.0), fixed);                        // no clamp on dp (D25)
        if (P.dim == 2u) { dp.z = 0.0; }                                   // 2D never integrates z (7.13)
        p = vec4f(p.xyz + dp, p.w);
        pos[i] = p;
        if (SWING_MODE == 0u) { store_old(i, f); }                         // fixed nodes too, so a later unpin sees no stale swing (7.11)
        free = !fixed;
    }
    // uniform control flow from here (3.5 rule 1): partials A (sum p, sum |p - c|^2, min, max over valid rows) and C (sum |dp|, free count)
    let c = S.centroid.xyz;
    var sumv = vec4f(0.0);
    var lo = vec4f(F32_MAX);
    var hi = vec4f(-F32_MAX);
    var dl = 0.0;
    var fr = 0u;
    if (valid) {
        let q = p.xyz - c;
        sumv = vec4f(p.xyz, dot(q, q));
        lo = vec4f(p.xyz, 0.0);
        hi = vec4f(p.xyz, dot(q, q));                                  // max.w carries max |p - c|^2 so K1 can write the exact layoutRadius (spec 3.3)
    }
    if (free) { dl = length(dp); fr = 1u; }
    let tSum = wg_reduce_vec4(sumv, lid.x, 0u);
    let tLo = wg_reduce_vec4(lo, lid.x, 1u);
    let tHi = wg_reduce_vec4(hi, lid.x, 2u);
    let tDl = wg_reduce_f32(dl, lid.x, 0u);
    let tFr = wg_reduce_u32(fr, lid.x, 0u);
    if (lid.x == 0u) {
        let g = group_id(wid);
        partials[g].sum = tSum;
        partials[g].min = tLo;
        partials[g].max = tHi;
        partials[g].dispFree = vec2f(tDl, f32(tFr));
    }
}
```

#### fa2-to-scene.wgsl.ts (`fa2ToSceneWgsl`, entry `to_scene`; spec 7.13, 7.18)

```wgsl
@compute @workgroup_size(WG)
fn to_scene(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = linear_id(wid, lid.x);
    if (i >= P.n) { return; }
    let s = pos[i].xyz * P.scale + P.center.xyz;
    scene[3u * i] = s.x;
    scene[3u * i + 1u] = s.y;
    scene[3u * i + 2u] = select(s.z, P.center.z, P.dim == 2u);       // 2D writes z = center.z on every readback (7.13)
}
```

### 4.6 The two NetworkX corrections (CONTRACT DECISIONS; section 9 items 1-2)

Read from `networkx/drawing/layout.py` at tag `networkx-3.4.2` (the fixture
generator pins `networkx>=3.4`): (a) `forceatlas2_layout`'s gravity is
`-gravity * mass * pos / |pos|` (lines 1466-1471), i.e. toward the ORIGIN,
unguarded, and `strong_gravity` multiplies by `|pos|`; the spec's 7.2 table
says NetworkX shares the port's centroid. `compat: "networkx"` therefore
compiles `GRAVITY_CENTER = 1` (and keeps the `|q| > 0.01` guard, which the
fixtures avoid by construction, 5.4); `compat: "paper"` keeps the centroid
(Q-1). (b) NetworkX's PER-NODE factor uses `swinging = mass * |update|`
(line 1497) while its GLOBAL sums use `mass * |pos - update|` and `0.5 mass
|pos + update|` (lines 1479-1480); the spec's 7.11 sketch used `|p - F|`
for the local factor in mode 1. K5 therefore computes `swing_i = m |F|` in
`SWING_MODE = 1`. Both are one-line `SWING_MODE` / `GRAVITY_CENTER`
consequences, so the paper mode is untouched.

## 5. test/ contracts

Rule of every test (spec 11.2, 11.9): a wrong result is never a skip; every
kernel result is compared to an oracle or an invariant; every kernel test
runs its kernel twice and asserts bitwise equality before comparing; every
`uncapturederror` fails the current test. Test files are in the relaxed
lint block of the root config but keep the ASCII rule and the house style.

### 5.1 test/setup/*.ts

#### test/setup/gpu.ts -- P0-T3 / P1-T1 (spec 11.2, 2.3, D16, D19)

Imports: `../../scripts/gpu-policy.js` (through its `.d.ts`), `../../src/node/index.js` (createNodeGpu), `../../src/context.js` (P1), `../../src/device/acquire.js`, vitest hooks. Reads `process.env`: `GRAPHTY_GPU_REQUIRE`, `GRAPHTY_GPU_ADAPTER`, `GRAPHTY_DAWN_FEATURES`, `GRAPHTY_GPU_NO_SUBGROUPS`, `GRAPHTY_GPU_INSPECT`, `GRAPHTY_PIPELINE_KEY_LOG` (a directory; when set, every context's pipeline keys are appended in `afterAll`), and sets `XDG_RUNTIME_DIR=/tmp` when unset. It is the node projects' `setupFiles` entry AND the module tests import, so one instance per worker.

```ts
/** The parsed policy of GRAPHTY_GPU_REQUIRE (scripts/gpu-policy.js shape). */
export function gpuPolicy(): GpuPolicy;
/** The adapter summary of the probe the setup ran once in beforeAll (null when acquisition failed). */
export function adapterSummary(): AdapterSummary | null;
/** Why acquisition failed ("E_NO_ADAPTER: ..." text), or null. */
export function skipReason(): string | null;
/** Skips the calling test when no adapter exists and the policy is "skip"; THROWS (fails) under any / hardware / <vendor> when the adapter is absent or violates the policy; returns otherwise. */
export function requireGpu(t: TestContext): void;
/** 1 on hardware, 1 / 50 on a software adapter (spec 11.2); scales fixture sizes and iteration counts. */
export function gpuScale(): number;
/** A FRESH adapter and device (raised limits) with NO context around them; registered for destroy() in afterAll. */
export function acquireRaw(options?: { readonly limits?: LimitPolicy | undefined; readonly optionalFeatures?: readonly GPUFeatureName[] | undefined } | undefined): Promise<{ readonly gpu: GPU; readonly adapter: GPUAdapter; readonly device: GPUDevice; readonly info: GPUAdapterInfo }>;
/** Options of acquire(). `subgroups` defaults to GRAPHTY_GPU_NO_SUBGROUPS !== "1"; false -> optionalFeatures []. */
export interface AcquireOptions {
    readonly subgroups?: boolean | undefined;
    readonly limits?: LimitPolicy | undefined;
    readonly label?: string | undefined;
    readonly optionalFeatures?: readonly GPUFeatureName[] | undefined;
    readonly rejectSoftware?: boolean | undefined;
    readonly warnUnreleasedSnapshots?: number | undefined;
}
/** P1: a GpuContext over a FRESH adapter each call (spec 11.2), created with runtime "node", onError collecting into the per-test uncaptured list (the afterEach hook fails the test when it is non-empty), ctx.debug.inspect from GRAPHTY_GPU_INSPECT, registered for dispose() in afterAll. */
export function acquire(options?: AcquireOptions | undefined): Promise<GpuContext>;
/** P1: a GpuContext over Dawn's `backend=null` adapter (spec 5.1: compiles pipelines, runs nothing) from a SECOND GPU handle created lazily once per worker through createNodeGpu({ backend: "null", installGlobals: false }); a fresh adapter per call; disposed in afterAll with the handle; the compile tests (compile.test.ts, wgsl-compile.test.ts) use it beside acquire(). */
export function acquireNullBackend(options?: AcquireOptions | undefined): Promise<GpuContext>;
/** The errors the uncapturederror sink collected during the current test (drained by the afterEach hook). */
export function uncapturedErrors(): readonly WebGpuGraphError[];
/** True when the setup's probe found a software adapter. */
export function isSoftware(): boolean;
```

Hooks the file installs: `beforeAll` (load Dawn once per worker through
`createNodeGpu({ adapter: GRAPHTY_GPU_ADAPTER, dawnFeatures })`, probe one
adapter, print `adapter.info` and the four limits with `console.warn`,
evaluate the policy), `afterEach` (assert `uncapturedErrors().length ===
0` with the messages, then drain), `afterAll` (dispose every context and
device from `acquire` / `acquireRaw` / `acquireNullBackend`, write the
pipeline-key log, drop the GPU handle and the lazily created `backend=null`
handle so the fork exits, spec 11.2).

#### test/setup/browser.ts -- P0-T3 (spec 2.3, 11.6)

Reads `import.meta.env.GRAPHTY_GPU_REQUIRE` / `GRAPHTY_BROWSER_GPU` (2.5).

```ts
/** The policy forwarded by vitest.config.ts. */
export function browserPolicy(): GpuPolicy;
/** "nvidia" | "swiftshader" as forwarded. */
export function browserGpu(): "nvidia" | "swiftshader";
/** navigator.gpu or undefined (never throws). */
export function browserWebGpu(): GPU | undefined;
/** Skips (policy "skip") or fails (any / hardware / vendor) when navigator.gpu or an adapter is absent or the adapter violates the policy; in the browser a vendor policy additionally requires isFallbackAdapter === false (spec 11.2). */
export function requireBrowserGpu(t: TestContext): Promise<void>;
/** A fresh context through requestGpuContext (P1) with the same uncaptured-error hook as the Node setup; disposed in afterAll. */
export function acquireBrowser(options?: BrowserGpuOptions | undefined): Promise<GpuContext>;
/** The browser's gpuScale(): 1 on nvidia, 1 / 50 on swiftshader. */
export function browserScale(): number;
```

Hooks: `afterEach` drains the browser context's pending-error slot after
`device.queue.onSubmittedWorkDone()` and fails the test when an error was
stored (spec 5.7: asynchronous delivery in browsers); `afterAll` disposes.

#### test/setup/global.ts -- P0-T3 (empty) / P2-T2 (spec 5.1: the override-matrix coverage)

```ts
/** Clears the pipeline-key log directory (tmp/pipeline-keys/) and exports its path through GRAPHTY_PIPELINE_KEY_LOG. */
export function setup(): void;
/** Reads every worker's key log and asserts test/helpers/override-matrix.ts covers each key (throws with the uncovered keys, which fails the run). */
export function teardown(): void;
```

#### test/setup/browser-commands.d.ts -- P0-T3 (the three commands of 2.5, structural payload types) / P3-T6 (narrows appendBenchRecord's payload)

P0 form (no imports: benchmarks/harness.ts does not exist yet):

```ts
/// <reference types="vite/client" />
declare module "@vitest/browser/context" {
    interface BrowserCommands {
        appendBenchRecord(payload: { readonly runnerClass: string; readonly session: unknown }): Promise<string>;
        writeNoiseFixture(payload: { readonly kernel: string; readonly fixture: string; readonly adapterClass: string; readonly values: readonly number[]; readonly dtype: "f32" | "u32" }): Promise<string>;
        recordNoiseRow(row: Record<string, unknown>): Promise<string>;
    }
}
interface ImportMetaEnv {
    readonly GRAPHTY_GPU_REQUIRE: string;
    readonly GRAPHTY_BROWSER_GPU: "nvidia" | "swiftshader";
    readonly GRAPHTY_NOISE_FLOOR_WRITE: string;
}
```

P3-T6 edit: `import type { BrowserBenchPayload } from
"../../benchmarks/harness.js";` and `appendBenchRecord(payload:
BrowserBenchPayload)`; `recordNoiseRow(row: NoiseRow)` with `NoiseRow` from
`../helpers/noise-floor.js` is P1-T7's edit when it creates
`noise-floor-browser.ts` (a type import only, so the browser bundle never
pulls `node:fs`).

### 5.2 test/helpers/*.ts

```ts
// test/helpers/device.ts -- P1-T2
/** A STORAGE | COPY_SRC | COPY_DST buffer holding `data` (one writeBuffer). */
export function uploadBuffer(ctx: GpuContext, data: ArrayBufferView, label: string, extraUsage?: number | undefined): GPUBuffer;
/** A zeroed STORAGE | COPY_SRC | COPY_DST buffer of byteLength. */
export function scratchBuffer(ctx: GpuContext, byteLength: number, label: string): GPUBuffer;
/** Whole-buffer Binding. */
export function bindingOf(buffer: GPUBuffer): Binding;
/** Reads a buffer back as U32 / F32 through ctx.readback. */
export function readU32(ctx: GpuContext, buffer: GPUBuffer, count: number, byteOffset?: number | undefined): Promise<U32>;
export function readF32(ctx: GpuContext, buffer: GPUBuffer, count: number, byteOffset?: number | undefined): Promise<F32>;
/** Runs `fn` with a context from acquire(options) and disposes it afterwards (also on throw). */
export function withContext<T>(options: AcquireOptions | undefined, fn: (ctx: GpuContext) => Promise<T>): Promise<T>;

// test/helpers/kernel.ts -- P1-T3 (imports only src/kernel/** and src/context.js, so P1-T2's helpers and P1-T3's stay independent)
/** Records one dispatch of a kernel with the given bindings and params values into a fresh encoder and submits (P1's pre-CommandBatch driver); the params slot is a fresh UNIFORM | COPY_DST buffer of the block's byteLength. */
export function runKernel(ctx: GpuContext, spec: WgslModuleSpec, bindings: KernelBindings, plan: DispatchPlan, params?: { readonly block: UniformBlock; readonly values: UniformValues } | undefined): Promise<void>;

// test/helpers/linear-id.ts -- P1-T4 (the constants the node AND browser 17M-item tests share, spec 11.5)
/** 16,776,961 = MAX_1D_ITEMS + 1: the first item count that needs the 2D dispatch. */
export const LINEAR_ID_ITEMS: number;
/** The fill `value` of the test (a fixed odd constant). */
export const LINEAR_ID_VALUE: number;
/** Sampled indices across the 1D / 2D boundary (0, 1, 255, 256, 65_535 x 256 - 1, 65_535 x 256, LINEAR_ID_ITEMS - 1). */
export const LINEAR_ID_SAMPLES: readonly number[];
/** The u32 checksum (sum mod 2^32 of every word) the fill must produce, recorded once and pinned (bitwise across adapters). */
export const LINEAR_ID_CHECKSUM: number;
/** The checksum of a result array (the same arithmetic the pin was recorded with). */
export function linearIdChecksum(words: Uint32Array): number;

// test/helpers/graphs.ts -- P1-T2 (pure data; copied helpers cite graph-format's parts.ts / gpu-upload.test.ts)
export type EdgeSpec = readonly [number, number] | readonly [number, number, number];
/** Zachary's karate club, 78 edges, node indices 0..33. */
export const KARATE_EDGES: readonly EdgeSpec[];
export function gridEdges(w: number, h: number): EdgeSpec[];
export function pathEdges(n: number): EdgeSpec[];
export function starEdges(leaves: number): EdgeSpec[];
export function cycleEdges(n: number): EdgeSpec[];
export function completeEdges(n: number): EdgeSpec[];
/** Seeded G(n, m) WITHOUT self-loops or parallels (LCG; the gpu-upload.test.ts generator). */
export function randomEdges(n: number, m: number, seed: number): EdgeSpec[];
/** Seeded G(n, m) WITH self-loops and parallels and integer weights 1..10 (the graph-format benchmark generator). */
export function randomEdgesLoose(n: number, m: number, seed: number): EdgeSpec[];
/** Seeded R-MAT-like hub graph (a, b, c, d = 0.57, 0.19, 0.19, 0.05). */
export function rmatEdges(scale: number, edgeFactor: number, seed: number): EdgeSpec[];
/** Options of snapshotOf. */
export interface SnapshotOptions {
    readonly directed?: boolean | undefined;
    readonly nodeCount?: number | undefined;
    readonly weighted?: boolean | undefined;
    readonly arena?: boolean | undefined;
    readonly label?: string | undefined;
}
/** fromEdgeArrays over the edge list (undirected by default; weights present iff some edge carries one or `weighted`). */
export function snapshotOf(edges: readonly EdgeSpec[], options?: SnapshotOptions | undefined): GraphSnapshot;
/** The same graph through fromCsr on SEPARATE arrays (arena === null). */
export function csrSnapshotOf(edges: readonly EdgeSpec[], options?: SnapshotOptions | undefined): GraphSnapshot;
/** The named fixtures of spec 11.3 / 11.4 sized by gpuScale(): "empty", "one", "self-loop", "karate", "grid10", "path1k", "star200", "complete6", "random1k", "hub10k" (a 10k-degree star inside a random graph), "coincident" (karate with two node pairs at equal positions, positions supplied), "isolated" (giant component + 1% isolated + 100 small components), "parallel" (with parallels and zero weights). */
export function fixture(name: string, scale?: number | undefined): { readonly snapshot: GraphSnapshot; readonly positions: F32 | null; readonly name: string };
export const FIXTURE_NAMES: readonly string[];

// test/helpers/matchers.ts -- P1-T2
export interface Tolerance { readonly rel: number; readonly abs: number; }
/** |a - e| <= abs + rel * |e| element-wise; reports the worst index. */
export function expectAllClose(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance: Tolerance, label?: string | undefined): void;
/** Bitwise equality of two typed arrays (Object.is on every element, so NaN patterns and -0 count). */
export function expectBitwiseEqual(a: TypedArrayData, b: TypedArrayData, label?: string | undefined): void;
/** The 11.4 per-node relative error with the floored denominator max(|F_cpu(i)|, floorFraction * max_j |F_cpu(j)|), over stride-3 vectors. */
export function flooredRelError(gpu: ArrayLike<number>, cpu: ArrayLike<number>, floorFraction: number): { readonly max: number; readonly argmax: number; readonly rms: number; readonly p99: number };
/** max_i |a_i - e_i| / max(|e_i|, absFloor). */
export function maxRelError(actual: ArrayLike<number>, expected: ArrayLike<number>, absFloor: number): number;

// test/helpers/leak-counter.ts -- P1-T7 (spec 11.3)
/** Counting proxy around createBuffer / destroy / mapAsync of a device (installed BEFORE GpuContext.from / create sees the device). */
export class LeakCounter {
    static wrap(device: GPUDevice): LeakCounter;
    readonly created: number;
    readonly destroyed: number;
    readonly live: number;
    readonly mapAsyncCalls: number;
    /** Resets the mapAsync counter (per batch bounds). */
    resetMapAsync(): void;
    /** Removes the proxies. */
    restore(): void;
}

// test/helpers/caps-tables.ts -- P1-T2 (lead b; spec 5.2, note 05 section 4); P2-T2 adds CAPS_INTEL_XE
export const CAPS_SPEC_DEFAULT: PlanCaps;    // 256 MiB buffer, 128 MiB binding, 8 storage buffers, 65535, 256 invocations, 16 KiB workgroup storage; no subgroups; software false
export const CAPS_SWIFTSHADER: PlanCaps;     // Chromium SwiftShader: spec defaults, subgroups 4/4, software true
export const CAPS_LAVAPIPE: PlanCaps;        // Dawn llvmpipe: 256 MiB / 128 MiB (not raisable), subgroups 8/8, software true
export const CAPS_NVIDIA_4070: PlanCaps;     // raised: 2 GiB binding, 1 TiB maxBufferSize clamped to the device value recorded at G1, 32/32, software false
export const CAPS_INTEL_XE: PlanCaps;        // subgroups 8/32 (min != max), otherwise defaults (P2)
export const CAPS_TABLES: readonly { readonly name: string; readonly caps: PlanCaps }[];
/** A table with some limits overridden. */
export function fakeCaps(base: PlanCaps, overrides: Partial<PlanLimits>, flags?: Partial<Pick<PlanCaps, "software" | "subgroupMinSize" | "subgroupMaxSize">> | undefined): PlanCaps;

// test/helpers/override-matrix.ts -- P2-T2 (spec 5.1, 11.3)
/** One compile case: an id, an override set and whether the subgroup axis applies. */
export interface OverrideCase {
    readonly id: KernelId;
    readonly overrides: Readonly<Record<string, number | boolean>>;
    readonly snippets?: Readonly<Record<string, string>> | undefined;
    readonly twin: boolean;
}
/** The bounded, explicit table: every entry's defaults, each override toggled alone, and the exact combinations the factories emit. */
export const OVERRIDE_MATRIX: readonly OverrideCase[];
/** True when every key in `keys` (PipelineCache.key strings) is produced by some case of the matrix on `caps`. */
export function matrixCovers(keys: Iterable<string>, caps: PlanCaps): { readonly ok: boolean; readonly missing: readonly string[] };

// test/helpers/sabotage.ts -- P1-T5 (degree, reduce, K3, K4 rows) / P2-T2 (segmented-reduce rows) / P3-T5 (K1, K2, K5 rows) (spec 11.9 item 1, 13 rule f)
/** One named mutation of a kernel body: a unique `find` string replaced by `replace`, and the minimum factor by which it must break the named test's tolerance. */
export interface Mutation {
    readonly name: string;
    readonly find: string;
    readonly replace: string;
    readonly minFactor: number;
    readonly test: string;
}
/** At least three mutations per kernel that has rows (spec 13 rule f); PARTIAL so a phase's kernels can land before its rows (the coverage test below gates by phase). */
export const SABOTAGE: Readonly<Partial<Record<KernelId, readonly Mutation[]>>>;
/** The phases whose kernels ALL have their rows: ["P1"] at P1-T5, + "P2" at P2-T2, + "P3" at P3-T5; test/sabotage/coverage.test.ts asserts every KERNELS entry whose `phase` is listed here has >= 3 rows, except SABOTAGE_EXEMPT. */
export const SABOTAGE_PHASES: readonly ("P1" | "P2" | "P3")[];
/** Kernels with no oracle-sensitive arithmetic to mutate (a wrong fill / toScene fails the exact-equality tests directly): ["fill", "fa2-to-scene"]. */
export const SABOTAGE_EXEMPT: readonly KernelId[];
/** The mutated body (throws when `find` is absent or not unique in the normative body). */
export function sabotagedBody(id: KernelId, mutation: Mutation): string;
/** Installs the mutation through setKernelBodyOverride, runs `fn` with a FRESH context, restores. */
export function withSabotage<T>(id: KernelId, mutation: Mutation, fn: (ctx: GpuContext) => Promise<T>): Promise<T>;

// test/helpers/noise-floor.ts -- P1-T5 (spec 11.9 item 3; Node only: node:fs)
/** Where a per-adapter raw output lives: test/fixtures/noise/<kernel>-<fixture>-<adapterClass>.json. */
export function noiseFixturePath(kernel: string, fixture: string, adapterClass: string): string;
/** Writes this adapter's raw output (only when GRAPHTY_NOISE_FLOOR_WRITE=1) in the same JSON shape as the vitest.config.ts `writeNoiseFixture` command (2.5). */
export function writeNoiseFixture(kernel: string, fixture: string, adapterClass: string, values: ArrayLike<number>, dtype: "f32" | "u32"): void;
/** Every committed adapter output for (kernel, fixture). */
export function readNoiseFixtures(kernel: string, fixture: string): readonly { readonly adapterClass: string; readonly values: Float64Array; readonly dtype: "f32" | "u32" }[];
/** A noise-floor row (5.6 schema); appended to benchmarks/results/noise-floor.json when GRAPHTY_NOISE_FLOOR_WRITE=1 (a row with the same id is replaced). */
export function recordNoiseRow(row: NoiseRow): void;
/** The committed tolerance of a test id and its basis row; throws when the id is unknown (a tolerance without a floor is a finding). */
export function noiseFloorFor(testId: string): { readonly value: number; readonly basis: string };
/** The adapter class string of the running adapter: <vendor>-<architecture>-<runtime>. */
export function adapterClass(caps: GpuCaps): string;

// test/helpers/noise-floor-browser.ts -- P1-T7 (the browser twin of the two writers; imports nothing from node:*)
/** commands.writeNoiseFixture(...) through the Vitest bridge (2.5); resolves the path written, or "" when GRAPHTY_NOISE_FLOOR_WRITE is not "1". */
export function writeNoiseFixtureBrowser(kernel: string, fixture: string, adapterClass: string, values: ArrayLike<number>, dtype: "f32" | "u32"): Promise<string>;
/** commands.recordNoiseRow(row) through the bridge. */
export function recordNoiseRowBrowser(row: NoiseRow): Promise<string>;
/** The same <vendor>-<architecture>-<runtime> string as adapterClass() (duplicated 3-line function; test/browser/skeleton.test.ts asserts it equals the Node form on a faked GpuCaps). */
export function adapterClassBrowser(caps: GpuCaps): string;

// test/helpers/frame-loop.ts -- P3-T6 (spec 7.19, 11.4)
export interface FrameLoopOptions {
    readonly ticks: number;
    readonly iterationsPerStep: number;
    readonly maxInFlight: number;
    /** Stop calling step() at this tick for `pauseTicks` ticks (null: no pause). */
    readonly pauseAt?: number | null | undefined;
    readonly pauseTicks?: number | undefined;
    /** setPosition calls to issue at given ticks. */
    readonly setPositionAt?: readonly { readonly tick: number; readonly index: number; readonly x: number; readonly y: number; readonly z: number }[] | undefined;
    readonly onTick?: ((tick: number) => void) | undefined;
}
export interface FrameLoopReport {
    readonly submissions: number;
    readonly coalesced: number;
    readonly maxObservedInFlight: number;
    readonly iterationsDoneByTick: readonly number[];
    readonly settledAtTick: number | null;
    readonly errors: readonly unknown[];
    /** For each setPosition issued: whether the written coordinates were still in the owner's array at every later tick until a batch submitted after the write landed. */
    readonly positionHolds: readonly { readonly tick: number; readonly index: number; readonly held: boolean }[];
    readonly submissionsDuringPause: number;
}
/** The element's bridge (spec 7.19): one synchronous step(k) per tick, .catch attached once per DISTINCT promise, ticks separated by a macrotask so readbacks land; never awaits step(). */
export function runFrameLoop(sim: GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>, positions: F32, options: FrameLoopOptions): Promise<FrameLoopReport>;

// test/helpers/metrics.ts -- P3-T5 (spec 11.4 distributional parity)
export function stress(s: GraphSnapshot, positions: ArrayLike<number>, dim: 2 | 3): number;
export function edgeLengthQuantiles(s: GraphSnapshot, positions: ArrayLike<number>, dim: 2 | 3, quantiles: readonly number[]): number[];
export function nearestNeighbourHistogram(positions: ArrayLike<number>, n: number, dim: 2 | 3, bins: number): number[];
export function componentSeparation(s: GraphSnapshot, positions: ArrayLike<number>, dim: 2 | 3): number;
export function spread(positions: ArrayLike<number>, n: number, dim: 2 | 3): number;
/** Every metric above as one record, for the "within 10%" comparisons. */
export function layoutMetrics(s: GraphSnapshot, positions: ArrayLike<number>, dim: 2 | 3): Readonly<Record<string, number>>;
```

### 5.3 test/oracle/*.ts (spec 11.3, 11.4, 11.9 item 2)

```ts
// test/oracle/degree.ts -- P1-T5
/** rowPtr differences; equals snapshot.outDegree() by construction (the independent reference is the loop, not the view). */
export function outDegreeOracle(s: GraphSnapshot): U32;

// test/oracle/reduce.ts -- P1-T5
/** f64 sequential reduction of f32 / u32 / vec4f inputs; sum returns the f64 total (tests scale the tolerance by count); min / max exact. */
export function reduceOracle(values: ArrayLike<number>, op: ReduceOp, dtype: ReduceDtype): number | readonly [number, number, number, number];

// test/oracle/segmented-reduce.ts -- P2-T2
/** Per-row f64 reduction of value(row, arc, target, weight) over the CSR rows (the TypeScript callback keeps `target`; only the WGSL snippet vocabulary says `nbr`, 4.5). */
export function segmentedReduceOracle(s: GraphSnapshot, value: (row: number, arc: number, target: number, weight: number) => number, op: ReduceOp): Float64Array;

// test/oracle/forceatlas2.ts -- P3-T4 (the SPEC of the L1 ForceAtlas2Simulation; index-based; the 7.2 table with the 4.6 corrections)
export interface OracleOptions extends ForceAtlas2Options {
    readonly compat: "paper" | "networkx";
    readonly precision: "f64" | "f32";
    /** 0 centroid, 1 origin (defaults from compat: paper -> 0, networkx -> 1). */
    readonly gravityCenter?: 0 | 1 | undefined;
    readonly fixed?: NodeMask | null | undefined;
    readonly mass?: ArrayLike<number> | null | undefined;
    readonly weights?: ArrayLike<number> | null | undefined;
}
/** The per-iteration intermediates the GPU exposes through inspect() (spec 11.9 item 2), stride 3. */
export interface OracleStages {
    readonly attraction: Float64Array;
    readonly repulsion: Float64Array;
    readonly gravity: Float64Array;
    readonly force: Float64Array;
    readonly oldForce: Float64Array;
    readonly swingPerNode: Float64Array;
    readonly tractionPerNode: Float64Array;
    readonly displacement: Float64Array;
    readonly partials: { readonly swing: number; readonly traction: number; readonly sum: readonly [number, number, number]; readonly sumSq: number; readonly min: readonly [number, number, number]; readonly max: readonly [number, number, number]; readonly disp: number; readonly free: number };
}
export interface OracleTraceRecord extends ForceAtlas2TraceRecord {
    readonly rmsRadius: number;
    /** max |p - c| about the same previous centroid as rmsRadius (what K1 writes into state.radius, 4.5). */
    readonly layoutRadius: number;
    readonly centroid: readonly [number, number, number];
}
/** The CPU ForceAtlas2 reference: positions in LAYOUT units (stride 3), f64 scratch (or f32 scratch summing in tile order when precision is "f32"). */
export class ForceAtlas2Oracle {
    constructor(s: GraphSnapshot, positions: ArrayLike<number>, options: OracleOptions);
    readonly n: number;
    readonly dim: 2 | 3;
    /** Layout-unit positions (Float64Array or Float32Array by precision). */
    readonly positions: Float64Array | Float32Array;
    /** The controller state. */
    readonly speed: number;
    readonly speedEfficiency: number;
    readonly swing: number;
    readonly traction: number;
    readonly settledCount: number;
    readonly iteration: number;
    /** One iteration: K1 fold (from the previous stages), K2, K3 + epilogue, K4, K5; returns the trace record. */
    step(): OracleTraceRecord;
    /** The intermediates of the LAST step(). */
    readonly stages: OracleStages;
    /** The trace of every step() so far. */
    readonly trace: readonly OracleTraceRecord[];
    setFixed(mask: NodeMask | null): void;
    setPosition(index: number, x: number, y: number, z: number): void;
    /** reheat() semantics of D8 (settledCount = 0; mode 1: swing = traction = 1). */
    reheat(): void;
}
/** Positions in SCENE units for a snapshot: seedPositions into a fresh F32 (the same LCG start as the GPU, spec 11.4 "same start"). */
export function seededScenePositions(s: GraphSnapshot, seed: number | null, dim: 2 | 3, scale: number, center: ArrayLike<number> | null): F32;
/** Runs the oracle for `iterations` from a scene-unit array and returns the scene-unit result (the toScene inverse applied). */
export function forceAtlas2Oracle(s: GraphSnapshot, scenePositions: F32, options: OracleOptions, iterations: number): { readonly positions: F32; readonly trace: readonly OracleTraceRecord[]; readonly oracle: ForceAtlas2Oracle };
```

Oracle contract: implements exactly the 7.2 table as the WGSL of section
4.5 does (one law, `max(d, 0.01)` floor, the coincident kick along the same
`kick_dir` hash re-implemented in TypeScript, gravity by `gravityCenter`,
`SWING_MODE` by `compat`, fresh sums over free nodes in paper mode,
accumulated sums over every node in networkx mode, `estimateFactor` line
for line, no displacement clamp, 2D z never integrated); mass default
`outDegree()[i] + 1`; the settle rule of 7.17; the RMS radius and the
layout radius (max |p - c|) from the previous centroid exactly as K1
computes them. With `precision: "f32"` every
accumulation is `Math.fround`ed in the GPU's tile / lane order (the tight
leg of the trace test, 11.4).

### 5.4 test/fixtures/networkx/ (spec 11.4 oracle independence)

`generate.py` (Python 3.10+, `networkx>=3.4` from the venv `tmp/nx-venv`;
the suite never runs Python): CLI `python generate.py --out
test/fixtures/networkx/ [--graph karate|grid10|star200|gnm200] [--iters 1 5
50] [--seed 7]`; for every graph x variant (`base`, `linlog`,
`distributed`, `strong`, `weighted`, `gravity0`) x iteration count it
writes `<graph>-<variant>-iter<k>.json`, passing the SAME initial
positions through NetworkX's `pos` argument, with weights as an edge
attribute for the weighted variant. The generator asserts: no self-loops,
no parallel edges (NetworkX degree would differ from `outDegree()`); the
early exit did not fire (the `k+1` run differs from the `k` run); no node
came within 0.02 of the origin and no pair within 0.02 of each other in
the initial positions (the floor / guard regions); the recorded NetworkX
version is `>= 3.4`. Fixture JSON schema (one file):

```json
{
    "generator": "test/fixtures/networkx/generate.py",
    "networkxVersion": "3.4.2",
    "pythonVersion": "3.10.12",
    "command": "python generate.py --out ... --graph karate --iters 1 5 50 --seed 7",
    "graph": { "name": "karate", "directed": false, "nodeCount": 34, "src": [0, 0], "dst": [1, 2], "weights": null },
    "options": { "max_iter": 5, "jitter_tolerance": 1.0, "scaling_ratio": 2.0, "gravity": 1.0, "distributed_action": false, "strong_gravity": false, "linlog": false, "dissuade_hubs": false, "weight": null, "dim": 2 },
    "initialPositions": [[0.1, -0.3], [0.5, 0.2]],
    "positions": [[0.12, -0.31], [0.49, 0.21]],
    "rescaled": false
}
```

`src` / `dst` are index pairs in edge order, `weights` an array or null,
positions node-index order, `dim` 2 in every committed fixture (a 3D karate
fixture is added for the 3D parity leg). CONTRACT DECISION: positions are
compared RAW (NetworkX 3.4.2 does not rescale `forceatlas2_layout`'s
output; the spec's "after rescaling on both sides" is therefore a no-op
and is not performed).

### 5.5 Test files per phase (what each asserts)

P0 (all P0-T3):

| File | Project | Asserts |
| --- | --- | --- |
| `test/errors.test.ts` | node | WebGpuGraphError: code, message, frozen details copy, name, isWebGpuGraphError / hasErrorCode across a structural clone; PASSTHROUGH_FORMAT_CODES contents |
| `test/device/constants.test.ts` | node | after createNodeGpu installed the globals, BufferUsage / MapMode / ShaderStage equal GPUBufferUsage / GPUMapMode / GPUShaderStage field by field |
| `test/device/acquire.test.ts` | node | requireGpu; acquireRaw gives an adapter; two acquireRaw calls give distinct adapters; isSoftwareAdapter(info) matches the policy expectation (hardware / nvidia -> false, GRAPHTY_GPU_ADAPTER=llvmpipe -> true); summarizeAdapter shape; info and the four limits printed |
| `test/device/policy.test.ts` | node | parseGpuRequire over unset / "" / any / hardware / nvidia / intel; checkAdapter over faked infos (software under any -> ok; under hardware -> not ok; vendor mismatch; browser isFallbackAdapter); scripts/gpu-policy.js's isSoftwareInfo agrees with src isSoftwareAdapter on a 6-row table; scripts/runner-class.js's runnerClass over a 4-row table (NVIDIA "NVIDIA GeForce RTX 4070 SUPER ... 580.173.02" -> `nvidia-ada-lovelace-driver580`, lavapipe -> `mesa-software-driver<mesa major>`, a description with no digits -> `...-driver0`, a vendor with a space -> `_`) and the GRAPHTY_RUNNER_CLASS override returning the env value verbatim (6.1, 6.6) |
| `test/browser/webgpu-check.test.ts` | browser | navigator.gpu present; requestAdapter non-null; isSoftwareAdapter vs browserPolicy (nvidia -> false and isFallbackAdapter false); import.meta.env.GRAPHTY_BROWSER_GPU equals the flag set in use |
| `test/index.test.ts` | node | the value export list of 3.15 (P0 list) exactly; no default export |
| `test/build-output.test.ts` | node | package.json shape (name, type, main, types, sideEffects false, exports keys == Object.keys(ENTRIES) with "." for the root, types condition first, no require), files, publishConfig, engines; devDependencies include webgpu (exact 0.4.0) and @vitest/browser; tsconfig.build stripInternal true; dist checks (P1-T7 adds the specifier assertions; hard-fail under CI when dist is absent, skip locally) |
| `test/layers.test.ts` | node | the import graph of src/**/*.ts parsed with ts.createSourceFile (NOT ts.preProcessFile, which does not distinguish `import type`): VALUE edges only -- ImportDeclaration / ExportDeclaration nodes with `isTypeOnly` and type-only specifiers are skipped -- has no cycle (the three declared type-only pairs buffer-pool <-> lease, batch <-> profiler, types/layout <-> types/accelerator are listed in the test with the side that must stay `import type`, and a second assertion checks that side IS type-only); every relative import (type edges included) respects the layer order of 2.4 (a table in the test); src/wgsl/** imported only by src/kernels.ts; src/browser and src/node imported by nothing else in src; "navigator" appears only under src/browser; "process." appears only under src/node; the string `"webgpu"` as an import specifier appears only in src/node/index.ts inside `import(`; the literal `caps.software` (a grep) appears in src/ only in src/kernel/dispatch.ts (planGridStride) -- `summary.software` / `info.software` in src/device/acquire.ts, src/context.ts (probe's rejectSoftware) and the AdapterSummary writers are outside the rule (spec 2.4 restricts `caps.software`); `ESLint.calculateConfigForFile("src/types/context.ts")` yields `@typescript-eslint/no-restricted-imports` options carrying BOTH `paths` (the CPU ban) and `patterns` (the types zone), and the same for `src/types/accelerator.ts` with `allowTypeImports` on its paths (2.4: flat config replaces rule options, so a later block must not clobber the zone) |
| `test/types/public-api.test-d.ts` | strict-consumer tsc (not vitest) | imports the three entries from their package names; constructs a WebGpuGraphError; narrows a code; uses the constants; under noUncheckedIndexedAccess + exactOptionalPropertyTypes |

P1:

| File | Task | Asserts |
| --- | --- | --- |
| `test/device/context.test.ts` | P1-T1 | probe (OK / software / rejectSoftware -> E_SOFTWARE_ONLY, gpu undefined -> E_NO_WEBGPU); create with gpu / adapter / device; "raise" gives >= default limits; an explicit limit above the adapter -> E_NO_DEVICE { reason: "limit" }; a consumed adapter -> E_NO_DEVICE { reason: "consumed" }; optionalFeatures [] -> features lacks subgroups; caps from device.limits (never the adapter's 1 TiB); runtime tag; from(device) runtime "unknown"; state transitions; dispose idempotent; assertReady after dispose -> E_DISPOSED; onError sink receives an uncaptured error from a deliberately broken bind group (and the afterEach hook does not fire because onError consumed it) |
| `test/device/lost.test.ts` | P1-T1 | device.destroy() mid-readback -> the pending read rejects E_DEVICE_LOST, ctx.state "lost", residency cleared without uncaptured errors; a new context from a fresh adapter uploads and runs degree afterwards |
| `test/device/error-scope.test.ts` | P1-T1 | withValidationScope surfaces a bad bind group as E_VALIDATION with the label; AllocationTracker: below the threshold no scope, above it the pop is collected, check() resolves; resident / liveBuffers bookkeeping; formatCompilationInfo subtracts the prelude |
| `test/node/entry.test.ts` | P1-T1 | createNodeGpu / createNodeGpuContext / probeNodeWebGpu; dawnFlags mapping; loadModule rejecting -> E_NO_WEBGPU with the install hint; installGlobals false leaves globalThis untouched; ctx.dispose() disposes the handle (the process-exit rule cannot be asserted, documented) |
| `test/memory/upload-plan.test.ts` | P1-T2 | planUpload over CAPS_TABLES x { arena, no arena } x { fits, exceeds buffer, exceeds binding }: the hot prefix by default, the full arena when a cold segment is needed and fits, perArray when the arena exceeds maxBufferSize, windowed when an array exceeds the binding; the 10M / 100M arithmetic of spec 4.2's table (25 windows); planArcWindows: start % 64 === 0 via `%`, a hub row split across windows, a synthetic rowPtr above 2^31 arcs; arcsPerWindowFor |
| `test/memory/residency.test.ts` | P1-T2 | the upload contract of spec 11.3 ported from gpu-upload.test.ts: arena bindings equal CPU views; perArray on fromCsr and transpose(); arena.byteOffset !== 0 from fromBytes (rich-v1.gsnp); packed u8 / bool columns through column(); identity permutations never materialised (byteLength({ views: true }) unchanged after core()); the same array object uploads once (view "outDegree" twice -> one buffer); column version bump re-uploads in place; ctx.release destroys every buffer (stats().buffers 0, allocator.liveBuffers 0, no uncaptured error) and trims the pool (pool.idleBytes === 0 after a scratch acquire / release preceded it); E_RELEASED on the next core() of a live user (isReleased); release idempotent and safe on an unknown snapshot; siblings via withColumns share one record; the once-only warning at warnUnreleasedSnapshots + 1 |
| `test/memory/buffer-pool.test.ts` | P1-T2 | sizeClass table (4 KiB .. 64 MiB powers of two, then 16 MiB steps); acquire / release reuse; maxIdlePerClass eviction; trim; liveBytes / idleBytes; E_TOO_LARGE above maxBufferSize |
| `test/memory/readback.test.ts` | P1-T2 | 100 back-to-back read() calls reuse slots without validation errors; chunking above slotBytes; readU32; dest given -> dest.buffer returned; borrowSlot grows the ring when every slot is busy and returnSlot frees it |
| `test/kernel/wgsl.test.ts` | P1-T3 | composeWgsl emits the exact format of 4.2 for a two-binding spec; unknown override -> E_SHADER_COMPILE { stage: "compose" }; snippet without marker and marker without snippet -> E_SHADER_COMPILE; a body containing `@group(` or `override ` -> E_SHADER_COMPILE; a body or snippet using a WGSL reserved word (`let target = 0u;`) -> E_SHADER_COMPILE { slot: "reserved:target" } while a comment containing the word passes; the subgroup block is spliced iff needs and caps agree, with SUBGROUP_MAX and SUBGROUP_MIN set (fakeCaps min 8 / max 32 -> SG_SLOTS >= WG / 8 from the composed overrides; min 0 -> SUBGROUP_MIN 4); the literal grep over src/wgsl/** and src/kernel/prelude.ts (no `65535u`, `256u`, `0xFFFFFFFFu`) |
| `test/kernel/struct-block.test.ts` | P1-T3 | offsets of a mixed block (u32 x5, f32, vec4f, vec2f) equal the WGSL strict layout; byteLength padded to 16; write / read round trip; padTo; E_INVALID_ARGUMENT cases; the FA2 blocks' offsets equal 3.10.2 |
| `test/kernel/pipeline-cache.test.ts` | P1-T3 | key format and stability (JSON key order independent); get twice -> one pipeline; a body with a WGSL error -> E_SHADER_COMPILE { stage: "compile", messages } with body-relative lines; warm compiles every spec once; keys() |
| `test/kernel/kernel.test.ts` | P1-T3 | bind creates one group per layout incl. empty ones, labelled `<id>/<group>`; the same buffers -> the same bind groups; a missing name -> E_INVALID_ARGUMENT; one buffer bound to a `storage` slot and to any other slot of the same call with intersecting ranges -> E_INVALID_ARGUMENT { argument: "aliasing" } synchronously (disjoint ranges of one buffer are accepted); a wrong-size uniform binding -> bind() returns, and the E_VALIDATION carrying the `<id>/<group>` label is delivered through the pending-error slot: under Dawn-node thrown by the next assertReady() after a submit (P2: the batch's readback), in the browser (browser/batch.test.ts, P2-T1) by assertReady() after onSubmittedWorkDone(); dispatch of an empty plan records nothing |
| `test/kernel/dispatch.test.ts` | P1-T3 | plan1d(16_776_960) is 1D and plan1d(16_776_961) is 2D with wg 256 (NOT 2^24); plan1d(0) -> x 0; y above the limit -> E_TOO_LARGE; plan2d; planGridStride / planIndirect -> E_UNSUPPORTED; groupsOf; every CAPS_TABLES entry incl. CAPS_INTEL_XE gives the same plans (no dependence on subgroup sizes) |
| `test/kernel/registry.test.ts` | P1-T4 | every KERNELS entry has a unique id, one entry point (entryPointOf agrees), <= 8 storage bindings per stage, the binding tables of 3.10.1 exactly; no entry's `body` contains `@group(` or `override ` (a direct string assertion over every entry, independent of the composer -- the unit test of spec 3.5); kernelSpec applies overrides / snippets; setKernelBodyOverride changes the body seen by kernelSpec; graphBindings / graphOverrides dummy rules on a weighted, an unweighted and an arcCount === 0 core, and with an explicit `weights` argument (null on a weighted core -> HAS_WEIGHTS false and colIdx in the slot; a binding on an unweighted core -> HAS_WEIGHTS true) |
| `test/kernel/compile.test.ts` | P1-T4 | every P1 entry compiles with its defaults on the real device (acquire()) AND on Dawn `backend=null` (acquireNullBackend(), 5.1), with and without subgroups (twin), through PipelineCache |
| `test/kernel/linear-id.test.ts` | P1-T4 | fill mode 1 over LINEAR_ID_ITEMS = 16,776,961 items (2D dispatch, 68 MB) on the current adapter: the LINEAR_ID_SAMPLES positions across the 1D / 2D boundary equal i + LINEAR_ID_VALUE and linearIdChecksum(result) === LINEAR_ID_CHECKSUM (test/helpers/linear-id.ts, shared with the browser skeleton so the two cannot drift; bitwise across adapters) |
| `test/primitives/reduce.test.ts` | P1-T5 | sum / min / max x f32 / u32 / vec4f over sizes 0, 1, 255, 256, 257, 4097, 65_536 x 256 + 1 (three levels; scaled by gpuScale) vs reduceOracle (u32 exact; f32 within count x 2^-24 relative); both twins in-process (acquire({ subgroups: false })) within 1e-6; two runs bitwise identical; lastDispatches 2 or 3 as expected; the f32 sum over random1k written as a cross-adapter noise fixture (`reduce-random1k-<class>.json`) and compared within the derived floor (u32 bitwise) |
| `test/algorithms/degree.test.ts` | P1-T5 | every named fixture incl. empty / one / self-loop / parallel / arcCount 0 / directed vs outDegreeOracle bitwise; dest honoured and returned; a wrong dest -> E_INVALID_ARGUMENT; signal pre-aborted -> E_ABORTED; twice bitwise; cross-adapter noise fixture written / compared bitwise (u32); a fromCsr snapshot (perArray) equal to the arena path |
| `test/sabotage/degree.test.ts` | P1-T5 | the SABOTAGE.degree mutations (`n` -> `n - 1` last workgroup skip: `P.end` -> `P.end - 1u`; swapped USE_PERM select; ignored rebase `arc - P.arcBase` -> `arc`; `nbr < P.n` -> `nbr <= P.n` on a fixture with a neighbour == n impossible, so replaced by counting `2u`) each FAIL the degree test |
| `test/sabotage/reduce.test.ts` | P1-T5 | the SABOTAGE.reduce mutations (the FINAL level writes `out[0]` instead of `out[P.outOffset + g]`, i.e. `P.outOffset + g` -> `g`; the min identity swapped: `return U32_MAX;` -> `return 0u;` and `return F32_MAX;` -> `return 0.0;`; the level-1 bound `i < P.count` -> `i <= P.count`) each FAIL the reduce test by >= minFactor |
| `test/sabotage/coverage.test.ts` | P1-T5 | for every KERNELS entry whose `phase` is in SABOTAGE_PHASES and whose id is not in SABOTAGE_EXEMPT, SABOTAGE[id] has >= 3 rows, every `find` occurs exactly once in the entry's body, every `test` names an existing test file (spec 13 rule f; the P2 / P3 tasks extend SABOTAGE_PHASES when their rows land) |
| `test/layouts/skeleton.test.ts` | P1-T6 | one exact-tile iteration (K3 then K4) on karate in both modes from host-written state, fixed mask all clear: force after K3 equals the f64 pair sum (k m m / d law + gravity) within 1e-5 relative / 1e-6 absolute; the trace slot's (swing, traction, speed, speedEfficiency) equal hand-computed values from the same forces (estimateFactor in TypeScript); twice bitwise; twin within 1e-6; cross-adapter within 1e-5 through the noise fixtures |
| `test/sabotage/fa2-skeleton.test.ts` | P1-T6 | SABOTAGE["fa2-repulsion-exact"] (gravity sign flipped; `k / d2` -> `k / (d2 * length(d))` i.e. 1/d^2; the `jj != i` guard removed; mass lane `.w` -> `.x`) and SABOTAGE["fa2-speed-finalize"] (`* 0.5` -> `* 0.9`; the 1.3 rise removed; the swing / traction swap) each fail the skeleton test by >= 10x its tolerance |
| `test/noise-floor.test.ts` | P1-T6 | for each (kernel, fixture) of the noise set: writes this adapter's raw output when GRAPHTY_NOISE_FLOOR_WRITE=1, compares against every committed adapter output (u32 bitwise; f32 max relative error recorded as a row), and asserts every tolerance in noise-floor.json is >= its basis row and <= 10x it |
| `test/leak.test.ts` | P1-T7 | LeakCounter around a fresh device: after degree + release + dispose, live === 0; mapAsync count per degree call === 1 |
| `test/browser/entry.test.ts` | P1-T7 | probeBrowserWebGpu (software flag vs the policy); requestGpuContext; GpuContext.probe({ gpu: undefined }) -> E_NO_WEBGPU |
| `test/browser/skeleton.test.ts` | P1-T7 | the 11.5 skeleton in the browser: arena + fromCsr upload; degree on karate, random1k and arcCount 0 vs outDegreeOracle bitwise; fill mode 1 over LINEAR_ID_ITEMS (the 2D dispatch), the LINEAR_ID_SAMPLES equal i + LINEAR_ID_VALUE and linearIdChecksum(result) === LINEAR_ID_CHECKSUM (the SwiftShader / NVIDIA-Chromium leg of spec 11.5's "17M-item map bitwise across adapters"); the K3 + K4 iteration on karate in paper mode from host-written state: force vs the f64 pair sum and the trace slot vs the hand-computed controller values within the noise-floor tolerance (`fa2-skeleton.force`), twice bitwise; asserts `ctx.caps.features.has("subgroups")` (both CI browser adapters expose it: SwiftShader size 4, NVIDIA 32) so the subgroup twin is the form exercised (spec 11.6 item 5; the workgroup twin is covered by browser/compile-matrix.test.ts at P2); writes the SwiftShader / NVIDIA-Chromium noise fixtures for degree (random1k), reduce (random1k f32 sum), the 17M map and the K3 + K4 force / trace through writeNoiseFixtureBrowser when GRAPHTY_NOISE_FLOOR_WRITE=1; adapterClassBrowser equals adapterClass on a faked GpuCaps; release leaves no buffers |
| `benchmarks/*` | P1-T7 | see section 6 |

P2:

| File | Task | Asserts |
| --- | --- | --- |
| `test/memory/lease.test.ts` | P2-T1 | storage / uniform / acquire go back to the pool on release (also on a thrown error in a try / finally pattern); count; E_DISPOSED after release |
| `test/kernel/batch.test.ts` | P2-T1 | a batch of 8 dispatches submits once; readback resolves the bytes of two requests at their offsets; a deliberately bad bind group rejects THE SAME batch's readback with E_VALIDATION { batchId } under Dawn-node (synchronous delivery) and is thrown from the next assertReady() in the browser (browser/batch.test.ts); discard() returns the slot after mapAsync (readback.borrowed returns to 0); generation carried; submit twice -> E_INVALID_ARGUMENT; dispatches count |
| `test/kernel/uniform-ring.test.ts` | P2-T1 | offsetOf(k) === 256 k; reserve wraps; write + flush + a kernel reading slot k sees the right values through the dynamic offset; count > slots -> E_INVALID_ARGUMENT |
| `test/kernel/profiler.test.ts` | P2-T1 | with timestamp-query granted: a pass timing per pass label, ns > 0, quantised false under Dawn; without the feature: enabled false and every method a no-op |
| `test/kernel/state-roundtrip.test.ts` | P2-T1 | Fa2State written by the host through FA2_STATE.write, incremented by a one-workgroup kernel, read back through FA2_STATE.read: every field round-trips at the 3.10.2 offsets (both runtimes: browser/state-roundtrip.test.ts) |
| `test/device/lost.test.ts` (extended) | P2-T1 | mid-batch device loss rejects the batch readback with E_DEVICE_LOST; the residency warning; residentBytes accounting |
| `test/browser/lost.test.ts` | P2-T1 | the browser leg of spec 5.7 / 11.3 "both runtimes": device.destroy() mid-batch (a submitted CommandBatch whose readback is pending) rejects that readback with E_DEVICE_LOST, ctx.state is "lost", residency cleared without uncaptured errors, every registered onLost listener ran; a new context from requestGpuContext() (a fresh adapter) uploads karate and runs degree afterwards; the afterEach pending-error drain sees nothing |
| `test/memory/residency.test.ts` (extended) | P2-T1 | residentBytes equals the sum of uploaded byte lengths; stats().perSnapshot |
| `test/browser/uniform-layout.test.ts` | P2-T1 | a hand-written misaligned uniform struct (a vec3f followed by an f32 declared with @size(12)) is REJECTED by Chromium's createShaderModule / pipeline creation while UniformBlock's generated text for the same fields compiles (spec 5.3, R-10); the same test on Dawn-node (kernel/uniform-layout.test.ts) records that Dawn accepts it |
| `test/primitives/segmented-reduce.test.ts` | P2-T2 | thread-per-row sum / min / max with `v = weight;` and `v = 1.0;` vs segmentedReduceOracle over the fixtures incl. all-equal weights, one hub row (star 10k scaled), empty rows, arcCount 0; USE_PERM false always; tiers !== null -> E_UNSUPPORTED; twice bitwise; a bad snippet identifier (incl. `target`) -> E_SHADER_COMPILE; the weighted f32 sum over random1k written as a cross-adapter noise fixture (`segmented-reduce-random1k-<class>.json`) and compared within the derived floor |
| `test/sabotage/segmented-reduce.test.ts` | P2-T2 | the SABOTAGE["segmented-reduce"] mutations (the row bound off by one: `row >= P.end` -> `row > P.end`; the `//@@VALUE@@` weight read replaced: `weight = weights[arc - P.arcBase]` -> `weight = 1.0`; the HAS_WEIGHTS select inverted: `if (HAS_WEIGHTS)` -> `if (!HAS_WEIGHTS)`) each FAIL the segmented-reduce test by >= minFactor; SABOTAGE_PHASES gains "P2" in the same task so coverage.test.ts covers the entry |
| `test/kernel/wgsl-compile.test.ts` | P2-T2 | every OVERRIDE_MATRIX case compiles on the real device (acquire()) and on Dawn backend=null (acquireNullBackend()); the browser twin (browser/compile-matrix.test.ts) on Chromium; the matrix is bounded (a fixed count asserted) |
| `test/kernel/bind-group-budget.test.ts` | P2-T2 | every bind-group-layout descriptor of every KERNELS entry has <= 8 storage entries per stage; the counts of 3.10.1 exactly |
| `test/setup/global.ts` teardown | P2-T2 | every PipelineCache key seen by the node suite is covered by OVERRIDE_MATRIX |
| `test/kernel/determinism.test.ts` | P2-T2 | reduce and segmented-reduce: two runs bitwise identical on both twins; the subgroup form's slot-sorting verified by a kernel that records the slot order (an ad hoc spec built in the test) |
| `test/limits/README.md` | P2-T2 | placeholder naming the P4 tests (no test file) |

P3:

| File | Task | Asserts |
| --- | --- | --- |
| `test/layouts/force-simulation.test.ts` | P3-T1 | with a FAKE ForceModel (one kernel that adds a constant to every position, a trivial state block): state machine transitions and every error of 3.13 (E_NOT_LOADED, E_DISPOSED, E_RELEASED after release during a live simulation, E_SNAPSHOT on a directed input, E_INVALID_ARGUMENT cases); coalescing at maxInFlight (the coalesced counter); maxInFlight 3 with step(256) three times back to back (the fake model writes its params slot value into every position): each batch's readback shows ITS OWN params, i.e. the ring never rewrote a slot a submitted batch still read; setParams({ maxInFlight }) -> E_INVALID_ARGUMENT; generation discard on load() with a batch in flight; the override list (a setPosition during flight survives the older batch); settled at maxIter and at settleWindow; reheat resets only the two counters; flush(); run() with signal -> E_ABORTED; dispose leak 0 (LeakCounter); device loss -> disposed and pending promises E_DEVICE_LOST |
| `test/layouts/seed.test.ts` | P3-T1 | Lcg constants and the first 10 draws of seed 42 equal the port's values (hard-coded); seed 0 / null draws a random seed; seedPositions: all-NaN rows -> [-1, 1) x scale + center in index order, partial rows keep finite axes, the bbox rule when some rows are finite, 2D writes center.z, E_INVALID_ARGUMENT cases |
| `test/layouts/inputs.test.ts` | P3-T1 | resolveNodeMass: null with a role-mass column (f32, f64, u32), null without -> outDegree + 1, F32 as is, name -> column, Record -> E_UNSUPPORTED with the hint, wrong length / non-numeric / non-positive -> E_INVALID_ARGUMENT; resolveWeights: true weighted / unweighted, a column name (f32 and f64) expanded through expandEdges, null; E_INVALID_ARGUMENT on a missing column; pass-through E_GPU_INELIGIBLE on a string column |
| `test/layouts/fa2-options.test.ts` | P3-T2 | resolveForceAtlas2Options defaults and every E_INVALID_ARGUMENT range; nodeSize -> E_UNSUPPORTED { option }; repulsion "grid" / auto above exactMaxNodes -> E_UNSUPPORTED { feature } at load; dissuadeHubs ignored; setParams({ dim }) and setParams({ maxInFlight }) -> E_INVALID_ARGUMENT; setParams of a law recompiles and resets speed, a numeric tweak does not; arcCount 0 runs (force zeroed, positions only move by gravity); the weights axis (3.10 CONTRACT DECISION): a weighted snapshot with `weight: false` produces the same one-iteration positions as the same graph built unweighted (bitwise on one device: both compile HAS_WEIGHTS false), `weight: true` on it differs, and an UNWEIGHTED snapshot with `weight: "<f32 edge column>"` equals the weighted snapshot carrying the same values as arc weights (bitwise); stats shape (grid fields null, repulsionTier "exact", trace length k; layoutRadius and rmsRadius equal the f64 oracle's `layoutRadius` / `rmsRadius` of the same iteration -- both about the previous centroid, 4.5 K1 -- within the traced tolerance) |
| `test/oracle/forceatlas2-networkx.test.ts` | P3-T4 | the oracle in compat "networkx" (gravityCenter 1, f64) reproduces every committed fixture: 1e-9 at 1 and 5 iterations, 1e-6 at 50, raw positions; the iteration-0 FORCES of compat "paper" equal the networkx run's (same laws) on every fixture; the fixtures' preconditions (no pair under 0.01, no node under 0.01 of the origin along the trajectory) hold in the oracle's run |
| `test/oracle/swing-mode.test.ts` | P3-T4 | on a three-node path with hand-computed forces: paper-mode per-node swing / traction (force form, fixed node excluded), networkx-mode (position-mixed, accumulated from 1, every node), the local factor in both modes (m|F - Fold| vs m|F|), and estimateFactor's outputs for two iterations, all against hand-written numbers in the test |
| `test/layouts/fa2-force-parity.test.ts` | P3-T5 | one iteration K2 + K3: `force` via inspect() vs the f64 oracle's `force` stage with the floored denominator at 1e-4 (DEPARTURE-6) on karate, grid10, star200, random1k x { weights, linlog, distributed, strong, gravity 0, nodeMass F32, 2D / 3D, compat paper / networkx, one pinned node }; tolerance traced to noise-floor.json |
| `test/layouts/fa2-trace-parity.test.ts` | P3-T5 | 50 x step(1) on 10-1,000-node graphs in both modes, twice bitwise; CONTRACT DECISION (P3-T5 PLAN DECISION 17; G3.md G3-F3): in BOTH modes the RE-SYNCHRONISED legs -- a fresh f32 and a fresh f64 oracle seeded with the GPU's iteration-start state before every iteration (`ForceAtlas2Oracle.resync()`), K4's controller fields and K1's fold of the same iteration within `fa2-trace-parity.resync.f32` / `.resync.f64` (cap 1e-4 each) -- and the free-running legs of spec 11.4 (the trace vs the f32 oracle within 1e-4 for the first 10 iterations, vs the f64 oracle within 5e-2 through 50) asserted in `compat: "networkx"` and printed in `compat: "paper"`, whose free-running trajectory is chaotic beyond any derivable tolerance (the f64 oracle misses both caps against itself under a one-ulp start perturbation); tolerances traced |
| `test/layouts/fa2-distributional.test.ts` | P3-T5 | same seed, 100 iterations: layoutMetrics of GPU vs the f64 oracle within 10% (coordinates never compared) on the cases whose metrics the f64 oracle reproduces within a third of the cap under one-ulp start perturbations (CONTRACT DECISION, P3-T5 PLAN DECISION 18; G3.md G3-F4: the 10 x 10 grid and the paper-mode isolated fixture land in different basins run to run and are not cases) |
| `test/layouts/fa2-behaviour.test.ts` | P3-T5 | the behaviour pins of 11.4: empty graph (load + step resolve, no GPU work), single node, disconnected components separated by > 0.03 after 100 iterations, maxIter respected, completeGraph(6) spread > 0.3, same seed -> bitwise same layout on the same device, different seeds differ; z === center.z in 2D whatever z was uploaded |
| `test/layouts/fa2-properties.test.ts` | P3-T5 | fast-check (numRuns 200): fixed nodes never move (random setFixed between steps incl. the all-fixed mask, which settles within settleWindow steps); setPosition visible in the next readback and never clobbered by an older batch; settled within maxIter; reheat on unpin / setPosition / load, not on pin; speed NOT reset by setPosition (D8: untouched at the call, and the next iteration continues from it -- proved through the re-synchronised oracle, P3-T5 PLAN DECISIONS 13 / 17); pin A, remove B < A, load(next) with the remapped array and a re-issued mask -> A still fixed; results ArrayBuffer-typed of exact length; per-node displacement <= speed |F| / (1 + sqrt(speed swing_i)) |
| `test/layouts/fa2-force-sum.test.ts` | P3-T5 | gravity 0 and distributedAction false: after one iteration |sum_i F_i| <= 1e-4 sum_i |F_i| on every fixture incl. "coincident" |
| `test/layouts/fa2-twins.test.ts` | P3-T5 | K1 / K3 / K4 / K5 with and without subgroups in-process: every stage's output of one iteration, and one iteration from each of ten oracle-trajectory states, within the traced twin tolerances (1e-6 relative for the trace record); the workgroup twin's 50-iteration trajectory against the re-synchronised oracles; the free-running 10-iteration trace within 1e-6 in `compat: "networkx"`, printed in `compat: "paper"` (CONTRACT DECISION, P3-T5 PLAN DECISION 17) |
| `test/layouts/fa2-lifecycle.test.ts` | P3-T5 | dispose leak 0; release(snapshot) during a live simulation -> the next step rejects E_RELEASED; residency.stats().snapshots === 1 after load + release of a previous snapshot; device loss mid-run |
| `test/layouts/fa2-inspect.test.ts` | P3-T5 | debugRunStages("K2") then inspect("force") equals the oracle's attraction stage; after "K3" the force stage and the partials' swing / traction; after "K4" the state's controller fields; after "K5" positions, partials A / C; after "toScene" scenePositions; each at the traced tolerance |
| `test/sabotage/fa2.test.ts` | P3-T5 | SABOTAGE["fa2-stats-finalize"] (settledCount never reset; centroid / n -> / (n - 1); rmsRadius without the sqrt), SABOTAGE["fa2-attraction"] (the self-loop `continue` removed on the self-loop fixture; `d * mag` -> `-d * mag`; LINLOG select arguments swapped), SABOTAGE["fa2-integrate"] (dp clamp inserted; `store_old` skipped; 2D z integrated), plus the P1 K3 / K4 mutations against the P3 parity tests (the trace check being the re-synchronised legs of fa2-trace-parity.test.ts over 50 iterations, PLAN DECISION 17): each fails its named test by >= minFactor x the tolerance |
| `test/layouts/frame-loop.test.ts` | P3-T6 | runFrameLoop 600 ticks on random1k: submissions <= ticks, maxObservedInFlight <= maxInFlight, iterationsDone monotone, every positionHolds true, settledAtTick non-null; the pause variant: exactly the in-flight batches land, submissionsDuringPause 0, flush() resolves, the later step continues (iterationsDone and the speed trace continuous) |
| `test/browser/forceatlas2.test.ts` | P3-T6 | smoke (3): 500-node graph, load, step(10) x 5 with maxInFlight 2, positions written back, setPosition / setFixed honoured, dispose clean; plus the frame-loop test |
| `test/browser/bench.test.ts` | P3-T6 | `bench`-tagged, skipped unless GRAPHTY_BROWSER_GPU === "nvidia": 10k exact-tier step(1) + readback timings appended through commands.appendBenchRecord (T-5 at 10k; 100k is P4) |
| `test/accelerator.test.ts` | P3-T3 | createAccelerator: kind, ctx, frozen options, forceAtlas2 inherits options.layout (exactMaxNodes / compat visible in the simulation's tuning), release / dispose delegate, no other own members |
| `test/types/accelerator.test-d.ts` | P3-T3 | expectTypeOf(createAccelerator(ctx)).toMatchTypeOf<AlgorithmAccelerator & LayoutAccelerator>() and the reverse structural checks against the mirrors; GpuLayoutSimulation extends LayoutSimulation |
| `test/types/options.test-d.ts` | P3-T3 | every option interface accepts explicit undefined under exactOptionalPropertyTypes; ForceAtlas2Options & GpuLayoutTuning is accepted by createForceAtlas2 |
| `test/index.test.ts` / `public-api.test-d.ts` (updated) | P3-T3 | the P3 export lists of 3.15 |

### 5.6 The noise-floor file (spec 11.9 item 3)

`benchmarks/results/noise-floor.json`, created at P1-T6 from the `degree`
and one-iteration FA2 outputs across the three adapters (NVIDIA Dawn,
lavapipe Dawn, SwiftShader Chromium) and extended at P3-T5 with K1-K5:

```json
{
    "recordedAt": "2026-09-20T00:00:00.000Z",
    "adapters": [{ "class": "nvidia-ada-lovelace-node", "vendor": "nvidia", "architecture": "ada-lovelace", "description": "...", "runtime": "node" }],
    "rows": [
        { "id": "fa2-skeleton.force.twin", "kernel": "fa2-repulsion-exact", "fixture": "karate", "comparison": "twin", "a": "nvidia-ada-lovelace-node", "b": "nvidia-ada-lovelace-node/no-subgroups", "maxRelError": 3.1e-7, "maxAbsError": 2.0e-6, "samples": 102 },
        { "id": "fa2-skeleton.force.cross", "kernel": "fa2-repulsion-exact", "fixture": "karate", "comparison": "cross-adapter", "a": "nvidia-ada-lovelace-node", "b": "mesa-software-node", "maxRelError": 1.2e-6, "maxAbsError": 3.0e-6, "samples": 102 },
        { "id": "degree.cross", "kernel": "degree", "fixture": "random1k", "comparison": "cross-adapter", "a": "nvidia-ada-lovelace-node", "b": "google-swiftshader-browser", "maxRelError": 0, "maxAbsError": 0, "samples": 1000 }
    ],
    "tolerances": {
        "fa2-skeleton.force": { "value": 1e-5, "basis": "fa2-skeleton.force.cross", "factor": 8.3 },
        "fa2-force-parity": { "value": 1e-4, "basis": "fa2-force-parity.oracle-f64", "factor": 6.0 }
    }
}
```

`comparison` is `"twin"`, `"cross-adapter"` or `"oracle-f64"` (the same
kernel against the f64 oracle with f32-rounded inputs); a `tolerances`
entry is valid only when `value <= 10 x rows[basis].maxRelError` and
`factor` records the ratio; `noiseFloorFor(id)` reads this file and the
parity tests take their tolerances from it, never from a literal.

## 6. benchmarks/ and scripts/ contracts

### 6.1 benchmarks/harness.ts -- P1-T7 (spec 11.7; graph-format's harness with an async run body)

```ts
/** One measured benchmark (graph-format's shape, unchanged). */
export interface BenchResult {
    readonly group: string;
    readonly name: string;
    readonly medianMs: number;
    readonly minMs: number;
    readonly maxMs: number;
    readonly runs: number;
    readonly memoryDeltaBytes: number;
    readonly rate: number | null;
    readonly rateUnit: string | null;
}
/** A prepared benchmark: setup outside the timer, an ASYNC run inside it, optional teardown. */
export interface BenchCase<T> {
    readonly setup: () => T | Promise<T>;
    readonly run: (input: T) => Promise<unknown> | unknown;
    readonly teardown?: ((input: T) => void | Promise<void>) | undefined;
}
/** Options of bench(): the device whose queue the timer waits on, runs (default 5), items / unit for the rate column. */
export interface BenchOptions {
    readonly device: GPUDevice;
    readonly runs?: number | undefined;
    readonly items?: number | undefined;
    readonly unit?: string | undefined;
}
/** One warm-up then `runs` timed runs; the timer brackets `await run(input); await device.queue.onSubmittedWorkDone()`; median of the runs. */
export function bench<T>(group: string, name: string, benchCase: BenchCase<T>, options: BenchOptions): Promise<BenchResult>;
/** Aligned table of a group's results. */
export function printTable(results: readonly BenchResult[]): void;
/** The gpu field of a session (spec 11.7): adapter identity, driver, the requested limits, software flag, runtime. */
export interface GpuSessionInfo {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
    readonly driver: string;
    readonly limits: { readonly maxBufferSize: number; readonly maxStorageBufferBindingSize: number; readonly maxStorageBuffersPerShaderStage: number; readonly maxComputeWorkgroupsPerDimension: number };
    readonly software: boolean;
    readonly runtime: "node" | "browser";
    readonly subgroupMaxSize: number;
}
/** One session as stored (graph-format's fields plus gpu and runnerClass). */
export interface BenchSession {
    readonly date: string;
    readonly host: string;
    readonly node: string;
    readonly cpu: string;
    readonly exposeGc: boolean;
    readonly gpu: GpuSessionInfo;
    readonly runnerClass: string;
    readonly results: readonly BenchResult[];
}
/** The browser bridge payload (5.1 browser-commands.d.ts). */
export interface BrowserBenchPayload {
    readonly runnerClass: string;
    readonly session: BenchSession;
}
/** Re-exported from scripts/runner-class.js (6.9): the ONE copy of the runner-class rule, shared with scripts/gpu-report.js and scripts/bench-compare.js so the file names of 6.4 cannot drift. */
export { runnerClass } from "../scripts/runner-class.js";
/** The GpuSessionInfo of a context. */
export function gpuSessionInfo(ctx: GpuContext): GpuSessionInfo;
/** Appends a session to benchmarks/out/<runnerClass>.json (created when absent); returns the path. */
export function appendSession(results: readonly BenchResult[], gpu: GpuSessionInfo, options?: { readonly dir?: string | undefined } | undefined): string;
/** xorshift32, graph-format's. */
export function makeRandom(seed: number): () => number;
```

### 6.2 benchmarks/datasets.ts -- P1-T7 (spec 11.7)

```ts
export interface EdgeArrays { readonly nodeCount: number; readonly src: U32; readonly dst: U32; readonly weights: F32; }
/** G(n, m) with self-loops and parallels, weights 1..10 (graph-format's). */
export function randomEdges(nodeCount: number, edgeCount: number, seed?: number | undefined): EdgeArrays;
/** R-MAT-like hub graph (0.57 / 0.19 / 0.19 / 0.05), no self-loops. */
export function rmatEdges(scale: number, edgeFactor: number, seed?: number | undefined): EdgeArrays;
/** A w x h grid. */
export function gridEdges(w: number, h: number): EdgeArrays;
/** Zachary's karate club as EdgeArrays. */
export const KARATE_EDGES: EdgeArrays;
/** The design-15.3 tiers as (nodes, edges) pairs: 10k/100k, 100k/1M, 1M/10M. */
export const TIERS: readonly { readonly name: string; readonly nodes: number; readonly edges: number }[];
/** An undirected weighted fromEdgeArrays snapshot of an EdgeArrays. */
export function snapshotOf(edges: EdgeArrays, options?: { readonly directed?: boolean | undefined; readonly label?: string | undefined } | undefined): GraphSnapshot;
```

### 6.3 benchmarks/run.ts and the group files

`run.ts` (P0-T2 stub; P1-T7 real): `tsx benchmarks/run.ts [group ...]
[--no-save] [--allow-software] [--runs N]`; creates the context through
`createNodeGpuContext({ adapter: process.env.GRAPHTY_GPU_ADAPTER })`,
prints the adapter summary first, REFUSES to time on a software adapter
unless `--allow-software` (prints "software adapter: nothing timed" and
exits 0 -- spec 11.7 "Software adapters never time anything"), runs the
selected groups (unknown group -> exit 1), prints one table per group,
appends the session unless `--no-save`, disposes the context. Groups and
the T-targets they record:

| Group file | Export | Contents | Target |
| --- | --- | --- | --- |
| `upload.bench.ts` (P1-T7) | `runUploadBenchmarks(ctx): Promise<BenchResult[]>` | residency.core of the 100k / 1M and 1M / 10M weighted hot prefixes (fresh snapshot per run; release in teardown) | T-1 |
| `roundtrip.bench.ts` (P1-T7) | `runRoundtripBenchmarks(ctx)` | degree + 400 KB readback at 100k; an empty submit + 4-byte readU32 | T-2, T-3 |
| `layout-exact.bench.ts` (P3-T7) | `runLayoutExactBenchmarks(ctx)` | createForceAtlas2 step(1) per-iteration wall time at the exact ladder 1k / 4k / 8k / 16k / 32k / 65k (E = 10n), 2D, after warm; the per-frame cost step(1) + readback at 10k | T-4 (and the Node side of T-5) |

`layout-run.ts` (P3-T7): `tsx benchmarks/layout-run.ts --nodes N --edges M
[--iterations 100] [--batch 8] [--seed 1] [--dim 2] [--compat paper]` lays
the G(n, m) graph out end to end with `run()`, prints ms / iteration, the
final stats and the wall time, verifies every position is finite and that
`settled` or `maxIter` was reached, exits 1 otherwise; the 100k / 1M
exact-tier run is the P3 deliverable of spec 13.

### 6.4 benchmarks/results/<runner-class>.json (spec 10.4, 11.7)

A JSON array of `BenchSession` (6.1) ordered by `date`; the checked-in
baseline for a runner class is the LAST session of its file; `bench:compare`
reads the last session of `benchmarks/out/<class>.json` against it. Files
expected after P3: `nvidia-ada-lovelace-driver580.json` (the dev box, written
by the owner's `pnpm run bench`; P1-T7 commits the first) and
`gpu-linux-t4.json` (CONTRACT DECISION: the T4 lane's class is FIXED to
`gpu-linux-t4` -- the runner name -- regardless of its driver, so the
nightly file name never drifts with the partner image; `runnerClass()` of
`scripts/runner-class.js` (6.9) returns it when `GRAPHTY_RUNNER_CLASS` is
set, which gpu.yml sets to `gpu-linux-t4` -- P0 delta 7 of section 2.9;
`gpu-report.js`, the harness and `bench-compare.js` all go through that one
function).

### 6.5 scripts/gpu-policy.js and gpu-policy.d.ts -- P0-T2 (spec 2.3, 11.2, D19)

```ts
// scripts/gpu-policy.d.ts (the .js implements exactly this)
/** The parsed GRAPHTY_GPU_REQUIRE. */
export interface GpuPolicy {
    readonly level: "skip" | "any" | "hardware" | "vendor";
    readonly vendor: string | null;
    readonly raw: string;
}
/** unset / "" -> skip; "any"; "hardware"; any other string -> vendor (lower-cased). */
export function parseGpuRequire(value: string | undefined): GpuPolicy;
/** architecture "software" | "swiftshader" or isFallbackAdapter === true (a copy of src isSoftwareAdapter; test/device/policy.test.ts keeps them equal). */
export function isSoftwareInfo(info: { readonly vendor: string; readonly architecture: string; readonly isFallbackAdapter?: boolean | undefined }): boolean;
/** The policy verdict for an adapter (null info = no adapter): skip -> { ok: false, reason, skip: true } when absent; any -> ok iff present; hardware -> ok iff !software; vendor -> ok iff vendor === policy.vendor and (browser ? isFallbackAdapter === false : true). */
export function checkAdapter(info: { readonly vendor: string; readonly architecture: string; readonly isFallbackAdapter?: boolean | undefined } | null, policy: GpuPolicy, options?: { readonly browser?: boolean | undefined } | undefined): { readonly ok: boolean; readonly skip: boolean; readonly reason: string | null };
```

### 6.6 scripts/gpu-report.js -- P0-T2 (spec 12.3)

Runs after `pnpm run build` (it imports `../dist/node.js` for `createNodeGpu`
and `dawnFlags` ONLY -- never `GpuContext`, so the script works from P0 and
never depends on the core): `createNodeGpu` with `GRAPHTY_GPU_ADAPTER` /
`GRAPHTY_DAWN_FEATURES`, `gpu.requestAdapter()`, `adapter.info`,
`gpu-policy.js`'s `checkAdapter`, `runner-class.js`'s `runnerClass(info,
process.env)` (so the report's `runnerClass` honours `GRAPHTY_RUNNER_CLASS`,
which gpu.yml sets to `gpu-linux-t4`, 2.9 delta 7), `adapter.requestDevice()` for the four
limits / features / subgroup sizes and the 4-byte round-trip latency (10 x
`writeBuffer` + `mapAsync` of a 4-byte staging buffer, median), then a 10 s
`nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader`
sample at 1 Hz when `nvidia-smi` exists. Prints ONE JSON document to
stdout:

```json
{
    "ok": true,
    "policy": { "level": "vendor", "vendor": "nvidia", "raw": "nvidia" },
    "adapter": { "vendor": "nvidia", "architecture": "ada-lovelace", "device": "", "description": "NVIDIA GeForce RTX 4070 SUPER ...", "software": false, "subgroupMinSize": 32, "subgroupMaxSize": 32, "features": ["subgroups", "timestamp-query"], "limits": { "maxBufferSize": 268435456, "maxStorageBufferBindingSize": 134217728, "maxStorageBuffersPerShaderStage": 8, "maxComputeWorkgroupsPerDimension": 65535 } },
    "deviceLimits": { "maxBufferSize": 2147483648, "maxStorageBufferBindingSize": 2147483648, "maxStorageBuffersPerShaderStage": 8, "maxComputeWorkgroupsPerDimension": 65535 },
    "roundTripMs": 0.04,
    "runnerClass": "nvidia-ada-lovelace-driver580",
    "nvidiaSmi": { "available": true, "samples": [{ "utilizationGpu": 0, "memoryUsedMiB": 512 }], "maxUtilization": 0, "maxMemoryUsedMiB": 512 },
    "webgpu": "0.4.0",
    "node": "v22.22.1",
    "reason": null
}
```

Exit codes: 0 ok; 2 the policy is violated (`ok: false`, `reason` set --
the "fails loudly on a software adapter" of spec 12.3); 3 no module / no
adapter (`reason` set); 1 an unexpected error (the message on stderr).
Never pipes through `tee` (the step reads the exit code).

### 6.7 scripts/run-browser-project.js -- P0-T2 (spec 11.6)

Spawns `timeout -k 10 600 pnpm exec vitest run --project=browser
--reporter=default --reporter=json --outputFile=browser-results.json`
(inherits the environment; on a platform without GNU `timeout` it runs
vitest directly with a 600 s Node-side kill). Exit-code rule: vitest exit 0
-> exit 0; exit 124 (the timeout hit, the `browser.close()` hang of R-13)
-> exit 0 iff `browser-results.json` parses and has `numTotalTests > 0 &&
numFailedTests === 0`, else exit 1 with the summary; any other exit -> that
exit code. Always prints `numTotalTests / numPassedTests / numFailedTests`
from the JSON when present.

### 6.8 scripts/bench-compare.js -- P0-T2 (stub) / P1-T7 (spec 10.4 T-13, 11.7)

Reads `gpu-report.json` (the runner class and the nvidia-smi sample),
`benchmarks/out/<class>.json` (last session) and
`benchmarks/results/<class>.json` (last session). Rules, in order: no
`gpu-report.json` or no out file -> print "nothing to compare", exit 0;
`nvidiaSmi.available && (maxUtilization > 10 || memory in use by other
processes > 0)` during the sample -> print "SKIPPED: GPU not quiet", exit 0
(T-13); no baseline file -> print every result as "new (no baseline)",
exit 0; for every result present in both, `medianMs > 3 x baseline.medianMs`
-> listed as a regression; any regression -> exit 1 with the table, else
exit 0 with the table. Options: `--threshold 3`, `--class <name>`
(overrides the report). The "memory in use by other processes" figure is
`maxMemoryUsedMiB` minus the report's own process footprint estimate
(the sample's minimum), so a quiet card with a resident desktop compositor
passes; the rule is documented in the script header. The runner class comes
from `gpu-report.json` (which `scripts/gpu-report.js` computed through
`scripts/runner-class.js`, 6.9), so the file `bench:compare` looks for is
the one the harness wrote through the same function.

### 6.9 scripts/runner-class.js and runner-class.d.ts -- P0-T2 (spec 10.4, 11.7)

The ONE copy of the runner-class rule (the 2.3 one-copy principle applied
to it, as `gpu-policy.js` is for the adapter policy): `scripts/gpu-report.js`
imports it (plain `.js`, so a `.ts` module could not be shared),
`benchmarks/harness.ts` re-exports it (6.1) and `test/device/policy.test.ts`
pins it (5.5).

```ts
// scripts/runner-class.d.ts (the .js implements exactly this)
/** The adapter facts the rule reads. */
export interface RunnerClassInfo {
    readonly vendor: string;
    readonly architecture: string;
    readonly description: string;
}
/** `env.GRAPHTY_RUNNER_CLASS` when set and non-empty (the T4 lane fixes `gpu-linux-t4`, 6.4), else `<vendor>-<architecture>-driver<major>` (spec 10.4). CONTRACT DECISION: `major` = the first run of digits in `description`, "0" when none (Dawn's NVIDIA description carries the driver version; SwiftShader / lavapipe carry Mesa's); every character outside [A-Za-z0-9_.-] becomes "_"; lower-cased. */
export function runnerClass(info: RunnerClassInfo, env?: Readonly<Record<string, string | undefined>> | undefined): string;
```

`env` defaults to `process.env` in the `.js`; the test passes explicit
records so the override rule is pinned without touching the environment.

## 7. Phase and task map

Rules: task ids are `P<phase>-T<n>`; "files" lists ownership (created or
modified) -- within a phase no two tasks name the same file (lead d);
"depends on" is the strict ordering; tasks with no dependency between them
run in parallel in one working tree; no task runs `git add` / `commit` /
`push` (lead c) -- the owner commits from `tmp/commit-<phase>.sh` scripts
the phase's last task drafts; every task ends with `pnpm run build:all &&
pnpm run lint && pnpm exec vitest run --project=node` green on the dev
box (NVIDIA) and with `GRAPHTY_GPU_ADAPTER=llvmpipe GRAPHTY_GPU_REQUIRE=any`
(the default lane) where the task touched GPU code.

### 7.1 P0 -- reset + package skeleton (spec 13 row P0; gate G0)

| Task | Title | Files (created unless "edit") | Depends on | Parallel |
| --- | --- | --- | --- | --- |
| P0-T1 | Scaffold triage, root rewrite, docs move | deletes the root scaffold list of 1.1; edits `/README.md`, `/.gitignore`; `git mv HEADLESS_GPU_REPORT.md` -> `packages/webgpu-graph-algorithms/docs/HEADLESS_GPU_REPORT.md`; copies `tmp/webgpu-plan/*.md` and `review/` -> `docs/research/` (1.1) | -- | with T2 |
| P0-T2 | Manifest, configs, scripts, CI, workspace entries | `package.json` (replaces the placeholder), `project.json`, `tsconfig.json`, `tsconfig.build.json`, `tsconfig.strict-consumer.json`, `eslint.config.js`, `vitest.config.ts`, `scripts/entries.js`, `scripts/build-bundle.js`, `scripts/bundle-types.js`, `scripts/gpu-policy.js`, `scripts/gpu-policy.d.ts`, `scripts/runner-class.js`, `scripts/runner-class.d.ts`, `scripts/gpu-report.js`, `scripts/run-browser-project.js`, `scripts/bench-compare.js` (stub rule), `benchmarks/run.ts` (stub), `.github/workflows/ci.yml`, `.github/workflows/gpu.yml`; edits `packages/pnpm-workspace.yaml`, `packages/knip.config.ts`, `packages/package.json` (description), `packages/.gitignore`; runs `pnpm install` at `packages/` (lockfile updated) | -- | with T1 |
| P0-T3 | src skeleton, test setup, first device tests, index / build-output / layers tests, package CLAUDE.md / README | `src/index.ts`, `src/errors.ts`, `src/constants.ts`, `src/device/webgpu-constants.ts`, `src/device/acquire.ts` (isSoftwareAdapter, summarizeAdapter), `src/browser/index.ts` (empty), `src/node/index.ts` (createNodeGpu, NodeGpuOptions, NodeGpuHandle, dawnFlags), `src/types/context.ts` (AdapterInfoLike, AdapterSummary only at P0; P1-T1 completes), `test/setup/gpu.ts` (P0 form), `test/setup/browser.ts`, `test/setup/global.ts` (empty), `test/setup/browser-commands.d.ts` (P0 form, 5.1), `test/errors.test.ts`, `test/device/constants.test.ts`, `test/device/acquire.test.ts`, `test/device/policy.test.ts`, `test/browser/webgpu-check.test.ts`, `test/index.test.ts`, `test/build-output.test.ts`, `test/layers.test.ts`, `test/types/public-api.test-d.ts`, `README.md`, `CLAUDE.md` (spec 3.1 sections, the verified `launch` spelling of 2.5, the env table of spec 12.2) | T2 | -- |
| P0-T4 | Owner checklist (G0) | `docs/decisions/G0.md`; edit `CLAUDE.md` ("Verified Platform Facts": the spec 12.2 image facts from the first gpu.yml dispatch -- modprobe needed or not, Xvfb needed or not, libegl1 present, the image's Ubuntu release / glibc / driver -- and the `test.env` vs `define` outcome of section 9 item 4; T3 leaves the section with "unverified" placeholders and T4 is strictly after T3, so the lead-d parallel-edit hazard does not arise); `tmp/commit-p0.sh` (drafted by T3's author, run by the owner) | T1, T2, T3 | -- |

P0-T4 items the OWNER performs and G0.md records (spec 12.4, 13 row P0):
create the public repository and push; `repository.url` confirmed; the
`gpu` label; GitHub Team plan; the `gpu-linux-t4` runner in group `gpu`
with access limited to this repository; the $50 spending limit and the 75%
alert; read-only default workflow permissions and approval for external
contributors; the first `workflow_dispatch` of `gpu.yml` (records the
image's Ubuntu release, glibc, driver, whether `modprobe` was needed,
whether Chromium found the T4 without Xvfb, `libegl1` present); the
deliberate red run (`GRAPHTY_GPU_REQUIRE=nvidia` on the software lane must
fail: the `gpu-report.js` step red, no `tee`); the default lane green on
GitHub; the 7.2 formula-table sign-off recorded in the P0 PR (D21) WITH the
two section-9 corrections either accepted or re-decided; the Vitest 3
`launch` spelling recorded in CLAUDE.md; `benchmarks/results/gpu-linux-t4.json`
committed from the first green run once P1 produces results (deferred to
P1-T7 when P0 has no benchmarks: G0 records "no baseline yet").

### 7.2 P1 -- walking skeleton (spec 13 row P1; gate G1)

| Task | Title | Files | Depends on | Parallel |
| --- | --- | --- | --- | --- |
| P1-T1 | Device layer, context, entries | edit `src/device/acquire.ts`; `src/device/caps.ts`, `src/device/error-scope.ts`, `src/device/lost.ts`, `src/context.ts`, edit `src/types/context.ts` (complete), `src/types/run.ts`, `src/types/memory.ts`, `src/kernel/profiler.ts` (shell), edit `src/browser/index.ts`, edit `src/node/index.ts`, edit `test/setup/gpu.ts` (acquire -> GpuContext, acquireNullBackend, uncaptured hook, inspect flag, key log), edit `test/setup/browser.ts` (acquireBrowser), `test/device/context.test.ts`, `test/device/lost.test.ts`, `test/device/error-scope.test.ts`, `test/node/entry.test.ts` | -- (P0 done) | with T2 and T3: the three form ONE integration unit (below) |
| P1-T2 | Upload planner, residency, pool, readback, caps tables, upload-contract tests | `src/memory/upload-plan.ts`, `src/memory/residency.ts`, `src/memory/buffer-pool.ts`, `src/memory/readback.ts`, `test/helpers/caps-tables.ts`, `test/helpers/device.ts`, `test/helpers/graphs.ts`, `test/helpers/matchers.ts`, `test/fixtures/rich-v1.gsnp` (copy), `test/memory/upload-plan.test.ts`, `test/memory/residency.test.ts`, `test/memory/buffer-pool.test.ts`, `test/memory/readback.test.ts` | T1's `src/types/memory.ts` text (copied from 3.3) | with T1 and T3 |
| P1-T3 | WGSL composer, prelude, UniformBlock, PipelineCache, Kernel, dispatch, their tests, the kernel test helper | `src/kernel/wgsl.ts`, `src/kernel/prelude.ts`, `src/kernel/struct-block.ts`, `src/kernel/pipeline-cache.ts`, `src/kernel/kernel.ts`, `src/kernel/dispatch.ts`, `test/helpers/kernel.ts`, `test/kernel/wgsl.test.ts`, `test/kernel/struct-block.test.ts`, `test/kernel/pipeline-cache.test.ts`, `test/kernel/kernel.test.ts`, `test/kernel/dispatch.test.ts` | T1's `src/types/memory.ts` text | with T1 and T2 |
| P1-T4 | kernels.ts registry, the P1 WGSL bodies and blocks, compile and registry tests, the 17M linear_id test and its shared constants | `src/kernels.ts`, `src/wgsl/degree.wgsl.ts`, `src/wgsl/reduce.wgsl.ts`, `src/wgsl/fill.wgsl.ts`, `src/wgsl/fa2-repulsion-exact.wgsl.ts`, `src/wgsl/fa2-speed-finalize.wgsl.ts`, `test/helpers/linear-id.ts`, `test/kernel/registry.test.ts`, `test/kernel/compile.test.ts`, `test/kernel/linear-id.test.ts` | T1 + T2 + T3 (the integration unit green) | -- |
| P1-T5 | degree kernel driver, reduce primitive, oracles, degree / reduce tests, the sabotage table (degree, reduce, K3, K4 rows) and its coverage test, the noise-floor helpers | `src/primitives/reduce.ts`, `src/algorithms/degree.ts`, `test/oracle/degree.ts`, `test/oracle/reduce.ts`, `test/helpers/sabotage.ts` (every P1 row: the K3 / K4 `find` strings come from the normative bodies of 4.5; `SABOTAGE_PHASES = ["P1"]`), `test/helpers/noise-floor.ts`, `test/fixtures/noise/degree-*.json`, `test/fixtures/noise/reduce-*.json`, `test/primitives/reduce.test.ts`, `test/algorithms/degree.test.ts`, `test/sabotage/degree.test.ts`, `test/sabotage/reduce.test.ts`, `test/sabotage/coverage.test.ts` | T4 | with T6 |
| P1-T6 | The FA2 skeleton iteration (K3 + K4 stage), its tests, the K3 / K4 sabotage test, the noise-floor file and test | `src/layouts/repulsion-exact.ts`, `test/fixtures/noise/fa2-*.json` (this adapter's), `benchmarks/results/noise-floor.json`, `test/layouts/skeleton.test.ts`, `test/sabotage/fa2-skeleton.test.ts`, `test/noise-floor.test.ts` | T4; T5 for `test/helpers/sabotage.ts` and `noise-floor.ts` (its src work starts in parallel with T5) | with T5 (src); after T5 (tests) |
| P1-T7 | Leak counter, benchmarks harness + upload / roundtrip groups, bench-compare, build-output bundle assertions, browser skeleton (incl. the SwiftShader noise fixtures and the 17M map), index exports, coverage, G1 record | `test/helpers/leak-counter.ts`, `test/helpers/noise-floor-browser.ts`, edit `test/setup/browser-commands.d.ts` (NoiseRow), `test/leak.test.ts`, `benchmarks/harness.ts`, `benchmarks/datasets.ts`, edit `benchmarks/run.ts`, `benchmarks/upload.bench.ts`, `benchmarks/roundtrip.bench.ts`, `benchmarks/results/<nvidia class>.json`, edit `scripts/bench-compare.js` (full rule), edit `test/build-output.test.ts`, `test/browser/entry.test.ts`, `test/browser/skeleton.test.ts`, edit `src/index.ts` (the P1 list), edit `test/index.test.ts`, edit `test/types/public-api.test-d.ts`, edit `README.md`, edit `CLAUDE.md`, `docs/decisions/G1.md` (T-1 / T-2 / T-3 recorded; cross-adapter results; coverage numbers), `tmp/commit-p1.sh` | T5, T6 | -- |

The P1 integration unit (CONTRACT DECISION; the section 7 per-task green
rule applies to T4-T7 individually): `src/context.ts` (T1) constructs
`GraphResidency` / `BufferPool` / `Readback` (T2) and `PipelineCache` /
`Profiler` (T1 / T3), `test/helpers/device.ts` (T2) and
`test/helpers/kernel.ts` (T3) both need `acquire()` (T1), and
`test/kernel/kernel.test.ts` (T3) uploads through T2's `uploadBuffer`, so
no one of T1 / T2 / T3 lints and tests green alone. The three are written
in parallel against this contract in one working tree (every declaration
they share -- `Binding`, `ArcWindow`, `PlanCaps`, `AllocationTracker`,
`Kernel`, `PipelineCache` -- is copied from section 3, never re-derived),
and `pnpm run build:all && pnpm run lint && pnpm exec vitest run
--project=node` is run ONCE over the union when the last of the three
lands; the type imports of `src/types/memory.ts` and the split of
`runKernel` into `test/helpers/kernel.ts` keep every remaining edge between
the three a TYPE edge or a constructor call inside `context.ts`, so the
cycle test of 5.5 stays green.

G1 checklist (spec 13 row P1 and 11.5), each item a named test or a
recorded number: the 11.5 skeleton on lavapipe, SwiftShader and NVIDIA;
u32 results (`degree` AND the 17M-item map, `test/kernel/linear-id.test.ts`
on lavapipe / NVIDIA and `test/browser/skeleton.test.ts` on SwiftShader /
NVIDIA, one pinned checksum in `test/helpers/linear-id.ts`) bitwise across
adapters; the FA2 iteration within 1e-5 across adapters and bitwise per
adapter; the upload-contract tests; the bundle specifier test hard-failing
under CI; T-1 / T-2 / T-3 in `benchmarks/results/`; coverage >= 80 / 80 /
75 / 80 on `--project=node`; the noise-floor file with the degree, reduce
and FA2-skeleton rows from all three adapters (the SwiftShader fixtures
written through the browser commands bridge, 2.5); the `degree`, `reduce`,
K3 and K4 sabotage mutations failing their tests by >= 10x
(`test/sabotage/coverage.test.ts` green).

### 7.3 P2 -- batch + dispatch infrastructure (spec 13 row P2; gate G2)

| Task | Title | Files | Depends on | Parallel |
| --- | --- | --- | --- | --- |
| P2-T1 | Lease, CommandBatch, UniformRing, Profiler, warm, device-loss propagation, residentBytes warning, and their tests | `src/memory/lease.ts`, edit `src/memory/buffer-pool.ts` (lease()), `src/kernel/batch.ts`, `src/kernel/uniform-ring.ts`, edit `src/kernel/profiler.ts` (full), edit `src/context.ts` (BatchHost members, profiler creation, onLost fan-out), edit `src/memory/residency.ts` (residentBytes warning, clearOnLoss), `test/memory/lease.test.ts`, `test/kernel/batch.test.ts`, `test/kernel/uniform-ring.test.ts`, `test/kernel/profiler.test.ts`, `test/kernel/state-roundtrip.test.ts`, `test/kernel/uniform-layout.test.ts`, edit `test/device/lost.test.ts`, edit `test/memory/residency.test.ts`, `test/browser/batch.test.ts`, `test/browser/lost.test.ts`, `test/browser/state-roundtrip.test.ts`, `test/browser/uniform-layout.test.ts` | -- (P1 done) | with T2 |
| P2-T2 | segmentedReduce thread-per-row + twin, its sabotage rows and noise fixtures, oracle skeleton, override matrix, compile matrix on both runtimes, bind-group budget, determinism, the key-log teardown | `src/wgsl/segmented-reduce.wgsl.ts`, edit `src/kernels.ts` (the entry), `src/primitives/segmented-reduce.ts`, `test/oracle/segmented-reduce.ts`, `test/helpers/override-matrix.ts`, edit `test/helpers/caps-tables.ts` (CAPS_INTEL_XE), edit `test/helpers/sabotage.ts` (the segmented-reduce rows; `SABOTAGE_PHASES` gains "P2"), edit `test/setup/global.ts` (the coverage teardown), `test/fixtures/noise/segmented-reduce-*.json`, edit `benchmarks/results/noise-floor.json` (the segmented-reduce rows), `test/primitives/segmented-reduce.test.ts`, `test/sabotage/segmented-reduce.test.ts`, `test/kernel/wgsl-compile.test.ts`, `test/kernel/bind-group-budget.test.ts`, `test/kernel/determinism.test.ts`, `test/browser/compile-matrix.test.ts`, `test/limits/README.md` | -- (P1 done) | with T1 |
| P2-T3 | Default-lane time budget check and the G2 record | `docs/decisions/G2.md` (lane durations from the GitHub run, the `gpu-report.json` upload, every G2 item mapped to its test), edit `CLAUDE.md` (Verified Platform Facts: the synchronous uncapturederror order, the negative uniform test outcome per runtime), `tmp/commit-p2.sh` | T1, T2 | -- |

G2 checklist: `planUpload` unit tests for every path x caps table; the
1D / 2D boundary and the WGSL linear_id test; reduce and thread-per-row
segmentedReduce equal their oracles at scaled sizes incl. all-equal keys and
one hub row, both twins in-process, bitwise across two runs; the
UniformBlock negative test rejected on Chromium and the state round trip
green; E_DEVICE_LOST mid-readback on both runtimes with recovery
(`test/device/lost.test.ts` and `test/browser/lost.test.ts`); a bad
bind group rejects its own batch's readback under Dawn-node; leak counter 0
after release and dispose; the segmented-reduce sabotage rows fail their
oracle test by >= 10x (`test/sabotage/segmented-reduce.test.ts`,
`coverage.test.ts` green with "P2" listed) and its noise-floor rows
recorded from the three adapters; the GPU lane green with
`gpu-report.json` uploaded; the default lane <= 15 min (T-12).

### 7.4 P3 -- ForceAtlas2, exact tier (spec 13 row P3; gate G3)

| Task | Title | Files | Depends on | Parallel |
| --- | --- | --- | --- | --- |
| P3-T1 | seed.ts, inputs.ts, the three type files, ForceSimulation core, the fake-model test | `src/layouts/seed.ts`, `src/layouts/inputs.ts`, `src/types/options.ts`, `src/types/layout.ts`, `src/types/accelerator.ts`, `src/layouts/force-simulation.ts`, `test/layouts/force-simulation.test.ts`, `test/layouts/seed.test.ts`, `test/layouts/inputs.test.ts` | -- (P2 done) | with T4 (the oracle depends only on the option TYPES; T4 may start from this contract's type text and rebase on T1's file) |
| P3-T2 | The FA2 model: K1 / K2 / K5 / toScene bodies and registry entries, ForceAtlas2Model, createForceAtlas2, the repulsion stage wired in, option tests | `src/wgsl/fa2-stats-finalize.wgsl.ts`, `src/wgsl/fa2-attraction.wgsl.ts`, `src/wgsl/fa2-integrate.wgsl.ts`, `src/wgsl/fa2-to-scene.wgsl.ts`, edit `src/kernels.ts` (the four entries; `fill` reuse), edit `src/layouts/repulsion-exact.ts` (recordRepulsion / recordSpeedFinalize split, model wiring), `src/layouts/forceatlas2.ts`, `test/layouts/fa2-options.test.ts` | T1 | -- |
| P3-T3 | createAccelerator, index exports, type tests, strict-consumer sample | `src/accelerator.ts`, edit `src/index.ts` (the P3 list), `test/accelerator.test.ts`, `test/types/accelerator.test-d.ts`, `test/types/options.test-d.ts`, edit `test/types/public-api.test-d.ts`, edit `test/index.test.ts` | T2 | with T4-T7 |
| P3-T4 | The f64 / f32 FA2 oracle, the NetworkX fixture generator, the committed fixtures, oracle-vs-NetworkX tests, the SWING_MODE unit test | `test/oracle/forceatlas2.ts`, `test/fixtures/networkx/generate.py`, `test/fixtures/networkx/*.json`, `test/oracle/forceatlas2-networkx.test.ts`, `test/oracle/swing-mode.test.ts` (creates `tmp/nx-venv` with `networkx>=3.4`; the venv is gitignored) | T1 (types) | with T2, T3 |
| P3-T5 | Parity tests, sabotage matrix K1-K5, inspect stage parity, noise-floor tracing, metrics | `test/helpers/metrics.ts`, edit `test/helpers/sabotage.ts` (K1 / K2 / K5 rows; `SABOTAGE_PHASES` gains "P3"), edit `test/fixtures/noise/*.json` (K1-K5 rows), edit `benchmarks/results/noise-floor.json`, edit `test/noise-floor.test.ts`, `test/layouts/fa2-force-parity.test.ts`, `test/layouts/fa2-trace-parity.test.ts`, `test/layouts/fa2-distributional.test.ts`, `test/layouts/fa2-behaviour.test.ts`, `test/layouts/fa2-properties.test.ts`, `test/layouts/fa2-force-sum.test.ts`, `test/layouts/fa2-twins.test.ts`, `test/layouts/fa2-lifecycle.test.ts`, `test/layouts/fa2-inspect.test.ts`, `test/sabotage/fa2.test.ts` | T2, T4 | with T3, T6, T7 |
| P3-T6 | Frame-loop helper and test (node + browser), browser smoke (3), the bench-tagged T-5 test, the bridge types | `test/helpers/frame-loop.ts`, `test/layouts/frame-loop.test.ts`, `test/browser/forceatlas2.test.ts`, `test/browser/bench.test.ts`, edit `test/setup/browser-commands.d.ts` (BrowserBenchPayload; the `appendBenchRecord` command itself ships in vitest.config.ts at P0-T2, 2.5) | T2 | with T3, T5, T7 |
| P3-T7 | layout-exact benchmarks, layout-run.ts, the T-4 ladder, the exactMaxNodes re-fix, the lavapipe budget, README / CLAUDE.md, G3 record | `benchmarks/layout-exact.bench.ts`, `benchmarks/layout-run.ts`, edit `benchmarks/run.ts` (the group), edit `benchmarks/results/<nvidia class>.json`, edit `src/constants.ts` (EXACT_MAX_NODES re-fixed by the 7.8 rule with the benchmark session cited in the JSDoc), edit `README.md` (the Node and browser recipes, the performance table from results/), edit `CLAUDE.md` (Adding a Layout Model recipe; Verified Platform Facts), `docs/decisions/G3.md` (T-4 ladder, T-5 at 10k, the crossover decision, the lavapipe suite time, every 11.4 tolerance with its noise-floor basis), `tmp/commit-p3.sh` | T2, T3, T5, T6 | -- |

G3 checklist (spec 13 row P3): 11.4 layout parity in full (oracle
independence vs the NetworkX fixtures at 1 / 5 / 50 iterations in networkx
mode; iteration-0 forces covering paper mode; the SWING_MODE unit test;
force 1e-4; trace 1e-4 vs the f32 oracle / 5e-2 vs f64; distributional 10%;
the behaviour pins; every option combination incl. compat, 2D / 3D, seeded
/ unseeded, arcCount 0, a pinned node); the 11.3 layout properties and the
force-sum invariant incl. the coincident fixture; the FA2 twins within 1e-6;
lifecycle (dispose leak 0, E_RELEASED, device loss); the frame-loop test on
node, SwiftShader and NVIDIA; T-4 met and the exact curve recorded;
`exactMaxNodes` re-fixed and cited; T-5 at 10k in Chromium; lavapipe runs
the FA2 suite at gpuScale sizes in <= 3 min; the sabotage matrix for K1-K5
(>= 3 mutations each, each failing by >= 10x); inspect() stage parity for
K2, K3, the epilogue, K4, K5; every 11.4 tolerance traced to the noise-floor
file.

## 8. Environment and toolchain facts

| Fact | Value | Verified |
| --- | --- | --- |
| Node | 22.22.1 | given (2026-09-14) |
| pnpm | 10.0.0 (`packages/package.json` packageManager) | read |
| vitest / @vitest/browser | 3.2.7 in the packages lockfile (`^3.2.4` declared); @vitest/browser 3.2.7 is NOT installed under `packages/` today -- P0-T2's `pnpm install` adds it; the root scaffold's node_modules holds an unrelated @vitest/browser 2.1.9 that P0-T1 deletes | read |
| Per-instance Playwright launch spelling (Vitest 3.2.7 types + @vitest/browser 3.2.4, the nearest installed copy) | `browser.instances: [{ browser: "chromium", launch: { args, env } }]` -- `BrowserInstanceOption extends BrowserProviderOptions` (vitest 3.2.7 `reporters.d.BuRON0I0.d.ts` line 2412; `browser.fileParallelism` line 2477; `commands` line 2539), and `providers/playwright.d.ts` augments `BrowserProviderOptions` with `launch?: LaunchOptions`; `BrowserProviderOptions` is `{}` unless that file is referenced (hence the triple-slash reference in 2.5) | `packages/node_modules/vitest/dist/chunks/reporters.d.BuRON0I0.d.ts` and `~/Projects/pupt-monorepo/node_modules/.pnpm/@vitest+browser@3.2.4_*/node_modules/@vitest/browser/providers/playwright.d.ts` (no 3.2.7 copy of @vitest/browser exists on the machine; re-read the 3.2.7 file after P0-T2's `pnpm install` and record any difference in CLAUDE.md) |
| Browser commands bridge | config `test.browser.commands: Record<string, BrowserCommand>`; browser side `import { commands } from "@vitest/browser/context"` ("a shortcut to `server.commands`") | same files (`context.d.ts` line 551) |
| `test.env` in browser mode | `env?: Partial<NodeJS.ProcessEnv>` populates Node workers' `process.env`; the contract forwards to the browser with `define` (2.5) and P0-T3 verifies `import.meta.env` receives it | types read; runtime to verify at G0 |
| webgpu (Dawn) | 0.4.0 installed for graph-format (`packages/node_modules/.pnpm/webgpu@0.4.0`; 0.6.1 is also in the store but needs glibc 2.38); `types.d.ts`: `create(options: string[]): GPU`, `globals: Object` | read |
| @webgpu/types | 0.1.72; `GPUSupportedLimits`, `GPUAdapterInfo` are `__brand`ed (hence `PlanCaps` / `AdapterInfoLike`); `GPUAdapterInfo.isFallbackAdapter: boolean`, `subgroupMinSize?` / `subgroupMaxSize?` | read |
| TypeScript | 5.9.3 (`Uint32Array<ArrayBuffer>` generics available) | read |
| Playwright | 1.54.1 at the root scaffold (deleted); `^1.54.1` declared for the package; Chromium build 1181 (Chromium 139) present in `~/.cache/ms-playwright/chromium-1181` | read |
| OS / glibc / Mesa | Ubuntu 22.04, glibc 2.35, Mesa lavapipe ICD `/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` | given |
| NVIDIA | RTX 4070 SUPER, driver 580.173.02; headless Chromium reaches it only with `LD_LIBRARY_PATH=/home/apowers/Projects/webgpu-graph-algorithms/tmp/egl/root/usr/lib/x86_64-linux-gnu` (libEGL.so.1) and the four flags `--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan --disable-vulkan-surface` | given (HEADLESS_GPU_REPORT.md) |
| SwiftShader flags | `--enable-unsafe-webgpu --use-angle=swiftshader --enable-unsafe-swiftshader` | given |
| Python / NetworkX | Python 3.10; system networkx 3.1 has NO `forceatlas2_layout`; the fixture generator uses `tmp/nx-venv` with `networkx>=3.4` (P3-T4 creates it); the 3.4.2 source was read from the networkx-3.4.2 tag for section 4.6 | read (`tmp/.../scratchpad/nx-layout-3.4.2.py` lines 1250-1505) |
| graph-format API used | `INVALID_INDEX`, `GraphSnapshot`, `fromEdgeArrays`, `fromCsr`, `fromRecords`, `fromBytes`, `expandEdges`, `foldArcs`, `makeMask` / `maskTest` / `maskSet` / `maskCount`, `paddedU32View`, `renumberPartition`, `GraphFormatError`, `GraphBuilder`; types `F32` / `U32` / `NodeMask` / `TypedArrayData` / `ViewName` / `CoreArrayName` / `AttributeTable` / `Column` / `ArenaLayout` / `SnapshotFlags` / `NumericVector` / `KnownColumnRole` / `NodeId` / `GpuEligibility` -- all present in `packages/graph-format/src/index.ts` | read |
| Repository | `graphty-org/webgpu-graph-algorithms` exists as origin; CI has not run; the 0.0.0 placeholder package is committed at `packages/webgpu-graph-algorithms/` (package.json, README.md, LICENSE) | given / read |
| Root ESLint config | ignores `**/scripts/**`, `**/benchmarks/**`, `**/docs/**`, `**/*.config.*`; relaxes `**/test/**`; no `eslint-plugin-import` (hence the core `no-restricted-imports` zones, lead a) | read |

## 9. Open items the plan writers must resolve

1. Spec 7.2, gravity-centre row (owner sign-off artefact): NetworkX 3.4.2
   pulls toward the ORIGIN (`layout.py` lines 1466-1471), not the centroid.
   This contract compiles `GRAVITY_CENTER = 1` in `compat: "networkx"`
   (4.6); the owner either confirms at G0 (then the 7.2 row and Q-1's
   "Gravity toward the centroid (port, NetworkX)" are amended in the Review
   log) or re-decides, in which case the NetworkX fixtures cannot be an
   oracle for that mode and P3-T4's test plan changes.
2. Spec 7.2 "Local speed / apply" row and the 7.11 sketch: NetworkX's
   per-node factor uses `swinging = mass * |update|` (`layout.py` line
   1497), not `m |p - F|`. This contract uses `m |F|` in `SWING_MODE = 1`
   (4.5 K5); same sign-off as item 1.
3. P4 only: K1's grid-tier bindings (`cellHist`, `hubCounters`) differ from
   the exact tier's, and a layout is derived per spec from `spec.bindings`;
   the P4 plan writer chooses a second module id (`fa2-stats-finalize-grid`)
   or dummy bindings in the exact variant. Not a P0-P3 decision.
4. `test.env` versus `import.meta.env` in Vitest browser mode: the contract
   forwards both ways (2.5); if G0 shows `define` is unnecessary the P0
   record says so and the `env` line stays for the Node projects only.
5. The `gpu-linux-t4` runner class (6.4) is fixed by the environment
   variable `GRAPHTY_RUNNER_CLASS` that gpu.yml sets -- P0-T2 adds `env:
   GRAPHTY_RUNNER_CLASS: gpu-linux-t4` to the `test-gpu` job (P0 delta 7,
   listed here because section 2.9's delta list was written before 6.4).
6. Whether the owner wants `benchmarks/results/<nvidia class>.json` committed
   from the dev box at P1-T7 (spec 12.1 says the T-table is measured there
   "by hand") -- assumed yes; the file is produced by the owner running
   `pnpm run bench` and the task's author only prepares the script.
7. Spec 3.5 lines 1032-1033 and D16 (Review-log amendment): "scratch arrays
   are sized `WG / SUBGROUP_MAX` (rounded up), which is enough for the
   smallest size the compiler may pick" is inverted -- the subgroup COUNT
   is `WG / subgroup_size`, largest at the SMALLEST size. This contract
   sizes the scratch by `SUBGROUP_MIN` (4.3), adds `SUBGROUP_MIN` to the
   standard overrides (3.9) and keeps `SUBGROUP_MAX` as a declared but
   unused override; the owner amends the two spec sentences in the Review
   log, or re-decides.
8. Spec 4.1 lines 1130-1136 versus spec 6 row 3 / 7.3 / 7.5 (Review-log
   amendment): 4.1 makes `USE_PERM` the arcToEdge / edgeToArc identity
   guard, the other three make it the degreeOrder row permutation with
   `rowPtr` as the dummy. This contract follows the majority (3.10 CONTRACT
   DECISION): `USE_PERM` / `perm` = the row permutation; the arc-permutation
   guard of P7+ is the reserved `USE_ARC_PERM`; `CoreBinding` has no
   `perm`. The owner records which reading 4.1 keeps.
9. Spec 3.3 `layoutRadius = max |p - centroid|` is implemented exactly
   (K5 carries max |p - c|^2 in `partials.max.w`, 4.5); no amendment
   needed, listed because an earlier draft of this contract used the
   bounding-box corner radius and the oracle's trace record gained
   `layoutRadius` (5.3).
10. Spec 7.10 snippet line `if (swing / tr > 2.0)` (Review-log amendment;
    CONTRACT DECISION K4-1 in 4.5): the K4 body and the oracle write the
    halving predicate as `swing > 2.0 * tr`, its exact form, because the
    paper-mode first iteration always has `traction` exactly half the
    `swing` and a 2.5-ULP f32 division decides the branch by device
    (`docs/decisions/G3.md` finding G3-F6). The owner amends the snippet
    (and the "line-for-line port" wording, which now holds up to this one
    algebraic rewrite) or re-decides.
