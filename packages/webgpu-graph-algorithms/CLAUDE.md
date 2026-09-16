# CLAUDE.md

This file provides guidance to Claude Code when working with the @graphty/webgpu-graph-algorithms package.

## Project Overview

@graphty/webgpu-graph-algorithms runs graph algorithms and force-directed layouts on the GPU through
WebGPU, over the frozen CSR snapshot of @graphty/graph-format, from Node (Dawn via the `webgpu` npm
package) and from browsers (Chromium). One runtime-agnostic core, two thin acquisition entries
(`./browser`, `./node`). There is NO fallback of any kind: no CPU path, no WebGL path, no silent
acceptance of a software adapter in `src/`; when no adapter or device exists the package throws
`WebGpuGraphError` (repository rule in the root CLAUDE.md; plan section 2.4).

The normative documents are, in order of authority:

- `docs/superpowers/plans/2026-09-14-webgpu-p0-p3-interfaces.md` (the "contract"): every file, signature,
  WGSL body and test of phases P0-P3; copy declarations from it, never re-derive them.
- `design/webgpu-acceleration-plan.md` (the "spec"): the runtime model (2), the package architecture (3),
  memory (4), kernels (5), primitives (6), layouts (7), algorithms (8), integration (9), targets (10),
  testing (11), CI (12), phases and gates (13).
- `docs/decisions/G<n>.md`: what each gate measured and signed off; re-fixed constants cite them.
- `docs/research/`: the research notes and drafts the plan was synthesised from; `docs/HEADLESS_GPU_REPORT.md`
  is the verified recipe for the real GPU under headless Chromium on the dev box.

Where the contract is silent, choose the simplest option consistent with the spec and record it as a
`PLAN DECISION` in the phase plan; where it is wrong, change the contract first.

## Package Structure

```
webgpu-graph-algorithms/
+-- package.json                  # ESM only, sideEffects false, "." + "./browser" + "./node"; webgpu an optional peer + an exact devDependency
+-- project.json                  # Nx project "webgpu-graph-algorithms" (cwd values are the monorepo's)
+-- tsconfig.json                 # lint/typecheck: src/ test/ benchmarks/ scripts/*.d.ts + ../graph-format/src (paths); types node, vitest/globals, vite/client, @webgpu/types
+-- tsconfig.build.json           # emit: src/ only, rootDir ".", outDir dist (-> dist/src/), stripInternal
+-- tsconfig.strict-consumer.json # test/types/*.test-d.ts against dist/*.d.ts under noUncheckedIndexedAccess + exactOptionalPropertyTypes
+-- eslint.config.js              # the root flat config + the layer zones, the entry isolation, the no-navigator / no-process rules, the CPU-package ban
+-- vitest.config.ts              # projects node / node-limits / browser; thresholds 80/80/75/80 when the project set is exactly `node`; BROWSER_FLAGS; the browser commands bridge
+-- scripts/entries.js            # the three bundle entries (shared by build-bundle.js and bundle-types.js)
+-- scripts/build-bundle.js       # one multi-entry vite lib build -> dist/webgpu-graph-algorithms.js, dist/browser.js, dist/node.js, dist/chunks/*
+-- scripts/bundle-types.js       # dist/<entry>.d.ts, one-line re-exports of dist/src/**
+-- scripts/gpu-policy.js (+.d.ts)   # parseGpuRequire / checkAdapter / isSoftwareInfo -- the ONE copy of the adapter policy
+-- scripts/runner-class.js (+.d.ts) # runnerClass(info, env) -- the ONE copy of the benchmark runner-class rule
+-- scripts/gpu-report.js         # adapter report + policy exit code + nvidia-smi sample (imports dist/node.js only)
+-- scripts/run-browser-project.js   # timeout -k 10 600 around the browser project; exit 124 passes iff the JSON says all tests passed
+-- scripts/bench-compare.js      # the 3x regression check against benchmarks/results/<runner-class>.json
+-- src/
|   +-- index.ts                  # the ONLY public barrel; explicit named exports; /// <reference types="@webgpu/types" preserve="true" />
|   +-- errors.ts                 # WebGpuGraphError, WebGpuGraphErrorCode, PASSTHROUGH_FORMAT_CODES, isWebGpuGraphError, hasErrorCode
|   +-- constants.ts              # every numeric constant the package and the WGSL prelude share (interpolated, never retyped as literals)
|   +-- types/                    # public option / result / accelerator TYPES only (context.ts, run.ts, memory.ts, options.ts, layout.ts, accelerator.ts)
|   +-- device/                   # webgpu-constants.ts, acquire.ts, caps.ts, error-scope.ts, lost.ts -- acquisition and capability only (D26)
|   +-- context.ts                # GpuContext: the composition root (probe / create / from; owns PipelineCache, BufferPool, Readback, GraphResidency, Profiler)
|   +-- memory/                   # upload-plan.ts, residency.ts, buffer-pool.ts, readback.ts, lease.ts
|   +-- kernel/                   # wgsl.ts (composeWgsl), prelude.ts, struct-block.ts, pipeline-cache.ts, kernel.ts, dispatch.ts, uniform-ring.ts, batch.ts, profiler.ts
|   +-- kernels.ts                # THE registry of every WgslModuleSpec (the only importer of src/wgsl/**)
|   +-- wgsl/                     # <name>.wgsl.ts: kernel BODIES only (no @group, no override lines)
|   +-- primitives/               # reduce.ts (P1), segmented-reduce.ts (P2); the section 6 primitives each phase pulls in
|   +-- algorithms/               # degree.ts (P1), the walking-skeleton diagnostic; the section 8 families from P7
|   +-- layouts/                  # repulsion-exact.ts (P1), seed.ts, inputs.ts, force-simulation.ts, forceatlas2.ts (P3)
|   +-- accelerator.ts            # createAccelerator(ctx, options?)
|   +-- browser/index.ts          # the ./browser entry: the ONLY directory that may reference navigator
|   +-- node/index.ts             # the ./node entry: the ONLY file that names the "webgpu" module, inside a dynamic import()
+-- test/
|   +-- setup/gpu.ts              # node projects: Dawn once per worker, a FRESH adapter per device, requireGpu(), gpuScale(), the policy
|   +-- setup/browser.ts          # browser project: the policy from import.meta.env, requireBrowserGpu()
|   +-- setup/global.ts           # globalSetup of the node project (the override-matrix coverage from P2)
|   +-- setup/browser-commands.d.ts  # BrowserCommands augmentation + ImportMetaEnv keys (a MODULE: `export {}` first)
|   +-- helpers/ oracle/ fixtures/   # test-side helpers, the f64 CPU references, committed fixtures
|   +-- device/ memory/ kernel/ primitives/ algorithms/ layouts/ sabotage/   # the node project
|   +-- limits/                   # the node-limits project (GPU lane only; P4)
|   +-- browser/                  # the browser smoke project
|   +-- types/*.test-d.ts  index.test.ts  build-output.test.ts  layers.test.ts  errors.test.ts
+-- benchmarks/                   # run.ts (tsx), harness.ts, datasets.ts, upload / roundtrip / layout-exact .bench.ts, layout-run.ts (the end-to-end driver), results/<runner-class>.json and noise-floor.json (checked in), out/ (gitignored)
+-- docs/                         # HEADLESS_GPU_REPORT.md, research/, decisions/G<n>.md
```

The layer rule (spec 3.2), enforced by the eslint zones AND `test/layers.test.ts` (import graph, cycles, greps):
`device < context < memory < kernel < kernels.ts < primitives < algorithms / layouts < accelerator`. A lower
layer never imports a higher one; `context.ts` is the one file that imports memory and kernel from below the
primitives (it constructs them); `src/kernel/**` may `import type` from `src/memory/**` but never a value (the
shared `Binding` / `ArcWindow` types live in `src/types/memory.ts`); `src/wgsl/**` is imported only by
`kernels.ts`; `src/types/**` holds types only; `src/browser` and `src/node` are imported by nothing else in
`src/`; `src/errors.ts` and `src/constants.ts` import nothing. The core never references `navigator`,
`window`, `document`, `process` or the `webgpu` module (spec 2.1); the numeric WebGPU constants live in
`src/device/webgpu-constants.ts` so nothing reads `GPUBufferUsage` at module top level (Dawn installs the
globals only after `Object.assign(globalThis, dawn.globals)`).

## Essential Commands

```bash
pnpm run build:all          # tsc -p tsconfig.build.json, then the multi-entry vite bundle and the d.ts shims
pnpm run lint               # eslint + tsc --noEmit + tsc -p tsconfig.strict-consumer.json (build first)
pnpm run test:node          # vitest run --project=node (the whole node suite; the default adapter)
pnpm run coverage           # the node suite with the 80/80/75/80 thresholds
pnpm run test:browser:ci    # node scripts/run-browser-project.js (SwiftShader unless GRAPHTY_BROWSER_GPU=nvidia)
pnpm run test:limits        # vitest run --project=node-limits (GPU lane only)
pnpm run bench              # tsx benchmarks/run.ts -> benchmarks/out/<runner-class>.json
pnpm run bench:compare      # the 3x regression check against benchmarks/results/<runner-class>.json
pnpm exec tsx benchmarks/layout-run.ts --nodes 100000 --edges 1000000   # the end-to-end exact-tier layout; exit 1 on a non-finite position or an unfinished run
pnpm run gpu:report         # node scripts/gpu-report.js (after build:all): adapter report, policy exit code
pnpm run ready:commit       # build:all, lint, test:node
cd .. && pnpm exec knip     # unused files / exports / dependencies, every workspace
```

## Benchmarks

`benchmarks/run.ts` (tsx) creates a Node context through `createNodeGpuContext`, refuses to time a software adapter unless
`--allow-software`, runs the groups and appends a session to `benchmarks/out/<runner-class>.json`. `benchmarks/harness.ts`
is graph-format's harness with an async `run` bracketed by `device.queue.onSubmittedWorkDone()`, a `teardown` hook and the
`gpu` / `runnerClass` session fields; `runnerClass` is re-exported from `scripts/runner-class.js`, the ONE copy of the rule
that `scripts/gpu-report.js` and `scripts/bench-compare.js` also use, so the file names never drift. Groups and targets
(spec 10.4):

| Group          | Rows                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Target                                                                                |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `upload`       | `residency.core` of the 100k / 1M and 1M / 10M weighted hot prefixes                                                                                                                                                                                                                                                                                                                                                                                                                                           | T-1                                                                                   |
| `roundtrip`    | `degree` + 400 KB readback at 100k; an empty submit + 4-byte `readU32`                                                                                                                                                                                                                                                                                                                                                                                                                                         | T-2, T-3                                                                              |
| `layout-exact` | one `createForceAtlas2` simulation per rung of the exact ladder 1k / 4k / 8k / 16k / 32k / 65k (E = 10n, 2D, `repulsion: "exact"`) plus the 10k frame rung; an untimed clock warm-up burst (>= 500 ms of back-to-back `step(1)`), then `step(1)` after an untimed `reheat()`; two rows per rung: `step(1) wall n=<n> ...` (one iteration + `toScene` + the 12n readback) and `ms/iteration (profiler\|wall) n=<n> ...` (`stats.msPerIteration`: the GPU time of the passes when `timestamp-query` was granted) | T-4 (the `ms/iteration` rows at 10k and 16k), the Node side of T-5 (the 10k wall row) |

The checked-in baselines live in `benchmarks/results/<runner-class>.json` (the dev box: `nvidia-lovelace-driver580.json` --
Dawn spells the RTX 4070 SUPER's architecture `lovelace`; the GPU lane: `gpu-linux-t4.json`, fixed by
`GRAPHTY_RUNNER_CLASS`); the LAST session of a file is the baseline and must carry every group (the append procedure of
`docs/decisions/G3.md` appendix A appends the last out session and refuses one that lacks a group or ran on a software
adapter). `bench:compare` fails the GPU lane above 3x and SKIPS when `gpu-report.json`'s nvidia-smi sample shows
utilisation > 10% or memory growth (T-13). Never commit a session measured while anything else used the card, and never a
software session; watch the SM clock too (`nvidia-smi --query-gpu=clocks.sm,pstate`): NVIDIA's power management leaves
the card at its idle 210 MHz (P8) under sparse sub-millisecond dispatches, and a kernel timed there reads 4-15x slower
(G3 finding G3-F1; the `layout-exact` group's clock warm-up burst is the countermeasure, the `roundtrip` rows measured
after the `upload` group still see it). `test/benchmarks.test.ts` proves the harness, the datasets, the seven branches of
`bench-compare.js`, the ladder table, the 7.8 rule and the driver's helpers without a GPU. Browser numbers (T-3 in
Chromium, T-5) arrive through the `appendBenchRecord` command of `vitest.config.ts` from `bench`-tagged browser tests
(`test/browser/bench.test.ts`, run only under `GRAPHTY_BROWSER_GPU=nvidia`) into `benchmarks/out/<the browser's runner
class>.json` (`nvidia-lovelace-driver0.json` on the dev box: Chromium redacts the driver string); they are recorded in the
gate record and the README, never merged into a Node class file.

`exactMaxNodes` (`EXACT_MAX_NODES` in `src/constants.ts`) is re-fixed from the ladder by the spec 7.8 rule, coded once as
`exactMaxNodesFromLadder` in `benchmarks/layout-exact.bench.ts`: the largest rung with <= 4 ms per iteration, rounded down
to a power of two; its "not slower than the grid tier" clause is re-checked at G4 (`docs/decisions/G3.md` section 3 records
the run of the rule over the committed baseline that produced the value, appendix A the script that applied it).
`benchmarks/layout-run.ts` is the end-to-end driver (seeded G(n, m), `run({ batch })`, prints ms / iteration and the stats,
exits 1 on a non-finite position or a run that neither settled nor reached `maxIter`); it never refuses a software adapter
but labels its timings as not representative. The measured numbers and the missed targets are in `docs/decisions/G1.md`
(P1) and `docs/decisions/G3.md` (P3).

@graphty/graph-format must be built before this package's tests or build run (pnpm's workspace symlink
resolves its `exports` to `dist/`; tsc resolves its sources through `paths`). From the workspace root
`pnpm -r run build:all` orders the packages correctly.

The dev box needs the extracted libEGL tree for BOTH Dawn-node and headless Chromium to see the NVIDIA GPU
(`docs/HEADLESS_GPU_REPORT.md` appendix D): `LD_LIBRARY_PATH=/home/apowers/Projects/webgpu-graph-algorithms/tmp/egl/root/usr/lib/x86_64-linux-gnu`
for the node projects and `scripts/gpu-report.js`; `GRAPHTY_EGL_LIB_DIR=<that dir>` (or the same
`LD_LIBRARY_PATH`) for the browser project. Without it Dawn lists only llvmpipe and Chromium falls back to
SwiftShader -- which the tests turn RED under `GRAPHTY_GPU_REQUIRE=hardware` / `nvidia` and under
`GRAPHTY_BROWSER_GPU=nvidia`, never into a silent pass.

## WGSL Conventions

- WGSL is TypeScript (D9): `src/wgsl/<name>.wgsl.ts` exports ONE constant, the kernel BODY, as
  `` const <name>Wgsl = /* wgsl */ `<body>` ``. A body never contains `@group(` or `override ` lines and has
  exactly one `@compute` entry point; function declarations and `var<workgroup>` declarations are allowed.
- Composition is string concatenation in `src/kernel/wgsl.ts` (`composeWgsl(spec, caps)`), driven by ONE
  `WgslModuleSpec` per module (contract 3.9, 4.2): the composer emits the `@group(g) @binding(b) var<storage, read> name: T;`
  block from `spec.bindings` and the `override NAME: T = default;` lines from `spec.overrideDecls`; `Kernel`
  derives the explicit `GPUBindGroupLayout` and the `bind()` record keys from the same list, so the shader
  and the layout cannot disagree. An unknown override key, a `//@@NAME@@` marker without a snippet, a
  snippet without a marker, or a reserved word in a body is `E_SHADER_COMPILE { stage: "compose" }`.
- The prelude (`src/kernel/prelude.ts`, contract 4.1) is composed too: `INVALID_INDEX`, `WG`,
  `MAX_WORKGROUPS_PER_DIM`, `U32_MAX`, the FA2 floors, `linear_id` (2D grid linearisation), `lowbias32`,
  `mask_bit`, `unpack_u8` are interpolated from `src/constants.ts` and graph-format; a unit test greps every
  `.wgsl.ts` and the prelude for `65535u`, `256u`, `0xFFFFFFFFu`.
- Uniformity (spec 3.5 rule 1): `workgroupBarrier`, `subgroupAdd` and every synchronisation / subgroup
  builtin is reached in UNIFORM control flow: per-invocation work under the `i < n` guard goes into locals,
  every reduction runs unconditionally after the guard; an early `return` before a barrier keys on
  `workgroup_id` and uniforms only. Tint rejects the guarded forms ("must only be called from uniform
  control flow").
- Operator precedence (spec 3.5 rule 2): WGSL refuses to mix `*` with `^` / `&` / `|` without parentheses;
  every hash expression is fully parenthesised.
- Subgroups (D16): kernels read `subgroup_size` / `subgroup_invocation_id` at runtime, obtain a subgroup
  index through an elected-lane `atomicAdd` + `subgroupBroadcast` (no `@builtin(subgroup_id)`: Chromium
  139 lacks the language feature), and size scratch by `SUBGROUP_MIN` (contract 4.3); `SUBGROUP_MAX` stays a
  declared but unused override. Every kernel with a subgroup variant has a twin without it, tested
  in-process.
- Bind groups (spec 3.5): group 0 = the graph (`rowPtr`, `colIdx`, `weights` | dummy, `perm` | dummy --
  four slots whether or not the permutation is the identity), group 1 = algorithm state, group 2 = the
  params uniform (dynamic offset into the `UniformRing`), group 3 = cold arrays. Never more than 8 storage
  buffers per stage: a kernel that needs more is SPLIT, never given a raised limit as a requirement.
- Uniform and storage structs are GENERATED by `UniformBlock` (D20) with the strict 16-byte layout; no
  hand-written struct text, no reliance on `uniform_buffer_standard_layout` (Chromium lacks it).
- Pipeline identity is `(id, overrides, needs present on device, snippets)`; `src/kernels.ts` enumerates
  every spec with its override axes, so the compile matrix, the bind-group-budget test and
  `PipelineCache.warm()` iterate one list.

## House Style

Same as @graphty/graph-format and @graphty/graph-io (spec 3.6): JSDoc on every export (`@param name -
description`, `@returns`); explicit return types; `import { type X, y }` inline qualifiers; `.js` suffixes on
relative imports; camelCase fields; `curly`; `default-case`; no default exports; no `console.log` in `src/`
(`console.warn` only for the residency warning of spec 4.1); options `?: T | undefined` on interface
properties (an optional PARAMETER is `?: T` -- the root eslint config's `no-duplicate-type-constituents`
rejects the redundant `| undefined` there); absent output is `null`, never `undefined`; every throwing call
leaves state unchanged; no `eslint-disable`, no `@ts-expect-error` outside negative type tests; plain ASCII
in every source, test, script, doc and commit message (`--` for dashes; non-ASCII test data built with
`String.fromCharCode`); prettier 4 / 120 / all; knip clean. Results are `Uint32Array<ArrayBuffer>` /
`Float32Array<ArrayBuffer>` through graph-format's `U32` / `F32`; `INVALID_INDEX` is the only sentinel;
never a bitwise operator on an arc index or a byte offset; never a write into a view; never a zero-length
binding (a dummy is bound instead); never a CPU fallback. Tests use the vitest globals (`describe` / `it` /
`expect`) and fast-check for properties. A JSDoc `@internal` tag stands alone on its line (the jsdoc plugin's
`empty-tags` rule); `stripInternal` drops the member from the published declarations either way.

## Testing

Three vitest projects in one config (spec 11.1; benchmarks are a tsx harness, not a project):

| Project          | Setup                                                                                   | Contents                                                                                                                                                                                                                                                                    | Default lane | GPU lane |
| ---------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | -------- |
| `node` (PRIMARY) | `test/setup/gpu.ts` (Dawn once per worker, a FRESH adapter per device; `pool: "forks"`) | every unit / kernel / primitive / algorithm / layout / planner test, `test/*.test.ts` and `test/{device,node,memory,kernel,primitives,algorithms,layouts,oracle,sabotage,types}/**`; carries the 80/80/75/80 thresholds whenever the selected project set is exactly `node` | lavapipe     | NVIDIA   |
| `node-limits`    | same                                                                                    | tests that need limits or time above lavapipe's (`test/limits/**`)                                                                                                                                                                                                          | no           | yes      |
| `browser`        | `test/setup/browser.ts`                                                                 | `test/browser/**`: the light smoke suite on Playwright Chromium, `browser.fileParallelism: false`, flags by `GRAPHTY_BROWSER_GPU`                                                                                                                                           | SwiftShader  | NVIDIA   |

Rules of every test (spec 11.2, 11.9): a wrong result is never a skip; every kernel result is compared to an
oracle or an invariant; every kernel test runs its kernel twice and asserts bitwise equality first; every
`uncapturederror` fails the current test; fixture sizes scale with `gpuScale()` (1 on hardware, 1/50 on a
software adapter). `acquireRaw()` / `acquire()` give a FRESH adapter per device because an adapter is consumed
by its first `requestDevice` (spec 2.2 step 1). The one policy variable (D19), parsed and checked by
`scripts/gpu-policy.js` for the Node setup, the browser setup and `scripts/gpu-report.js`:

| `GRAPHTY_GPU_REQUIRE`        | Meaning                                                                                                        |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------- |
| unset                        | no adapter -> `t.skip(reason)` with the printed `E_NO_ADAPTER` reason (local convenience only)                 |
| `any`                        | an adapter must exist (lavapipe / SwiftShader count); none -> hard failure (the DEFAULT lane)                  |
| `hardware`                   | additionally `!isSoftwareAdapter(info)`                                                                        |
| `nvidia` (any vendor string) | additionally `adapter.info.vendor === value` and, in the browser, `isFallbackAdapter === false` (the GPU lane) |

Environment (spec 12.2; read ONLY by `test/setup/gpu.ts`, `vitest.config.ts` and `scripts/`; `src/` never
reads an environment variable):

| Variable                                  | Default lane                                                | GPU lane                                                | Local (dev box)                                                            |
| ----------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------- |
| `GRAPHTY_GPU_ADAPTER`                     | `llvmpipe`                                                  | unset (Dawn picks the discrete GPU)                     | unset (NVIDIA) or `llvmpipe` to mirror CI                                  |
| `GRAPHTY_GPU_REQUIRE`                     | `any`                                                       | `nvidia`                                                | unset (skip with reason) or `hardware`                                     |
| `GRAPHTY_BROWSER_GPU`                     | `swiftshader` (flag set)                                    | `nvidia` (flag set)                                     | `nvidia`                                                                   |
| `GRAPHTY_GPU_NO_SUBGROUPS`                | a second pass over `test/primitives test/layouts` with `1`  | a second pass over the whole `node` project with `1`    | unset (the twins are also tested in-process)                               |
| `GRAPHTY_DAWN_FEATURES`                   | unset                                                       | unset                                                   | optional Dawn toggles                                                      |
| `GRAPHTY_EGL_LIB_DIR` / `LD_LIBRARY_PATH` | --                                                          | unset (the partner image has `libegl1`; verified at G0) | the extracted tree (`docs/HEADLESS_GPU_REPORT.md` appendix D)              |
| `GRAPHTY_RUNNER_CLASS`                    | unset                                                       | `gpu-linux-t4`                                          | unset                                                                      |
| `VK_DRIVER_FILES`                         | `/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` (determinism) | unset                                                   | unset                                                                      |
| `XDG_RUNTIME_DIR`                         | `/tmp` (silences Mesa)                                      | `/tmp`                                                  | `/tmp`                                                                     |
| `CI`                                      | set by GitHub                                               | set by GitHub                                           | unset: the build-output test's bundle assertions hard-fail only under `CI` |

The two Chromium flag sets (`BROWSER_FLAGS` in `vitest.config.ts`): `nvidia` = `--enable-unsafe-webgpu
--enable-features=Vulkan --use-angle=vulkan --disable-vulkan-surface`; `swiftshader` = `--enable-unsafe-webgpu
--use-angle=swiftshader --enable-unsafe-swiftshader`.

The Vitest 3 per-instance Playwright `launch` spelling (recorded at P0, spec 13 row P0; verified against
vitest 3.2.7 + @vitest/browser 3.2.7 in `packages/node_modules`): `test.browser.instances: [{ browser:
"chromium", launch: { args: [...BROWSER_FLAGS[gpu]], env } }]` -- `BrowserInstanceOption extends
BrowserProviderOptions`, which `@vitest/browser/providers/playwright.d.ts` augments with `launch?:
LaunchOptions` (hence the `/// <reference types="@vitest/browser/providers/playwright" />` at the top of the
config); `fileParallelism` is spelled `test.browser.fileParallelism: false` per project; `test.browser.commands`
is `Record<string, BrowserCommand>` and the browser side reaches it as `commands` from
`@vitest/browser/context`.

Browser-project environment forwarding (contract 2.5): `vitest.config.ts` reads `GRAPHTY_GPU_REQUIRE`,
`GRAPHTY_BROWSER_GPU` and `GRAPHTY_NOISE_FLOOR_WRITE` at config-evaluation time and forwards them to the
browser BOTH through vite `define` of `import.meta.env.<NAME>` and through `test.env`; `test/setup/browser.ts`
reads `import.meta.env` (Chromium has no `process.env`) and `test/browser/webgpu-check.test.ts` asserts the
values arrive. Whether `test.env` alone would suffice is the section 9 item 4 question; the answer is in
"Verified Platform Facts".

Running the suites locally:

```bash
# node project on the NVIDIA GPU (the setup prints [gpu] adapter vendor=nvidia ...)
LD_LIBRARY_PATH=/home/apowers/Projects/webgpu-graph-algorithms/tmp/egl/root/usr/lib/x86_64-linux-gnu GRAPHTY_GPU_REQUIRE=hardware pnpm exec vitest run --project=node
# node project as the default lane runs it (lavapipe)
GRAPHTY_GPU_ADAPTER=llvmpipe GRAPHTY_GPU_REQUIRE=any VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json XDG_RUNTIME_DIR=/tmp pnpm exec vitest run --project=node --coverage
# browser project on SwiftShader (the default lane) and on the NVIDIA GPU
GRAPHTY_BROWSER_GPU=swiftshader GRAPHTY_GPU_REQUIRE=any node scripts/run-browser-project.js
LD_LIBRARY_PATH=/home/apowers/Projects/webgpu-graph-algorithms/tmp/egl/root/usr/lib/x86_64-linux-gnu GRAPHTY_BROWSER_GPU=nvidia GRAPHTY_GPU_REQUIRE=nvidia node scripts/run-browser-project.js
```

The browser runner wraps vitest in `timeout -k 10 600` with `--reporter=json --outputFile=browser-results.json`
and treats exit 124 (the `browser.close()` hang after GPU work on the NVIDIA path, spec 11.6) as a pass iff
the JSON has `numTotalTests > 0 && numFailedTests === 0`. Benchmarks: `benchmarks/results/<runner-class>.json`
is the checked-in baseline per runner class (`scripts/runner-class.js`: `<vendor>-<architecture>-driver<major>`,
or `GRAPHTY_RUNNER_CLASS`), `benchmarks/out/` the gitignored run output.

## Verified Platform Facts

Settled by the review probes (spec R-22, [M] = measured; the probes live in `docs/research/review/probes/`):

| Fact                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Value                                                                                                                                                                                                                                                                                                                         | Probe / source                                                                                                                                                                                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A default Dawn-node device carries the spec-default limits (256 MiB / 128 MiB / 8 storage buffers / offset alignments 256) although the ADAPTER reports 1 TiB and alignment 16                                                                                                                                                                                                                                                                                                                        | plan against `device.limits`, never `adapter.limits`; never assert `=== 256`                                                                                                                                                                                                                                                  | `dawn-facts.mjs` (NVIDIA and llvmpipe)                                                                                                                                                                                                  |
| Dawn-node timestamp queries are unquantised (1,024 ns ticks); Chromium quantises to 100 us                                                                                                                                                                                                                                                                                                                                                                                                            | per-kernel profiling is a Node activity                                                                                                                                                                                                                                                                                       | `design-probe-nvidia.log` item 6                                                                                                                                                                                                        |
| An adapter is consumed by one `requestDevice`; the second call rejects with `OperationError: adapter is "consumed"`                                                                                                                                                                                                                                                                                                                                                                                   | a FRESH adapter per device everywhere                                                                                                                                                                                                                                                                                         | `uncaptured-order-probe.mjs`                                                                                                                                                                                                            |
| `uncapturederror` fires SYNCHRONOUSLY under Dawn-node, before `queue.submit()` returns; asynchronously in browsers                                                                                                                                                                                                                                                                                                                                                                                    | the batch checks the pending-error slot after submit; the browser setup drains after `onSubmittedWorkDone`                                                                                                                                                                                                                    | `uncaptured-order-probe.mjs`, `dawn-latency.mjs`                                                                                                                                                                                        |
| `backend=null` yields an adapter under webgpu@0.4.0 (compiles pipelines, runs nothing)                                                                                                                                                                                                                                                                                                                                                                                                                | the compile matrix uses it                                                                                                                                                                                                                                                                                                    | `dawn-select.mjs`                                                                                                                                                                                                                       |
| Dawn-node adapter info: NVIDIA = vendor `nvidia`, architecture `lovelace`, description `NVIDIA: 580.173.02 580.173.2.0`, subgroups 32/32; llvmpipe = vendor `mesa`, architecture `software`, subgroups 8/8                                                                                                                                                                                                                                                                                            | `isSoftwareAdapter` tests `architecture` first, `isFallbackAdapter` second                                                                                                                                                                                                                                                    | research note 05 section 2.4, `dawn-probe.mjs`                                                                                                                                                                                          |
| Dawn-node 0.4.0 `adapter.info.isFallbackAdapter` is a BOOLEAN (`true` on llvmpipe, `false` on the RTX 4070 SUPER), as in Chromium; note 05 / spec 2.6 measured the deprecated `GPUAdapter.isFallbackAdapter` attribute, which is `undefined` -- spec 2.6 needs the correction (owner)                                                                                                                                                                                                                 | `test/device/acquire.test.ts` asserts the boolean; the policy still keys on `architecture` first                                                                                                                                                                                                                              | re-measured 2026-09-15 (P0-T2 and P0-T3 probes on the installed module)                                                                                                                                                                 |
| `adapter=<substring>` that matches nothing makes `requestAdapter()` (not `create()`) throw "no suitable backends found"                                                                                                                                                                                                                                                                                                                                                                               | the Node setup maps it to an `E_NO_ADAPTER: ...` reason                                                                                                                                                                                                                                                                       | note 05 section 2.2, `dawn-select.mjs`; re-seen 2026-09-15 by the P0-T3 skip-form run                                                                                                                                                   |
| Headless Chromium 139 (Playwright build 1181) reaches the RTX 4070 SUPER only with `libEGL.so.1` on `LD_LIBRARY_PATH` and the four `nvidia` flags; without the library it silently runs SwiftShader (flag sets without `--enable-features=Vulkan`) or hangs at `newPage()` (the four-flag set; the runner's 600 s limit turns that into exit 124 + no JSON = FAIL). Re-measured 2026-09-15 on the installed playwright 1.63.0 (Chrome for Testing 153.0.8010.12, headless shell 1243): same behaviour | `GRAPHTY_EGL_LIB_DIR` / the flag-set assertion of `test/browser/webgpu-check.test.ts`                                                                                                                                                                                                                                         | `docs/HEADLESS_GPU_REPORT.md`                                                                                                                                                                                                           |
| `webgpu@0.6.1` needs `GLIBC_2.38` and fails to load on Ubuntu 22.04 / glibc 2.35; `0.4.0` loads                                                                                                                                                                                                                                                                                                                                                                                                       | the devDependency pin `0.4.0` until P-ENV                                                                                                                                                                                                                                                                                     | note 05 section 2.1                                                                                                                                                                                                                     |
| Vitest 3.2.7 per-instance `launch` spelling                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `instances: [{ browser: "chromium", launch: { args, env } }]`, `browser.fileParallelism: false`                                                                                                                                                                                                                               | this file, "Testing" (P0-T3 re-read the installed 3.2.7 copy: `providers/playwright.d.ts` line 17 `launch?: LaunchOptions`; `vitest/dist/chunks/reporters.d.*.d.ts` `instances?: BrowserInstanceOption[]`, `fileParallelism?: boolean`) |
| tsc 5.9 drops a bare `/// <reference types="@webgpu/types" />` from the declaration output (TS 5.5+ copies a directive only when written `preserve="true"`), and emits it AFTER a detached header comment -- still a valid pragma                                                                                                                                                                                                                                                                     | every source entry starts with `/// <reference types="@webgpu/types" preserve="true" />`; `test/build-output.test.ts` checks the emitted d.ts carries it in the leading-comment region; a consumer without `@webgpu/types` in its `types` resolves `GPU` through it (probe `tmp/p0-t3-probe/`, positive and negative control) | measured 2026-09-15 (P0-T3)                                                                                                                                                                                                             |
| `pool: "forks"` with the Dawn addon                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | verified by graph-format's suite; `threads` untested and not used                                                                                                                                                                                                                                                             | `packages/graph-format/vitest.config.ts`                                                                                                                                                                                                |
| Chromium SwiftShader adapter                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | vendor `google`, architecture `swiftshader`, `isFallbackAdapter` true, subgroups 4 / 4, `maxComputeInvocationsPerWorkgroup` 256 (WG 256); adapter class `google-swiftshader-browser`                                                                                                                                          | `test/browser/entry.test.ts` `[entry] probe ...` line and `test/browser/skeleton.test.ts` `[skeleton] adapter ...` line, 2026-09-15 (P1-T7)                                                                                             |
| Chromium on the RTX 4070 SUPER                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | vendor `nvidia`, architecture `lovelace` (the SAME spelling as Dawn-node, so the adapter classes differ only by the runtime suffix: `nvidia-lovelace-browser` / `nvidia-lovelace-node`), `isFallbackAdapter` false, subgroups 32 / 32                                                                                         | same, NVIDIA run                                                                                                                                                                                                                        |
| The 17M-item map in Chromium                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | `LINEAR_ID_CHECKSUM` (877493955) reproduced bitwise on SwiftShader and NVIDIA-Chromium; the 2D fill plus the 64 MiB readback through the staging ring takes 0.14 s on SwiftShader, 0.08 s on NVIDIA-Chromium                                                                                                                  | `test/browser/skeleton.test.ts`, P1-T7 Step 30                                                                                                                                                                                          |
| Chromium f32 tile noise vs the f64 pair sum (K3 + K4 on karate)                                                                                                                                                                                                                                                                                                                                                                                                                                       | SwiftShader 3.413e-7 (bitwise identical to lavapipe's force), NVIDIA-Chromium 2.571e-7 (bitwise identical to NVIDIA-Dawn's force, trace, degree and reduce outputs); both under `fa2-skeleton.force` = 3.413e-6                                                                                                               | the `oracle-f64` rows of `benchmarks/results/noise-floor.json`; `docs/decisions/G1.md` section 4                                                                                                                                        |
| `browser.close()` after GPU work                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | NVIDIA: exit 0 (no hang, 14 / 14 tests); SwiftShader: exit 0 (14 / 14); the wrapper's exit-124 rule stays for the GPU lane                                                                                                                                                                                                    | `node scripts/run-browser-project.js`, P1-T7 Step 33, 2026-09-15                                                                                                                                                                        |
| Default-lane coverage run                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 00:04 wall on lavapipe for `--project=node --coverage` (34 files, 494 tests; T-12 budget 15 min for the lane); 97.83 / 100 / 96.17 / 97.83 against 80 / 80 / 75 / 80                                                                                                                                                          | P1-T7 Step 35                                                                                                                                                                                                                           |
| Dawn-node object model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `createBuffer` / `destroy` / `mapAsync` live on the prototypes of extensible wrappers; own-property proxies and `delete` work (the LeakCounter relies on it); `degree` maps exactly one staging slot per call                                                                                                                 | `test/leak.test.ts`, 2026-09-15 probe on llvmpipe and NVIDIA                                                                                                                                                                            |
| Dawn-node upload throughput of one `queue.writeBuffer` (the arena hot prefix)                                                                                                                                                                                                                                                                                                                                                                                                                         | 2.76 GB/s at 16.4 MB (5.9 ms), 1.29 GB/s at 164 MB (127 ms, above the 100 ms T-1 target; owner decision in G1.md section 7); `degree` + 400 KB readback 0.87 ms; the empty-submit round trip 0.083 ms median with a 0.04-0.69 ms spread                                                                                       | `benchmarks/results/nvidia-lovelace-driver580.json`, 2026-09-15                                                                                                                                                                         |

To verify at G0 (P0-T4 replaces every "unverified" below with the measured answer and the run that
measured it):

| Fact                                                                                                                                                       | Status                                                                                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `test.env` alone populates `import.meta.env` in the browser project, or `define` is required (contract 2.5; section 9 item 4)                              | unverified -- P0-T3's define-off experiment result is in the P0 PR description; P0-T4 records it here |
| The `gpu-linux-t4` image: Ubuntu release / glibc / NVIDIA driver version (decides whether the `webgpu@0.4.0` pin could be lifted on the lane before P-ENV) | unverified (first `gpu.yml` dispatch)                                                                 |
| `sudo modprobe nvidia nvidia_uvm` needed on the T4 image before the driver answers, or a no-op                                                             | unverified (the "Driver up" step of `gpu.yml`)                                                        |
| Headless Chromium finds the T4 without `xvfb-run -a`                                                                                                       | unverified (the "Browser smoke on NVIDIA" step of `gpu.yml`)                                          |
| `libegl1` present on the T4 image (so the `LD_LIBRARY_PATH` workaround stays local)                                                                        | unverified                                                                                            |
| lavapipe on an actual GitHub `ubuntu-latest` runner acquires an adapter under `GRAPHTY_GPU_REQUIRE=any`                                                    | unverified (first `ci.yml` run)                                                                       |
| `GRAPHTY_GPU_REQUIRE=nvidia` on the software lane is RED at the `gpu-report.js` step (no `tee`)                                                            | unverified (the deliberate red run)                                                                   |
| Firefox / Safari exposure of `subgroups` / `timestamp-query` on Linux CI                                                                                   | not verified; Chromium only (Q-22)                                                                    |

### Settled at G2 (P2-T3; the evidence is docs/decisions/G2.md)

| Fact                                                                                                                                                                                        | Dawn-node (webgpu@0.4.0)                                                                                                                                                                                                                                                                                                                                                                                                                                                | Chromium (Playwright 1.63.0, Chrome for Testing 153.0.8010.12, headless shell 1243)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Settled by                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| When `uncapturederror` is delivered for a bad bind group (a STORAGE-only buffer in a UNIFORM slot) or an invalid command buffer (a copy out of range)                                       | SYNCHRONOUSLY: the listener has run before `createBindGroup()`, `finish()` and `queue.submit()` return (event count 0 -> 1 across the failing call), on lavapipe (the default lane's adapter) and on the RTX 4070 SUPER alike; `CommandBatch.submit()` therefore reads the context's pending-error slot right after `submit()` and rejects ITS OWN readback with `E_VALIDATION { batchId, batchLabel }` (spec 5.7). The Tesla T4 lane has not run yet (G2.md section 1) | ASYNCHRONOUSLY: the count is still 0 when the failing call returns and the event has arrived by the time `queue.onSubmittedWorkDone()` resolves (SwiftShader and NVIDIA Vulkan alike; the second event of the invalid-command-buffer case one macrotask later). The P2 batch meets the error at the `assertReady()` after the staging slot's `mapAsync` resolves, so the same batch's readback still rejects with its batch id (measured path `readback` on both adapters); an error that arrives after the batch is thrown from the next `assertReady()`, which is why `test/setup/browser.ts` awaits `onSubmittedWorkDone()` and drains the slot in `afterEach`                                                                                                                                                                                                                                                                                                                                      | `tmp/g2/uncaptured-order.mjs` (Dawn-node) and `tmp/g2/uncaptured-order-browser.mjs` (Chromium), 2026-09-15, outputs in `tmp/g2/uncaptured-order-*.txt`; `test/kernel/batch.test.ts` (a deliberately bad bind group rejects the same batch's readback), `test/browser/batch.test.ts` (`delivered through readback`); first seen in the review probe `docs/research/review/probes/uncaptured-order-probe.mjs` (NVIDIA only) |
| A hand-written misaligned uniform struct (the WGSL 14.4.5 "Invalid" example: a 4-byte struct-typed member followed by an `f32` at offset 4, `test/browser/uniform-layout.test.ts` BAD_WGSL) | ACCEPTED by `createShaderModule` and pipeline creation: Dawn-node lists `uniform_buffer_standard_layout` in `wgslLanguageFeatures` (lavapipe and NVIDIA), so a uniform layout bug is invisible under Node (R-10)                                                                                                                                                                                                                                                        | ACCEPTED as well: Chromium 153 lists `uniform_buffer_standard_layout` (13 language features) and accepts the struct with and without `--disable-dawn-features=allow_unsafe_apis`; Chromium 145 (headless shell 1208) lists it and accepts too. The REJECTION spec 5.3 pinned from Chromium 139 exists on this box only on Chromium 143 (headless shell 1200, feature not listed) with `--disable-dawn-features=allow_unsafe_apis`, at `createShaderModule` (Tint: "'uniform' storage requires that the number of bytes between the start of the previous member of type struct and the current member be a multiple of 16 bytes, but there are currently 4 bytes between 'a' and 'b'"); under `--enable-unsafe-webgpu` alone (Dawn's `allow_unsafe_apis`) 143 accepts it. `UniformBlock`'s generated text for the same fields compiles on every runtime and version, which is why every params and state struct is generated (D20); the checklist's "rejected on Chromium" leg is owner decision G2-D1 | `test/kernel/uniform-layout.test.ts` and `test/browser/uniform-layout.test.ts` (both assert `accepted === wgslLanguageFeatures.has("uniform_buffer_standard_layout")` and print the measured facts), `tmp/g2/uniform-layout-probe.mjs` over headless shells 1200 / 1208 / 1243 (2026-09-15, `tmp/g2/uniform-layout-probe.txt`)                                                                                            |

### Settled at G3 (P3-T7; the evidence is docs/decisions/G3.md)

| Fact                                                                                                                 | Value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Probe / source                                                                                                                                                                                   |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `timestamp-query` under Dawn-node                                                                                    | granted on the RTX 4070 SUPER (`ctx.profiler.enabled` true: the `layout-exact` rows say `(profiler)`) and granted on lavapipe as well (the same rows say `(profiler)`; `layout-run.ts` prints `profiler on`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `pnpm exec tsx benchmarks/run.ts --no-save layout-exact` on both adapters (P3-T7 Step 10)                                                                                                        |
| NVIDIA power management under sparse dispatches                                                                      | after the `upload` group's CPU-heavy setup (six `fromEdgeArrays` of 16-164 MB) the SM clock falls to P8 / 210 MHz (2475-2730 MHz at P0) and stays there under the sub-millisecond `step(1)` pattern: the profiler then reports the ladder 4-15x slower (1k 0.99 ms instead of 0.096 ms); ~200 ms of dense GPU work raises the clock again; a plain 164 MB `writeBuffer` does not trigger it                                                                                                                                                                                                                                                                                    | `nvidia-smi --query-gpu=clocks.sm,pstate -lms 100` during `benchmarks/run.ts upload layout-exact` (`tmp/g3/clocks-upload-ladder.csv`, `tmp/g3/upload-probe2.ts`); G3.md section 10 finding G3-F1 |
| The exact ladder on the RTX 4070 SUPER (GPU ms per iteration, full FA2 iteration, E = 10n, 2D, at the working clock) | 1k 0.096 / 4k 0.255 / 8k 0.478 / 10k 0.586 / 16k 1.052 / 32k 2.560 / 65k 8.405; `step(1)` wall at 10k 0.738 ms                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `benchmarks/results/nvidia-lovelace-driver580.json` session 2026-09-16T02:07:45.933Z (P3-T7 Step 11)                                                                                             |
| `exactMaxNodes`                                                                                                      | 32768 by the 7.8 budget clause (2.560 ms at 32k, 8.405 ms at 65k); the grid clause is re-checked at G4                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `exactMaxNodesFromLadder` over the baseline's last session (G3.md section 3, appendix A; P3-T7 Step 13)                                                                                          |
| The exact tier at 100k / 1M end to end (Node, batches of 8, 100 iterations)                                          | 18.971 ms per iteration wall, 16.975 ms GPU (the last batch); every position finite                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `benchmarks/layout-run.ts` (P3-T7 Step 8); G3.md section 8                                                                                                                                       |
| T-5 at 10k in Chromium on the RTX 4070 SUPER (`step(1)` + the 12n readback)                                          | 2.400 ms (target <= 6; 50 frames after 5 warm frames, 100 us quantised); Node: 0.738 ms                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `test/browser/bench.test.ts` through `appendBenchRecord` into `benchmarks/out/nvidia-lovelace-driver0.json` (P3-T6; P3-T7 Step 18)                                                               |
| The FA2 suite on lavapipe at `gpuScale` sizes                                                                        | 39 s wall for `test/layouts test/oracle test/sabotage/fa2*.test.ts test/noise-floor.test.ts` (budget 180 s; 689 tests, 8 red -- P3-T5's open items, G3.md section 10); the whole `--project=node --coverage` run 44 s (T-12 budget 900 s; coverage 94.88 / 98.5 / 94.58 / 94.88 with `--coverage.reportOnFailure`); the no-subgroups `test/primitives test/layouts` pass 26 s                                                                                                                                                                                                                                                                                                  | P3-T7 Step 16; G3.md section 5                                                                                                                                                                   |
| The frame loop on the four adapters                                                                                  | `test/layouts/frame-loop.test.ts` green on NVIDIA-Dawn and lavapipe (11 / 11); `test/browser/forceatlas2.test.ts` green on SwiftShader and NVIDIA-Chromium; `browser.close()` after the FA2 work: NVIDIA exit 0 (39 / 39 in 15 s), SwiftShader exit 0 (38 passed + the bench test skipped, 27 s)                                                                                                                                                                                                                                                                                                                                                                               | P3-T7 Step 18                                                                                                                                                                                    |
| f32 division in a compute shader under Dawn                                                                          | NOT correctly rounded on the RTX 4070 SUPER (WGSL / Vulkan grant 2.5 ULP): 27% of random quotients 1 ulp off, 5 of 32,768 2 ulp off, and `x / (x / 2)` != 2 for 15% of the inputs (2.2% above 2); lavapipe divides exactly. A predicate on an exact ratio (K4's `swing / traction > 2` at the paper-mode first iteration, where traction is exactly half the swing) must be written in its multiplied form `swing > 2.0 * traction` (contract 4.5 CONTRACT DECISION K4-1; G3.md finding G3-F6)                                                                                                                                                                                 | `tmp/p3-fix/div-probe.mjs` (G3.md appendix A) on both Dawn adapters, 2026-09-16                                                                                                                  |
| The paper-mode ForceAtlas2 trajectory                                                                                | chaotic beyond any derivable tolerance: the free-running trace diverges x1.1 .. x1.5 per iteration from f32 rounding (x4 .. x5 per iteration over iterations 2 .. 9 on random1k / grid10, where the per-node swing m\|F(t) - F(t-1)\| cancels), and the f64 oracle misses the spec 11.4 caps against itself under a one-f32-ulp start perturbation (0.86 .. 1.71 through 50); one GPU iteration re-synchronised to the oracle's state agrees within 1e-5 at every iteration. The trace, twin, distributional and setPosition tests compare per iteration (G3.md G3-F3 / G3-F4; P3-T5 PLAN DECISIONS 17 / 18)                                                                   | `tmp/p3-fix/measure-nvidia.txt`, `dist-ensemble*-nvidia.txt`, 2026-09-16                                                                                                                         |
| WebKit (Safari 26 on iPadOS / macOS, Metal) and pipeline constants                                                   | `createComputePipeline` fails with `Compute library failed creation` whenever `constants` names an `override` the entry point never reads (`HAS_WEIGHTS` in `degree`, `TIER` in the thread-per-row `segmented-reduce` / `fa2-attraction`, an unused module override in a minimal body); constants for read overrides compile (`SWING_MODE`, `STRONG_GRAVITY`, `OP`, ...). The spec allows unused constants ("validating GPUProgrammableStage": the constant is not required to be statically used by the entry point), so this is a WebKit bug; `composeWgsl` returns `constants` = the referenced subset and `PipelineCache` supplies only those (`ComposedModule.constants`) | `demo/compile.html` on an iPad (68 of 111 matrix cases red before the fix, all bisect variants with unread constants red, `bisect:no-overrides` green), 2026-09-16                               |

## Adding an Algorithm / a Kernel

1. Types: add the option / result types to `src/types/` (types only; structural mirrors of the CPU packages
   until W1, D27).
2. Body: `src/wgsl/<name>.wgsl.ts` exporting `<name>Wgsl` -- the body only, written to the uniformity and
   precedence rules; no `@group(`, no `override `, constants interpolated from `src/constants.ts`.
3. Registry: one `WgslModuleSpec` entry in `src/kernels.ts` with its `bindings` (group 0 graph / 1 state /
   2 params / 3 cold), `overrideDecls`, the override axes, `needs`, `uniforms` (a `UniformBlock`) and
   `snippets`; the compile matrix, the bind-group-budget test and `PipelineCache.warm()` pick it up from
   there.
4. Driver: a plain async function in `src/primitives/` or `src/algorithms/` (`(ctx, snapshot, ...args,
options?) => Promise<Result>`, index-aligned typed arrays, no id mapping); every scratch buffer through a
   `Lease`; every dispatch through `CommandBatch`; empty ranges bind a dummy and skip the dispatch.
5. Accelerator: the method on `createAccelerator`'s object in `src/accelerator.ts`, and the barrel export
   in `src/index.ts` (update `test/index.test.ts` and `test/types/public-api.test-d.ts`).
6. Oracle: `test/oracle/<name>.ts`, an independent index-based f64 reference (never derived from the kernel).
7. Tests: the differential test against the oracle on the fixture list (empty graph, one node, self-loop,
   karate, grids, paths, stars, complete, seeded G(n, m), hubs; directed / undirected; weighted / not), the
   run-twice bitwise check, the subgroup twin in-process, at least three sabotage mutations in
   `test/helpers/sabotage.ts` each failing the test by >= 10x the tolerance, an `inspect()` stage comparison,
   and a noise-floor row (spec 11.9); a browser smoke test where the runtime differs.
8. Integration: the dispatcher method in @graphty/algorithms (spec 9.2) or the adapter in @graphty/layout
   (spec 9.4) in the CPU packages.

## Adding a Layout Model

The shared state machine is `ForceSimulation` (`src/layouts/force-simulation.ts`: buffers, in-flight batches, the
readback into the owner's stride-3 array, the settle window, the fixed mask, the `setPosition` override list, the
trace, `run()`); a layout is a `ForceModel<Options, Stats>` it consumes by composition (spec 7.19; contract 3.13).
`ForceAtlas2Model` (`src/layouts/forceatlas2.ts`) is the reference; the FR model of P5 and the spring-electrical
preset follow the same steps:

1. Types: the option record in `src/types/options.ts` (the CPU package's names and defaults, every field
   `?: T | undefined`; plus its `Resolved<Model>Options`), the stats record extending `LayoutStatsBase` in
   `src/types/layout.ts`, and the method on the `LayoutAccelerator` mirror in `src/types/accelerator.ts`.
2. Kernels: the bodies in `src/wgsl/<model>-*.wgsl.ts` and their registry entries in `src/kernels.ts` (group 0
   the graph through `graphBindings` / `graphOverrides`, group 1 the model state, group 2 the params slot of the
   `UniformRing` with a dynamic offset; reuse `fill` for zeroing and `fa2-to-scene` for the scene unpack when
   the positions are `vec4f`). The model's `UniformBlock`s: `params` must declare the eight shared fields `n`,
   `dim`, `flags`, `iterationIndex`, `seed`, `scale`, `center` (a `vec4f`), `settleThreshold` and fit in
   `UNIFORM_SLOT_BYTES`; `state` must declare `centroid`, `min`, `max` (`vec4f`), `rmsRadius`, `radius`,
   `meanDisplacement` (`f32`), `iteration`, `settledCount` (`u32`) and fit in `STATE_HEADER_BYTES`; `trace` is
   one record per iteration (`MAX_ITERATIONS_PER_STEP` of them follow the header in the same buffer).
3. The model class in `src/layouts/<model>.ts` implementing every `ForceModel` member: `kind`, `stages` (the
   kernel stage names `inspect()` / `debugRunStages` accept), `params` / `state` / `trace`, `buffers(n, dim)`
   (the model-owned buffers beyond the shared set, e.g. `oldForce`, `velocity` -- never alias a writable slot
   with another binding of the same dispatch), `inputs(s, options)` (mass through `resolveNodeMass`, weights
   through `resolveWeights`), `overrides(options)` (the model's override set; `USE_PERM` / `HAS_WEIGHTS` are
   merged in by the simulation), `specs(overrides, subgroups)` (for `warm()` and the compile matrix),
   `bind(resources, overrides)` (compile through `resources.pipelines`, build the bind groups from
   `resources.buffer(name)`), `paramsFor(iteration, options)` (the per-iteration uniform values; `iteration` is
   the global index), `recordIteration(batch, slot, tier, upTo?)` (the kernel sequence of one iteration; honour
   `upTo` by stopping after the named stage), `onLoad` / `onReheat` / `onSetParams(patch, writer)` (the controller
   resets through the `StateWriter`), `readStats(state, trace)` (decode the header and the k trace records; leave
   `msPerIteration` null -- the simulation fills it).
4. The factory: `resolve<Model>Options(options, previous?)` applying the defaults from `src/constants.ts` and
   validating every range with `E_INVALID_ARGUMENT { argument }`, and `create<Model>(ctx, options?)` returning
   `new ForceSimulation(ctx, model, resolved, resolveLayoutTuning(options), resolve)`; the method on
   `createAccelerator`'s object; the barrel export and the pinned lists (`test/index.test.ts`,
   `test/types/public-api.test-d.ts`, `test/types/options.test-d.ts`).
5. Oracle: `test/oracle/<model>.ts` in f64 with an f32 variant, index-based, exposing the same per-iteration
   stages the kernels have (`stages`) and the trace; cross-checked against an independent implementation's
   committed trajectory fixtures when one exists (`test/fixtures/<impl>/generate.py` + JSON, as the NetworkX
   fixtures of P3-T4).
6. Tests (spec 11.4 applied to the model): force parity per stage through `inspect()`, trace parity against the
   f32 and f64 oracles, distributional parity (`test/helpers/metrics.ts`), the behaviour pins, the fast-check
   properties (fixed nodes, `setPosition`, settle, reheat, remapped `load`), the force-sum invariant where the
   law is antisymmetric, the subgroup twins in-process, lifecycle (leak 0, `E_RELEASED`, device loss), the
   frame-loop test (`test/helpers/frame-loop.ts`) on node and in the browser, at least three sabotage rows per
   kernel in `test/helpers/sabotage.ts` (and the phase in `SABOTAGE_PHASES`), a noise-floor row and a
   `tolerances` entry per tolerance (`noiseFloorFor(id)`; never a literal), a browser smoke test.
7. Benchmarks and the gate: a `layout-<model>.bench.ts` group registered in `benchmarks/run.ts` with a
   per-iteration row per rung of the ladder (keep the clock warm-up burst of `layout-exact.bench.ts`), the T-row
   in the README table, and the gate record `docs/decisions/G<n>.md` mapping every gate item to its test or number
   (G3.md is the template).
8. Integration: the adapter in @graphty/layout (spec 9.4) and, in the element, the route from the layout type
   to the accelerator method.

## Distribution

- Root entry: `dist/webgpu-graph-algorithms.js` (ES module; `@graphty/graph-format`, `webgpu`,
  `@graphty/algorithms` and `@graphty/layout` external); types `dist/webgpu-graph-algorithms.d.ts` (a one-line
  re-export of `dist/src/index.d.ts`).
- Subpaths `@graphty/webgpu-graph-algorithms/browser` and `/node`: `dist/browser.js`, `dist/node.js` and their
  one-line d.ts shims, built by the same vite invocation so shared code lives once under `dist/chunks/` (one
  `WebGpuGraphError` class whichever entry a consumer loads). Each source entry starts with
  `/// <reference types="@webgpu/types" preserve="true" />` so the published declarations resolve the `GPU*`
  names (the `preserve="true"` is what makes tsc 5.5+ copy the directive into the emitted d.ts).
- `webgpu` is IMPORTED by exactly one file, `src/node/index.ts`, only inside `await import("webgpu")` within a
  function body; `test/layers.test.ts` asserts it at the source level and `test/build-output.test.ts` asserts no `webgpu` import specifier in the root and browser bundles and a dynamic one only in
  `dist/node.js`, hard-failing when the bundle is absent under `CI`.
- No `"browser"` field, no `"node"` export condition: an explicit subpath is unambiguous.
- `stripInternal: true`: members tagged `@internal` (`GpuContext.residency`, the `loadModule` test seam and the other internal members the contract marks)
  are absent from the published d.ts; the strict-consumer compile proves the published surface under
  `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes`.
- `webgpu@0.4.0` is the devDependency pin and the `E_NO_WEBGPU` install hint; both move together at P-ENV.

## Rules that never bend

- `./node` is imported only by Node entry points and tests (spec 2.5 item 5): no browser-side file may import
  it, because Safari fails on a dynamic import of a missing module even before the import runs.
- Never create fallbacks if WebGPU isn't supported (root CLAUDE.md): no CPU path, no WebGL path, no silent
  software-adapter acceptance in `src/`; the TEST layer may skip with a printed `E_NO_ADAPTER` reason only
  under the unset policy, and a wrong result is never a skip.
- Never `git add` / `commit` / `push` (the owner commits from `tmp/commit-p<N>.sh`); never `sudo`; plain
  ASCII everywhere; no `eslint-disable`, no `@ts-expect-error` outside negative type tests.
