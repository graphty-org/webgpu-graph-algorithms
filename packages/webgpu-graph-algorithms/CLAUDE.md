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
+-- benchmarks/                   # run.ts (tsx), harness.ts, datasets.ts, <group>.bench.ts, results/<runner-class>.json (checked in), out/ (gitignored)
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
pnpm run gpu:report         # node scripts/gpu-report.js (after build:all): adapter report, policy exit code
pnpm run ready:commit       # build:all, lint, test:node
cd .. && pnpm exec knip     # unused files / exports / dependencies, every workspace
```

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

## Adding an Algorithm / a Kernel / a Layout Model

1. Types: add the option / result types to `src/types/` (types only; structural mirrors of the CPU packages
   until W1, D27).
2. Body: `src/wgsl/<name>.wgsl.ts` exporting `<name>Wgsl` -- the body only, written to the uniformity and
   precedence rules; no `@group(`, no `override `, constants interpolated from `src/constants.ts`.
3. Registry: one `WgslModuleSpec` entry in `src/kernels.ts` with its `bindings` (group 0 graph / 1 state /
   2 params / 3 cold), `overrideDecls`, the override axes, `needs`, `uniforms` (a `UniformBlock`) and
   `snippets`; the compile matrix, the bind-group-budget test and `PipelineCache.warm()` pick it up from
   there.
4. Driver: a plain async function in `src/primitives/` or `src/algorithms/` (`(ctx, snapshot, ...args,
options?) => Promise<Result>`, index-aligned typed arrays, no id mapping), or a `ForceModel` in
   `src/layouts/` consumed by `ForceSimulation`; every scratch buffer through a `Lease`; every dispatch
   through `CommandBatch`; empty ranges bind a dummy and skip the dispatch.
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
  function body; `test/layers.test.ts` asserts it at the source level and `test/build-output.test.ts` (from
  P1-T7) asserts no `webgpu` import specifier in the root and browser bundles and a dynamic one only in
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
