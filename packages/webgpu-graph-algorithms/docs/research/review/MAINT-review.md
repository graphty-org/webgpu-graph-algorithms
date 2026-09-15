# Maintainability review of design/webgpu-acceleration-plan.md

Reviewer lens: Maintainability (the person who maintains this code for five years inside
graphty-monorepo). Document reviewed in full: `/home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md`
(3328 lines, dated 2026-09-14). Line numbers below are of that file.

Sources consulted: graph-format exports (`packages/graph-format/src/index.ts`, `src/types/snapshot.ts`),
the graph-io skeleton the plan mirrors (`packages/graph-io/{package.json,tsconfig*.json,vitest.config.ts,scripts/*.js,CLAUDE.md}`),
`packages/tsconfig.base.json`, `packages/knip.config.ts`, the accepted design
`/home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md` (10, 14.2-14.6, 16),
the monorepo `algorithms/vitest.config.ts`, `.github/workflows/ci.yml`, `nx.json`, `commitlint.config.js`,
research notes 02 and 07, and four probe scripts under `tmp/webgpu-plan/review/probes/` run against the
installed `webgpu@0.4.0` (Dawn) on the RTX 4070 SUPER and on Mesa llvmpipe.

Probe results relied on (all outputs reproduced in the probe files' directory):

- `maint-sync-probe.mjs` (NVIDIA): a TS `constants` key that the WGSL does not declare is a
  `GPUPipelineError` ("Pipeline overridable constant "LINLOG" not found"); an override declared in the
  prelude but unreferenced by the entry point is accepted when set; `maxComputeWorkgroupsPerDimension`
  is 65535 on the adapter and cannot be raised; default `maxStorageBuffersPerShaderStage` is 8,
  raisable to 16.
- `binding-mismatch-probe.mjs` (llvmpipe): with an EXPLICIT pipeline layout, Dawn rejects a
  read-only layout entry for a `read_write` shader var, a `storage` layout entry for a `read` var, AND
  a binding index that differs from the shader's -- each as a `GPUPipelineError` at pipeline creation,
  i.e. only when that exact variant is compiled.
- `uncaptured-order-probe.mjs` (NVIDIA and llvmpipe): Dawn-node 0.4.0 delivers `uncapturederror`
  SYNCHRONOUSLY inside the API call that fails (both events are counted before `submit()` returns);
  a second `requestDevice()` on the same adapter throws `adapter is "consumed"`.
- `nodom/` (tsc 5.9.3, `skipLibCheck: true` as in `packages/tsconfig.base.json`): `lib: ["ES2020"]` +
  `types: ["node", "@webgpu/types"]` compiles a core module that names `GPUDevice` / `GPUBuffer`, and
  `navigator` in that core is `error TS2304` while a browser entry reading
  `(globalThis as { navigator?: ... }).navigator` compiles.

## Summary judgement

The plan is unusually thorough on WHAT the kernels do and on verification, and its big structural
calls are right (algorithms as plain functions; two thin entries; CPU packages own the accelerator
interfaces; no registry). Its maintainability weak spots are the places where two artefacts must agree
by hand: WGSL text versus TypeScript (storage bindings, the `state` struct, three copies of 256 / 65535,
`DIM` as both override and uniform), the CPU-owned option types versus the GPU package's copies, the
CI lane presets restated in three places, and one base class (`ForceSimulation`) plus one composition
root (`GpuContext`) that are set up to absorb every future feature. None of these blocks
implementation; several would be expensive to change after P3. 20 findings: 9 major, 11 minor, 0
blockers.

## Findings

### MAINT-1 (major) -- storage bindings are declared twice (WGSL text and TS BindingSpec) with no generator, unlike uniforms

- Section 3.5 / 5.1 / 7.5, lines 767-775 (`WgslModuleSpec` has `overrides`, `needs`, `uniforms`,
  `snippets` -- no bindings), 797-806 (bind-group conventions), 1097-1101 (`Kernel` binds a compiled
  pipeline to a separate `BindingSpec` of `{ group, binding, kind, name }`), 1438-1445 (the
  fa2-attraction module hand-writes `@group(0) @binding(0) var<storage, read> rowPtr` ... eight lines).
- Claim: every one of the ~50-80 kernel modules the plan implies (five FA2 + toScene + fill, seven
  grid, ~15 primitive kernels, one to five per algorithm) will carry a hand-written declaration block
  whose group / binding index / access mode must match a TS `BindingSpec` written elsewhere, and a
  third name (the `Record<name, Binding>` key of `Kernel.bind`) must match both. The plan invented
  `UniformBlock` (D20) precisely because "the two cannot disagree" is worth a generator, then left the
  larger surface (storage bindings) as two hand-kept copies.
- Evidence: `binding-mismatch-probe.mjs` shows that an index or access-mode mismatch is a
  `GPUPipelineError` raised only when that variant is created -- so drift is invisible until the
  compile matrix (which the plan never says how to enumerate, MAINT-6) reaches that variant on that
  runtime. 5.1 already needs a rule ("declares that binding `read_write` in every variant") that
  exists only because the declaration is duplicated per variant.
- Fix (plan text): in 3.5 add `readonly bindings: readonly BindingDecl[]` to `WgslModuleSpec`
  (`{ group, binding, name, kind: "storage" | "storage-ro" | "uniform", wgslType: "array<u32>" | ... }`)
  and state that the composer EMITS the `@group/@binding var<...>` block from it (as `UniformBlock`
  emits the struct), that `Kernel` derives the explicit `GPUBindGroupLayout` and the `bind()` record
  keys from the same list, and that a `.wgsl.ts` body contains no `@group(` token (a unit test greps
  for it). Delete the separate `BindingSpec` from 5.1. Move 7.5's example to show the body only.

### MAINT-2 (major) -- `//@@NAME@@` snippet splicing is an untyped home-grown preprocessor with four splice axes and no compose-time validation

- Section 3.5, lines 767-775 (`snippets?: Record<string, string>` substituted at `//@@NAME@@`
  markers), 791-795 (pipeline identity includes `hash(spec.snippets)`), 1183 (segmentedReduce takes a
  "value snippet"), 1198 (advance takes `functor: { visit: snippet, filter: snippet }`), 1262-1266
  (`needs: ["subgroups"]` splices `enable subgroups;` plus helpers), 1134-1140 (uniform struct text
  spliced).
- Claim: the composer has four independent text-splicing mechanisms (overrides, `needs`, uniform
  structs, operator snippets). A snippet is a free-form string: nothing declares which variables it
  may reference, whether every marker was filled, whether a supplied snippet has a marker, or what
  its parameter names are; the Gunrock functors it imitates are C++ templates that the compiler
  checks. Compile errors are reported against the COMPOSED source (line 1090), so the maintainer
  maps line numbers back by hand. This is the file that becomes a preprocessor.
- Evidence: `maint-sync-probe.mjs` shows override-name errors surface only at pipeline creation;
  a snippet that references a variable that exists in one host kernel and not another fails the
  same way, per variant, at runtime.
- Fix (plan text): replace `snippets` with typed functors: `readonly functors?: readonly WgslFunctor[]`
  where `WgslFunctor = { readonly slot: "visit" | "filter" | "value" | "law"; readonly params: readonly
  [name: string, type: string][]; readonly returns: string; readonly body: string }`; the composer
  emits a complete `fn <slot>(params) -> returns { body }` and the host kernel calls it by that fixed
  name, so a functor can reference only its parameters. Compose-time errors (`E_SHADER_COMPILE
  { stage: "compose" }`) for an unfilled slot or an unknown slot. Turn `needs` into a list of named
  helper modules (`"subgroups" | "hash" | "mask" | "pos3"`), keep the prelude to constants and
  overrides only, and record in 5.1 that the compilation-info formatter subtracts the prelude and
  helper line counts so messages point at the kernel body.

### MAINT-3 (major) -- the `state` storage struct and the `partials` regions are hand-declared WGSL structs whose byte offsets TypeScript reads; D20 covers uniform blocks only

- Section 7.3 / 7.10 / 7.19 / 11.4, lines 1380-1381 (`partials` "per-workgroup regions ... padded to
  64 B", `state` = "128 + k x 32" bytes with 14 named scalars, three xyz triples and a `k`-slot trace
  region), 1659 (`S.trace[P.iterationIndex] = vec4f(...)`), 1847 ("copy the state trace into
  `stats`"), 2727 ("read from `state` after each `step(1)`"), 1134-1140 (`UniformBlock` generates
  uniform structs only).
- Claim: the host reads `speed, speedEfficiency, swing, traction, centroid xyz, radius, min xyz, max
  xyz, cellSize, meanDisplacement, iteration, settledCount` and the trace from a storage buffer whose
  layout is a WGSL struct `S` written by hand. `vec3` members in a struct next to `f32` scalars are
  the classic 16-byte-alignment offset trap; storage layout rules differ from uniform rules, so the
  Chromium negative test of D20 does not cover it. Every LayoutStats field, every trace-parity test
  and the grid `GridSpec` block depend on those offsets.
- Fix (plan text): in 5.3 generalise `UniformBlock` to `StructBlock.define(fields, { layout: "uniform"
  | "storage" })` that emits the WGSL struct text, `write(view, values)` AND `read(view): values`, and
  state that `state`, `partials`, `GridSpec` and every other host-visible block are declared through
  it; add to the 11.3 kinds a round-trip test (write on the host, copy through a trivial kernel, read
  back) per declared block on both runtimes.

### MAINT-4 (major) -- `GpuContext` is the composition root but is placed in the lowest layer and references every layer above it, contradicting the plan's own layering rule

- Section 2.2 / 3.1 / 3.2 / 3.3, lines 306-308 (step 6 "Create the singletons: `PipelineCache`,
  `BufferPool`, `Readback` staging ring, `GraphResidency`, `Profiler`"), 329-338 (`ctx.calibrate()`
  runs "the exact-tile repulsion kernel ... and the grid build"), 496 (`device/` contains
  `calibrate.ts`), 535 (device layer = `GpuContext`, `calibrate`), 543 ("A lower layer never imports a
  higher one"), 548 (`GpuContext` "owns ... pipeline cache, buffer pool, staging ring and residency;
  `accelerator()`"), 608-609 (`accelerator(): GpuAccelerator`, `calibrate()` on the class).
- Claim: `device/gpu-context.ts` must value-import `memory/`, `kernel/`, `layouts/` (for calibrate)
  and `src/accelerator.ts` (for `accelerator()`), while `accelerator.ts` imports algorithms and layouts
  which import the context type: the device layer depends on the top layer. The rule at line 543 is
  therefore unenforceable from day one, and a maintainer reading the layer table will put new
  cross-cutting features on `GpuContext` because it already knows everything.
- Fix (plan text): rename the layer `device` to acquisition-only (`GpuDevice` handle: adapter /
  device / caps / error scopes / lost) and introduce `src/context.ts` ABOVE memory and kernel as the
  named composition root (`GpuContext` = device handle + caches + residency); remove `accelerator()`
  and `calibrate()` from the class; export `createAccelerator(ctx)` from `src/accelerator.ts` and
  `calibrateLayout(ctx, options?)` from `src/layouts/calibrate.ts`; update 2.2 step 6, 3.1, 3.2, 3.3
  and 9.5 (`element.setAccelerator(createAccelerator(ctx))`). Add an eslint `import/no-restricted-paths`
  zone table to 3.2 so the rule is checked.

### MAINT-5 (major) -- `ForceSimulation` (a base class with subclasses) is the file that will grow without bound

- Section 3.2 / 7.19 / 7.20, lines 540 (layouts layer: `ForceSimulation` (shared) plus two
  subclasses), 563 (class one-liner: "buffers, in-flight batches, readback, settle, fixed mask,
  `setPosition` overrides, trace; subclasses supply the per-iteration kernel sequence"), 1824-1830
  (state machine incl. generation counter), 1832-1849 (`step()` algorithm), 1893-1911 (FR needs
  temperature slots written per batch, `FR_APPLY`, a different reheat rule), 1912-1922 (the
  spring-electrical preset adds a velocity buffer and a Verlet integrator), 689-691 (`setParams`,
  `run`, `flush`, `reheat`, `stats` on the public object).
- Claim: one class already owns eleven concerns (device buffers, units, seeding, fixed mask, drag
  override list, in-flight batch queue with generation, readback copy, settle window, params /
  uniform slots, trace / stats, batch driver) and the plan adds per-model variation by inheritance.
  Every later feature (nodeSize, cluster tree, shared Babylon device, worker transfer, KK) lands in
  the base or in a subclass override of a base method. After five years this is a 3,000-line file
  with an implicit protocol between base and subclasses.
- Fix (plan text): in 7.19 replace inheritance with composition and name the pieces: `ForceModel`
  (per-model strategy: `buffers(n)`, `overrides`, `recordIteration(batch, slot, tier)`, `onReheat()`,
  `paramsFor(iteration)`), `InFlightQueue` (generation, `maxInFlight`, coalescing, the readback copy
  minus overridden rows), `PositionBridge` (owner array, units, seeding, `setPosition` override list,
  fixed mask), `SettleDetector` (window / threshold from the state block). `ForceSimulation` becomes
  the ~200-line coordinator that wires them; `createForceAtlas2` / `createFruchtermanReingold` / the
  spring preset are `ForceModel`s. Adjust the 3.2 class list accordingly.

### MAINT-6 (major) -- adding an algorithm (or kernel, or layout) touches ~15 hand-maintained places and the plan has no recipe; two of the lists cannot even be derived

- Section 3.1 / 3.3 / 5.1 / 11.3, lines 490 (barrel: explicit named exports), 495 (`types/`
  options / results / accelerator mirror), 499-503 (kernel WGSL in `src/wgsl/`, driver in
  `src/algorithms/<family>/`, `accelerator.ts`), 630-651 (19 exported functions), 704-716
  (`GpuAccelerator` restates each function's signature by hand, "one per shipped algorithm"),
  721-727 (option types re-declared structurally), 1091-1096 (compile matrix compiles "every module
  in every override combination the package uses" -- no enumeration mechanism), 2713 ("the barrel
  export list pinned"), 2711 / 2714 (compile matrix and bind-group budget as separate tests), 1241
  (a CPU reference per primitive in `oracle.ts`), plus the monorepo side 2282-2306 (interface method
  + `*ResultLike`), 2308-2314 (dispatcher method), 2401-2409 (element adapter).
- Claim: for one new algorithm the maintainer edits the kernel file(s), the driver, `types/options.ts`,
  the result type, the barrel, `GpuAccelerator` (interface and object), the accelerator mirror, the
  oracle, the differential test, the pinned export test, the compile-matrix list (however it is
  built), a benchmark group, the README table, and then three files in two other packages. The
  sibling package's `CLAUDE.md` has an "Adding a format" recipe with exactly this kind of list
  (`packages/graph-io/CLAUDE.md` lines 126-159); the plan defers everything to a package `CLAUDE.md`
  whose contents it never specifies (MAINT-19). Two of the lists are not derivable from code: "every
  module in every override combination the package uses" and the accelerator method list.
- Fix (plan text): (a) define `KernelDef` in 3.5 as the single object per kernel (`id`, `body`,
  `bindings` (MAINT-1), `overrideAxes: Record<name, readonly values[]>`, `uniforms`, `functors`,
  `needs`) and a `src/kernels.ts` array that lists every `KernelDef`; the compile matrix, the
  bind-group-budget test and `PipelineCache.warm()` iterate that array and the cartesian product of
  `overrideAxes` (knip flags a kernel that is not listed); (b) co-locate the WGSL with its driver
  (`src/layouts/fa2/attraction.wgsl.ts` next to `attraction.ts`; coverage exclude becomes
  `src/**/*.wgsl.ts`); (c) build `GpuAccelerator` from `const ALGORITHMS = { pageRank, ... } as const`
  with a generic `bindContext(ctx, ALGORITHMS)` so the type and the object come from one table; (d)
  add a numbered "Adding an algorithm / Adding a kernel / Adding a layout model" recipe to section 3
  (and mandate it in the package `CLAUDE.md`) that lists every file, so the count is visible and can
  be driven down.

### MAINT-7 (major) -- after W1 the published d.ts imports types from `@graphty/algorithms` / `@graphty/layout` that are devDependencies only; before W1 the option types are copies; no runtime contract version exists across the three packages

- Section 3.3 / 9.1 / 9.8 / 13, lines 668-670 (`GpuLayoutSimulation extends LayoutSimulation` in the
  PUBLIC API, "a structural copy lives in src/types until W1"), 721-727 (option types "re-declare the
  CPU packages' option shapes STRUCTURALLY ... at W1 a type test asserts mutual assignability"),
  2248-2251 (CPU packages imported "ONLY as devDependencies for type conformance"), 2533 (W1:
  "replace the structural mirrors with `import type` from the real packages (devDependencies)"),
  2541-2545 ("it declares NO peer on algorithms / layout (types only, dev) ... the type conformance
  test in W1 is what guards drift"), 3169 (P10 repeats it).
- Claim: `tsc -p tsconfig.build.json` preserves `import type { LayoutSimulation } from "@graphty/layout"`
  in `dist/src/types/*.d.ts`; a consumer that installs `@graphty/webgpu-graph-algorithms` without
  `@graphty/layout` gets TS2307 (or silent `any` under `skipLibCheck`). That is an undeclared
  dependency in a published package, and neither `build-output.test.ts` (line 2713) nor the
  strict-consumer compile (which resolves through the workspace) catches it. Before W1 the copies of
  `ForceAtlas2Options` etc. drift with only a manual assignability test; at run time the accelerator's
  `kind: string` (line 2283) is the only identity, so an app that pairs GPU package X with algorithms
  package Y whose interface changed (a required member, a renamed option, a changed default) fails
  late or silently.
- Fix (plan text): (a) in 2.5 / 3.1 declare `@graphty/algorithms` and `@graphty/layout` as OPTIONAL
  `peerDependencies` (types only; the runtime never imports them) so the d.ts imports are declared,
  and keep them as devDependencies for the conformance test; (b) delete `src/types/options.ts` at W1
  (option types come from the owners via `import type`); (c) add `readonly contract: 1` (a numeric
  literal, bumped with any breaking interface change, a `feat!:` in the owning CPU package) to
  `AlgorithmAccelerator` and `LayoutAccelerator` in 9.2 / 9.3, and state that `accelerated()` and
  `createSimulation()` throw `E_ACCELERATOR_CONTRACT` when `acc.contract !== 1`; (d) in 9.2 derive the
  method list from `typeof indexed` (`Partial<Accelerated<typeof indexed>>` with one `Widen<F64 ->
  NumericVector>` mapped type for results) instead of hand-listing 19 methods and 12 `*ResultLike`
  interfaces, so an algorithm added to `indexed` extends the interface without a second edit.

### MAINT-8 (major) -- the 80/80/75/80 coverage thresholds are never enforced on any lane as specified

- Section 11.1 / 11.8 / 12.3 / 12.5, lines 2662 (the `node` project "carries the 80/80/75/80
  thresholds when run whole"), 2818-2822 (thresholds "skipped when a single `--project` is selected,
  `algorithms/vitest.config.ts` pattern"), 2950 (default lane: `vitest run --project=node --coverage`),
  3094-3096 (monorepo shard: the same command); G1 at 3159 requires "coverage >= 80/80/75/80 on the
  node project".
- Claim: every CI invocation passes `--project=node`, which by the cited pattern disables thresholds
  (`/home/apowers/Projects/graphty-monorepo/algorithms/vitest.config.ts` lines 62-72: thresholds are
  `undefined` when `--project=` is present, "checked at CI level after merging coverage"). The staging
  repo has no merge step, so until W1 no lane checks the number the gate demands; running "whole"
  (no `--project`) would also run `browser`, `node-limits` and `bench`, which is not something any lane
  does.
- Fix (plan text): in 11.8 invert the condition -- thresholds are enforced when `--project=node` is
  selected (the only project that produces coverage) and disabled only when `COVERAGE_DIR` is set
  (monorepo sharded runs, where `tools/merge-coverage.sh --ci` checks them); update 11.1 and G1 to
  say "on the `node` project with `--coverage`".

### MAINT-9 (major) -- four per-call resource objects (`Lease`, `CommandBatch`, `UniformRing`, `Readback`) with an unstated choreography that every algorithm repeats, and two owners of the staging buffer

- Section 3.2 / 4.4 / 5.3 / 5.8 / 7.3, lines 554-561 (four classes), 561 (`CommandBatch` "carries the
  staging buffer for the batch's readback"), 973-981 (`Readback` "owns a ring of ... staging buffers
  ... pick a free staging buffer ... or grow the ring"), 1387 (staging ring owned by `Readback`, "one
  slot per in-flight batch"), 966-971 (`Lease` `try/finally` per algorithm), 1141-1147 (`UniformRing`
  slot writes per batch), 1205 (abort: "releases its lease and rejects"), 1216-1223 (submission
  model), 1995-1999 (every algorithm "records k rounds per submit ... checks `signal` between batches").
- Claim: each of the 19 algorithms hand-writes the same sequence (lease, batch, uniform slots,
  error scope for large buffers, signal check, readback, release on both paths) and can get any step
  wrong in a way the tests of that one algorithm must catch. The staging buffer has two described
  owners (`CommandBatch` "carries" it, `Readback` "owns" and "picks" it), which is exactly the kind of
  boundary that produces the "buffer in use by a queued copy" validation error the plan cites.
- Fix (plan text): in 5.8 introduce one scope, `ctx.run(label, options, async (run) => ...)`, where
  `run` exposes `scratch(bytes, label)` (lease), `uniform(block, values)` (ring slot), `pass()` /
  `dispatch()` (batch recording), `submit()`, `read(src, bytes, dest?)` / `readU32()` (staging from the
  ring), `checkpoint()` (signal + pending uncaptured error), and releases everything on resolve or
  reject; `Lease`, `CommandBatch`, `UniformRing` become internal to `run` (drop them from the 3.2
  public class list) and `Readback` is the single owner of staging buffers, handing a slot to a run
  for its lifetime. Section 8's algorithm contract then reads "every algorithm is one `ctx.run`".

### MAINT-10 (minor) -- the numbers 256, 65535 and the meaning of `DIM` are each declared in two or three places that must agree by hand

- Lines 494 (`constants.ts`: `WORKGROUP_SIZE = 256`, `MAX_WORKGROUPS_PER_DIM = 65535`), 782
  (prelude: `override WG: u32 = 256u;`), 786 (prelude: `linear_id` hard-codes `65535u`), 1101-1102
  (`WG = min(256, caps.limits.maxComputeInvocationsPerWorkgroup)`), 1115-1121 (planner uses
  `limits.maxComputeWorkgroupsPerDimension` for the 1D test and a literal 65535 for the 2D split),
  1080 (`DIM` listed among the overrides), 1729 ("`dim` is a uniform"), 1752 (7.14: `dim` binding =
  uniform), 1382 (params uniform lists `dim`).
- Claim: `linear_id` in WGSL and `plan1d` in TS must use the same x-extent; today both say 65535 by
  coincidence of two literals (the probe confirms the 4070 reports exactly 65535 and cannot raise it,
  but the plan lists the limit as raisable at lines 258-261 and the planner reads it from caps). `DIM` is
  an override in 5.1 and a uniform in 7.13 / 7.14 -- a maintainer cannot tell which.
- Fix (plan text): state in 3.5 that the composer interpolates `constants.ts` values into the
  prelude (`override WG: u32 = ${WORKGROUP_SIZE}u;`, `${MAX_WORKGROUPS_PER_DIM}u` in `linear_id`) and
  that no numeric literal that also exists in `constants.ts` may appear in a `.wgsl.ts` (a unit test
  greps for `65535u` / `256u`); remove `maxComputeWorkgroupsPerDimension` from `RaisableLimit` or make
  the 2D split use the raised value consistently in both places; decide `dim` once (uniform, per
  7.13) and delete `DIM` from the 5.1 override list.

### MAINT-11 (minor) -- the layer table, the class list and the API section disagree on names, and six public type names are never defined

- Lines 535-541 (layer table names `UploadPlanner`, `DispatchPlanner`, `WgslComposer`, `calibrate`,
  `Profiler` as modules), 546-564 (class list omits `Profiler`; names `DispatchPlanner` "pure
  functions"), 886 (`planUpload(...)` "is a pure function"), 3159 (`composeWgsl`), 318
  (`AdapterSummary`), 329 (`GpuCalibration`, `CalibrateOptions`), 689 (`ForceAtlas2Params |
  FruchtermanReingoldParams` -- everywhere else the type is `ForceAtlas2Options`), 842 (`view()` takes
  a hand-written union of eight view names while graph-format exports `ViewName`,
  `packages/graph-format/src/types/snapshot.ts` lines 226-243).
- Claim: an implementer will create both a `WgslComposer` class and a `composeWgsl` function, or a
  `Params` type next to `Options`, because the document licenses both; the residency's view-name
  union will drift from the format's `ViewName` (`"symmetric"`, the scalar views).
- Fix (plan text): one name per thing (functions: `planUpload`, `plan1d`, `composeWgsl`,
  `calibrateLayout`; classes only where the class list says so); add `Profiler` to the class list;
  define `AdapterSummary`, `GpuCalibration`, `CalibrateOptions` in 3.3 or drop them; replace
  `ForceAtlas2Params` with `ForceAtlas2Options`; type `view()`'s name as `Extract<ViewName, "reverse" |
  "coo" | ...>` so the format's union is the source.

### MAINT-12 (minor) -- `lib: DOM` for the whole package makes "the core never touches `navigator`" a lint rule instead of a compile error, and `caps.runtime` is a leak channel into the core

- Lines 525-527 (`lib: ["ES2020", "DOM", "DOM.Iterable"]` because "the browser entry needs
  `navigator` typings"), 296-299 (`runtime: "browser" | "node" | "unknown"` on `GpuCaps`, "set by the
  ENTRY"), 592 (public field), 366-368 (`no-restricted-globals` lint rule as the guard), 112 (G2).
- Claim: the probe under `review/probes/nodom/` shows `lib: ["ES2020"]` + `types: ["node",
  "@webgpu/types"]` with the base config's `skipLibCheck: true` compiles the core and makes
  `navigator` a TS2304 error, while the browser entry reads `globalThis.navigator` through a local
  interface. graph-format already uses `lib: ["ES2020"]` (`packages/graph-format/tsconfig.json` line
  9). A `runtime` field on the caps record is the natural place for the next `if (caps.runtime ===
  "node")` branch; nothing in the plan reads it (2.6 explicitly sizes for the browser number).
- Fix (plan text): 3.1 tsconfig `lib: ["ES2020"]`; 3.4 browser entry reads
  `(globalThis as { navigator?: { gpu?: GPU } }).navigator?.gpu`; keep the lint rule as belt and
  braces; drop `runtime` from `GpuCaps` (keep it, if wanted, on `AdapterSummary` for display).

### MAINT-13 (minor) -- the core planner branches on `caps.software` although 2.2 and 7.8 promise it never does

- Lines 1127 (`planGridStride` default `maxGroups = caps.software ? 64 : 4096`) versus 336-338
  (`"auto"` "never depends on `caps.software`") and 1613-1614 ("the package never guesses from
  adapter strings").
- Claim: one policy stated twice and violated once; the next maintainer will add the second
  `software` branch citing line 1127.
- Fix (plan text): make `maxGroups` a `GpuContextOptions.tuning.gridStrideGroups` value with one
  default (4096) that `test/setup/gpu.ts` lowers on software adapters; strike the `caps.software`
  clause at 1127 and add "no `caps.software` read outside `probe` / `isSoftwareAdapter`" to 2.4's lint
  list.

### MAINT-14 (minor) -- CI / environment presets are restated in three places, the benchmark results path has three spellings, the subgroup twin needs a second CI invocation that covers only `test/primitives`, and `.mjs` scripts fall outside the knip / eslint globs

- Lines 2886-2898 (env table), 2941-2951 (staging yaml default lane env + a second `vitest run` with
  `GRAPHTY_GPU_NO_SUBGROUPS=1 ... test/primitives`), 2986-3001 (GPU lane env, `--outputJson
  bench/results.json`, `scripts/bench-compare.mjs bench/results.json`), 3091-3096 (monorepo diff
  repeats the env block), 2622 (`benchmarks/results/<host>-node<version>.json`), 2664
  (`benchmarks/results/<runner-class>.json`), 3007 (`scripts/gpu-report.mjs`), 2709 (subgroup twin
  "forced ... on any adapter" by the env var), 1637 / 1262 (the repulsion epilogue and the layout
  kernels also have subgroup twins -- never run without subgroups on any lane).
- Claim: the presets will drift between the staging yaml, the monorepo yaml and the table (they
  already differ in the results path); the twin test is a second process per lane and misses
  layouts / algorithms; `packages/knip.config.ts` globs are `scripts/**/*.{ts,js}` so `.mjs` is
  neither linted nor tracked (the siblings use `scripts/*.js` under `"type": "module"`).
- Fix (plan text): one `scripts/ci-lane.js <default|gpu>` (package scripts `test:lane:default`,
  `test:lane:gpu`) that sets the env and runs the sequence, called by both workflows, and the 12.2
  table becomes "generated from that script"; one `BENCH_RESULTS_DIR = benchmarks/results/` with the
  file name `<runner-class>.json`; replace `GRAPHTY_GPU_NO_SUBGROUPS` with a per-context option
  (`GpuContextOptions.features: { subgroups?: false }`) so the differential tests run both twins on
  the same device in one process (add a `variants` axis to the 11.3 differential kinds); `.js` for
  every script.

### MAINT-15 (minor) -- `test/helpers/oracle.ts` is one file holding ~30 CPU reference implementations and becomes a second algorithms package until W1

- Lines 506 (`test/helpers/... oracle.ts` as one file), 538 (every primitive "with a CPU reference in
  `test/helpers/oracle.ts`"), 1241-1243, 2692-2696 (algorithm differentials "vs `test/helpers/oracle.ts`
  (index-based ports of design Ports 1-6) until W1, then vs `indexed.*`"), 2533 (W1 switch), 3162
  (P3: the FA2 oracle "becomes the spec of the L1 `ForceAtlas2Simulation`"), 3173-3178 (P8 / P9 / P11
  may run before P10).
- Claim: 12 primitive references (radix sort, scan, segmented reduce, ...), two layouts in f64 and
  every algorithm's reference (Brandes, Dijkstra, Louvain, ...) in ONE file is thousands of lines;
  the algorithm half is thrown away at W1, and the FA2 half is a second implementation of the 7.2
  table that must track the real L1 rewrite by hand until W1 ("a mismatch ... blocks W1").
- Fix (plan text): `test/oracle/<name>.ts`, one per primitive / algorithm, mirroring `src/` paths;
  cap oracles to what ships before W1 (FA2, FR, PageRank family, WCC); order P10 (W1) before P8 / P9 /
  P11 so the frontier, BC and community slices test against `indexed.*` directly and no oracle is
  written for them; state that the FA2 / FR oracle files are moved (not rewritten) into
  `layout/src/simulation/` at L1 and that after W1 `test/oracle/` contains only primitive references.

### MAINT-16 (minor) -- the settlement / in-flight knobs live in two option types owned by two packages, and 7.14 contradicts 9.3 about which package owns them

- Lines 692-697 (`GpuLayoutTuning` has `settleThreshold`, `settleWindow`, `iterationsPerStep`,
  `maxInFlight`), 1757-1758 (7.14 lists `settleThreshold` / `settleWindow` / `iterationsPerStep` /
  `maxInFlight` as "new", GPU-only), 2344-2349 (layout-owned `ForceAtlas2Options` ALSO has
  `settleThreshold`, `settleWindow`, `iterationsPerStep`), 2352-2357 (`LayoutAccelerator.forceAtlas2?(
  options?: ForceAtlas2Options)`), 2459 (element config `behavior.layout.maxInFlight`).
- Claim: the same three names in two types with two owners means two defaults to keep equal and an
  undefined precedence when both are given; `maxInFlight` is in the GPU type only, so the element's
  config value cannot reach the simulation through the typed `LayoutAccelerator` interface without a
  cast.
- Fix (plan text): one home -- the layout-owned `ForceAtlas2Options` / `FruchtermanReingoldOptions`
  carry `settleThreshold`, `settleWindow`, `iterationsPerStep` and `maxInFlight` (the CPU simulation
  ignores `maxInFlight`); `GpuLayoutTuning` keeps only `repulsion`, `exactMaxNodes`, `nearMax`,
  `deterministic`, `gridMax2D`, `gridMax3D`, `compat`; fix the 7.14 rows to say "layout-owned option".

### MAINT-17 (minor) -- the public surface exposes the memory layer and several speculative members

- Lines 606 (`readonly residency: GraphResidency` on the public class, which drags `Binding`,
  `CoreBinding`, `ViewBinding`, `ColumnBinding`, `ArrayBinding`, `ArcWindow`, `ResidencyStats` and the
  `core()` / `view()` / `column()` / `array()` upload API into the public d.ts), 588
  (`wgslFeatures` "informational only (never relied on)"), 657 (`precision: "f32"` on
  `GpuBetweennessResult` only, although every score result is f32), 643 (`degree()` "walking-skeleton
  kernel; kept as a diagnostic" in the root barrel), 1915-1922 (`forceLaw: "coulomb"`,
  `velocityVerlet` reserved), 1350 (`ADJUST_SIZES` reserved), 1351 (`edgeWeightInfluence` reserved).
- Claim: once `ctx.residency` is public, every rename inside `memory/` is a breaking change and
  consumers will upload arrays themselves; the reserved names are API promises with no tests.
- Fix (plan text): mark `residency` `@internal` and add `stripInternal: true` to `tsconfig.build.json`
  (the graph-format pattern, `packages/graph-format/tsconfig.build.json` line 9); expose `ctx.stats()`
  and `ctx.residentBytes` only; delete `wgslFeatures`; put `precision: "f32"` on `GpuScoresResult` (all
  GPU scores) or nowhere; move `degree` to `test/`; strike the reserved names from 7.2 / 7.20 and keep
  them in 14.2 as questions.

### MAINT-18 (minor) -- an undeclared departure from the design's injection spelling, and one result shape that breaks the "one result-writing loop"

- Lines 207 ("Everything else in sections 10, 14.3-14.6, 15 and 16 is honoured as written"), 2401-2409
  (adapters call `accelerated(this.graph.accelerator).pageRank(s, opts)`), 2273 / 2296
  (`SsspResultLike { dist, predArc }`), 2508-2509 (adapter does "`isInPath` via `pathTo()`
  reconstruction on the CPU from `predArc`"), versus design 14.5 lines 4239-4240 ("graphty-element
  injects it as `runAlgorithm(snapshot, { accelerator: gpu })`") and design 14.2 line 3734
  (`SsspResult { dist; predArc; pathTo(t); pathEdges(t) }`).
- Claim: the injection spelling differs from the design without a DEPARTURE entry, and the GPU SSSP
  result lacks the `pathTo` / `pathEdges` methods the design gives the CPU result, so the element's
  SSSP adapter needs a GPU-specific branch -- contrary to 9.4 item 3.
- Fix (plan text): add DEPARTURE-5 to 1.5 (per-method `accelerated()` dispatcher instead of
  `runAlgorithm(snapshot, { accelerator })`, or add the latter as an alias in 9.2); give
  `GpuSsspResult` the same `pathTo` / `pathEdges` closures (CPU-side walks over `predArc`, shared
  helper owned by `@graphty/algorithms` per design 14.6) and align `SsspResultLike` with the design.

### MAINT-19 (minor) -- the document mixes decisions, the judge trail and reference tables that must track code; six places defer to a package `CLAUDE.md` whose contents are never specified

- Lines 26-59 ("How to read": graft markers, Review notes), 209-218 and every section's "Review
  notes", 1324-1362 (7.2 formula table: the shared spec of the L1 CPU rewrite AND the GPU kernel),
  1737-1758 (7.14 parity), 2501-2523 (9.7 parity), 2629-2645 (T-table regenerated from
  `benchmarks/results/`), 2886-2898 (env table), 3013-3045 (host-side runner recipe), 410, 2670, 3158,
  3228, 3249 (defer to `CLAUDE.md`), 3251 (Q-18 where the plan lives).
- Claim: a 3.3k-line file where a maintainer must know that 7.2 is normative for two packages, that
  10.4 is regenerated, and that "Review notes" are history, will not be kept current; the sibling
  packages keep exactly this material in `CLAUDE.md` (`packages/graph-io/CLAUDE.md` sections
  "Package Structure", "Adding a format", "House Style", "Distribution").
- Fix (plan text): add a "Where this document goes on acceptance" paragraph to Q-18: decisions stay in
  `design/webgpu/plan.md`; "Review notes" and graft markers move to `design/webgpu/review-log.md`;
  7.2 + 7.14 become `design/webgpu/forceatlas2-reference.md` cited by both `layout` and the GPU
  package; 3.5-3.6 conventions, the "Adding a ..." recipes (MAINT-6), the verified platform facts
  (R-22), the env table and the Vitest launch spelling are the enumerated contents of the package
  `CLAUDE.md`; 12.4 becomes `docs/ci-runner.md`; 10.4's table lives in the README and is regenerated.
  List those `CLAUDE.md` sections in 3.1.

### MAINT-20 (minor) -- two test-harness contract gaps the probes expose: an adapter is consumed by one `requestDevice`, and the deferred "throw from the next public call" misattributes kernel errors

- Lines 265 / 277-279 (`options.adapter` given -> `requestDevice` on it), 2676-2684 (`acquire()`
  generalised from `gpu-upload.test.ts`, "acquire once per worker"), 2710 (device-loss test: "a
  new context + `load()` works afterwards"), 1202 (uncaptured errors "stored and thrown from the
  NEXT public call"), 2690-2691 ("An `uncapturederror` listener fails the current test").
- Claim: `uncaptured-order-probe.mjs` shows Dawn-node 0.4.0 rejects a second `requestDevice()` on the
  same adapter (`adapter is "consumed"`), so a cached adapter in `acquire()` breaks every test that
  creates a second context (device loss, `from(device)`), and shows the `uncapturederror` event is
  delivered synchronously inside the failing call -- so the plan's "next public call" default throws
  from an unrelated call site (a debugging trap) when it could reject the batch that caused it.
- Fix (plan text): 11.2 -- `acquire()` returns `{ gpu, newDevice(): Promise<GPUDevice> }` that requests
  a FRESH adapter per device, and 2.2 documents that `options.adapter` is consumed by `create`; 5.7 --
  after `submit()` `CommandBatch` checks the context's pending-error slot and rejects ITS OWN readback
  promise with `E_VALIDATION { batchId, label }`; the "next public call" throw remains only for errors
  that arrive between batches; the browser project (asynchronous delivery) drains pending errors in
  `afterEach` after `queue.onSubmittedWorkDone()`.

## Things checked and found acceptable (not findings)

- Naming against graph-format: `nodeCount` / `arcCount` / `edgeCount`, `INVALID_INDEX`, `F32` / `U32` /
  `NumericVector` / `NodeMask` / `CoreArrayName`, `serial`, `hotByteLength`, `renumberPartition`,
  `foldArcs`, `paddedU32View`, `degreeOrder({ of })` all match `packages/graph-format/src/index.ts`
  and `src/types/snapshot.ts`; `parent` (singular) matches design line 3725-3726.
- Two entry files confine the runtime split; `webgpu` appears in one file behind a dynamic import; no
  `"browser"` field / conditional export; the build-output test guards it.
- Algorithms as plain functions; no registry; `sideEffects: false`; explicit named exports.
- The vitest four-project layout, `pool: "forks"`, the Playwright flag constant, the CI shard shape and
  the root touch points match the monorepo's `ci.yml` (lines 236-300) and `packages/README.md`.
- Flakiness: the timing targets are gate measurements in `bench` (GPU lane, never required), not
  assertions in the required lane; determinism is asserted only where the plan claims bitwise
  reproducibility on one device; the `browser.close()` hang has a hard timeout.

## Overall confidence

I would bet on this plan producing a working GPU ForceAtlas2 (P0-P3) and on its integration
mechanism surviving in the monorepo; the CPU-owned interfaces plus one dispatcher per package are the
right shape and the two-entry runtime split is well guarded. The single biggest maintainability risk
is the WGSL / TypeScript seam: the plan generates uniform structs so "the two cannot disagree" but
leaves storage bindings, the `state` struct, the functor snippets and three numeric constants as
hand-kept duplicates across ~50-80 kernel modules, each mismatch detectable only when that exact
variant is compiled on that runtime. Fixing that (MAINT-1/2/3/10 -- one `KernelDef` per kernel from
which declarations, layouts, the compile matrix and the budget test are derived) costs a few days in
P1-P2 and nothing later; not fixing it is the five-year tax. Second is the pair of growth files
(`ForceSimulation`, `GpuContext`) and the undeclared type dependency the W1 `import type` step creates.
