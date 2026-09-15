# MAINT verdicts -- skeptical verification of the Maintainability review

Target: /home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md
Verifier method: every cited plan line re-read; every cited file opened; the three
reviewer probes re-run on BOTH adapters (NVIDIA RTX 4070 SUPER with
LD_LIBRARY_PATH=tmp/egl/root/usr/lib/x86_64-linux-gnu, and Mesa llvmpipe) from
tmp/webgpu-plan/review/probes/; two new probe files added under
tmp/webgpu-plan/review/probes/nodom-verify/ (core2.ts, core3.ts) to test the
MAINT-12 fix. Probe outputs quoted where they decide a verdict.

Probe facts established (Dawn webgpu@0.4.0, both adapters unless stated):

- binding-mismatch-probe.mjs: access-mode mismatch, binding-index mismatch AND the
  "looser" case (layout `storage` for a shader `var<storage, read>`) are all
  `GPUPipelineError` at `createComputePipelineAsync`. Only an exact match creates.
- maint-sync-probe.mjs (NVIDIA): adapter `maxComputeWorkgroupsPerDimension` 65535,
  raised device still 65535; a TS `constants` key the composed WGSL does not declare
  (`LINLOG`, or misspelled `USE_PREM`) is `GPUPipelineError: Pipeline overridable
  constant ... not found`; an override declared in the prelude but never referenced
  is accepted when set.
- uncaptured-order-probe.mjs (NVIDIA and llvmpipe): both `uncapturederror` events are
  counted BEFORE `queue.submit()` returns (`submitted@0.03; events so far: 2`);
  `adapter.requestDevice()` a second time -> `OperationError: adapter is "consumed"`.
- nodom-verify/core3.ts: with `lib: ["ES2020"]` + `@webgpu/types` + `skipLibCheck:
  true`, `queue.writeBuffer(b, 0, "not a buffer")` and `writeBuffer(b, 0, 42)` COMPILE
  (exit 0); with `lib: ["ES2020","DOM"]` both are TS2345. Reason: `BufferSource` is a
  DOM-lib type; without it `@webgpu/types` only type-checks because `skipLibCheck`
  hides 8+ unresolved names (`BufferSource`, `AddEventListenerOptions`,
  `EventListenerOrEventListenerObject`, `ImageBitmap`, ...) and the affected
  parameters degrade to unchecked.

## Verdicts

### MAINT-1 -- confirmed, major
Plan 767-775 (`WgslModuleSpec` has no bindings), 1097-1099 (`Kernel` takes a separate
`BindingSpec`), 1438-1445 (eight hand-written `@group/@binding` lines) are as cited.
D20 (194) and 5.3 (1140-1147) generate uniform structs "so the two cannot disagree";
storage bindings get no such treatment. Probe: every mismatch is a per-variant
`GPUPipelineError` at pipeline creation (and the "looser layout" case is rejected too,
so the plan's 5.1 rule "declare `read_write` in every variant" is the only workable
sharing rule -- the plan is right there). The compile matrix (1091-1096, 2711) can
catch a mismatch only if it builds the layout from the same `BindingSpec` the driver
uses, which needs the enumeration MAINT-6 says does not exist. The maint-sync probe
adds a THIRD hand-kept duplicate the reviewer did not list: the per-module `override`
declarations (1435-1437, 1488) must match the keys of `spec.overrides` exactly (an
extra or misspelled key is a `GPUPipelineError`).
Fix (revised): in 3.5 add `readonly bindings: readonly BindingDecl[]` ({ group,
binding, name, kind: "storage" | "storage-ro" | "uniform", wgslType }) AND
`readonly overrideDecls` (name, type, default) to `WgslModuleSpec`; the composer emits
the `@group/@binding var<...>` block and the `override` lines; `Kernel` derives the
explicit `GPUBindGroupLayout` and the `bind()` record keys from `bindings`; a unit
test asserts no `.wgsl.ts` body contains `@group(` or `override `. Delete the separate
`BindingSpec` from 5.1; 7.5 shows only the kernel body. (Deriving the spec by parsing
the WGSL text is the acceptable alternative if the owner prefers WGSL as the source;
either way there must be ONE source.)

### MAINT-2 -- downgraded to minor
Line citations are wrong: 1183 / 1198 are in 5.5 / 5.6; the snippet-taking primitives
are at 1251 (`segmentedReduce ... value snippet`) and 1256 (`advance ... functor: {
visit: snippet, filter: snippet }`). The substance holds: 773 substitutes snippets at
`//@@NAME@@` markers and nothing in 3.5 / 5.1 validates them. The one hazard worth a
plan change: an UNFILLED marker is a WGSL line comment, so a misspelled snippet key
("vist") yields a kernel that compiles cleanly and silently does nothing -- caught
only by the differential test as "wrong result", never as "unfilled slot". The
typed-functor redesign is a preference; the plan's string snippets keyed into the
pipeline identity (789-792) are workable.
Fix (narrowed): in 3.5 state that `composeWgsl` throws `E_SHADER_COMPILE { id, stage:
"compose", slot }` when any `//@@` marker remains after substitution or a snippet key
has no marker; document that a snippet may reference only the parameters named in the
primitive's functor signature (6 rows 3 and 8); optionally have the compilation-info
formatter subtract the prelude line count.

### MAINT-3 -- downgraded to minor
1381 (`state` = 128 + k x 32 with named fields, read by every kernel and copied to
staging), 1659 (`S.trace[...]`), 1847, 2727 and 1134-1147 (UniformBlock is uniform-only)
are as cited. But the scope is ONE struct, not "every host-visible block": `partials`
(1380) is never read by the host, and `GridSpec` is passed inside the params uniform
(1382, 1547) so it IS generated. A wrong TS offset into `state` is caught immediately by
the trace-parity test (11.4, 2726-2730) -- with poor diagnostics, not silently.
Fix (narrowed): give `UniformBlock` a `{ layout: "uniform" | "storage" }` mode and a
`read(view)` counterpart (rename to `StructBlock` if wanted) and declare `state`
through it; one host->kernel->host round-trip test for `state` on both runtimes.

### MAINT-4 -- confirmed, major
543 states the rule; 306-308 (create() builds `PipelineCache`, `BufferPool`,
`Readback`, `GraphResidency`, `Profiler`), 496 / 535 (`device/calibrate.ts` runs the
exact-tile kernel and the grid build, 329-333), 608-609 (`accelerator()` and
`calibrate()` on the class) violate it in the same section: the lowest layer
value-imports memory, kernel, layouts and the top-level accelerator. It would run (the
imports are used inside `create()`, not at module evaluation) but every file in the
package would import the file that imports everything, and `import/no-cycle` /
`no-restricted-paths` could never be enabled. Cheap to fix in the plan, expensive after
50 files import `device/gpu-context.ts`.
Fix: as proposed -- `device/` = acquisition, caps, error scopes, lost; `src/context.ts`
above memory and kernel as the named composition root (`GpuContext` may keep its public
name there); `createAccelerator(ctx)` in `src/accelerator.ts`, `calibrateLayout(ctx)`
in `src/layouts/calibrate.ts`; update 2.2 step 6, 3.1, 3.2, 3.3, 9.5, 9.6; add the
import-boundary lint zones to 3.2.

### MAINT-5 -- downgraded to minor
563, 1824-1849, 1893-1922, 689-691 are as cited. The template-method shape ("subclasses
supply the per-iteration kernel sequence") is a legitimate design, and the FR
differences (per-batch temperature slots, a different reheat, no K4) fit hooks. What is
actually missing is the NAME of the hook protocol: the plan never lists what a
subclass must provide, so the base/subclass contract is implicit.
Fix (narrowed): in 7.19 name the per-model hook interface (e.g. `ForceModel {
buffers(n), overrides(), recordIteration(batch, slot, tier), paramsFor(i), onReheat()
}`) that FA2, FR and the spring preset implement; whether `ForceSimulation` consumes it
by inheritance or composition is left to implementation. Update the 3.2 class list.

### MAINT-6 -- downgraded to minor
The touch-point count is real (490, 495, 499-503, 704-716, 721-727, 2713, 2282-2314,
2401-2409 all verified) and `packages/graph-io/CLAUDE.md` lines 126-159 does carry an
"Adding a format" recipe the plan's package never specifies. The one enforceable gap:
1091-1096 / 2711 "every module in every override combination the package uses" names
no enumeration mechanism, so a new variant can be omitted from the matrix without any
test failing. The `KernelDef` proposal is absorbed by MAINT-1; co-locating WGSL with
drivers and the `bindContext` generic are preferences.
Fix (narrowed): one `src/kernels.ts` (or one array per family) listing every
`WgslModuleSpec` with its override axes, consumed by the compile matrix, the
bind-group-budget test and `PipelineCache.warm()`; add "Adding an algorithm / kernel /
layout model" recipes to the package CLAUDE.md content list (see MAINT-19).

### MAINT-7 -- confirmed, major (fix revised)
668-670, 2533 ("replace the structural mirrors with `import type` from the real
packages (devDependencies)") and 2541-2545 ("declares NO peer on algorithms / layout")
contradict each other: `tsc -p tsconfig.build.json` (declaration: true, no bundling --
`packages/graph-io/scripts/bundle-types.js` writes one-line re-export shims over the
tsc output) preserves `import type { LayoutSimulation } from "@graphty/layout"` in
`dist/src/types/*.d.ts`, so a consumer without the CPU packages installed (a Node script
using graph-format + GPU PageRank, goal G2) gets an unresolvable d.ts. The `contract: 1`
runtime version and the `typeof indexed` mapped type are optional niceties, not
defects; structural typing at the app's typecheck is the plan's stated guard.
Fix (revised): at W1 KEEP the structural mirrors in `src/types/accelerator.ts` as the
published contract and ADD the mutual-assignability type test against the real packages
(devDependencies) -- this honours "no peer" as written; OR declare `@graphty/algorithms`
and `@graphty/layout` as optional peerDependencies (types) and switch to `import type`.
Reword 2533 and P10 (3169) accordingly.

### MAINT-8 -- downgraded to minor
Verified: 2818-2822 copies the `algorithms/vitest.config.ts` pattern (lines 62-72 set
`thresholds: undefined` whenever `--project=` is present or COVERAGE_DIR is set); 2950
and 3077 run `--project=node --coverage` on every lane; the staging repo has no merge
step; `tools/merge-coverage.sh` in the monorepo merges and uploads but contains NO
threshold check (grep: none), so the sibling comment "checked at CI level after
merging" is itself false today. G1 (3159) therefore names a coverage gate nothing
enforces. Real, silent, but a one-line config fix with no structural consequence.
Fix: as proposed -- thresholds apply when `--project=node` is the selected project and
are disabled only when COVERAGE_DIR is set; reword 11.8 / 11.1 / G1 to "on the node
project with --coverage"; note that the monorepo pattern does not enforce thresholds
either.

### MAINT-9 -- downgraded to minor
"Two described owners" is a misreading: 561 says `CommandBatch` "carries" (a borrowed
slot), 973 says `Readback` "owns a ring" -- consistent. The per-call choreography
(Lease + CommandBatch + UniformRing + Readback + signal checks) is a stated pattern
(1995-1999), not a defect. What the plan does leave implicit: the staging slot's
lifecycle when a batch is DISCARDED -- stale generation (1845-1846) or abort (1205) --
since 976 picks slots by `mapState === "unmapped"`, a discarded batch whose `mapAsync`
is never awaited/unmapped would keep its slot pending and grow the ring.
Fix (narrowed): one sentence in 4.4 / 5.8: `Readback` owns the ring; a `CommandBatch`
borrows one slot for its lifetime and ALWAYS unmaps and returns it (resolved,
discarded or aborted); a run-scope helper bundling Lease / CommandBatch / UniformRing is
an optional implementation convenience.

### MAINT-10 -- confirmed, minor
494, 782, 786, 1115-1121, 1080 vs 1729 / 1752 all verified. The mixed use in 5.2
(device limit for the 1D test, literal 65535 for the 2D split) is internally
consistent (the spec minimum is 65535 and the probe shows the 4070 cannot raise it), so
no bug -- but three places must agree by hand, and `DIM` is listed as an override in
5.1 while 7.13 / 7.14 make `dim` a uniform. Add `INVALID_INDEX` (781 hard-codes
`0xFFFFFFFFu` while graph-format exports the constant) to the same interpolation.
Fix: as proposed (composer interpolates constants.ts / graph-format values into the
prelude; grep test; drop `maxComputeWorkgroupsPerDimension` from `RaisableLimit` or use
the raised value in both places; `dim` is a uniform, delete `DIM` from 5.1).

### MAINT-11 -- confirmed, minor (fix narrowed)
Verified: 535-541 vs 886 / 3159 (module labels vs function names), 555 (`DispatchPlanner`
in the CLASS list but "pure functions"), no `Profiler` row in 546-564 although 607 types
`ctx.profiler: Profiler`, 689 `ForceAtlas2Params` vs `ForceAtlas2Options` everywhere
else, 842 hand-written 8-name union while graph-format exports `ViewName`
(`packages/graph-format/src/index.ts` line 123, `src/types/snapshot.ts` 226-243).
`AdapterSummary` (318), `GpuCalibration` (331-333) and `CalibrateOptions` (329 vs 609)
ARE described inline, just not as declarations -- "never defined" overstates it.
Fix: one name per thing; add `Profiler` to the class list; declare the three types in
3.3; `ForceAtlas2Options`; `Extract<ViewName, ...>` for `view()`.

### MAINT-12 -- refuted
The nodom probe compiles, but only because `skipLibCheck: true` hides that
`@webgpu/types` 0.1.72 depends on DOM-lib names. Verifier probe
`probes/nodom-verify/core3.ts`: with `lib: ["ES2020"]`, `queue.writeBuffer(b, 0, "not a
buffer")` and `writeBuffer(b, 0, 42)` compile without error (BufferSource unresolved ->
parameter unchecked); with DOM they are TS2345. Dropping DOM would silently weaken type
checking on the most-used upload call in the package (and on `addEventListener`
overloads). The plan's choice at 525-527 (DOM lib + the `no-restricted-globals` rule at
366-368) is the correct trade-off. `caps.runtime` is set by the entry (296-299) and read
by nothing in the core; a lint rule against reading it in `src/` is a one-liner if
wanted, not a defect.
Recommended one-line addition to 3.1: "DOM lib stays because `@webgpu/types` names
`BufferSource` / DOM event types; removing it degrades `writeBuffer` typing under
`skipLibCheck`" so a future maintainer does not "fix" it.

### MAINT-13 -- downgraded to minor
1127 reads `caps.software` for the grid-stride group default; 336-338 and 1597 / 1613
promise only that the REPULSION `"auto"` crossover never depends on it -- the promises
are scoped, so this is not a contradiction. The branch is a documented, unit-tested
performance default (1128-1130), affects no result (grid-stride maps are order-
independent; determinism is promised per device only, 7.16). The maintainability point
that survives: the plan does not say where `caps.software` may be read inside `src/`,
so the scoped promises are easy to widen by accident.
Fix (narrowed): state in 2.4 that `caps.software` is read in `src/` only by
`isSoftwareAdapter` and `planGridStride` (a performance default, never a behaviour
decision), and let tests override it through `GpuContextOptions` tuning if they need a
fixed dispatch shape.

### MAINT-14 -- confirmed, minor (fix narrowed)
Verified: env block in 2889-2898, 2944-2948 and 3093-3097 (the two YAMLs exist at
different times, so this is not permanent triplication); results paths 2622
`<host>-node<version>.json` vs 2664 `<runner-class>.json` vs 2999-3000 `bench/results.json`
(a `bench/` directory absent from the 3.1 tree, which has `benchmarks/results/`); the
`GRAPHTY_GPU_NO_SUBGROUPS=1` run covers `test/primitives` only (2951, 2998) while
1262-1266 and 1637-1638 give the layout epilogue a twin and 2709 promises "on any
adapter" -- so the layout twin is never exercised in CI. `.mjs` is outside knip's
`scripts/**/*.{ts,js}` (`packages/knip.config.ts`) but INSIDE eslint's `**/*.mjs` block
(`packages/eslint.config.js` line 282) -- the eslint half of the claim is wrong. One
more gap the reviewer missed: the plan never says HOW the env var reaches the core
(the cache key filters `needs` by `caps.features`, 1076-1077), given 349-350 forbids
`src/` from reading env vars.
Fix (narrowed): one results directory (`benchmarks/results/<runner-class>.json`);
replace the env var with a `GpuContextOptions.features: { subgroups?: false }` (or a
`GpuContext.from(device, { features })` override) that `test/setup/gpu.ts` applies, and
run the twin for every kernel that has one (a `variants` axis in 11.3) in the same
process; `.js` for scripts (package `"type": "module"`).

### MAINT-15 -- downgraded to minor
506, 538, 1241-1243, 2529, 2705, 3161, 3173-3179 verified: P7-P9-P11 may precede P10, so
oracles for the whole algorithm list can be written before W1 and "switched" away at
W1 (2533). But the fix "order P10 before P8/P9/P11" makes GPU work wait on three
monorepo landings (A2 / L1 / E1) outside this package's control -- a schedule change
the reviewer should not force. The oracles are naive index-based references (FIFO BFS,
heap Dijkstra, union-find: tens of lines each) and an INDEPENDENT oracle is stronger
than testing GPU against the CPU package alone (shared design, shared bugs).
Fix (narrowed): split into `test/oracle/<name>.ts` mirroring `src/`; at W1 ADD
`indexed.*` as a second oracle instead of replacing the first (reword 2533 and P10);
state that the FA2 / FR oracles are the SPEC of L1 (3161) and are moved, not copied,
into `layout/` if L1 wants them.

### MAINT-16 -- confirmed, minor
692-697 (GpuLayoutTuning carries settleThreshold, settleWindow, iterationsPerStep,
maxInFlight), 2344-2349 (layout-owned ForceAtlas2Options carries the first three),
1757-1758 (7.14 lists them under "new" GPU rows), 2352-2357 (`LayoutAccelerator.
forceAtlas2?(options?: ForceAtlas2Options)`), 2458-2459 (`behavior.layout.maxInFlight`
config) all verified. `maxInFlight` has no typed path from the element's config to the
GPU simulation because `ForceAtlas2Options` lacks it.
Fix: as proposed (layout-owned options carry all four, the CPU simulation ignores
`maxInFlight`; `GpuLayoutTuning` keeps only GPU-only keys; 7.14 rows read
"layout-owned option (L1)").

### MAINT-17 -- downgraded to minor (fix narrowed)
Verified: 606 (`residency: GraphResidency` public), 588, 657, 643, 1350-1351, 1915-1922;
`packages/graph-format/tsconfig.build.json` has `stripInternal: true`,
`packages/graph-io/tsconfig.build.json` does not. Real inconsistency: `precision: "f32"`
is on `GpuBetweennessResult` only (657) while 2412 ("a GPU result may carry
precision") and Q-24 (3257: "results carry precision: 'f32'") speak of results in
general. Reserved override names in 7.2 / 7.20 are normal plan content, not API
surface; `wgslFeatures` is harmless caps display.
Fix (narrowed): decide `precision` once (on `GpuScoresResult` or nowhere) and align
2412 / Q-24; mark `residency` `@internal` and add `stripInternal: true` to
`tsconfig.build.json` (or document `GraphResidency` as public API with its types);
move `degree` to `test/` or document it as a public diagnostic.

### MAINT-18 -- confirmed, minor (fix revised)
Design 4239-4240 reads "graphty-element injects it as `runAlgorithm(snapshot, {
accelerator: gpu })`"; the plan's own 1.3 row (164) quotes that spelling as honoured
by section 9, while 9.2 / 9.4 (2313, 2404-2406) use `accelerated(acc).pageRank(s)`.
`runAlgorithm` is a one-line sketch in the design (no signature anywhere: grep finds
only line 4239), so this is a spelling departure, not a mechanism departure. Design
3734: `SsspResult { dist; predArc; pathTo(t); pathEdges(t) }`; plan 2273 / 655 drop the
closures and 2509 has the adapter reconstruct on the CPU. The reviewer's fix ("a CPU-
side helper owned by @graphty/algorithms" inside the GPU result) is impossible under
D3 (no runtime import of the CPU packages).
Fix (revised): add a one-line note to 1.3 / 1.5 that `accelerated(acc).x(s)` is the
concrete form of the design's `runAlgorithm(snapshot, { accelerator })` sketch
(DEPARTURE-5 if the owner wants it explicit); have the DISPATCHER in
`@graphty/algorithms` (`AcceleratedAlgorithms.sssp`) decorate an `SsspResultLike` with
`pathTo` / `pathEdges` closures over `predArc` so adapters see the design's `SsspResult`
from both paths.

### MAINT-19 -- confirmed, minor (fix narrowed)
Verified: 410, 2670, 3158, 3228, 3249 defer to a package CLAUDE.md whose contents are
never listed, while `packages/graph-io/CLAUDE.md` (Package Structure / Adding a format /
House Style / Distribution) is the sibling model. The split of 7.2 / 7.14 / 10.4 / 12.4
into separate documents is a preference.
Fix (narrowed): in 3.1 enumerate the package CLAUDE.md sections (the 3.5-3.6
conventions, the env table, the `GRAPHTY_GPU_REQUIRE` policy, the Vitest launch
spelling, the verified platform facts of R-22, the "Adding a ..." recipes of MAINT-6);
in Q-18 say that "Review notes" and graft markers are dropped on acceptance.

### MAINT-20 -- confirmed, minor (fix narrowed)
Probe re-run on both adapters reproduces both facts. Note the plan never says "once
per worker" (no such text; 505 says "acquire() per project") and never claims adapter
reuse -- but 2.2's `options.adapter -> requestDevice` path plus a harness returning `{
gpu, adapter, device }` (the graph-format `acquire()` it generalises, lines 38-63) is
exactly the trap: passing that `adapter` into `GpuContext.create` after the harness
already called `requestDevice` fails. The uncaptured-error fact matters for 1202: an
invalid command buffer is a no-op submit, so the batch's readback resolves with STALE
staging bytes and the error surfaces one call later at an unrelated site.
Fix (narrowed): 2.2 documents "under Dawn-node 0.4.0 an adapter is consumed by one
`requestDevice`; `create({ adapter })` consumes it"; 11.2: `acquire()` requests a fresh
adapter per device; 5.7 / 5.8: after `submit()` `CommandBatch` checks the context's
pending-error slot (synchronous in Dawn-node) and rejects ITS OWN readback with
`E_VALIDATION { batchId, label }`, keeping the next-call throw for asynchronous
delivery (browsers); the browser project drains pending errors in `afterEach`.

## Missed defects (same lens)

### MAINT-M1 -- minor -- error contract contradiction
Section 1.4 D12 (line 186) / 3.3 (593-599): "The package throws its own
`WebGpuGraphError` (never `GraphFormatError`)". Section 4.3 (941-943): `column()` calls
`table.gpuView(name)` which "throws `E_GPU_INELIGIBLE` from the format ...; the GPU
package lets it propagate". `packages/graph-format/src/columns/column.ts` 1994 does
throw `new GraphFormatError("E_GPU_INELIGIBLE", ...)`. A consumer catching
`WebGpuGraphError` per D12 misses it.
Fix: either wrap in 4.3 as `E_INVALID_ARGUMENT { cause, column, dtype }`, or amend D12
to "never throws `GraphFormatError` for its OWN conditions; errors from snapshot /
column accessors propagate unchanged" and list the pass-through codes in 5.7.

### MAINT-M2 -- minor -- GPU-only layout tuning has no path through the accelerator
3.3 line 608: `accelerator(): GpuAccelerator // one per context, cached` (no options);
713: `forceAtlas2(o?: ForceAtlas2Options)` "accepts the CPU option type"; 9.2 line
2323-2324: "GPU-only tuning goes through the GPU package's own factory options, never
through the dispatcher"; 9.4 item 4 (2414-2415): the element creates simulations with
`createSimulation(type, opts, graph.accelerator)`. Yet 2.2 (334-335), 9.5 (2489-2490)
and R-2 (3208) say the app "calls `ctx.calibrate()` once and passes `exactMaxNodes`" --
there is no API through which the app-side value (or `nearMax`, `gridMax*`,
`deterministic`, `compat`) reaches a simulation the element creates. The detected-
acceleration flow (G5, D7) is therefore incomplete as written.
Fix: `ctx.accelerator(options?: { readonly layout?: GpuLayoutTuning })` (or
`createAccelerator(ctx, options)` per MAINT-4) whose defaults apply to every simulation
it creates; the app passes the calibration result there; document in 9.4 / 9.5 that
per-simulation GPU tuning through the element is not available in v1.

### MAINT-M3 -- minor -- the env-var reader list is wrong
2.3 (349-350): env vars "are read ONLY by the test setup (`test/setup/gpu.ts`, ...),
never by `src/`"; 12.2 (2886-2887): "read only by `test/setup/gpu.ts` and the vitest
config"; 12.3 (3010-3011): `scripts/gpu-report.mjs` "exits non-zero when
`GRAPHTY_GPU_REQUIRE` names a vendor that does not match" -- a third reader with its
own copy of the vendor-match rule (2688).
Fix: list the three readers in 12.2 and factor the policy parser
(`parseGpuRequire(env) -> { level, vendor }`, `checkAdapter(info, policy)`) into one
module both `test/setup/gpu.ts` and `scripts/gpu-report` import.

### MAINT-M4 -- minor -- the browser entry cannot satisfy the stated lint rule
2.4 (365-368): "`src/` contains ... no reference to `navigator`. A lint rule
(`no-restricted-globals` ...) enforces" it. 2.1 (232) and 2.3 (342-343): `./browser` =
`src/browser/index.ts` "uses `globalThis.navigator.gpu`". The rule as stated fails on
the file that must use it; the plan never scopes the exemption.
Fix: state the rule as "`src/**` except `src/browser/**`" (an eslint `files` /
`ignores` pair) and add the mirror rule that `src/browser/**` is imported by nothing
in `src/` outside itself (`no-restricted-imports` from `src/index.ts` and `src/node/`).

### MAINT-M5 -- minor -- two root barrels, two sibling conventions
3.1 (490 tree): a root `webgpu-graph-algorithms.ts` (graph-format's convention:
`packages/graph-format/graph-format.ts` = `export * from "./src/index.js"`) AND
`src/index.ts` "the only public barrel", while the package "mirrors `packages/graph-io`
file for file" -- graph-io has NO root `.ts` (ls: only `vitest.config.ts`) and maps the
root bundle to `src/index.ts` in `scripts/entries.js`. Two barrels invite drift and the
`bundle-types.js` shim already re-exports `dist/src/index.d.ts`.
Fix: drop the root `.ts`; `scripts/entries.js` = `{ "webgpu-graph-algorithms":
"src/index.ts", browser: "src/browser/index.ts", node: "src/node/index.ts" }`, with
`test/build-output.test.ts` checking `package.json` exports against it as graph-io does.

### MAINT-M6 -- minor -- one non-generic `GpuLayoutSimulation` for two models
3.3 (677-691): `stats: LayoutStats` carries FA2-only fields (`swing`, `traction`,
`speedEfficiency`, `trace` of the same) and `setParams(patch: Partial<ForceAtlas2Params
| FruchtermanReingoldParams>)` on the SAME interface returned by `createFruchterman
Reingold` (698); 7.20 (1908): FR has "No swing / traction ... no K4". An FR simulation
therefore reports meaningless swing / traction / speedEfficiency and accepts FA2 keys at
the type level; every future model widens the union.
Fix: `GpuLayoutSimulation<Options, Stats extends LayoutStatsBase>` with
`ForceAtlas2Stats` / `FruchtermanReingoldStats` subtypes (or `null` for absent
fields), `setParams(patch: Partial<Options>)`; `createForceAtlas2` /
`createFruchtermanReingold` return the specialised type; `LayoutAccelerator` keeps
returning `LayoutSimulation`.
