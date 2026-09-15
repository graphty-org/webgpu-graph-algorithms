# INTEG review -- Integration and API conformance

Document reviewed: `/home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md`
(3328 lines, read in full). Lens: does the plan conform to the implemented graph-format API, to the
accepted design (sections 10, 14.3-14.6, 15, 16), to the real @graphty/algorithms, @graphty/layout
and graphty-element sources, and to the monorepo's pnpm / Nx / CI mechanics?

Reviewer: adversarial, integration lens. Date: 2026-09-14.

## Method

- Every graph-format identifier the plan names was grepped against
  `packages/graph-format/src/index.ts` and the modules it re-exports (`types/snapshot.ts`,
  `types/columns.ts`, `snapshot/views.ts`, `snapshot/derived.ts`, `snapshot/graph-snapshot.ts`,
  `util/mask.ts`, `errors.ts`, `ids/node-id-map.ts`).
- Every cited line in `layout/src/layouts/force-directed/forceatlas2.ts`, `utils/random.ts`,
  `graphty-element/src/{algorithms,managers,layout,config,ai,behaviors}`, `algorithms/src`,
  the monorepo `ci.yml` / `release.yml` / `coverage.yml` / `nx.json` / `package.json`, and the design
  sections 10, 13.5, 14.2-14.6, 15.3, 16.2, 16.6-16.7 was opened and compared.
- Nx release behaviour was verified in the installed nx 22.3.3 source under
  `graphty-monorepo/node_modules/.pnpm/nx@22.3.3`.
- One probe was run: `tmp/webgpu-plan/review/probes/serial-sharing.mjs` (withColumns / toUndirected
  identity and serial behaviour).
- External facts checked: GitHub Actions queue limit (docs.github.com/en/actions/reference/limits),
  pnpm `workspace:` publish replacement (pnpm.io/workspaces), the published
  `@graphty/graphty-element` dependency pins (`npm view`).

## Identifier audit (graph-format)

Every graph-format name the plan uses exists with the stated shape: `INVALID_INDEX`, `GraphSnapshot`,
`AttributeTable.gpuView(name)`, `column.version`, `markDirty()`, `setAll()`, `paddedU32View()`,
`gpuEligibility`, `foldArcs(snapshot, perArc, reducer, out?)`, `expandEdges(snapshot, perEdge, out?)`,
`renumberPartition(labels, out?)` at `derived.ts:1155`, `makeMask` / `maskSet`, `NodeMask` / `EdgeMask`
(= `U32`), `F32` / `U32` / `F64` (`<ArrayBuffer>`-parameterised), `TypedArrayData`, `NumericVector`,
`ArenaLayout { buffer, byteOffset, byteLength, alignment, segments, hotByteLength }`, `ArenaSegment`,
`CoreArrayName`, `ViewName` (all eight names the residency binds), `DegreeOrderView { perm,
segmentOffsets }`, `DegreeOrderOptions.of`, `ReverseView.fwdArc`, `CooView.src`, `EdgeListView.arc`,
`SnapshotFlags { multigraph, hasSelfLoops, arcToEdgeIsIdentity, weighted, allWeightsOne,
nonNegativeWeights, finiteWeights }` (no `sortedRows`, as the plan says), `snapshot.serial`,
`withColumns()`, `dropCaches()`, `toUndirected()`, `transpose()`, `contract()`, `validate({ checksum })`,
`ids.requireIndex` / `idOf`, `toWire({ transfer })`, `E_GPU_INELIGIBLE`, `DerivedGraph.edgeRemap`.

Two imprecisions only (both minor, INTEG-13): `NumericVector` is `F32 | F64 | U32 | I32`
(`types/columns.ts:80`), not `F32 | F64` as D13 says; and `isFallbackAdapter` lives on
`GPUAdapterInfo` in `@webgpu/types` 0.1.72 (`dist/index.d.ts:2229`), so 2.6's row
`adapter.isFallbackAdapter` should read `adapter.info.isFallbackAdapter` (the code sketch in 2.2 already
reads it from `info`).

## Findings

### INTEG-1 (blocker) -- the GPU lane inside `ci.yml` blocks releases and coverage for a day when the dev box is off

Section 12.1 / 12.3 / 12.5, plan lines 2843-2846, 2968-2977, 3100-3101.

Claim: adding `test-gpu` to the monorepo `CI` workflow (and running it on push to master) makes the
GPU lane a release and coverage gate, contradicting "never blocks" (line 2845).

Evidence: `graphty-monorepo/.github/workflows/release.yml:4-5,19` and `coverage.yml:4-5,17,51` are
`workflow_run` on `CI` gated on `github.event.workflow_run.conclusion == 'success'`. A workflow run's
conclusion is `failure` when any job fails unless that job has `continue-on-error: true`, and a job that
waits for a self-hosted runner is queued for up to 24 hours before it is cancelled ("A job can be in the
queue for 24 hours before it is automatically cancelled", docs.github.com/en/actions/reference/limits).
`timeout-minutes: 45` (line 2976) only starts once the job runs. Not being in `all-checks.needs`
(line 3100) protects branch protection, not `workflow_run` consumers. So a powered-off box delays every
master release by 24 hours and a red GPU run cancels the release and the coverage merge.

Fix: put the GPU lane in its own workflow file (`gpu.yml`, triggers push / schedule / dispatch /
labelled same-repo PR, `needs` nothing from `CI`; it downloads the build artifact by run id or rebuilds),
so `CI`'s conclusion never depends on it; state in 12.1 that "never required" means "not in `CI` at all",
and add `continue-on-error: true` on the job as belt and braces. Apply the same to the staging 12.3
workflow so the shape that moves in at W1 is already right.

### INTEG-2 (major) -- W1 leaks devDependency types into the published d.ts and breaks the plan's own lint rule

Section 3.1 / 3.3 / 9.8 / 13, plan lines 495, 676, 2533, 3169 versus 365-368 and 177.

Claim: "replace the structural mirrors with `import type` from the real packages (devDependencies)"
in `src/types/accelerator.ts` makes `dist/*.d.ts` reference `@graphty/algorithms` and `@graphty/layout`,
which the package declares neither as dependency nor as peer (line 2541), and violates 2.4's
"`src/` contains ... no import of `@graphty/algorithms` or `@graphty/layout`" enforced by
`no-restricted-imports`. `GpuLayoutSimulation extends LayoutSimulation` (line 676) would then be an
unresolvable type for any consumer without `skipLibCheck`.

Evidence: plan lines 365-368 (lint rule), 2541 ("declares NO peer on algorithms / layout"), 2533 (W1
replaces mirrors), 676 ("a structural copy lives in src/types until W1").

Fix: keep the structural mirrors in `src/types/` permanently (they ARE the published contract, exactly
as note 02 section 4 intends) and confine the real imports to `test/types/conformance.test-d.ts`
(`expectTypeOf<GpuAccelerator>().toMatchTypeOf<AlgorithmAccelerator & LayoutAccelerator>()` plus the
reverse direction). Reword 3.1, 3.3 and the W1 row accordingly.

### INTEG-3 (major) -- GPU-only options have no route through the dispatcher or `LayoutAccelerator`

Sections 3.3, 7.8, 7.19, 9.2, 9.4 item 7, 9.5, 14.2 Q-13; plan lines 713-714, 1612-1613, 1857-1858,
2110-2111, 2321-2324, 2353-2361, 2458-2460, 2489-2490, 3246.

Claim: the plan promises knobs that cannot reach a simulation or algorithm created through the seams
it defines. `LayoutAccelerator.forceAtlas2(options?: ForceAtlas2Options)` and `GpuAccelerator.forceAtlas2`
accept only the CPU option type, `ctx.accelerator()` takes no arguments and is cached, and 9.2 forbids
GPU-only options through the dispatcher. Yet: the app "may call `ctx.calibrate()` once to pass
`exactMaxNodes`" (2490, 1612-1613, R-2); the element config `behavior.layout.maxInFlight` and
`iterationsPerStep` (2458-2460, 1857-1858) must reach `GpuLayoutTuning`; Q-13 says "the accelerator
exposes `sources` / `k`, `onProgress` and `signal`" while `AlgorithmAccelerator.betweennessCentrality?(s,
options?: BetweennessOptions)` reuses the CPU type, and the CPU `BetweennessCentralityOptions` has only
`normalized` (`algorithms/src/algorithms/centrality/betweenness.ts:15-19`); 9.7 compares "sampled CPU BC
with the same source list", which does not exist. `GpuRunOptions.dest` / `signal` / `onProgress` likewise
never pass the dispatcher.

Fix (text): (a) give the accelerator a configuration surface: `ctx.accelerator(defaults?: {
layout?: GpuLayoutTuning; algorithms?: { betweenness?: { sources?, k? }; ... } })` or
`GpuAccelerator.configure(patch)`; the app applies `calibrate()` there, the element bridge applies
`maxInFlight` / `iterationsPerStep` by calling `setParams` when the simulation exposes it (a documented
`"setParams" in sim` check); (b) state explicitly that the A2 `indexed.betweennessCentrality` gains
`sources` / `k` (so the option type is shared and 9.7's sampled comparison exists), and that
`signal` / `onProgress` / `dest` are reachable only through the GPU package's own functions; (c) delete
or qualify the sentence in 9.2 "GPU-only tuning ... never through the dispatcher" to match (a).

### INTEG-4 (major) -- option resolution (`nodeMass`, `nodeSize`, `weight`) is assigned to helpers the GPU package may not import

Sections 7.3, 7.5, 7.14, 9.3 versus 2.4 / D3; plan lines 1378, 1476-1480, 1748-1749, 2364-2366,
2379-2382 versus 177, 365-368.

Claim: the plan says `nodeMass` as a `Record` or column name is "resolved by the layout package's
`resolveNodeVector`" (1378, 1748), a named-column `weight` "binds the per-arc array the caller expanded
with `expandEdges`" (1477-1480), and "`seedPositions` is what both the GPU package and the element
call" (2379-2380) -- but the GPU package has "no runtime import of the CPU packages" (177, 365-368), and
the only call path from the element is `createSimulation(type, opts, accelerator)` ->
`accelerator.forceAtlas2(opts)` -> `sim.load(snapshot, positions)`, where the resolution needs the
snapshot that only `load()` sees. Nobody in that chain can call `resolveNodeVector` / `resolveWeights`
/ `seedPositions` on the GPU side, and the element's `expandEdges` cache is not reachable from `load()`.

Evidence: `ForceAtlas2Options.nodeMass?: F32 | string | Readonly<Record<NodeId, number>> | null`
(2347); `LayoutAccelerator.forceAtlas2?(options?: ForceAtlas2Options)` (2355); design 14.3 line
4028-4030 places `resolveNodeVector` in the layout port; design 14.6 assigns those helpers to `layout`.

Fix: define the seam so the layout package resolves: `createSimulation` wraps the accelerator's
simulation in a thin `LayoutSimulation` adapter whose `load(snapshot, positions)` runs
`resolveNodeVector` / `resolveWeights` / `seedPositions` and then calls
`gpuSim.load(snapshot, positions, { mass: F32, weights: F32 | null })`; `LayoutAccelerator.forceAtlas2`
takes a `ResolvedForceAtlas2Options` (vectors only, no `Record` / column names). Alternatively move
the three helpers to graph-format and record a change to design 14.6's ownership table as a declared
departure. Either way 7.3, 7.5, 7.14 and 9.3 must name the resolver and its caller.

### INTEG-5 (major) -- `release(previous)` leaks the undirected derived snapshot every layout and undirected adapter uploads

Sections 4.5, 9.4 item 2; plan lines 990-991, 2399-2402, 872-874.

Claim: the `snapshot-replaced` listener releases only `previous`, but layouts and the undirected
adapter group upload `dm.undirected(previous).snapshot`, which for a directed source is a DIFFERENT
snapshot with its own `serial` and its own core arrays; its GPU buffers are never released. Under
`data.directed: "auto"` the builder starts directed (design 14.4 line 4053-4056), so this is the common
case, and the plan's own warning fires after three resident snapshots (872-874).

Evidence: design 14.4 lines 4103-4107 ("Layout engines receive the same `undirected(s).snapshot`
object"), 4189-4197 (undirected adapter group), 4159-4162 (`undirectedCache` is a WeakMap: it drops the
JS object, not the GPU memory); probe `tmp/webgpu-plan/review/probes/serial-sharing.mjs`:
`undirectedIsSameObject: false, undirectedSerial: 2, undirectedSharesRowPtr: false`.

Fix: 9.4 item 2 must read "release `previous`, `dm.undirected(previous).snapshot` when it is not
`previous`, and the `visible(previous)` cache's induced and undirected snapshots"; `Graph.dispose()`
likewise. Add a test to 11.3: after a directed-graph freeze + layout + `snapshot-replaced`,
`residency.stats().snapshots === 1`.

### INTEG-6 (major) -- the residency's `refs: Set<serial>` cannot count `withColumns()` siblings, because siblings share the serial

Section 4.1, DEPARTURE-4; plan lines 205, 856, 862-868.

Claim: "a `ResidentBuffer` records `{ ..., refs: Set<serial>, ... }`; a core shared by sibling
snapshots is destroyed when the last sibling is released" is unimplementable as written: `withColumns()`
returns "a new snapshot object sharing the core, the id map and the serial"
(`graph-format/src/snapshot/graph-snapshot.ts:923-942`; probe: `serial: 1, siblingSerial: 1,
siblingSharesRowPtr: true`). A `Set<serial>` holds one entry for all siblings, so releasing any sibling
destroys the core under the others, and the strong `Map<serial, record>` "so `withColumns()` siblings
share one core" cannot tell them apart either. The DEPARTURE-4 rationale is therefore only half right:
the serial index finds the shared core but cannot express per-sibling lifetime.

Fix: choose and state one semantics: either (a) `release(s)` releases the shared core for every sibling
of `s` (document that `withColumns()` siblings are one residency unit; a live simulation on a sibling
gets `E_RELEASED`), or (b) count references per snapshot OBJECT (the `WeakMap<GraphSnapshot,
ResidencyRecord>` entries, not serials) and destroy the core when the count reaches zero. Rewrite 4.1's
key table and the `ResidentBuffer` record accordingly.

### INTEG-7 (major) -- `nx release` will version and publish the GPU package on every algorithms / layout release

Section 9.8 versioning, 9.1; plan lines 2537-2545, 2248-2250.

Claim: "it declares NO peer on algorithms / layout (types only, dev), so an app can combine any
versions" ignores that `devDependencies` are project-graph edges for `nx release`, and the monorepo runs
`updateDependents: "auto"`. Every `@graphty/algorithms` or `@graphty/layout` release will side-effect
patch-bump and publish `@graphty/webgpu-graph-algorithms` (with a changelog entry and a git tag), and a
`feat!:` in either forces nothing but still publishes.

Evidence: `graphty-monorepo/nx.json` `release.version.generatorOptions.updateDependents: "auto"`;
nx 22.3.3 `src/plugins/js/project-graph/build-dependencies/explicit-package-json-dependencies.js:68-85`
(`readDeps` walks `optionalDependencies`, `peerDependencies`, `devDependencies`, `dependencies`);
`src/command-line/release/utils/release-graph.js:190-207` builds `projectToDependents` from those
edges; `src/command-line/release/version/version-actions.js:207-212` filters out only `implicit`
dependencies; `release-group-processor.js:542-550` bumps dependents under `"auto"`.

Fix: state the coupling and pick one: accept the side-effect patch releases (cheap, but say so), or
keep the conformance test out of the package's `package.json` graph (a root `tools/` type test, or a
`release.groups` entry for `webgpu-graph-algorithms` with `updateDependents: "never"` -- a root
`nx.json` touch point to add to 12.5). Delete the "combine any versions" sentence or qualify it.

### INTEG-8 (major) -- the "line-for-line port" of `estimateFactor` is not line for line

Section 7.10 (and 7.2 row `estimateFactor`); plan lines 1348, 1640-1660.

Claim: the WGSL sketch writes `eff = max(eff * 0.5, 0.05)` and `eff = max(eff * 0.7, 0.05)`; the CPU
port (`layout/src/layouts/force-directed/forceatlas2.ts:205-216`) and Gephi both write `if (eff >
minSpeedEfficiency) eff *= 0.5` -- a conditional multiply that can leave `eff` BELOW 0.05 (0.06 -> 0.03)
and never raises it (0.03 stays 0.03; the sketch would lift it to 0.05). The sketch also divides by
`max(traction, 1e-30)` where the CPU divides by `traction` (Infinity / NaN behaviour differs). The
trace-parity test of 11.4 (`1e-4` for ten iterations) is meaningful only if both sides implement the
same law, which is the whole point of R-1.

Fix: replace the two `max(...)` lines by the conditional form, drop the `1e-30` floor (or add it to the
L1 CPU rewrite and to the 7.2 table as a documented change), and re-state that the WGSL is generated from
the 7.2 table, with the L1 CPU code as the executable spec.

### INTEG-9 (major) -- the CI builds with `tsc` only, so the "no `webgpu` in the bundle" guard, the subpath entries and the strict-consumer compile do not see what they claim to test

Sections 2.5 item 1, 3.1, 12.3, 12.5, 13 G1; plan lines 375-377, 387-392, 485, 2938, 2990, 3107,
3159.

Claim: the workflow runs `pnpm -r run build` (tsc, `tsconfig.build.json`) and then
`test/build-output.test.ts`, which reads `dist/webgpu-graph-algorithms.js` and `dist/browser.js`.
Following the graph-io skeleton the plan mirrors "file for file", `dist/browser.js` / `dist/node.js`
(and the `dist/browser.d.ts` shims) exist only after `build:bundle` (`graph-io/scripts/entries.js`,
`build-bundle.js`; `graph-io/tsconfig.build.json` includes `src/**/*.ts` only), and graph-io's
build-output test SKIPS its bundle checks when the bundle is absent (`graph-io/test/build-output.test.ts:
140-143`; `packages/README.md` section 4: "3 build-output checks skip without the bundle"). So the G1
gate "proves no `webgpu` in the root / browser bundles" is vacuous on both lanes, `typecheck:strict-
consumer` cannot see `./browser` / `./node` typings, and the `webgpu-graph-algorithms.ts` root shim
(line 485) only covers the root entry. Separately, the assertion "the string `webgpu` does not occur"
is unimplementable as worded: `dist/webgpu-graph-algorithms.js` ends with
`//# sourceMappingURL=webgpu-graph-algorithms.js.map`, and the E_NO_WEBGPU messages of `./browser`
mention WebGPU.

Fix: run `pnpm -r run build:all` in both jobs (and say that `tools/prepush.sh`'s tsc-only build means
the guard runs only in CI); make the test assert on import specifiers (`from "webgpu"`,
`import("webgpu")`), not the substring; either add root `browser.ts` / `node.ts` shims to the tsc
`include` (so the subpaths resolve after a tsc-only build, unlike graph-io) or state that the subpaths
are bundle-only.

### INTEG-10 (minor) -- the injection call shape differs from the design without a declared departure

Section 1.3 row 14.5 and 9.2; plan lines 164, 2313-2319.

Claim: 1.3 says the plan honours "injected as `runAlgorithm(snapshot, { accelerator: gpu })`"
(design 14.5 lines 4239-4240), but 9.2 defines `accelerated(acc).pageRank(s, options)`: an option on the
call in the design, a dispatcher object in the plan. Both are explicit injection, but the plan's 1.5
promises that "everything else ... is honoured as written".

Fix: add a one-line note in 1.3 (or a DEPARTURE-5) that the injection shape follows note 02 section
4.5 (`accelerated(acc)`), and why (one dispatcher, no per-function option plumbing).

### INTEG-11 (minor) -- the f32 parity tolerance departs from design 16.2 while 1.3 claims it is honoured

Section 1.3 row 16.2, 9.7, 11.4; plan lines 168, 2510, 2722-2723.

Claim: design 16.2 (lines 4545-4547) fixes "`1e-5` for f32 GPU parity"; the plan uses `1e-4` for
betweenness (2510) and for the one-iteration force parity (2722-2723) with a sound reason (f32
accumulation over many sources / tile order) but without declaring the departure.

Fix: add the two `1e-4` cases to 1.5 (or to the 1.3 row with the reason) so the W1 amendment of 16.2 is
part of the recorded design changes.

### INTEG-12 (minor) -- `SsspResultLike` drops the `pathTo` / `pathEdges` members of the design's `SsspResult`

Section 9.2, 3.3, 9.7; plan lines 655, 2273, 2509.

Claim: design 14.2 (line 3735) defines `SsspResult { dist; predArc; pathTo(t); pathEdges(t) }`; the
plan's `SsspResultLike` and `GpuSsspResult` carry only `dist` / `predArc`, and 9.7 says the adapter does
"`pathTo()` reconstruction on the CPU from `predArc`" -- so the dijkstra / bellman-ford adapters cannot
be the "ONE result-writing loop for CPU and GPU" of 9.4 item 3.

Fix: give `SsspResultLike` the two methods (a shared CPU helper in `@graphty/algorithms` that walks
`predArc` through `arcSource` / `arcToEdge`; the GPU result attaches it after readback), or state that
these two adapters branch.

### INTEG-13 (minor) -- small factual slips about existing code and types

Plan lines 187, 429, 2237, 2404, 2416-2417.

- D13: `NumericVector` is `F32 | F64 | U32 | I32` (`types/columns.ts:80`), not `F32 | F64`.
- 2.6: `adapter.isFallbackAdapter` is `adapter.info.isFallbackAdapter` in `@webgpu/types` 0.1.72
  (`dist/index.d.ts:2229`); 2.2's helper signature already reads it from `info`.
- `graphty-element/src/algorithms/Algorithm.ts:283` is `static register()`; the abstract
  `run(g: Graph): Promise<void>` is line 217 (the plan inherited note 02's number).
- "a bridge implementing today's abstract `LayoutEngine` (lines 36-63)": today's class has
  `addNode / addEdge / getNodePosition / getEdgePosition / nodes / edges`, not `load / reload / dispose /
  getNodePositionInto`; the sketch implements the design-E1 shape. Say so.

Fix: correct the four statements.

### INTEG-14 (minor) -- `SimulationType` names do not map onto `LayoutAccelerator` methods

Sections 9.3, 7.20, 13 P5; plan lines 2360-2361, 1911-1921, 3164.

Claim: `SimulationType = "forceatlas2" | "fruchtermanReingold" | "spring"` while `LayoutAccelerator`
has `forceAtlas2?` / `fruchtermanReingold?` only; the element's `spring` layout IS Fruchterman-Reingold
(`graphty-element/src/layout/SpringLayoutEngine.ts:1,108` -> `springLayout` = `fruchtermanReingoldLayout`,
`layout/src/layouts/force-directed/spring.ts:6`), and 7.20 / P5 add a `spring-electrical` (ngraph-like)
preset with no accelerator method or `SimulationType` value.

Fix: a three-column table (`SimulationType` -> CPU class -> accelerator method), with `spring` as an
alias of `fruchtermanReingold` and `spring-electrical` either added to both interfaces or removed from
P5's deliverables.

### INTEG-15 (minor) -- the "first A2 commit" dispatcher cannot delegate to ports that do not exist yet

Section 9.2, 9.8; plan lines 2308-2313, 2316-2317, 2530.

Claim: `AcceleratedAlgorithms` is "the same list, non-optional" and its CPU branch is
`Promise.resolve(indexed.x(s, ...))`; before the 95 ports land there is no `indexed.x`, so the file
cannot be additive-and-complete as the first A2 PR.

Fix: say the dispatcher's method list grows with the ports (each port adds its method), or that
unported methods throw `E_NOT_PORTED` from the CPU branch until A2 completes.

### INTEG-16 (minor) -- the `no-restricted-imports` rule is a root `eslint.config.js` touch point the plan does not list

Section 2.4, 12.5; plan lines 367-368, 3104-3110.

Claim: per-package lint blocks live in the ROOT flat config (`graphty-monorepo/eslint.config.js:208`
`compact-mantine/**/*.tsx`, etc.); the staging copy is "VERBATIM ... Do not edit the copied shared
configs here" (`packages/README.md:9-12, 25-27`). "In the package eslint block" therefore means a root
edit in the monorepo and a deliberate deviation in staging.

Fix: add `eslint.config.js` to the 12.5 root touch points and note the staging deviation in 3.1.

### INTEG-17 (minor) -- the coverage thresholds are never enforced by any CI invocation

Sections 11.8, 12.3, 12.5, 13 G1; plan lines 2818-2821, 2950, 3077, 3159.

Claim: 11.8 skips thresholds "when a single `--project` is selected (`algorithms/vitest.config.ts`
pattern)", and every CI command is `vitest run --project=node --coverage`; the merged coverage step has
no thresholds either (`tools/merge-coverage.sh` checks only presence). G1's "coverage >= 80/80/75/80 on
the node project" is therefore unenforced. (In algorithms the skip exists because `default` and
`browser` are two halves of one suite; here `node` IS the whole suite.)

Fix: keep thresholds active when only `node` is selected, or add an explicit `coverage:check` step.

### INTEG-18 (minor) -- `workspace:*` publishes an EXACT pin, not the caret range the inherited rule describes

Section 1.3 row 13.5, 3.1; plan lines 166, 516-517, 380-381.

Claim: design 13.5 rule 3 (line 3661) says `workspace:*` is "published as a caret range"; pnpm
replaces `workspace:*` with the exact version and only `workspace:^` with a caret (pnpm.io/workspaces,
"Publishing workspace packages"); the published `@graphty/graphty-element` indeed pins
`"@graphty/algorithms": "1.7.2"` (`npm view`). With `preserveLocalDependencyProtocols: true` nx keeps
the protocol, so the GPU package would ship `"@graphty/graph-format": "1.x.y"` exact next to a
`^1.0.0` peer, and an app on a newer format gets two copies.

Fix: use `workspace:^` in the manifest (and propose the same for graph-io / the design rule), or say
that exact pinning is accepted and why the brand check makes it tolerable.

### INTEG-19 (minor) -- the app detection sketch probes one adapter and creates on another

Sections 2.2, 3.4, 9.5; plan lines 314-321, 735-737, 2481-2483.

Claim: `probeBrowserWebGpu({ rejectSoftware })` and then `requestGpuContext({ limits: "raise" })`
issue two `requestAdapter()` calls; the second is not given `rejectSoftware` and `ProbeResult.adapter`
is a summary, not the `GPUAdapter`, although `GpuContextOptions.adapter` exists to skip
`requestAdapter()`.

Fix: let `ProbeResult` carry the `GPUAdapter` (probe never calls `requestDevice`, so the adapter is
still usable) and have the sketch pass it (`requestGpuContext({ adapter: probe.adapter, limits:
"raise" })`), or pass `rejectSoftware` to `requestGpuContext` too.

### INTEG-20 (minor) -- "renders on WebGL today" cites the lines that contain the WebGPUEngine branch

Sections 1.2, 7.19, R-4, Q-7; plan lines 128-132, 1874-1876, 3210, 3240.

Claim: `graphty-element/src/managers/RenderManager.ts:63-69` is `if (this.config.useWebGPU) {
this.engine = new WebGPUEngine(this.canvas); } else { new Engine(...) }`; the element already types
`engine: WebGPUEngine | Engine` (`Graph.ts:94`) and Babylon 8's `WebGPUEngine` exposes `_device:
GPUDevice`. Rendering IS WebGL by default (`useWebGPU` is never set), but the "future `WebGPUEngine`"
framing understates what exists, and the plan has no row for the interaction when `useWebGPU` is on
(two devices on one adapter, no buffer sharing, double memory).

Fix: reword 1.2 ("WebGL by default; the `useWebGPU` branch exists but is unwired"), and add a risk row
for `useWebGPU` + injected accelerator.

### INTEG-21 (minor) -- "graphty-element imports nothing new" versus real-GPU stories in the element

Sections 9.1, 9.4 item 8, 9.8 W2; plan lines 2252-2253, 2466-2469, 2534.

Claim: stories with "the real one behind a `navigator.gpu` check" live in `graphty-element/stories`
and must import `@graphty/webgpu-graph-algorithms/browser`, giving the element a devDependency on the
GPU package (and, per INTEG-7, an nx `updateDependents` edge in the other direction).

Fix: put the real-GPU stories in the app (which already imports the package) or declare the element's
devDependency and its consequences in 9.1.

### INTEG-22 (minor) -- the E1 precondition omits the design-E1 refactor the element changes presuppose

Sections 9.8, 13 P6; plan lines 2532, 3165.

Claim: `snapshot-replaced`, `DataManager.getSnapshot()`, `dm.undirected(s)`, `getNodePositionInto`,
`engine.load / reload` exist only after the design's E1 work (design 14.4); the plan's E1 row lists
"A2 first commit + L1" as the precondition and P6 sizes the element part at a share of 8-10 ed without
that dependency.

Fix: add "the design-E1 `DataManager` / position-column refactor merged (or the same branch)" to the
E1 precondition and to P6's dependencies.

### INTEG-23 (minor) -- the `webgpu` peer range `^0.4.0` excludes the 0.6.x the plan bumps to at P-ENV

Sections 2.5, 13 P-ENV; plan lines 381, 3162.

Claim: a caret on a 0.x version pins the minor (`^0.4.0` = `>=0.4.0 <0.5.0`); after P-ENV bumps the
devDependency to 0.6.x, a Node consumer installing 0.6.x gets a peer warning and pnpm's
`auto-install-peers` semantics differ.

Fix: state that P-ENV also widens the peer range (`>=0.4.0 <1.0.0` or `^0.6.0`) and re-pins the
devDependency.

## Statements checked and found correct (not findings)

- All graph-format identifiers (see audit above); `withColumns()` shares `rowPtr` (probe); `reverse()`
  of an undirected snapshot returns the forward arrays and `arcToEdge` aliases `fwdArc` on a directed
  identity snapshot (`views.ts:61-136`).
- `forceatlas2.ts` line citations 26-42, 57-65, 184-230, 266, 271-285, 288-297, 318, 322-329, 335-364,
  379-390, 403-428; `random.ts` `m = 2^35 - 31, a = 185852, c = 1`, `seed || random` (seed 0 = unseeded).
- graphty-element: `UpdateManager.updateLayout()` loop at 203-214, `LayoutManager.step()` at 241-245,
  `LayoutEngine` abstract at 36-63, `GraphBehavior.ts:13` default `ngraph`, `ai/providers/index.ts:9-12`
  Safari note, `vite.config.ts:39` web-llm external, `ForceAtlas2LayoutEngine.ts` schema 99-115 and
  `dissuadeHubs` 66-73, `NodeBehavior.onDragStart/Update/End` and `pinOnDrag`, `PageRankAlgorithm.ts:
  224-240`, `BetweennessCentralityAlgorithm.ts:59-78`, `LouvainAlgorithm.ts:166-184`, `package.json`
  depends on `@graphty/algorithms` and `@graphty/layout` (`workspace:*`).
- algorithms: every public algorithm is sync; `pageRank` at `pagerank.ts:83`; no `indexed` namespace yet.
- Monorepo: `ci.yml` shard matrix 235-350, Playwright cache 413-427, `all-checks` 706-708 needs `test`
  + Chromatic only, test job downloads every build artifact, coverage-upload `if` lists shards
  explicitly; `nx.json` `projectsRelationship: independent`; root pins `@vitest/browser ^3.2.4`,
  `playwright ^1.54.1`, `vite ^7`, `vitest ^3.2.4`; `packages/move/root-touch-points.diff` shape.
- cuda-ffi: `on: [push, workflow_dispatch]`, `runs-on: cudaffi-gpu-runner`, `container.options:
  --gpus all --user root`, lint-only without a GPU.
- graph-io skeleton facts: both deps and peer on graph-format, `build-bundle.js:33-47` externalises
  dependencies + peerDependencies, `coverage:preview` ports 9056 / 9057, `pool: "forks"`, prettier
  4 / 120 / all, `no-console` allows `warn`, `jsdoc/require-jsdoc` and explicit return types on.
- The four declared departures are real departures; besides INTEG-10 and INTEG-11 no other silent
  contradiction with design sections 10, 14.3-14.6, 15 or 16 was found. Design 14.6's landing order
  (F1 -> A1 -> F2 -> A2 / L1 / E1 -> W1) is reproduced correctly in 9.8, and the STATUS.md owner
  decisions (weight flags describe the f32 arc array) are consistent with the plan's use of
  `flags.allWeightsOne` / `nonNegativeWeights`.

## Overall

The plan is conformant at the identifier level -- every graph-format name it uses exists with the
stated shape, and its reading of the layout / element sources is accurate to the line in all but one
place. The defects are in the seams it invents: the CI wiring would gate releases on the dev box
(INTEG-1); the W1 type story leaks devDependencies into the published d.ts (INTEG-2); the two
accelerator interfaces have no channel for the tuning the plan promises the app and the element
(INTEG-3) and no owner for option resolution (INTEG-4); the release lifecycle and the residency both
mis-model graph-format's derived-snapshot and sibling identity (INTEG-5, INTEG-6); nx release couples the
package it says is decoupled (INTEG-7); and the one numeric "line-for-line" claim is not (INTEG-8). All
are fixable in the text; none invalidates the architecture.
