# INTEG verdicts -- skeptical verification of the "Integration and API conformance" review

Document: `/home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md`
Reviewer findings verified: INTEG-1 .. INTEG-23. Verifier date: 2026-09-14.

Method: every cited plan line was re-read; every cited source file was opened at the cited
lines; the one probe (`probes/serial-sharing.mjs`) was re-run; two external documents were
fetched (docs.github.com/en/actions/reference/limits, pnpm.io/workspaces); nx behaviour was
checked in the monorepo's resolved `nx@22.7.12` / `@nx/js@22.3.3` sources and in a real
release commit (`5ef67039`). Default was "refuted unless the evidence holds".

Probe re-run output (`node probes/serial-sharing.mjs`):
`{"serial":1,"siblingSerial":1,"siblingSharesRowPtr":true,"undirectedIsSameObject":false,"undirectedSerial":2,"undirectedSharesRowPtr":false}`

## Summary

| Id | Verdict | Severity | One line |
| --- | --- | --- | --- |
| INTEG-1 | confirmed | blocker | GPU job inside `CI` gates release.yml / coverage.yml (workflow_run, conclusion == success); offline runner = 24 h queue then a non-success conclusion, so NO release for that push. Fix revised: separate workflow, no `continue-on-error`. |
| INTEG-2 | confirmed | major | 9.8 W1 / P10 replace the src/types mirrors with `import type` from devDependencies: contradicts 2.4's lint rule and 3.3's own "type test asserts assignability"; leaks non-dependency imports into dist d.ts. |
| INTEG-3 | confirmed | major | No route for `exactMaxNodes`, `maxInFlight`, sampled-BC `sources`/`k`, `signal`, `onProgress` through the seams; `ForceAtlas2Params` in `setParams` is undefined. Evidence nit: CPU BC options are `normalized`, `endpoints`, `optimized` (not only `normalized`) -- still no `sources`/`k`. |
| INTEG-4 | confirmed | major | nodeMass / named-column weight / seed resolution assigned to layout helpers the GPU package cannot import, with no caller holding snapshot + helpers. Fix broadened: the GPU package can resolve at `load()` with graph-format primitives. |
| INTEG-5 | confirmed | major | `release(previous)` never releases `dm.undirected(previous).snapshot` (distinct object, serial 2, own rowPtr per probe); every layout on a directed source leaks and the 3-resident warning fires. |
| INTEG-6 | confirmed | major | `withColumns()` siblings share the serial (graph-snapshot.ts:923-942, probe), so `refs: Set<serial>` and "destroyed when the last sibling is released" cannot be implemented; tombstone-by-serial kills the original. |
| INTEG-7 | downgraded | minor | Real (devDependencies are release-graph edges; `updateDependents: "auto"`; commit 5ef67039 shows dependents bumped) but the effect is spurious patch releases the monorepo already tolerates, not breakage; one-line fix. |
| INTEG-8 | downgraded | minor | Real: WGSL `max(eff*0.5, 0.05)` vs CPU/Gephi conditional multiply (forceatlas2.ts:205-216); the 1e-30 floor is benign. An illustrative snippet the G3 trace-parity test would catch; two-line fix. |
| INTEG-9 | confirmed | major | Both 12.3 jobs run `pnpm -r run build` (tsc only): `dist/browser.js` / `dist/node.js` / subpath d.ts exist only after `build:bundle`, so the G1 guard is vacuous and strict-consumer cannot resolve subpaths. Fix narrowed: the "substring unsatisfiable" sub-claim is dropped (the plan says the quoted specifier). |
| INTEG-10 | confirmed | minor | Design 4239 `runAlgorithm(snapshot, { accelerator })` vs plan `accelerated(acc).x(s, o)`; equivalent in spirit, undeclared in 1.3/1.5. |
| INTEG-11 | confirmed | minor | Design 16.2 fixes 1e-5; plan uses 1e-4 for BC (2510) and force parity (2722); reasoned in 9.7 / Q-24 but absent from the "all of them" table 1.5. |
| INTEG-12 | confirmed | minor | Design SsspResult has `pathTo` / `pathEdges` (3735); `SsspResultLike` drops them; 9.7 relies on `pathTo()`. Fix revised: `accelerated()` attaches them to the GPU result. |
| INTEG-13 | confirmed | minor | All four factual slips verified (columns.ts:80; @webgpu/types 0.1.72 index.d.ts:2229; Algorithm.ts:217/283; LayoutEngine.ts:36-63). |
| INTEG-14 | confirmed | minor | `spring` is FR in both packages (SpringLayoutEngine.ts:1,108; spring.ts:6,33); mapping table missing; `spring-electrical` has no type value / method. |
| INTEG-15 | confirmed | minor | `algorithms/src` has no `indexed/`; a "first A2 commit" dispatcher with "the same list, non-optional" cannot compile. |
| INTEG-16 | downgraded | minor | Refuted as stated: graphty-element already has a package-local `eslint.config.js` extending the root, so no root edit is required; the residual is a wording clarification. |
| INTEG-17 | confirmed | minor | 11.8 skips thresholds under a single `--project`; every CI command selects `--project=node`; merge-coverage.sh only sums and publishes. Thresholds are never enforced. |
| INTEG-18 | confirmed | minor | pnpm publishes `workspace:*` as an exact pin (pnpm.io/workspaces; `npm view @graphty/graphty-element` shows `@graphty/algorithms: 1.7.2`); design rule 3's "caret range" is wrong and the plan inherits it. |
| INTEG-19 | confirmed | minor | Sketch probes with `rejectSoftware`, then `requestGpuContext({ limits })` re-requests an adapter; `GpuContext.create` never says it honours `rejectSoftware`. |
| INTEG-20 | confirmed | minor | RenderManager.ts:63-69 IS the `useWebGPU` branch (unwired: no other reference); Graph.ts:94 types `WebGPUEngine \| Engine`; Babylon `_device` exists. |
| INTEG-21 | confirmed | minor | Real-GPU stories in graphty-element/stories need `@graphty/webgpu-graph-algorithms/browser`, contradicting "imports nothing new". |
| INTEG-22 | confirmed | minor | The plan's E1 content presupposes the design-E1 DataManager refactor, unlisted as a precondition and outside P6's 8-10 ed. |
| INTEG-23 | confirmed | minor | `^0.4.0` on a 0.x excludes 0.6.x; P-ENV bumps to 0.6.x without widening the peer. |

Missed (found while verifying): M-1 (major) `accelerator-changed` has no subscriber -- late injection and
device loss leave the active layout on the wrong path / dead; M-2 (minor) `schedule` + `labeled` on the
monorepo `CI` workflow run the whole matrix and five Chromatic builds nightly and per label; M-3 (minor) the
"NVIDIA canary" reads no env var and skips instead of failing; M-4 (minor) 12.5 drops the design-16.6
strict-consumer CI step; M-5 (minor) the bridge's `reload()` ignores `report.nodeRemap` for the pin mask.

## Verdicts in detail

### INTEG-1 -- confirmed, blocker

Plan lines 2843-2846 (GPU lane "never required"), 2968-2977 (`test-gpu` `if:` includes
`github.event_name == 'push'`), 3100-3101 (12.5: the job joins `ci.yml`, "NOT added to
all-checks.needs"), 3107 ("release.yml ... gated on CI success -- unchanged for this package").

Evidence re-read:
- `graphty-monorepo/.github/workflows/release.yml:3-8,19`: `on: workflow_run: workflows: ["CI"]
  types: [completed] branches: [master]`; job `if: github.event.workflow_run.conclusion == 'success'`.
- `coverage.yml:3-7,17,51`: identical gate on both jobs.
- `ci.yml`: no `concurrency:` block at all (grep), `all-checks` (706-738) needs only test + Chromatic.
- docs.github.com/en/actions/reference/limits (fetched): "A job can be in the queue for 24 hours
  before it is automatically cancelled" (self-hosted runners).

Reasoning: with `test-gpu` a job of `CI` and `push` in its `if`, every master push queues it. Runner
offline -> the run is not `completed` for 24 h -> `release.yml` / `coverage.yml` do not fire; at the
24 h mark the job is cancelled, the run's conclusion is not `success`, and the release for that push
NEVER happens (the next push repeats the cycle). A real GPU-test failure has the same effect: every
package's release is blocked by the optional lane. "Not in all-checks.needs" protects only PR merges.
The plan considered release.yml (line 3107) and missed that the gate is the whole-workflow conclusion.

Fix (revised): put the GPU lane in its own workflow file (`gpu.yml`: `push` to master, `schedule`,
`workflow_dispatch`, and `pull_request` with `types: [labeled]` + the same-repo/label `if`), with its
own checkout/build steps (or `workflow_run` on CI success if the build artifact is wanted); never a job
of `CI`. Do NOT add `continue-on-error: true`: it would also hide real GPU regressions from the nightly
tracking-issue mechanism of 12.6, and it does not shorten a queued job anyway. State in 12.1 that
"never required" means "not a job of the `CI` workflow". Apply the same split to the staging 12.3
workflow (there the cost is only a pending/red `CI` status and the `${{ github.workflow }}-${{ github.ref }}`
concurrency group stalling master runs behind a queued GPU job). See also M-2.

### INTEG-2 -- confirmed, major

Plan lines 495 (`accelerator.ts = structural mirrors until W1`), 676 (`GpuLayoutSimulation extends
LayoutSimulation` ... "a structural copy lives in src/types until W1"), 2533 (W1: "replace the
structural mirrors with `import type` from the real packages (devDependencies)"), 3169 (P10: same),
2541 ("declares NO peer on algorithms / layout"), versus 365-368 (`src/` has "no import of
`@graphty/algorithms` or `@graphty/layout`", lint-enforced) and 730-734 (3.3: option types are
re-declared structurally; "at W1 a type test asserts mutual assignability").

The plan contradicts itself: 3.3 / 9.1 (2248-2250) keep the mirrors and add a TEST; 9.8 / P10
REPLACE the mirrors in `src/types` with real imports. Under the 9.8 / P10 reading the published
`dist/*.d.ts` carries `import type ... from "@graphty/layout"` for exported interfaces
(`GpuLayoutSimulation extends LayoutSimulation`, `forceAtlas2(o?: ForceAtlas2Options)`), which is
neither a dependency nor a peer. Under pnpm's strict `node_modules` an external consumer that does not
itself depend on `@graphty/layout` cannot resolve it from inside the GPU package (a devDependency is
not installed for a published package): TS2307 without `skipLibCheck`, silently-`any` members with it.
The package's own strict-consumer compile runs inside the workspace where devDependencies exist, so it
would not catch this. ESLint's `no-restricted-imports` flags `import type` unless `allowTypeImports`
is set, so the W1 PR as written also fails the plan's own lint rule.

Fix: as the reviewer says -- the structural mirrors in `src/types` ARE the published contract; the
real packages are imported only in `test/types/conformance.test-d.ts` (`expectTypeOf` in both
directions); reword 3.1 line 495, 3.3 line 676, the 9.8 W1 row and P10. If the owner prefers real
imports, then algorithms / layout must become peers, which contradicts 2541 -- say which.

### INTEG-3 -- confirmed, major

Plan lines 608 (`accelerator(): GpuAccelerator` -- no arguments), 689 (`setParams(patch:
Partial<ForceAtlas2Params | FruchtermanReingoldParams>)` -- neither type is defined anywhere in the
plan: grep finds only this line), 695 (`GpuLayoutTuning` incl. `exactMaxNodes`, `maxInFlight`),
713-714 (`forceAtlas2(o?: ForceAtlas2Options)` "accepts the CPU option type"), 2321-2324 ("GPU-only
tuning goes through the GPU package's own factory options, never through the dispatcher"),
2353-2361 (`LayoutAccelerator.forceAtlas2?(options?: ForceAtlas2Options)`), 2458-2460 (element knobs
`maxInFlight`, `gpuMinNodes`), 1858 ("The element config can set `maxInFlight: 1`"), 2489-2490 (the
app "may call `ctx.calibrate()` once to pass `exactMaxNodes`"), 3246 (Q-13: "The accelerator exposes
`sources` / `k`, `onProgress` and `signal`").

Verified: `algorithms/src/algorithms/centrality/betweenness.ts:15-28` -- `BetweennessCentralityOptions`
has `normalized`, `endpoints`, `optimized` (the reviewer wrote "only normalized"; the substance holds:
no `sources` / `k` / `signal` / `onProgress`). The design (grep "sources", "sampled") defines no
sampled variant for `indexed.*`. The element reaches the GPU only through `createSimulation(type,
opts, acc)` -> `acc.forceAtlas2(opts)` and `accelerated(acc).x(s, opts)`, both carrying CPU option
types; `iterationsPerStep` is the one GPU knob with a route (9.3's `ForceAtlas2Options` includes it
and `step(iterations)` carries it). `exactMaxNodes`, `maxInFlight`, `repulsion`, `compat` and the BC
sampling / cancellation options have none.

Fix: as the reviewer proposes (a tuning-defaults argument on `ctx.accelerator(defaults?)` or
`GpuAccelerator.configure(patch)`; the element bridge applies its knobs through a documented
`"setParams" in sim` duck-type check; `indexed.betweennessCentrality` gains `sources` / `k` in A2 so the
option type is shared and 9.7's "sampled vs sampled" comparison exists; `dest` / `signal` /
`onProgress` are reachable only via the GPU package's own functions; qualify the 9.2 sentence). Add:
define `ForceAtlas2Params` / `FruchtermanReingoldParams` (or replace them with the option types).

### INTEG-4 -- confirmed, major (fix broadened)

Plan lines 1378 (`mass` ... "or `nodeMass` resolved by the layout package's `resolveNodeVector`"),
1476-1480 (`weight === "<edge column>"` binds "the per-arc array the caller expanded with
`expandEdges` (graphty-element caches it ...)"), 1748-1749 (7.14: Record "resolved by the layout
package's `resolveNodeVector`"), 2364-2366 (`createSimulation` returns `accelerator.forceAtlas2(options)`
directly -- no snapshot at that point), 2379-2382 (the three helpers live in layout), versus 177 / 365-368
(no runtime import of layout, lint-enforced).

Verified: the only chain is `createSimulation` -> `acc.forceAtlas2(options)` -> `sim.load(snapshot,
positions)`; nobody in it holds the snapshot AND the layout helpers; the element's `expandEdges`
cache (design 4118-4120) is not reachable from `load()`. So the plan assigns the work to code that
cannot run it.

Fix: the plan must name the resolver. Two valid choices, either is fine: (a) the reviewer's thin
adapter in layout's `createSimulation` whose `load()` resolves and calls `gpuSim.load(snapshot,
positions, { mass, weights })` with `LayoutAccelerator.forceAtlas2` taking resolved vectors only; or
(b) the GPU simulation resolves at `load()` itself using graph-format primitives only --
`snapshot.ids.requireIndex` for a `Record`, `snapshot.nodes.gpuView(name)` for a column name,
`expandEdges(snapshot, column.data)` (exported from `graph-format/src/index.ts:27`) for a named edge
column, cached in the residency under the existing `(gpuView array, column.version)` key -- a
documented structural duplicate of layout's helpers, like the LCG copy already is. (b) needs no
interface change and keeps 14.6's ownership table intact; (a) keeps one resolver. Name one in 7.3,
7.5, 7.14 and 9.3.

### INTEG-5 -- confirmed, major

Plan lines 990-992 (4.5 diagram: listener releases `previous`; the layout reloads
`undirected(next)`), 2399-2402 (9.4 item 2: `release(previous)`; `Graph.dispose()` releases
`current`), 872-874 (3 resident snapshots = missing release warning).

Verified: design 4103-4107 (undirected cache), 4159-4162 (`undirected(s)` returns `s.toUndirected()`
for a directed source), 4189-4197 (undirected adapters and layouts use `dm.undirected(s).snapshot`);
design 4052-4057: under `data.directed: "auto"` the builder starts `directed: true` and record-pushed
data never changes it, so API-fed graphs are directed. Probe: `undirectedIsSameObject: false`,
`undirectedSerial: 2`, `undirectedSharesRowPtr: false` -- a different snapshot with its own core
arrays and serial. `release(previous)` therefore finds nothing the layout uploaded; the undirected
copy of every superseded snapshot stays resident; after three freezes with a live layout the plan's
own warning fires. Nothing in the GPU package can find the derived snapshot from the source (no
link), so the element must release it. (Note: `previous` is `null` on the first freeze per design
4088; the listener must guard it.)

Fix: as the reviewer proposes (release `previous`, `dm.undirected(previous).snapshot` when distinct,
and the `visible(previous)` cache's induced + undirected snapshots; same in `Graph.dispose()`; the
11.3 test `residency.stats().snapshots === 1` after freeze + layout + replace on a directed graph).
Also redraw the 4.5 diagram.

### INTEG-6 -- confirmed, major

Plan lines 205 (DEPARTURE-4: "`withColumns()` siblings share a core; `release(snapshot)` must find
every buffer regardless"), 856 (4.1 key table: "indexed by `snapshot.serial` so `withColumns()`
siblings find it"), 862-868 (`Map<number, ResidencyRecord>` keyed by serial; `ResidentBuffer.refs:
Set<serial>`; "destroyed when the last sibling is released"), 1002-1004 (4.5: "the record is
tombstoned by serial").

Verified: `packages/graph-format/src/snapshot/graph-snapshot.ts:923-925` ("sharing the core, the id
map and the serial"); probe `serial: 1, siblingSerial: 1, siblingSharesRowPtr: true`. With one serial
for all siblings, `refs: Set<serial>` has exactly one entry for every core (no two snapshots with
different serials share a `rowPtr` object in the implemented format), so "the last sibling" is
undefined and `release(sibling)` destroys the shared core and tombstones the ORIGINAL's serial: the
original's next bind throws `E_RELEASED`. A Node script doing `const s2 = s.withColumns({...});
run(s2); ctx.release(s2); run(s)` hits it. The element does not use `withColumns()`, which limits
exposure but not the contradiction in the residency spec.

Fix: as the reviewer proposes: choose (a) siblings are one residency unit (document that releasing
any sibling releases all; a live simulation on a sibling gets `E_RELEASED`) or (b) count distinct
snapshot OBJECTS per record (a `WeakSet`-guarded counter incremented on first `core(s)` per object,
decremented on `release(s)`) and destroy at zero; rewrite the 4.1 key table, the `ResidentBuffer`
record and the DEPARTURE-4 rationale (its "siblings share a core" half is right; the serial half is
why a serial index does not distinguish them).

### INTEG-7 -- downgraded to minor

Plan lines 2537-2545 (independent nx project; "NO peer on algorithms / layout (types only, dev), so
an app can combine any versions"), 2248-2250.

Verified: `nx.json` `release.version.generatorOptions.updateDependents: "auto"`,
`projectsRelationship: "independent"`; `@nx/js@22.3.3/src/release/version-actions.js:97,145`
(`devDependencies` in the dependency collections); nx 22.7.12 `release/utils/release-graph.js:200-216`
(`projectToDependents` built from `readDependencies`, which filters only `implicit`);
`release-group-processor.js:542-544`. Real-world confirmation: commit `5ef67039` "chore(release):
publish" bumps `graphty 0.6.0` as a dependent of `compact-mantine 0.7.0`. So a devDependency on
algorithms / layout would patch-bump and publish the GPU package on every algorithms / layout release.

Why downgraded: the consequence is spurious patch releases, which is how every other dependent in
this monorepo already behaves; nothing breaks, and the "combine any versions" sentence stays true for
CONSUMERS (a patch bump does not add a peer). If INTEG-2's fix is taken (real imports only in a test
file), the devDependency remains and so does the edge.

Fix: one sentence in 9.8: either "accept the side-effect patch releases (monorepo convention)" or add
a `release.groups` entry for the package with `version.updateDependents: "never"` (a root `nx.json`
touch point for 12.5), and qualify "combine any versions" to consumers.

### INTEG-8 -- downgraded to minor

Plan lines 1348 (7.2: `estimateFactor` "unchanged (Gephi ... line for line)" ... "ported to WGSL
verbatim (7.10)"), 1640-1660 (the snippet).

Verified: `layout/src/layouts/force-directed/forceatlas2.ts:205-216`: `if (swing / traction > 2.0)
{ if (eff > 0.05) eff *= 0.5; ... }` and `if (swing > jitter * traction) { if (eff > 0.05) eff *= 0.7;
} else if (speed < 1000) eff *= 1.3;` -- conditional multiply, no floor. The WGSL `eff = max(eff *
0.5, 0.05)` differs: eff 0.06 -> CPU 0.03 / GPU 0.05; eff 0.04 -> CPU 0.04 / GPU 0.05 (raised). Gephi's
`ForceAtlas2.java` uses the conditional form, so "line for line" is false for those two lines. The
`tr = max(traction, 1e-30)` floor changes nothing observable (swing/1e-30 > 2 iff swing/0 > 2 for
swing > 0; 0/1e-30 = 0 and 0/0 = NaN both fail the test), so that half of the claim is moot.

Why downgraded: the snippet is illustrative; the 11.4 trace-parity test at 1e-4 exists precisely to
catch this and would, at G3; the fix is two lines.

Fix: replace the two `max(...)` lines with `if (eff > 0.05) { eff = eff * 0.5; }` / `* 0.7`; keep or
drop the 1e-30 floor but say which (it is harmless); state that the L1 CPU code is the executable
spec of the WGSL.

### INTEG-9 -- confirmed, major (fix narrowed)

Plan lines 375-377 (test asserts no `"webgpu"` in `dist/webgpu-graph-algorithms.js` and
`dist/browser.js`), 387-392 (bundle script copied from graph-io), 485 (skeleton includes the root shim
`webgpu-graph-algorithms.ts`), 2938 and 2990 (both 12.3 jobs: `pnpm -r run build`), 3107, 3159 (G1).

Verified: `packages/graph-io/package.json` scripts: `build` = `tsc -p tsconfig.build.json`, `build:all`
= build + `build:bundle`; `tsconfig.build.json` includes `src/**/*.ts` only;
`scripts/build-bundle.js` + `bundle-types.js` are what write `dist/<entry>.js` and the one-line
`dist/<entry>.d.ts` shims; `graph-io/test/build-output.test.ts:140-143` guards the bundle checks with
`it.skipIf(!bundleExists)`; `packages/README.md:103,169` ("Build step is `pnpm -r run build` (tsc
only)"; "3 build-output checks skip without the bundle"). With the root shim, a tsc-only build DOES
yield `dist/webgpu-graph-algorithms.js`, but as `export * from "./src/index.js"` -- one line, so the
"no webgpu" assertion on it is vacuous either way -- and it yields NO `dist/browser.js`, `dist/node.js`,
`dist/browser.d.ts`, `dist/node.d.ts`. Hence on both 12.3 lanes the G1 guard skips (or fails, if the
plan's test is written without `skipIf`) and `tsconfig.strict-consumer.json`'s `paths` to
`./dist/*.d.ts` cannot resolve the subpaths. The monorepo lane (12.5) is fine: `nx run
webgpu-graph-algorithms:build` runs `build:all` via `project.json`, as graph-io's does.

Narrowed: the "unsatisfiable substring" sub-claim is over-stated. The plan says the string
`"webgpu"` "(the module specifier)", i.e. the quoted specifier; the sourcemap comment
`//# sourceMappingURL=webgpu-graph-algorithms.js.map` (verified on `dist/graph-io.js`) contains the
bare word but not the quoted form, and `E_NO_WEBGPU` is upper-case. A bare-substring test WOULD be
unsatisfiable, so the test must be written on the specifier form.

Fix: run `pnpm -r run build:all` in both 12.3 jobs (and note `tools/prepush.sh` is tsc-only, so the
guard runs in CI only); write the test on the specifier form (`from "webgpu"`, `import("webgpu")`,
both quote styles), not the substring; say the subpaths are bundle-only.

### INTEG-10 -- confirmed, minor

Plan 164 (1.3 row 14.5: "injected as `runAlgorithm(snapshot, { accelerator: gpu })`"), 2313-2319
(`accelerated(acc).pageRank(s, options)`); design 4239-4240. `Graph.runAlgorithm(namespace, type)` is
an element method today (`Graph.ts:360`), so the design's shape is a loose sketch of per-call
injection; the plan's dispatcher is per-call injection with a different spelling. Undeclared in 1.5's
"all of them" table. Fix as proposed (one line in 1.3, citing note 02 section 4.5).

### INTEG-11 -- confirmed, minor

Design 4545-4547 ("`1e-5` for f32 GPU parity"); plan 168 (1.3 row 16.2), 2510 (BC 1e-4), 2722-2723
(force parity 1e-4). Reasoned in 9.7 and Q-24 but 1.5 claims completeness. Fix as proposed.

### INTEG-12 -- confirmed, minor (fix revised)

Design 3735 (`SsspResult { dist; predArc; pathTo(t); pathEdges(t) }`); plan 655 (`GpuSsspResult {
dist; predArc; reachedCount }`), 2273 (`SsspResultLike { dist; predArc }`), 2509 ("`isInPath` via
`pathTo()` reconstruction on the CPU from `predArc`"). Fix: the `accelerated()` dispatcher (which
lives in `@graphty/algorithms`, the owner of the CPU helpers) attaches `pathTo` / `pathEdges` to a GPU
`sssp` / `bellmanFord` result by walking `predArc` through `snapshot.arcSource` / `arcToEdge`
(`graph-snapshot.ts:544`); then `SsspResultLike` can carry both methods and 9.4 item 3's single loop
holds.

### INTEG-13 -- confirmed, minor

Verified: `packages/graph-format/src/types/columns.ts:80` `NumericVector = F32 | F64 | U32 | I32`;
`@webgpu/types` 0.1.72 `dist/index.d.ts:2229` `isFallbackAdapter` on `GPUAdapterInfo` (plan 2.2 and
`isSoftwareAdapter(info)` already read it from `info`; only the 2.6 row label says `adapter.`);
`graphty-element/src/algorithms/Algorithm.ts:217` `abstract run(g: Graph)`, `:283` is inside `static
register`; `LayoutEngine.ts:36-63` has `init/addNode/addEdge/getNodePosition/setNodePosition/
getEdgePosition/step/pin/unpin/nodes/edges/isSettled` and none of `load/reload/dispose/
getNodePositionInto`. Fix as proposed.

### INTEG-14 -- confirmed, minor

`graphty-element/src/layout/SpringLayoutEngine.ts:1,76,108` (`static type = "spring"`, calls
`springLayout`); `layout/src/layouts/force-directed/spring.ts:6,33` (delegates to
`fruchtermanReingoldLayout`); plan 2360-2361 (`SimulationType` has `"spring"`, `LayoutAccelerator`
has no `spring`), 1911-1921 / 3164 (`spring-electrical` preset). Fix as proposed.

### INTEG-15 -- confirmed, minor

`ls graphty-monorepo/algorithms/src` -> no `indexed/`; plan 2308-2317 ("the same list, non-optional";
`Promise.resolve(indexed.x(s, ...))`) and 2530 ("first A2 commit ... does not wait for all 95 ports").
The dispatcher needs the 19 ported functions and their option types. Fix as proposed.

### INTEG-16 -- downgraded to minor (main claim refuted)

Plan 367-368 ("in the package eslint block"); 3104-3110 (root touch points). Verified:
`graphty-monorepo/eslint.config.js:208` has a per-package block for compact-mantine, BUT
`graphty-element/eslint.config.js:1-9` is a package-local flat config that `import`s and spreads the
root config and adds package rules -- an existing precedent that needs no root edit (eslint resolves
the nearest `eslint.config.js` from the package cwd used by the `lint` script). The staging README's
"do not edit the copied shared configs" does not cover a package-local file. So "requires editing
the ROOT" is refuted. Residual: the phrase "package eslint block" is ambiguous. Fix: say "a
package-local `eslint.config.js` extending the root, as graphty-element does" (or, if a root block is
preferred, add `eslint.config.js` to 12.5's touch points).

### INTEG-17 -- confirmed, minor

Plan 2818-2821 (thresholds "skipped when a single `--project` is selected"), 2950 / 3077
(`--project=node --coverage` everywhere). Verified: `algorithms/vitest.config.ts:63-76` skips
thresholds on `--project=default` / `--project=browser`; `tools/merge-coverage.sh:225-260` checks
artifact presence, merges and sums LF/LH, publishes to Coveralls, enforces no threshold. Fix as
proposed (thresholds active for `--project=node`, skipped only for `browser` / `bench` /
`node-limits`).

### INTEG-18 -- confirmed, minor

pnpm.io/workspaces (fetched): `"foo": "workspace:*"` -> `"foo": "1.5.0"`; `workspace:^` -> `^1.5.0`.
`npm view @graphty/graphty-element dependencies`: `@graphty/algorithms: 1.7.2`, `@graphty/layout:
1.6.2` (exact). Design 13.5 rule 3 (3661-3664) says "published as a caret range" -- the design is
wrong and the plan inherits it (1.3 row 13.5, 3.1 lines 516-517). `nx.json`
`preserveLocalDependencyProtocols: true` leaves the rewrite to pnpm. Fix as proposed (`workspace:^`,
and flag the design rule).

### INTEG-19 -- confirmed, minor

Plan 314-321 (`ProbeResult.adapter: AdapterSummary | null`), 265 (`GpuContextOptions.adapter?`),
735-737 (`BrowserGpuOptions` carries `rejectSoftware` for both helpers), 271-290 (`create()` steps
compute `software` but never reject on it), 2481-2483 (sketch: probe with `rejectSoftware`, then
`requestGpuContext({ limits: "raise" })`). Fix as proposed; additionally specify whether `create()`
honours `rejectSoftware` (-> `E_SOFTWARE_ONLY`).

### INTEG-20 -- confirmed, minor

`graphty-element/src/managers/RenderManager.ts:26,63-69` (`useWebGPU` -> `new WebGPUEngine(canvas)`;
no other reference to `useWebGPU` in `src/`, so unwired), `Graph.ts:94` (`engine: WebGPUEngine |
Engine`), `@babylonjs/core/Engines/webgpuEngine.d.ts:153` (`_device: GPUDevice`). Fix as proposed.

### INTEG-21 -- confirmed, minor

Plan 2252-2253 vs 2466-2469 (real accelerator "behind a `navigator.gpu` check" in element stories)
and 2534 (W2 puts GPU stories in "graphty (app), graphty-element stories"); `graphty-element/
package.json` has no such devDependency today. Fix as proposed.

### INTEG-22 -- confirmed, minor

Plan 2532 (E1 precondition "A2 first commit + L1"), 3165 (P6 8-10 ed); design 4048-4211 and 4255.
`graphty-element/src` has no `DataManager.getSnapshot` / `snapshot-replaced` today (grep). Fix as
proposed.

### INTEG-23 -- confirmed, minor

Plan 381 (`"webgpu": "^0.4.0"` peer), 3162 (P-ENV bumps to 0.6.x in both lanes and graph-format
devDependencies). `^0.4.0` on a 0.x = `>=0.4.0 <0.5.0`. Fix as proposed.

## Missed under this lens

### M-1 (major) -- `accelerator-changed` has no subscriber; late injection and device loss leave the active layout on the wrong path

Section 9.4 items 1 and 4, 9.5, 7.19. Plan lines 2388-2389 (`setAccelerator(acc)` and an
`accelerator-changed` event -- the event's only occurrence in the document, per grep), 2413-2445
(the bridge is created once in `_setLayoutInternal` with `createSimulation(type, opts,
graph.accelerator)`; nothing re-creates it), 1860-1862 (the bridge's `.catch` "routes the error to the
element's error channel and stops the layout"), 2484-2485 (app: `setAccelerator(null)` on
`ctx.lost`; "the element keeps working on the CPU path for NEW runs").

Evidence: `attachAccelerator` (2478-2486) is async (probe + `requestDevice`) while the element sets its
layout when data loads; if the layout is set first (declarative attributes, or the app awaiting the
probe after mounting), `createSimulation` runs with `accelerator === null` and the layout stays on the
CPU until the user changes layout type -- the "detected" GPU layout never engages. On device loss the
live `GpuLayoutSimulation.step()` rejects with `E_DEVICE_LOST`, the bridge stops the layout, and
`setAccelerator(null)` re-creates nothing, so the layout is dead (not "working on the CPU path").
The design gives the element everything needed to swap losslessly: positions are element-owned
(design 4067-4077) and `engine.reload(...)` exists (design 4187).

Fix: `LayoutManager` subscribes to `accelerator-changed` and, when the active engine is a
`SimulationLayoutEngine`, disposes the old simulation and re-creates it through
`createSimulation(type, opts, graph.accelerator)` + `load(dm.undirected(getSnapshot()).snapshot,
positions)` with the pinned mask re-applied (coordinates are preserved because the array is the
element's); state this in 9.4 item 1 and 4, say in 9.5 that after `setAccelerator(null)` the CPU
simulation takes over the running layout, and add the swap to the E1 element tests (fake accelerator
injected after `setLayout`; injected accelerator removed mid-run).

### M-2 (minor) -- `schedule` and `labeled` on the monorepo `CI` workflow run the whole matrix and five Chromatic builds nightly and on every label

Section 12.5, plan lines 3063-3064 (`+ types: [opened, synchronize, reopened, labeled]`, `+ schedule:
[{ cron: "17 6 * * *" }]` on the monorepo `CI`), 3100 (`test-gpu: needs: build`).

Evidence: `graphty-monorepo/.github/workflows/ci.yml` has no `concurrency:` block (grep), and its
`build` (43-45), `test` (235-238), `chromatic-element` (467-470), `chromatic-app` (505),
`chromatic-layout` (619) jobs carry no `if:` on the event -- unlike the staging 12.3 `test` job
(`if: github.event_name != 'schedule'`, line 2926). With the diff applied, the nightly cron runs
build + every shard + all Chromatic snapshot builds (billed), and every label added to any PR
re-runs the full workflow (Chromatic included). The 12.5 diff does not add the guards the 12.3 job has.

Fix: the INTEG-1 separate `gpu.yml` removes the need for either trigger on `CI`; if the job stays in
`CI`, guard every non-GPU job with `if: github.event_name != 'schedule'` and gate the PR path on
`workflow_dispatch` or a comment/label check inside the GPU job rather than a `labeled` trigger on
the whole workflow.

### M-3 (minor) -- the "graph-format GPU audit on NVIDIA (canary)" enforces neither NVIDIA nor non-skip

Section 12.1 / 12.3, plan line 2845 ("`gpu-upload.test.ts` on NVIDIA as a canary"), 2994-2996 (the
job step runs `pnpm exec vitest run test/audit/gpu-upload.test.ts` in `packages/graph-format` under the
job-level `GRAPHTY_GPU_REQUIRE=nvidia`).

Evidence: `packages/graph-format/test/audit/gpu-upload.test.ts:38-63` `acquire()` reads no environment
variable, calls `dawn.create([])` (default adapter) and returns a string on any failure; lines 70-74
turn the failure into `console.warn("SKIPPED")`; lines 93-98 `requireGpu()` calls `t.skip(...)`. A
skipped file is a green step. The variable the plan sets is read only by the GPU package's own
`test/setup/gpu.ts` (2.3, line 349). The preceding `gpu-report.mjs` step proves the default adapter
is NVIDIA in that environment only if it uses the identical default acquisition, which the plan does
not state.

Fix: state that the canary relies on `gpu-report.mjs` using the same `create([])` +
`requestAdapter()` default path, and make the step fail on skips (e.g. `--reporter=json` piped to a
check that `numPendingTests === 0` and `numTotalTests > 0`), or teach the audit to honour
`GRAPHTY_GPU_REQUIRE` (a graph-format change: add it to the touch points).

### M-4 (minor) -- 12.5 drops the design-16.6 strict-consumer CI step

Section 12.5 / 13 P10, plan lines 3060-3110 (the monorepo diff has no `typecheck:strict-consumer`
step), 2941 (12.3 runs it), 168 (1.3 row 16.6 "honoured"), 3169-3170 (G10 lists shards, coverage, nx
release, type conformance -- not the strict-consumer compile).

Evidence: design 4629-4631: "A CI step `tsc -p tsconfig.strict-consumer.json` compiles a
consumer-shaped sample against `dist/...d.ts`". Monorepo package `lint` scripts are `eslint && tsc
--noEmit` (`algorithms/package.json:49`, `graphty-element/package.json:97`); graph-format's
`package.json` runs `typecheck:strict-consumer` only from `ready:commit`, so the sibling has the same
gap and the plan's "mirror graph-io" inherits it. Note INTEG-9: this step also depends on the bundle
d.ts shims existing.

Fix: fold `typecheck:strict-consumer` into the package's `lint` script (runs under `nx affected -t
lint`) or add it as a step of the `webgpu-graph-algorithms-node` shard, and list it in G10.

### M-5 (minor) -- the bridge's `reload()` ignores `report.nodeRemap` for the pin mask and the drag overrides

Section 9.4 item 4 / 7.12 / 7.19, plan line 2423 (`reload(snapshot, report, positions) {
this.sim.load(snapshot, positions); }` -- `report` unused), 2430-2431 (`pin` / `unpin` keep a bridge-owned
`NodeMask` by index and call `setFixed(this.mask)`), 1702-1707 and 1874-1884 (`setFixed` / `load`
semantics: nothing says what `load()` does to the fixed mask or the `setPosition` override list when
`n` or the index space changes), 1315-1317 (7.1 item 6 promises topology changes "without losing
placed coordinates" -- pins unmentioned).

Evidence: design 4064-4066 and 4088-4090: on compaction `node.index = report.nodeRemap[node.index]`,
so a bridge mask keyed by the OLD index pins the wrong nodes after a `removeNode` freeze, and the
mask length is stale when `n` grows.

Fix: `reload()` rebuilds the mask from `node.pinned` over `dm.nodes` (or remaps it through
`nodeRemap`) and re-issues `setFixed` after `load()`; `load()` on a new `n` clears the simulation's
fixed words and the override list (state it in 7.12 / 7.19); add an 11.3 property (pin node A, remove
node B < A, freeze, reload: A is still fixed).
