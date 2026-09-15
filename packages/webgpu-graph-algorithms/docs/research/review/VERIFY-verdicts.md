# VERIFY lens -- verifier verdicts

Document: /home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md
Lens: Verifiability, testing and CI
Method: every cited plan line re-read; every cited file opened; docs fetched
(docs.github.com limits + workflow-syntax, actions/upload-artifact README,
pnpm/action-setup README, actions/runner images/Dockerfile); probes run from
tmp/webgpu-plan/review/probes/ (verify-checksum-cross-adapter.mjs on llvmpipe
and NVIDIA, verify-uncapturederror.mjs, tsc-only builds of graph-format and
graph-io into the scratchpad).

Summary: 21 findings -- 12 confirmed, 8 downgraded (real, narrower or less
severe than stated), 1 refuted-in-part (VERIFY-1's mechanism is wrong but a
narrower defect stands, recorded as downgraded). Blockers that survive: VERIFY-3
(release gating) and VERIFY-4 (PAT inside the job container). Five missed
defects added, all minor, plus one verified non-defect (uncapturederror works
under Dawn-node 0.4.0).

## Verdicts

### VERIFY-1 -- DOWNGRADED (blocker -> major)

Claim checked: `pnpm -r run build` is tsc-only so `@graphty/graph-format`
cannot be resolved, strict-consumer fails with TS2307, build-output skips.

What I found:
- `packages/graph-format/tsconfig.build.json` includes `graph-format.ts` (a
  root re-export shim) with `rootDir: "."`, so tsc alone DOES emit
  `dist/graph-format.js` and `dist/graph-format.d.ts` (probe: tsc into
  scratchpad/tsc-only-dist -> both files exist, content
  `export * from "./src/index.js";`). The root export of graph-format
  resolves after a tsc-only build; the "cannot resolve" and the TS2307 on
  `@graphty/graph-format` are REFUTED.
- graph-io (the template the new package mirrors "file for file", plan line
  480-487, D18) has NO root shim: `tsconfig.build.json` include is
  `["src/**/*.ts"]` and tsc-only emits only `dist/src/**` (probe:
  scratchpad/tsc-only-graph-io has no dist/graph-io.js). Its subpath entries
  `dist/<name>.js` / `.d.ts` come only from `scripts/build-bundle.js` +
  `bundle-types.js` (graph-io/scripts/entries.js). The new package's
  `./browser` and `./node` subpaths (plan 375-377) are the same shape, so
  after `pnpm -r run build`: `dist/browser.{js,d.ts}` and `dist/node.{js,d.ts}`
  do not exist; the strict-consumer paths map (`@graphty/<pkg>/*` ->
  `./dist/*.d.ts`, graph-io/tsconfig.strict-consumer.json lines 8-11; plan
  527-528 "against dist/*.d.ts") gives TS2307 for the subpath imports; the
  mirrored `build-output.test.ts` `it.skipIf(!bundleExists)` pattern
  (graph-io/test/build-output.test.ts 140-149; graph-format 137-139) skips the
  "no `\"webgpu\"` in dist/browser.js" assertion (plan 389-392, 2713, G1 line
  3159) -- the one bundle property the package exists to guarantee is never
  checked in CI.
- Root `packages/package.json` line 8 already defines `build` as
  `pnpm -r run build:all`, and the monorepo nx target runs `build:all`
  (graph-format/project.json), so the plan's staging lanes are the only place
  the bundle is never built.

Severity: major (the G0 gate command fails for the subpaths and the G1
bundle assertion is vacuous), not blocker (the root import works; fix is
one token).

Fix: replace every `pnpm -r run build` in 12.3 (lines 2938, 2990) and G0
(3158) with `pnpm run build` at the workspace root (= `pnpm -r run
build:all`); in 11.3 "Build output" and 3.1 state that the bundle assertions
hard-fail when `process.env.CI` is set (no `skipIf`), since CI always runs
`build:all` first.

### VERIFY-2 -- DOWNGRADED (blocker -> major)

Confirmed facts: ci.yml "Build affected (PR)" is `if: pull_request` (line
79-81) and "Build all (master)" is `if: ref == master && (push ||
workflow_dispatch)` (line 89-90); neither runs on `schedule`. The plan's own
added build step is `if: pull_request` (line 3068). The `test` matrix (235-
237, 16 shards) and the five `chromatic-*` jobs (467-621) have no `if`, so
they run on every schedule tick. Worse than stated: every `test` shard
downloads ALL build artifacts unconditionally (ci.yml 380-403, only the
Storybook download is conditional), so on a scheduled run all 16 shards fail
with "Artifact not found" (the failure mode `packages/move/root-touch-
points.diff` lines 133-141 documents), not only `test-gpu`.

Severity: major, not blocker: nothing blocks a merge (`all-checks` does not
need `test-gpu`), the defect is in the W1 slice (P10), and the fix is a text
change. The cost that does land is Chromatic snapshots burned nightly and a
nightly that can never be green.

Fix: as the reviewer wrote -- a separate `.github/workflows/gpu.yml` with
its own triggers (push master, schedule, workflow_dispatch, same-repo PRs
labelled `gpu`), its own install and
`pnpm exec nx run-many -t build --projects=graph-format,webgpu-graph-algorithms`
on the runner; do not touch ci.yml `on:`; no `needs:` on ci.yml jobs.

### VERIFY-3 -- CONFIRMED (blocker)

release.yml lines 3-19 and coverage.yml lines 3-16 + 51 are `workflow_run`
on "CI" with `if: conclusion == 'success'` (read). A `test-gpu` job inside
ci.yml that fails makes the CI run's conclusion `failure` (job-level
`continue-on-error` is absent from the 12.3 body), and a job queued for an
offline self-hosted runner keeps the run in progress: docs.github.com/en/
actions/reference/limits, fetched: "A job can be in the queue for 24 hours
before it is automatically cancelled." So the plan's own note 06 section
4.4 observation ("coverage.yml only runs on a successful CI run") applies to
the GPU job itself, and line 2845 "never blocks" plus line 3108-3110
"release.yml ... unchanged for this package" are false in the monorepo. This
contradicts the owner constraint quoted at 2845 ("a powered-off box must not
block the team").

Fix: the same separate workflow as VERIFY-2. If the job ever stays in
ci.yml, it needs job-level `continue-on-error: true` AND the plan must say
that an offline runner still delays every release/coverage publish by up to
24 h; the "never blocks" claim at 2845 must be reworded either way.

### VERIFY-4 -- CONFIRMED (blocker)

Plan 3024 `USER runner`; 3030-3034 mounts `/srv/gha-runner/token` into the
container and the in-container entry loop runs `gh api ... registration-
token` then `./config.sh ... ./run.sh; rm -rf _work; loop`; 2880-2882 says
job steps run DIRECTLY on the runner (no job container). Therefore job
steps run in the same container, as the same user, with the PAT readable
(the loop must read it, so `runner` can). Additional fact the reviewer
missed: the base image gives `runner` passwordless root --
raw.githubusercontent.com/actions/runner/main/images/Dockerfile (fetched):
`usermod -aG sudo runner` and `echo "%sudo ALL=(ALL:ALL) NOPASSWD:ALL" >
/etc/sudoers`. Any job can therefore become root in a container that
persists across jobs (`-d --restart unless-stopped`, only `_work` is
removed), read the `Administration: write` PAT, and rewrite the loop for the
next job. Exposure is limited to same-repo PR authors / pushers (fork PRs are
excluded by the `if`), but the PAT grants repository administration, which a
write-level collaborator does not otherwise have, and 12.2 line 2878-2880
explicitly claims the token "lives only in the host-side loop". Blocker
stands: it must be corrected before the runner is registered at P0.

Fix (revised): 12.4 becomes a HOST-side loop (outside Docker) that mints a
JIT config (`gh api -X POST /repos/$R/actions/runners/generate-jitconfig`,
note 06 section 3.3) and runs `docker run --rm --gpus all -e
NVIDIA_DRIVER_CAPABILITIES=all graphty-gpu-runner ./run.sh --jitconfig
"$JIT"` per job; the container exits after one job; the PAT never enters
the image or the container; the image is rebuilt from the Dockerfile. Note
in 12.2 that the image's `runner` user has NOPASSWD sudo (needed for
`playwright install --with-deps`), which is acceptable only because the
container is per-job.

### VERIFY-5 -- CONFIRMED (major)

Re-ran tmp/webgpu-plan/review/probes/verify-checksum-cross-adapter.mjs
(n = 2048, seeded): lavapipe `forceSha256 6c338e2470ec11cb`, `f0.z
317.65924072265625`; NVIDIA (LD_LIBRARY_PATH to tmp/egl for libEGL.so.1)
`forceSha256 bf271f4f80286f1f`, `f0.z 317.6592102050781`; swing sums
478208467.346 vs 478208461.491 (1.2e-8 relative). The exact-tile FA2
iteration is not bit-identical across adapters, exactly as plan 7.16
(1778-1782) says; 11.5 (2780-2782) and G1 (3159) nevertheless require
"identical checksums" for the whole skeleton file, which includes "one exact-
tile FA2 iteration on karate" (2776-2777). The only cross-adapter checksum
evidence cited (review note 2832-2834, note 06 3.5 checksum 999.712) is a
multiply-add gather. G1 is unpassable as written.

Fix: as the reviewer wrote -- bit-identical `Uint32Array` results for
`degree` and the 17M-item map on all three adapters; for the FA2 iteration,
swing / traction / per-node force within 1e-5 relative (1e-6 absolute
floor) across adapters, bitwise only across two runs on the SAME adapter.

### VERIFY-6 -- CONFIRMED (major)

graphty-monorepo/algorithms/vitest.config.ts lines 59-70 (read): thresholds
are `undefined` when `COVERAGE_DIR` is set or `--project=browser` /
`--project=default` is on argv. Plan 11.8 (2816-2818) adopts this pattern
("skipped when a single --project is selected") while every CI invocation is
`--project=node --coverage` (2949, 3076) and G1 (3159) requires ">= 80/80/
75/80 on the node project". tools/merge-coverage.sh and .github/workflows/*.yml
contain no threshold (grep: nothing). So no command the plan runs ever
enforces the numbers G1 cites.

Fix: in 11.8 and 11.1 state that thresholds apply when the selected
project set is exactly `node` (argv contains `--project=node` and no other
`--project`) and are disabled only under `COVERAGE_DIR`; keep
`test/limits/**`, `test/browser/**` and `benchmarks/**` out of the node
project's `include` so `--project=node` is the whole node suite.

### VERIFY-7 -- CONFIRMED (minor)

docs.github.com workflow-syntax (fetched): unspecified shell on Linux is
`bash -e {0}`; `shell: bash` is `bash --noprofile --norc -eo pipefail {0}`.
Plan 2993 `node scripts/gpu-report.mjs | tee gpu-report.json` therefore
returns tee's status; the "exits non-zero when GRAPHTY_GPU_REQUIRE names a
vendor that does not match" property (3007-3011) and the G0 red-run check
(3158) are masked. Fix as written: `shell: bash` on the step (also in the
12.5 body), or `> gpu-report.json && cat gpu-report.json`.

### VERIFY-8 -- DOWNGRADED (major -> major, claim narrowed)

Confirmed: GNU `timeout` exits 124 when the limit is hit regardless of the
child's result, so a browser run whose tests all passed but whose
`browser.close()` hung (R-13 "high" likelihood, line 3216; note 06 section
3.5 and 7 reproduced the hang on the NVIDIA path) is a red step, and rule (a)
of section 13 ("green on BOTH lanes") cannot be met reliably. NOT supported:
the orphaned-Chromium claim. `timeout` signals the whole process group unless
`--foreground` (`man timeout`), so vitest receives SIGTERM, and Playwright
spawns the browser `detached` and on SIGTERM runs `process.kill(-pid,
"SIGKILL")` (playwright-core/lib/server/utils/processLauncher.js lines 108-
111, 165-166, 201). With VERIFY-4's per-job container any leftover dies with
the container anyway.

Fix (narrowed): `vitest run --project=browser --reporter=default
--reporter=json --outputFile=browser-results.json` wrapped in `timeout -k 10
600`; on exit 124 pass iff the JSON has `numTotalTests > 0 &&
numFailedTests === 0`; state this in 11.6 and 12.3 (both lanes). Drop the
`pkill`.

### VERIFY-9 -- CONFIRMED (major)

Plan 2977-3004 (read): the GPU job's steps are checkout, pnpm, node,
install, build, gpu-report, canary, node tests, no-subgroups, bench, bench-
compare, `timeout 900 pnpm exec vitest run --project=browser`, upload -- no
`playwright install chromium`. The Dockerfile comment (3023) bakes only
`install-deps` (system libraries). `packages/node_modules/playwright/
package.json` (1.54.1) has no install/postinstall script, so `pnpm install`
downloads no browser. The browser step fails with "Executable doesn't exist"
on every GPU-lane run, and G0 (3158, "GPU lane registered and green") cannot
pass. Refinement: the runner image's `runner` user has NOPASSWD sudo (see
VERIFY-4), so the default lane's `playwright install chromium --with-deps`
step works verbatim in the container.

Fix: add the cached `playwright install chromium` steps (2952-2959) to the
GPU job in 12.3 and to the 12.5 body, or mount a host volume at
`PLAYWRIGHT_BROWSERS_PATH` into the per-job container.

### VERIFY-10 -- DOWNGRADED (major -> minor)

Confirmed contradiction: P1 (3159) merges "ONE exact-tile FA2 repulsion
iteration (K3 with its swing / traction epilogue plus the one-workgroup K4
speed finalize)", i.e. WGSL implementing the 7.2 rows REPULSION_LAW,
SWING_MODE and estimateFactor, at gate G1; 7.2 (line ~1359) says "Gate G3
requires the owner's sign-off on this table, recorded in the PR, before any
FA2 WGSL merges"; D21 (195) says "before any WGSL merges" (literally also
the `degree` kernel). Impact is one skeleton kernel that G3's full parity
re-validates, so minor. Fix as written: move the sign-off into G0 (recorded
in the P0 PR) and reword D21 to "before any WGSL that implements a 7.2 row
merges (P1's K3 / K4 included)".

### VERIFY-11 -- CONFIRMED (major), fix revised

Plan 3161: the oracle is "f64, index-based, the 7.2 table; becomes the spec
of the L1 ForceAtlas2Simulation"; 3168: the first independent cross-check is
W1 (P10). The WGSL and the oracle are two transcriptions of one table by one
author; 11.4's parity tests catch f32 / ordering bugs, not a shared reading
error, and R-1 (3207) rates that risk high / high with the sign-off as the
only mitigation. An independent oracle exists today: `@graphty/layout`
1.6.2 is on npm (`npm view` 2026-09-14) exporting `forceatlas2Layout` and
`rescaleLayout` (layout/src/index.ts line 12; forceatlas2.ts 26, 442), and
7.2's `compat: "port"` column reproduces its laws (REPULSION_LAW 1,
SWING_MODE 1, centroid gravity, same estimateFactor, same attraction,
distributed, gravity). Note 01 2.1.7: the port is deterministic given the
initial positions and node order; 2.1.4: the final `rescaleLayout` is a
similarity transform.

Fix (revised tolerances and mechanics): add to G3: "test/helpers/oracle.ts
in `compat: \"port\"` with settling disabled reproduces the published
`@graphty/layout@1.6.2` `forceatlas2Layout` (test-only devDependency) on
karate / 10x10 grid / star 200 / seeded G(200, 600), passing the SAME initial
positions through the port's `pos` argument (bypasses the LCG question),
for maxIter 1 and 5 within 1e-9 and maxIter 50 within 1e-6 after
`rescaleLayout` on both sides (a chaotic controller amplifies f64 ordering
noise; 1e-9 at 50 is over-tight)". For the paper rows (REPULSION_LAW 0,
centroid gravity, linlog, distributed), an iteration-0 FORCE fixture from
NetworkX `forceatlas2_layout` (note 01 2.1.9: NetworkX's repulsion is the
paper law and iteration-0 forces do not involve swing / traction) generated
once by a committed script under test/fixtures/. Do not use NetworkX beyond
iteration 0: its swing / traction accumulate across iterations (2.1.9 table).

### VERIFY-12 -- CONFIRMED (major), claim narrowed

Plan 2951 and 2998: the NO_SUBGROUPS pass covers only `test/primitives`;
12.2's table (2894) calls it a "matrix 0 / 1" but the YAML has no matrix.
Subgroup paths outside primitives: the FA2 attraction module compiles to
three tiers "exactly as segmentedReduce" with TIER 1 = subgroup-per-row
(1437, 1470-1472) and the repulsion / near-field epilogue reduces "subgroup
variant when available" (1637-1638); without subgroups the tier boundaries
(`tierStart / tierEnd`) also move, so the planner path differs. Every
adapter in the matrix has subgroups (435: 32 / 8 / 4), so the layout
kernels' fallback twins -- the shipped default on any device without
`subgroups` (R-8 default, 3214) -- are never executed. The reviewer's
`spmvPull` / `advance` examples are primitives and probably ARE covered; the
gap is the layout kernels (and any future kernel outside test/primitives).

Fix: run the NO_SUBGROUPS pass over the whole `node` project on the GPU
lane and over `test/primitives test/layouts` on the default lane; add
"results identical with the variant on and off" for the FA2 kernels to G3
(and SpMV to G7).

### VERIFY-13 -- DOWNGRADED (major -> minor)

Confirmed inconsistency: 11.1 (2664) defines a `vitest bench` project
whose JSON is compared with `benchmarks/results/<runner-class>.json`; 12.3
(2999-3000) runs `vitest bench --outputJson bench/results.json` then
`bench-compare.mjs bench/results.json`; 11.7 (2801-2811) says `benchmarks/`
copies graph-format's tsx harness (`bench()`, `appendSession` writing
`benchmarks/results/<host>-node<version>.json`, packages/graph-format/
benchmarks/harness.ts lines 16-34, 205-229; `package.json` line 43 `tsx
benchmarks/run.ts`); 10.4 (2622) records targets "in
benchmarks/results/<host>-node<version>.json". vitest 3.2.7 does have
`--outputJson` (dist/chunks/cac.*.js) with its own shape. Two more facts:
graph-format's `bench()` takes a synchronous `run: (input) => unknown`
(harness.ts line 52) so a GPU benchmark needs an async variant; and "runner
class" is never defined (host + node version in the harness file name vs
adapter in 11.7). It is a P1 design choice, not a failure, hence minor.

Fix: choose the tsx harness (async `bench()`), delete the `bench` vitest
project from 11.1 / 12.3, define `bench-compare.mjs` over the harness shape
with runner class = `${vendor}/${architecture}/${driver major}`, and give
the browser project a Vitest `server.commands` file write so T-3 / T-5 land
in the same results file.

### VERIFY-14 -- DOWNGRADED (major -> minor)

Confirmed: 2918 `permissions: { contents: read }` with no job-level
override; 2977-3004 has no `actions/github-script` step; 3124 promises the
nightly tracking issue; 2876-2877 says no secrets in the GPU job. An
unimplemented promise plus one permission line; minor. Fix as written: a
`gpu-nightly-report` job on `ubuntu-latest` with `needs: test-gpu`, `if:
always() && github.event_name == 'schedule' && needs.test-gpu.result !=
'success'`, job-level `permissions: { issues: write }`; `test-gpu` stays at
`contents: read`.

### VERIFY-15 -- CONFIRMED (minor)

packages/graph-format/test/audit/gpu-upload.test.ts lines 37-64 (read):
`dawn.create([])`, no environment variable, `requestAdapter()`; lines 66-75
skip with a warning when acquisition fails; lines 11-16 document the
llvmpipe fallback. As a "canary" (2845, 2994-2996) it is green on llvmpipe
and green when skipped. After a pipefail-safe gpu-report (VERIFY-7) exits 0
under `nvidia`, the same default `create([])` pick is NVIDIA, so the
residual gap is the skip path. Fix: `shell: bash` and `| tee canary.log &&
grep -q "vendor=nvidia" canary.log` (the test prints `[gpu-upload] adapter
vendor=...`, line 76-78), or list honouring the two variables as an F2
change to graph-format.

### VERIFY-16 -- CONFIRMED (minor)

Plan 349-351 and 2886-2888: environment variables are read by
`test/setup/gpu.ts` and "the vitest config"; 11.2 says `nvidia` requires
`isFallbackAdapter === false` in the browser, but no line says how
`GRAPHTY_GPU_REQUIRE` reaches browser test code (note 05 9.4 names
`test.env` / `import.meta.env`). Fix as written: config-time `test.env`
forwarding and `test/setup/browser.ts` reading `import.meta.env`, listed in
2.3 next to `test/setup/gpu.ts`.

### VERIFY-17 -- CONFIRMED (minor)

11.4 (2743-2745) marks only 1M "hardware only"; 11.2 (2683) scales sizes by
1/50 on software adapters; 7.7 (1560) puts finest-grid saturation "above n ~
65k". 262k / 50 = 5.2k nodes tests no saturation. Fix as written: 262k and
1M GPU-lane-only (`node-limits`), plus a software-adapter case with
`gridMax2D = 32` so saturation occurs at ~1k nodes.

### VERIFY-18 -- CONFIRMED (minor)

11.3 (2711) "every module in every override combination"; 5.1 (1077-1081)
lists 13 overrides; 11.6 item (6) runs the matrix in the browser project.
Unbounded as written, and SwiftShader pipeline creation (LLVM JIT) is slow
enough that an enumerated cross-product would dominate the "light" suite.
Fix as written: an explicit exported table (defaults, each override alone,
the combinations the factories emit) asserted to cover every `PipelineCache`
key seen in the node suite; the browser run is capped to that table.

### VERIFY-19 -- CONFIRMED (minor)

actions/upload-artifact README (fetched): `overwrite` -- "If false, the
action will fail if an artifact for the given name already exists";
artifacts are immutable and unique per run (issues #478 / #493: re-running a
job yields "(409) Conflict: an artifact with this name already exists on the
workflow run"). Plan 2966 and 3004 set neither `overwrite` nor a
`run_attempt` suffix, and a re-run is the normal recovery for an offline
runner. Fix as written.

### VERIFY-20 -- CONFIRMED (minor), fix narrowed

design 15.5 (4437-4440, read) justifies 3x against "shared-runner noise"
on hosted CPU runners; plan 2845 puts the GPU lane on the owner's dev box;
11.7 has no quiet-GPU rule; T-13 (2641) fails the lane on > 3x. Concurrent
interactive GPU use on the same 4070 producing > 3x medians on O(n^2)
kernels is plausible but not measured; with VERIFY-14 each such night opens
an issue. Fix (narrowed): record `nvidia-smi --query-gpu=utilization.gpu,
memory.used` sampled over 10 s in gpu-report.json; `bench:compare` SKIPS
(not fails) when utilisation > 10 %; compare the median of 5 runs; two
consecutive nightly failures before an issue is opened.

### VERIFY-21 -- CONFIRMED (minor)

10.4 (2622-2627) scopes the owner-decision rule to "a target ... in this
table"; 11.4 (2726-2730) fixes 1e-4 / 5e-2 with no rule for an unmeetable
number; 7.16 (1778-1782) already concedes f32 noise. Added risk: `swing_i =
m_i |F_i(t) - F_i(t-1)|` is a difference of nearly equal forces, so f32
force error (~1e-6 relative) becomes ~1e-4 relative in swing when forces
change by ~1 % per iteration -- the 1e-4 leg is borderline by construction.
Fix: extend the 10.4 rule to every 11.4 number; the 1e-4 leg compares
against an f32 oracle (Float32Array scratch, tile summation order), the f64
oracle is the reference for the 5e-2 leg and the distributional metrics.

## Missed defects

### MISSED-1 -- 12.3 / P0: staging-workspace wiring the workflow assumes does not exist (minor)

- `pnpm/action-setup@v4` (2932, 2984) has no `version` and no
  `package_json_file`; the README (fetched) says `version` is "Optional
  when there is a packageManager ... field in the package.json ... otherwise
  this field is required" and `package_json_file` defaults to
  `package.json` at the checkout root. The repository root `package.json`
  has no `packageManager` (it is the npm scaffold, `"name":
  "@graphty/webgpu-graph-algorithms"`; note 07 section 5 rewrites or deletes
  it); `packageManager: pnpm@10.0.0` lives in `packages/package.json` line
  21. Both jobs fail at the second step. The section 12 review note (3135-
  3138) fixed the `working-directory` for `run` steps but not the action.
- `packages/pnpm-workspace.yaml` lists `graph-format` and `graph-io`
  explicitly (lines 1-3) and `packages/knip.config.ts` enumerates workspaces
  by name (lines 17-43); P0 (3158) lists neither addition (12.5 / 3104-3105
  lists them only for the monorepo at W1), yet G0 requires "knip-clean" and
  `pnpm -r` must see the package.

Fix: `- uses: pnpm/action-setup@v4` `with: { package_json_file:
packages/package.json }` in both jobs (and in the deliberate-red-run
instructions); add "`packages/pnpm-workspace.yaml` entry, `packages/
knip.config.ts` workspace entry mirroring graph-io" to the P0 deliverables.

### MISSED-2 -- 11.4 "iteration 0 positions are bit-identical" needs f32 quantisation on the oracle side (minor)

11.4 (2719-2720): both sides seed NaN rows "with the same LCG in index
order ... so iteration 0 positions are bit-identical"; 11.3 says the
oracle is an f64 reference (2705, 3161 "f64, index-based"). The GPU
`positions` buffer is f32 (7.3), so the CPU LCG output must be rounded to
f32 before the oracle uses it, or the two starts differ by up to 2^-24 and
"bit-identical" is false by construction; the f64 oracle then also compares
a slightly different trajectory in the trace leg. Fix: state that
`seedPositions` writes a `Float32Array` and the oracle consumes that same
array (f64 arithmetic on f32-valued inputs).

### MISSED-3 -- 12.5: on a scheduled run every monorepo test shard fails, not only test-gpu (minor, strengthens VERIFY-2)

ci.yml 380-403: each `test` shard downloads `build-algorithms`, `build-
layout`, `build-graphty`, `build-graphty-element`, `build-remote-logger`,
`build-compact-mantine` unconditionally; on `schedule` no build step runs
(79-90), so all 16 shards fail at the first download, and the five
Storybook builds (104-117, no `if`) run and may fail on unbuilt deps. Fix:
subsumed by VERIFY-2's separate workflow; add to 12.5 the sentence "ci.yml
must never gain a `schedule` trigger".

### MISSED-4 -- 12.3 / 12.4: the runner image grants the job root (minor as a standalone item; folded into VERIFY-4)

actions/runner images/Dockerfile (fetched): `runner` is in `sudo` with
`NOPASSWD:ALL`. 12.2 / 12.4 never mention it. Consequence for the plan's
text: the "ephemeral" claim (2871-2873) is registration-only; with a
persistent container the filesystem is not ephemeral. Fix: VERIFY-4's per-
job container; document the sudo fact in 12.2 (it is what makes `playwright
install --with-deps` work in the image).

### MISSED-5 -- 11.7 / 12.3: benchmark harness needs an async `bench()` (minor; folded into VERIFY-13)

packages/graph-format/benchmarks/harness.ts line 52: `run: (input: T) =>
unknown` is synchronous and times with a plain wall clock; GPU benchmarks
must `await mapAsync` / `onSubmittedWorkDone`. "copies graph-format's
harness" (2801) is therefore not literal. Fix: the copied harness gets an
`async` `run` and awaits `device.queue.onSubmittedWorkDone()` inside the
timer.

## Verified non-defects (for the record)

- `uncapturederror` under webgpu@0.4.0 (Dawn-node): probe
  tmp/webgpu-plan/review/probes/verify-uncapturederror.mjs on llvmpipe --
  `device.addEventListener` is a function, `onuncapturederror` exists, a
  deliberately undersized storage binding delivered `GPUValidationError` to
  both the listener and the property, and `popErrorScope` also returned it.
  11.2's "an uncapturederror listener fails the current test" is
  implementable as written.
- The root `@graphty/graph-format` export resolves after a tsc-only build
  (VERIFY-1 refuted part) because `graph-format.ts` is a root shim in
  `tsconfig.build.json`'s include.
- GNU `timeout` without `--foreground` signals the whole process group, and
  Playwright SIGKILLs the browser's process group on SIGTERM (VERIFY-8's
  orphan clause is not supported).
