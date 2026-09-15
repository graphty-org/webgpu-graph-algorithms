# VERIFY review -- verifiability, testing and CI

Reviewer lens: "Verifiability, testing and CI" (adversarial).
Document: /home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md (3328 lines, read in full).
Sections attacked: 11, 12, 13 and the gates referenced from 1.4 (D21), 2.5, 3.1, 10.4.
Date: 2026-09-14.

Evidence sources used (all read, not assumed):

- Plan sections 1-15; research notes 01, 05, 06, 07; draft C's gate discipline via the plan's review notes.
- /home/apowers/Projects/graphty-monorepo/.github/workflows/ci.yml (738 lines), release.yml, coverage.yml,
  tools/merge-coverage.sh, tools/prepush.sh, algorithms/vitest.config.ts (read-only).
- /home/apowers/Projects/webgpu-graph-algorithms/packages/{graph-format,graph-io}/package.json, vitest.config.ts,
  tsconfig.strict-consumer.json, scripts/build-bundle.js, test/build-output.test.ts, test/audit/gpu-upload.test.ts,
  packages/move/root-touch-points.diff, packages/graph-format/benchmarks/harness.ts.
- Installed toolchain: webgpu@0.4.0, vitest 3.2.7 (packages/), vitest 2.1.9 + playwright 1.54.1 (root).
- Two probes written and run under tmp/webgpu-plan/review/probes/ (outputs quoted below).
- External: node-webgpu build.yml (raw GitHub), GitHub docs on workflow-syntax shell defaults and Actions limits.

## Probe results (tmp/webgpu-plan/review/probes/)

1. `verify-lavapipe.mjs` (webgpu@0.4.0, Dawn): `create(["adapter=llvmpipe"])` selects
   `vendor=mesa architecture=software device=llvmpipe-llvm-15-0-7-256-bits-`; `isFallbackAdapter` is undefined.
   Default `requestDevice()` (no requiredLimits) reports the SPEC DEFAULTS: maxBufferSize 268,435,456,
   maxStorageBufferBindingSize 134,217,728, maxStorageBuffersPerShaderStage 8, workgroup storage 16,384,
   invocations 256, both offset alignments 256, features = ["core-features-and-limits"] only.
   -> plan 2.2 step 2 (`architecture === "software"`) and the "defaults" assumptions of 4.2 / G2 hold
   (this closes note 05 unverified item 2). Note: without LD_LIBRARY_PATH for libEGL the NO-OPTS run ALSO
   lands on llvmpipe -- the silent-fallback failure mode is real on this box.
2. `verify-checksum-cross-adapter.mjs` (an FA2-shaped exact-tile iteration: paper law k/d2, max floor,
   gravity with length(), swing = m*|F|; n = 2048, seeded LCG positions):
   - mesa/software: forceSha256 6c338e2470ec11cb, swingSha256 4dbb035c48ab40c0, swingSum 478208467.34643555
   - nvidia/lovelace: forceSha256 bf271f4f80286f1f, swingSha256 7b75112b2d11e051, swingSum 478208461.49072266
   - f0 differs in the last ulp of z (317.65924072265625 vs 317.6592102050781).
   -> The FA2 exact-tile iteration is NOT bit-identical across adapters; see VERIFY-5.

## Findings (ordered by severity)

### VERIFY-1 [blocker] 12.3 / 13 G0 -- `pnpm -r run build` cannot produce what the lane then consumes

Lines 2938, 2990 (both jobs) and 3158 (G0: "`pnpm -r run build`, `lint`, `typecheck:strict-consumer`, `knip` pass").

Claim: `pnpm -r run build` runs each package's `build` script, which in the staged workspace is `tsc -p tsconfig.build.json`
only (packages/graph-format/package.json line 31, graph-io line 71); the vite bundle and the one-line `.d.ts` shims come only
from `build:all` / `build:bundle`. But (a) `@graphty/graph-format`'s `exports["."]` points at `./dist/graph-format.js`
(packages/graph-format/package.json lines 7-13), so every test in the new package that imports the format fails to resolve on
a fresh checkout; (b) the mirrored `tsconfig.strict-consumer.json` maps the package to `./dist/<pkg>.d.ts` and the format to
`../graph-format/dist/graph-format.d.ts` (packages/graph-io/tsconfig.strict-consumer.json lines 8-11), so the G0 step
`typecheck:strict-consumer` fails with TS2307; (c) `test/build-output.test.ts`, which the plan says "proves no `webgpu` in the
root / browser bundles" (2.5 mechanism 1, G1), mirrors graph-io's `it.skipIf(!bundleExists)` (packages/graph-io/test/
build-output.test.ts lines 133-145) and therefore SKIPS silently when only tsc ran. graph-io's own `ready:commit` script
runs `build:all` before `typecheck:strict-consumer` for exactly this reason (graph-io/package.json line 44).

Fix: replace every `pnpm -r run build` in 12.3 and G0 with `pnpm -r run build:all` (or the workspace-root `pnpm run build`,
which is `pnpm -r run build:all`, packages/package.json line 9); state in 11.3 "Build output" that the bundle assertions are
hard failures (no `skipIf`) whenever `process.env.CI` is set.

### VERIFY-2 [blocker] 12.5 -- adding `schedule` to the monorepo `ci.yml` breaks the nightly GPU lane by construction and runs 16 shards + 5 Chromatic jobs every night

Lines 3060-3105 (the diff), especially `+ schedule: [{ cron: "17 6 * * *" }]` (3064), `Build webgpu-graph-algorithms (PR)`
with `if: github.event_name == 'pull_request'` (3066-3068) and `test-gpu: (... needs: build ... downloading
build-graph-format and build-webgpu-graph-algorithms)` (3101).

Claim: in the monorepo `build` job both build steps are gated -- "Build affected (PR)" on `pull_request` and "Build all (master)"
on `push || workflow_dispatch` (ci.yml lines 79-88) -- so on a `schedule` event nothing is built, the upload steps produce no
artifact, and `test-gpu`'s `download-artifact` fails with "Artifact not found" (the exact failure mode
packages/move/root-touch-points.diff lines 133-141 documents). The `test` matrix (ci.yml 235-237) and the five `chromatic-*`
jobs (467, 505, 543, 581, 619) have no `if:`, so every nightly run also executes 16 hosted shards and five Chromatic uploads
(snapshot quota) for a change nobody made. The standalone 12.3 workflow got this right (`if: github.event_name != 'schedule'`
on `test`, line 2925); the diff forgot it for every other job.

Fix: do not add `schedule` (or `pull_request.types`) to `ci.yml`. Put the GPU lane in its own workflow file
(`.github/workflows/gpu.yml`) with `on: push (master), schedule, workflow_dispatch, pull_request [labeled, synchronize]`,
its own `pnpm install` + `nx run-many -t build --projects=graph-format,webgpu-graph-algorithms` on the dev box (fast), the
same `if`, and no `needs:` on `ci.yml`. This also resolves VERIFY-3.

### VERIFY-3 [blocker] 12.1 / 12.5 -- "never blocks" is false: a `test-gpu` job inside `ci.yml` gates the monorepo's release and coverage publishing

Lines 2845 ("NO (never blocks a merge; a powered-off box must not block the team)"), 3100-3112 ("NOT added to
all-checks.needs" ... "release.yml ... gated on CI success -- unchanged").

Claim: `release.yml` and `coverage.yml` are `workflow_run: workflows: ["CI"], types: [completed]` with
`if: github.event.workflow_run.conclusion == 'success'` (release.yml lines 3-19, coverage.yml lines 3-16, 51). A job in the
same workflow that fails (the R-13 `browser.close()` hang, a driver regression, a 3x bench blip) turns the run's conclusion
to `failure`; a job that cannot start because the dev box is off stays queued and GitHub cancels it after 24 hours
(docs.github.com/en/actions/reference/limits: "A job can be in the queue for 24 hours before it is automatically
cancelled"). Either way: no release and no Coveralls publish for that master push, and every release waits for the GPU
lane to finish. Keeping the job out of `all-checks.needs` protects PR merges only.

Fix: move the GPU lane to a separate workflow (VERIFY-2). If it must stay in `ci.yml`, add `continue-on-error: true` at job
level AND accept that the 24-hour queue wait still delays releases -- state that trade-off explicitly; do not claim "never
blocks".

### VERIFY-4 [blocker] 12.2 vs 12.4 -- the `Administration: write` PAT is mounted inside the container that runs untrusted job steps, and the container persists across jobs

Lines 2878-2880 ("the registration token (fine-grained PAT with `Administration: write`, or a GitHub App) lives only in the
host-side loop ... The job runs its steps DIRECTLY on the runner (no job-level `container:`)") versus 3025-3034
(`docker run ... -v /srv/gha-runner/token:/run/secrets/gh-token:ro graphty-gpu-runner` and "entry loop: TOKEN=$(gh api -X
POST ... registration-token) ... ./config.sh ... ./run.sh; rm -rf _work; loop").

Claim: the entry loop is the container's process, so the PAT is inside the container; job steps run "directly on the runner"
as the same `runner` user (Dockerfile `USER runner`, line 3024) in the same filesystem. Any job -- a same-repo PR from any
account with write access, a compromised devDependency's install script, a test -- can `cat /run/secrets/gh-token` (a PAT
with repository Administration: write, i.e. repo-admin level) and can edit `./run.sh`, `./config.sh` or `bin/` so the NEXT
loop iteration runs attacker code: only the registration is ephemeral, the machine is not (GitHub's own warning quoted in
12.2). `rm -rf _work` removes the checkout, not the runner binaries, npm caches or `~/.cache/ms-playwright`.

Fix: rewrite 12.4 so the PAT never enters the container: a host-side loop (outside Docker) calls
`POST /repos/.../actions/runners/generate-jitconfig` (or the registration token), then starts a FRESH `docker run --rm
--gpus all -e RUNNER_JITCONFIG=<one-shot> graphty-gpu-runner` whose entrypoint is `./run.sh --jitconfig "$RUNNER_JITCONFIG"`
and exits after one job. Say explicitly that the image is rebuilt from the Dockerfile, never snapshotted. Keep 12.2's
sentence, which is then true.

### VERIFY-5 [major] 11.5 / G1 -- "identical checksums on lavapipe, SwiftShader and NVIDIA" for the whole skeleton file contradicts 7.16 and is false for the FA2 iteration

Lines 2780-2782 ("the whole file produces identical checksums on lavapipe, SwiftShader and NVIDIA"), 3159 (G1: "11.5 in
full on lavapipe, SwiftShader and NVIDIA with identical checksums"), 2832-2834 (review note grafting draft A's criterion).

Claim: the skeleton file includes "one exact-tile FA2 iteration on karate and read the swing / traction trace" (line 2777).
7.16 says cross-GPU coordinates "differ at f32 noise level" and 11.4 says "coordinates are never compared". Probe 2 above
shows the FA2-shaped tile (division, max, length) yields DIFFERENT byte checksums on llvmpipe and NVIDIA under Dawn
(6c338e24... vs bf271f4f...; last-ulp differences). The evidence cited for the criterion (note 06 section 3.5, checksum
999.712) is a multiply-add gather, not the tile. As written G1 is unpassable, or passes only after a silent relaxation.

Fix: in 11.5 and G1 write "bit-identical `Uint32Array` results for `degree` and the 17M-item map on all three adapters; for
the FA2 iteration, swing / traction / per-node force within 1e-5 relative (floor 1e-6 absolute) across adapters".

### VERIFY-6 [major] 11.8 / G1 -- the coverage thresholds are never enforced by any command the plan runs

Lines 2818-2820 ("Thresholds 80 / 80 / 75 / 80 on the `node` project when run whole ... skipped when a single `--project`
is selected, `algorithms/vitest.config.ts` pattern"), 2949 (`vitest run --project=node --coverage`), 3076 (monorepo shard,
same command), 3159 (G1: "coverage >= 80/80/75/80 on the node project").

Claim: the cited pattern is `thresholds: process.env.COVERAGE_DIR || process.argv.includes("--project=browser") ||
process.argv.includes("--project=default") ? undefined : {...}` (graphty-monorepo/algorithms/vitest.config.ts lines 59-70):
thresholds are DISABLED whenever `--project` is passed. Every CI invocation passes `--project=node`. "When run whole" would
run all four projects, which cannot succeed on the default lane (`node-limits` needs limits lavapipe cannot raise, 11.1 line
2663) and includes the Playwright project. The monorepo checks nothing after merging (`tools/merge-coverage.sh` contains no
threshold logic; grep "threshold" returns nothing in tools/ or .github/workflows/). So G1's coverage gate is prose.

Fix: in 11.8 specify "thresholds apply when the selected project set is exactly `node` (`--project=node` alone) and are
disabled only when `COVERAGE_DIR` is set (sharded runs)"; keep the `include` of `test/limits/**` and `test/browser/**` out of
the `node` project so `--project=node` is the whole node suite.

### VERIFY-7 [minor] 12.3 -- `node scripts/gpu-report.mjs | tee gpu-report.json` cannot "fail loudly"

Lines 2993 and 3007-3011 ("exits non-zero when `GRAPHTY_GPU_REQUIRE` names a vendor that does not match"); the same shape
appears in the 12.5 job body (note 06 section 6 diff).

Claim: a `run:` step without an explicit `shell:` executes under `bash -e {0}`; only an explicit `shell: bash` adds
`-o pipefail` (docs.github.com workflow-syntax, jobs.<job_id>.steps[*].shell: "bash --noprofile --norc -eo pipefail {0}"
versus unspecified "bash -e {0}"). The step's exit status is `tee`'s (0), so a software adapter does not fail this step; the
job only turns red two steps later at the vitest run, after the graph-format "canary" has already passed on llvmpipe
(VERIFY-15).

Fix: add `shell: bash` to the step, or write `node scripts/gpu-report.mjs > gpu-report.json && cat gpu-report.json` as
note 06's sketch does.

### VERIFY-8 [major] 12.3 / 11.6 / R-13 -- the `timeout 600` "backstop" turns the known hang into a red step and leaves Chromium holding the GPU

Lines 2963 (`timeout 600 pnpm exec vitest run --project=browser # hard kill backstop`), 3001 (`timeout 900 ...`), 2797-2799
("a hard job timeout backstops the `browser.close()` hang"), 3083 (monorepo shard), R-13 line 3216 (likelihood "high").

Claim: `timeout` sends SIGTERM to `pnpm`, exits 124, and does not kill Chromium's process tree (three.js SIGKILLs the tree
for this reason, note 06 section 3.5). Result: every occurrence of a hang AFTER all tests passed is a failed step (exit 124),
and the orphaned Chromium keeps the GPU inside the persistent runner container (VERIFY-4) for the next job. With R-13 rated
"high" the GPU lane's browser step is red by construction, and on the required default lane a hang is a red PR.

Fix: (a) run vitest with `--reporter=default --reporter=json --outputFile=browser-results.json`; (b) wrap as `timeout -k 10 600
setsid pnpm exec vitest ...` and, on exit 124, decide pass/fail from `browser-results.json` (`numFailedTests === 0`) and
`pkill -f chromium` in the same step; (c) with VERIFY-4's per-job container, orphans die with the container. State this in
11.6 and 12.3.

### VERIFY-9 [major] 12.3 GPU job / 12.4 -- no Playwright browser is ever installed on the GPU lane

Lines 2977-3004 (the `test-gpu` steps: no `playwright install`), 3023 ("+ `npx playwright install-deps chromium` at build
time").

Claim: `install-deps` installs system libraries only; the browser binary comes from `playwright install chromium`.
`playwright@1.54.1` has no postinstall that downloads browsers (node_modules/playwright/package.json has no `scripts`
entry). The default lane has the cached install step (2952-2959); the GPU lane runs `timeout 900 pnpm exec vitest run
--project=browser` with nothing populating `~/.cache/ms-playwright` -> "Executable doesn't exist" on every run, and again
after every Playwright bump.

Fix: add the same three cached steps (or `pnpm exec playwright install chromium` with `PLAYWRIGHT_BROWSERS_PATH` on a host
volume mounted into the per-job container) to the GPU job in 12.3 and to the 12.5 job body.

### VERIFY-10 [major] 13 P1 vs G3 / D21 -- FA2 WGSL merges in P1, before the sign-off G3 requires "BEFORE the WGSL merges"

Lines 3159 (P1 deliverables: "ONE exact-tile FA2 repulsion iteration (K3 with its swing / traction epilogue plus the
one-workgroup K4 speed finalize)"), 3161 (G3: "owner sign-off on the 7.2 formula table recorded in the PR BEFORE the WGSL
merges"), 195 (D21: "the owner signs off the FA2 formula table before any WGSL merges").

Claim: K3 carries `REPULSION_LAW`, the coincident-node kick and the swing / traction form; K4 is the `estimateFactor` port --
all of them rows of the 7.2 table. They merge at P1 (gate G1), two phases before G3. D21's literal "any WGSL merges" would
also block P1's `degree` kernel. R-1 (high / high) is mitigated by the sign-off, so the timing matters.

Fix: move the sign-off into G0 ("owner sign-off on the 7.2 table recorded in the P0 PR; no FA2-law WGSL merges before it")
and reword D21 to "before any WGSL that implements a 7.2 row merges (P1's K3 / K4 included)".

### VERIFY-11 [major] 13 P3 / 9.8 W0 / 11.3 -- the P3 FA2 oracle has no independent check until W1

Lines 3161 ("the CPU FA2 oracle in `test/helpers/oracle.ts` (f64, index-based, the 7.2 table; becomes the spec of the L1
`ForceAtlas2Simulation`)"), 2528 ("CPU reference implementations in `test/helpers/oracle.ts` written from design Ports 1-6
and the 7.2 table"), 2705-2706 ("CPU FA2 / FR oracle (post-L1 the real `ForceAtlas2Simulation`)").

Claim: the L1 rewrite does not exist, so the oracle and the WGSL are written by the same engineer from the same table; a
shared misreading (sign of a term, the `0.5 m |F + Fold|` traction, the `1e-30` clamp) passes every 11.4 parity test and is
first caught at W1 ("a mismatch ... blocks W1", line 3168) -- after P4-P9. Note 01 section 8.8 item 1 explicitly says the
CPU formulas must be settled in the layout package "BEFORE writing WGSL". An independent reference exists today: the
published `@graphty/layout` `forceatlas2Layout` (npm, no graph-format dependency) is deterministic under `seed` (note 01
2.1.7) and implements exactly the `compat: "port"` rows of 7.2; its final `rescaleLayout` is a similarity transform, so a
k-iteration run compares after applying the same transform to the oracle's positions.

Fix: add to G3 "the oracle in `compat: "port"` mode reproduces `@graphty/layout@<published>` `forceatlas2Layout` (devDependency
of the test only) on karate / grid / star / random 200 for maxIter 1, 5 and 50 with the same seed, positions equal within
1e-9 after `rescaleLayout` on both sides; the `paper` rows are covered by an iteration-0 force fixture generated once from
NetworkX `layout.py` (committed JSON, generator script under `test/fixtures/`)". This gives P3 an oracle that is not the
author's own transcription.

### VERIFY-12 [major] 12.3 / 12.2 / R-8 -- the non-subgroup twins of the layout and algorithm kernels are never executed

Lines 2951 and 2998 (`GRAPHTY_GPU_NO_SUBGROUPS=1 pnpm exec vitest run --project=node test/primitives`), 2893 (12.2 table:
matrix `0` / `1`), 1274-1276 (6: "The non-subgroup variant is always compiled and tested too"), R-8 line 3211
("high / high").

Claim: every adapter in the matrix advertises `subgroups` (lavapipe 8, SwiftShader 4, NVIDIA 32; 2.6 table), so the
workgroup-memory fallback runs only where the variable forces it, and the variable is applied to `test/primitives` alone.
The FA2 attraction tiers are their own module with `override TIER` (7.5: "the same module compiles to three tiers ...
exactly as the segmentedReduce primitive"), the repulsion / near-field epilogue has its own "subgroup variant when
available" (7.10), and `spmvPull` / `advance` callers select the variant per kernel. None of these fallbacks is in
`test/primitives`; R-8's "always ship the non-subgroup variant" ships it untested.

Fix: run the `GRAPHTY_GPU_NO_SUBGROUPS=1` pass over the whole `node` project on the GPU lane (cost: one more 2-6 minute
run) and over `test/primitives test/layouts` on the default lane; add "results identical with the variant on and off" to
G3 and G7 for the layout and SpMV kernels specifically.

### VERIFY-13 [major] 11.1 / 11.7 / 3.1 / 10.4 -- two incompatible benchmark mechanisms; `bench:compare` compares files of different shapes

Lines 2664 (`bench` vitest project: "`vitest bench` ... JSON output compared with `benchmarks/results/<runner-class>.json`"),
2999-3000 (`vitest bench --project=bench --outputJson bench/results.json` then `bench-compare.mjs bench/results.json`),
2801-2806 (11.7: "copies graph-format's harness (`bench()`, `printTable`, `appendSession` ...)"), 511 (skeleton:
`benchmarks/ harness.ts datasets.ts run.ts <group>.bench.ts results/`), 2622 (10.4: targets recorded in
`benchmarks/results/<host>-node<version>.json`), 2812-2814 ("Browser numbers come from the frame-loop test printing
`performance.now()` deltas into the report artifact").

Claim: graph-format's harness is a `tsx benchmarks/run.ts` script writing `BenchResult { group, name, medianMs, ... }`
sessions (packages/graph-format/benchmarks/harness.ts lines 1-40, package.json line 43); `vitest bench --outputJson` writes
vitest's own `{ files: [{ groups: [{ benchmarks: [{ hz, mean, p75 ...` shape. The plan's `bench-compare.mjs` reads the
vitest file and compares with a harness-format baseline; 10.4's T-1..T-11 are "recorded" into the harness file; T-3 and T-5
(Chromium) come from browser tests, which cannot write files at all (no fs in the page), so they land in the job log only.
Nothing in the plan says which format is authoritative, and `vitest run` without `--project` would also try to run the
bench project as tests.

Fix: choose the tsx harness (it already has `appendSession`, host / GPU metadata and `datasets.ts`), delete the `bench`
vitest project from 11.1 and 12.3 (`tsx benchmarks/run.ts --json bench/results.json`), define `bench-compare.mjs` over the
harness shape, and give the browser project a `server.commands` (Vitest browser `commands`) file-write so T-3 / T-5 land
in the same results file.

### VERIFY-14 [major] 12.6 vs 12.2 / 12.3 -- the nightly "tracking issue" needs `issues: write` on the self-hosted job, which the plan forbids and the YAML lacks

Lines 3124 ("opens / refreshes a tracking issue on failure (`actions/github-script`)"), 2918 (`permissions: { contents:
read }`), 2876-2877 ("workflow permissions read-only; NO secrets in the GPU job (artifacts only)").

Claim: `actions/github-script` creating an issue requires `issues: write` on the job's `GITHUB_TOKEN`; the 12.3 YAML has
neither the permission nor the step, so 12.6 promises a mechanism 12.3 does not implement, and the obvious implementation
(adding `permissions: { issues: write }` to `test-gpu`) puts a write-capable token on the self-hosted runner, contradicting
12.2.

Fix: add to 12.3 a job `gpu-nightly-report` on `ubuntu-latest` with `needs: test-gpu`, `if: always() &&
github.event_name == 'schedule' && needs.test-gpu.result != 'success'` and job-level `permissions: { issues: write }`,
which downloads `gpu-results-*` and opens / updates the issue; keep `test-gpu` at `contents: read`.

### VERIFY-15 [minor] 12.3 / 12.1 -- the graph-format "canary" cannot fail on a software adapter

Lines 2994-2996 (step "graph-format GPU audit on NVIDIA (canary)" running `pnpm exec vitest run
test/audit/gpu-upload.test.ts`), 2845 ("graph-format's `gpu-upload.test.ts` on NVIDIA as a canary").

Claim: that test's `acquire()` calls `dawn.create([])`, ignores `GRAPHTY_GPU_ADAPTER` and `GRAPHTY_GPU_REQUIRE`, and
`t.skip`s with a printed reason when no adapter exists (packages/graph-format/test/audit/gpu-upload.test.ts lines 11-16,
37-64, 68-75). It is green on llvmpipe and green when skipped; combined with VERIFY-7 the job reaches it before anything has
enforced `nvidia`.

Fix: state that the canary is meaningful only after a pipefail-safe `gpu-report.mjs` step has exited 0 under
`GRAPHTY_GPU_REQUIRE=nvidia`, and add `| tee canary.log && grep -q "vendor=nvidia" canary.log` (with `shell: bash`) so a
skip or an llvmpipe run is red; or extend the graph-format test to honour the two variables at F2 (a graph-format change,
listed as such).

### VERIFY-16 [minor] 11.6 / 2.3 / 12.2 -- how `GRAPHTY_GPU_REQUIRE` reaches browser test code is unspecified

Lines 2788-2789 ("the vendor assertion holds under `GRAPHTY_GPU_REQUIRE=nvidia`" in the browser project), 349-351
("Environment variables are read ONLY by the test setup (`test/setup/gpu.ts` ...)"), 2886-2894 (the variable table).

Claim: browser-mode test code runs in Chromium where `process.env` does not exist; note 05 section 9.4 already flagged that
the value must be "forwarded via `test.env` or `import.meta.env`". Without a stated mechanism the NVIDIA-lane browser step
silently accepts SwiftShader -- the exact failure HEADLESS_GPU_REPORT.md documents -- while the plan claims the lane is red
on a software fallback.

Fix: in 11.1 / 11.2 add "the browser project sets `test.env: { GRAPHTY_GPU_REQUIRE: process.env.GRAPHTY_GPU_REQUIRE ?? "" }`
at config-evaluation time and `test/setup/browser.ts` reads `import.meta.env.GRAPHTY_GPU_REQUIRE`"; list the browser setup
file next to `test/setup/gpu.ts` in 2.3.

### VERIFY-17 [minor] 11.4 / G4 -- the "262k finest-grid saturation" fixture does not saturate at `gpuScale` on the default lane

Lines 2743-2745 ("sizes 20k, 100k, 262k (finest-grid saturation) and 1M (hardware only)"), 3163 (G4: "lavapipe runs the
grid suite at `gpuScale` sizes in <= 4 min"), 2683-2685 (`gpuScale()` = 1/50 on software adapters).

Claim: 262k / 50 = 5.2k nodes; 7.7 says the 512^2 cap saturates only above ~65k nodes in 2D, so the default-lane version of
this fixture never exercises the saturated near field, and the only lane that does is never required. The exact oracle at
262k on 4-thread lavapipe would take ~9 minutes per iteration (388 ms/iter at 20k on 32 threads x (262/20)^2 x ~8), so
scaling is unavoidable -- the fixture's name is what is wrong.

Fix: mark 262k and 1M as GPU-lane-only (`node-limits`), and add a software-adapter saturation case that lowers `gridMax2D` to
32 so saturation occurs at ~1k nodes.

### VERIFY-18 [minor] 11.3 / 11.6 -- "every module in every override combination" is unbounded and sits in the "light" browser suite

Lines 2711 (WGSL compile matrix: "every module in every override combination compiles ... AND on Chromium in the browser
project"), 2795 (browser item (6)).

Claim: the FA2 modules alone carry `LINLOG`, `DISTRIBUTED`, `STRONG_GRAVITY`, `REPULSION_LAW`, `SWING_MODE`,
`GRAVITY_CENTER`, `TIER`, `USE_PERM`, `HAS_WEIGHTS`, `SUBGROUP_SIZE`, `LEVELS` (5.1 list); the cross product is hundreds of
pipelines per module and grows with every algorithm. "the package uses" is not a bound anyone can check, and on SwiftShader
each `createComputePipelineAsync` costs tens of milliseconds.

Fix: define the matrix as an explicit exported table (`test/helpers/override-matrix.ts`: the defaults, each override toggled
alone, and the exact combinations the factories emit), assert the table covers every `PipelineCache` key seen in the node
suite, and cap the browser run to that table.

### VERIFY-19 [minor] 12.3 -- fixed artifact names fail on job re-run, the normal recovery for an offline runner

Lines 2966 (`name: coverage-webgpu-graph-algorithms`), 3004 (`name: "gpu-results-${{ github.run_id }}"`).

Claim: `actions/upload-artifact@v4` refuses an upload whose name already exists in the run unless `overwrite: true`; a
re-run of a failed `test-gpu` (or of the default job) within the same run id hits the name from the first attempt.

Fix: add `overwrite: true` to both upload steps (and to the 12.5 job body), or suffix `-${{ github.run_attempt }}`.

### VERIFY-20 [minor] 10.4 T-13 / 11.7 / 12.6 -- the 3x regression check on the owner's active dev box is flaky by construction

Lines 2641 (T-13: "any tracked median > 3x its checked-in baseline for the runner class fails the GPU lane's `bench:compare`
step"), 2810-2811, 3123-3124.

Claim: the runner shares the RTX 4070 SUPER with the owner's interactive work (Storybook on the real GPU, other test runs,
the P6 stories); a concurrent GPU workload during the 3-10 minute bench produces > 3x medians on O(n^2) kernels. The
design's 3x rule (design 15.5) was written for hosted CPU runners; here the noise source is the operator. With VERIFY-14's
issue automation this becomes issue spam; without it, a red nightly nobody reads.

Fix: before the bench step, poll `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` for 10 s and skip (not
fail) the bench when utilisation > 10%; compare the median of 5 runs; require two consecutive nightly failures before the
issue is opened; record the utilisation sample in `gpu-report.json`.

### VERIFY-21 [minor] 11.4 -- the trace tolerance schedule has no rule for when it is unmeetable

Lines 2726-2730 ("relative error `<= 1e-4` for the first 10 iterations and `<= 5e-2` through iteration 50 ... (chaotic
divergence beyond is expected)").

Claim: the oracle is f64 with a sequential sum; the kernel is f32 with tile-order sums and vendor-dependent fma contraction
(7.16). The controller feeds `speed` back into every position, so per-iteration relative error grows geometrically; 5e-2 at
iteration 50 on a 1,000-node graph is a guess the plan itself hedges. 10.4's "a target is never relaxed silently" rule
covers T-targets only, not the 11.4 numbers, so a failing schedule would be edited in the test file.

Fix: extend the 10.4 owner-decision rule to the 11.4 tolerances ("a change to any 11.4 number is a recorded decision in the
PR"), and specify that the 1e-4 leg compares against an f32 oracle that sums in tile order (the same TypeScript with
`Float32Array` scratch), the f64 oracle being the reference for the 5e-2 leg and the distributional metrics.

## What holds up (checked, no finding)

- lavapipe under the Dawn node package: `adapter=llvmpipe` selection, `architecture === "software"`, spec-default device
  limits (probe 1); node-webgpu's own CI runs on `ubuntu-24.04` with `mesa-vulkan-drivers libvulkan1` and
  `WEBGPU_USE_CI_AVAILABLE_RENDERER=1` (raw build.yml fetched). Still not executed on a hosted runner by this project
  (note 06 item 7) -- G0 covers that.
- `pool: "forks"` with the Dawn addon: graph-format's config (line 7) and its suite.
- Monorepo shard matrix: extra `include` keys (`needs-vulkan`) are legal; `merge-coverage.sh`'s shard match
  (`coverage-$pkg-*`, lines 97-116) accepts `coverage-webgpu-graph-algorithms-node`; the browser shard uploads no coverage,
  consistent with `--ci` failing on missing packages only.
- The `test-gpu` `if` expression, label gating and `head.repo.full_name` clause are correct; `concurrency` at job level with
  `cancel-in-progress: false` is valid; the artifact `path` with an escaped `\n` inside a double-quoted flow scalar is valid
  YAML.
- Timing assertions: no test asserts a wall-clock bound; T-1..T-12 are recorded numbers with an owner-decision rule; the
  only CI-asserted timing is T-13 (VERIFY-20).
- Node-primary / browser-light: the browser project is seven items, with the compile matrix (VERIFY-18) as the only
  unbounded one.
- Walking skeleton coverage of the risky things: Dawn in Node, `./browser` on SwiftShader and NVIDIA, arena hot prefix with
  segment offsets and `arena.byteOffset !== 0`, per-array path, readback ring reuse, `release` + leak counter, uncaptured
  error hook, 2D dispatch at the boundary, device loss -- all present in 11.5 / 11.3. The `.wgsl.ts` decision (D9) removes
  the import-path risk; `src/wgsl/**` is excluded from coverage (11.8), so WGSL strings do not depress the thresholds.
- Property tests per kernel, exact-vs-approximate bounds (RMS 5% / p99 25% / unbiasedness 5%), differential tolerances
  (9.7, 11.4) are all numeric.

## Overall

The testing design (11.3-11.5) is unusually concrete and mostly mechanically checkable. The CI design is where the plan
would fail on first contact: the build command (VERIFY-1) breaks the default lane before any test runs, the monorepo diff
(VERIFY-2 / VERIFY-3) breaks the nightly lane and gates releases on a dev box, and the runner recipe (VERIFY-4) puts a
repo-admin PAT where job code can read it. All four are text fixes. The single biggest risk to the first deliverable is
VERIFY-11: a self-authored oracle validates a self-authored kernel until W1.
