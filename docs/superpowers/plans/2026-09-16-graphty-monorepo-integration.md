# WebGPU Package -> graphty-monorepo Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Every task names the repository it runs in; most run in `/home/apowers/Projects/graphty-monorepo`. NEVER run `git add`, `git commit` or `git push` yourself: the owner commits through the landing script this plan writes (Task M1-T2), exactly as `tools/commit-graph-format-landing.sh` was run for graph-format.

**Goal:** Move `@graphty/webgpu-graph-algorithms` and its entire design corpus out of the staging repository `graphty-org/webgpu-graph-algorithms` into the pnpm/Nx monorepo `graphty-org/graphty-monorepo` with its git history, make it a first-class workspace member (build, lint, knip, coverage, pre-push, CI shards, release), port the three GitHub Actions lanes (software adapters, the label-gated NVIDIA T4 lane, the informational macOS/Windows host matrix), and then integrate the GPU layout into `@graphty/layout` first and the other packages after, so every later WebGPU phase (P4+, P7+) continues inside the monorepo.

**Architecture:** The package lands at `graphty-monorepo/webgpu-graph-algorithms/` (a sibling of `graph-format/`, exactly as design section 3.1 and graph-format design 14.5 fix it) through a `git filter-repo` rewrite of a fresh clone merged with `--allow-unrelated-histories` -- the procedure the monorepo used to assemble itself (`design/monorepo/nx-monorepo-implementation-plan.md:1473-1565`). The design documents land at `design/webgpu/` (design decision Q-18). The move happens NOW, before the design's own W1 slot (design 13 row P10, which waited for A2/L1/E1); the parts of P10 that need those phases (deleting the D27 structural mirrors, the second oracle, the seedPositions cross-test, the design amendments) are split off into a later step (Phase M5b) so nothing lands half-done. The layout seam is design section 9.3: `layout/src/simulation/` with `LayoutSimulation`, `LayoutAccelerator`, `createSimulation` and steppable CPU simulations; the GPU package then switches its mirrors to `import type` from the real package.

**Tech Stack:** pnpm 10.0.0 workspaces, Nx 22 (`nx:run-commands` targets, `nx release` with conventional commits, independent projects, `{projectName}@{version}` tags), vitest 3.2.7 (node / node-limits / browser projects; Playwright Chromium), Dawn for Node (`webgpu@0.4.0`), Mesa lavapipe on `ubuntu-latest`, GitHub Actions (`ci.yml` shard matrix, `release.yml` with npm OIDC trusted publishing, `coverage.yml`), `git filter-repo` a40bce548d2c, TypeScript 5.9.

**Spec:** `design/webgpu-acceleration-plan.md` (this repository; sections 2.5, 3.1, 9, 12, 13, 14, Q-18, Q-28, D27) and `/home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md` (sections 13.3, 13.5, 14.3, 14.5, 14.6). The plan argues from both; executors read both. After Phase M1 the WebGPU spec lives at `graphty-monorepo/design/webgpu/webgpu-acceleration-plan.md`.

**Plan of record for the earlier phases:** `docs/superpowers/plans/2026-09-14-webgpu-p0-p3-interfaces.md` (the contract) and `docs/superpowers/plans/2026-09-15-webgpu-p{0,1,2,3}.md` (the executed phase plans); they move with the design (Phase M1) and are cited here by their new paths `design/webgpu/plans/<basename>`.

## Global Constraints

- Never run `git add`, `git commit` or `git push` (global rule). Every commit in this plan is made by the owner running `tools/land-webgpu-graph-algorithms.sh <step>` (Phase M1-M3) or `tools/commit-changes.sh` (later phases); a task's "Commit" step means "leave the working tree in the described state and tell the owner which script step to run".
- Never add a `Co-Authored-By` or `Claude-Session` trailer to any commit message, script or file. The landing script refuses them (as `tmp/ci-push.sh` does today).
- Plain ASCII in every file this plan creates or edits; `--` for dashes, straight quotes. The two files whose EXISTING lines carry non-ASCII (the monorepo root `CLAUDE.md` tree and build-order line) are edited by a Python script that spells those characters as `chr()` code points (Task M2-T7), never by pasting them.
- Never run `sudo`; nothing in this plan needs it on the dev box (the CI runners have passwordless sudo for `apt-get`).
- Ports 9000-9099 only: the package's coverage preview is 9058, its demo dev server gets 9030 (both free in the monorepo today: `git grep -n 9058` hits only `design/ui/UAT.md`, `git grep 9030` hits nothing).
- Project rule: "Never create fallbacks if WebGPU isn't supported" -- carried into the monorepo root `CLAUDE.md` (Task M2-T7) because the package `CLAUDE.md` cites it as a repository rule.
- No `eslint-disable`, `@ts-expect-error` (outside negative type tests) or `@ts-ignore`; never lower a coverage threshold (the package's are 80/80/75/80 on `--project=node`).
- The design is the specification. Where this plan departs from it, the departure is listed in section 0.4 with its reason.
- Do not edit the owner's in-progress files in the monorepo main worktree (`graph-format/vitest.config.ts`, `graph-io/vitest.config.ts`, `graph-format/test/setup/`, `graph-io/test/setup/`, `remote-logger/src/server/dual-server.ts`, `remote-logger/test/server/*.test.ts`, `graphty-element/src/algorithms/*Algorithm.ts`); every task of Phases M1-M3 runs in a FRESH worktree of `master` (Task M1-T1) so those edits are never swept into a landing commit.
- Version pins that must survive the move: `webgpu` devDependency `0.4.0` exact (0.6.x needs glibc 2.38; the dev container is Ubuntu 22.04 / glibc 2.35; the bump is the design's P-ENV, scheduled after this plan), `@webgpu/types ^0.1.72`, `vitest`/`@vitest/browser ^3.2.4` (resolved 3.2.7 by the monorepo lockfile), `playwright ^1.54.1` (resolves to the monorepo's 1.57.0, not staging's 1.63.0 -- see D-11).
- The GPU lane's identifiers are load-bearing and must not be renamed: workflow display name `GPU`, file name `gpu.yml`, runner label `gpu-linux-t4`, runner group `gpu`, env `GRAPHTY_RUNNER_CLASS=gpu-linux-t4`, baseline file `benchmarks/results/gpu-linux-t4.json`, the `gpu` label. (`gh run list --workflow GPU` and `workflow_id: "gpu.yml"` inside the workflow key on the first two.)

---

## 0. Read this first

### 0.1 Where both repositories stand (2026-09-16)

| Fact | Evidence |
| --- | --- |
| Source: `graphty-org/webgpu-graph-algorithms`, branch `master` = `origin/master` = `a66c889`, 38 commits, no tags, one author, 21 GPG-signed + 17 unsigned commits, every message ASCII with no trailers. | `git log --format='%h %G? %s'` |
| The package is `packages/webgpu-graph-algorithms/` (435 tracked files). The design corpus outside it: `design/webgpu-acceleration-plan.md` (449 KB, 4758 lines) and `docs/superpowers/plans/` (5 files, 4.2 MB; p1 and p3 exceed 1 MB). Inside it: `docs/HEADLESS_GPU_REPORT.md`, `docs/decisions/G0-G3.md`, `docs/research/**` (notes, drafts, review, 4 committed `.log` probes). | `git ls-files`; docs inventory |
| Two files are UNTRACKED in the source: `packages/webgpu-graph-algorithms/docs/decisions/G0.md` (75 `[[fill: ...]]` occurrences) and `G2.md` (22 `{{...}}` occurrences); both are open gate records waiting on the never-provisioned GPU runner; the package `CLAUDE.md` cites them. | `git status --porcelain` |
| P0-P3 are done (gate G3 GREEN on the dev box 2026-09-16); CI (`ci.yml`, lavapipe + SwiftShader) and Hosts (`hosts.yml`, Metal/WebKit + WARP) are green on `a66c889`; `gpu.yml` has push and schedule commented out because the `gpu-linux-t4` runner does not exist (graphty-org is on the GitHub Free plan; hosted GPU runners need Team). | run 35142649256 (CI), 35142674793 (Hosts); `gh api orgs/graphty-org` plan `free` |
| Monorepo: `graphty-org/graphty-monorepo`, `origin/master` = `dc08826b` (the night of 2026-09-16/17: `6fc56c1b`, nine repair and feature commits ending at `6b4777df`, then the release commit `dc08826b chore(release): publish [skip ci]`; the owner's local `master` was still at `6b4777df` when execution started, so M1-T1 fast-forwards it first), 1046 commits, 5 root commits, 85 merges (four of them history imports of algorithms / layout / graphty-element / graphty via `git filter-repo --to-subdirectory-filter` + `merge --allow-unrelated-histories`), 818 imported unsigned commits, no branch protection, `.git` 71 MB. | `git rev-list --max-parents=0 HEAD`; `design/monorepo/nx-monorepo-implementation-plan.md:1473-1565` |
| graph-format 0.1.0 and graph-io 0.1.0 landed 2026-09-16 as ONE squashed commit each (`5000b631`, `0d856404`), the root wiring in `0069386c` (byte-identical to the rehearsed `packages/move/root-touch-points.diff`), the records in `b7ed9b16` (`design/graph-format/STATUS.md`, `CONFORMANCE.md` with a prepended LANDED note). No tag exists for either yet. | `git show --stat` |
| Monorepo master CI was RED on `6fc56c1b` (run 35136935430: `Test (graph-format)`, `Test (graph-io)`, `Test (remote-logger)`, Chromatic graphty and graphty-element). The owner landed the repair on the evening of 2026-09-16 as four commits (`b3e64b26 fix(workspace): stop a blocked test worker from failing a green run`, `3adc73fc fix(remote-logger): bind test servers to a free port`, `58bbf728`, `71a35012`) and five more by the time execution started (`b1bf8a00`, `efdf50ca`, `914e8cdc`, `169a1edf`, `6b4777df fix(graph-io): admit the format's new minor into the peer range`); the main worktree is clean at `6b4777df`; CI run 35181978899 on `169a1edf` was green and run 35188079599 on `6b4777df` was in progress when execution started. `6b4777df` RESOLVES the D-18 release blocker the owner's way (graph-io's peer becomes `^0.2.0`), CI run 35188079599 on it was green, and Release run 35189211639 then PUBLISHED graph-format@0.2.0, graph-io@0.2.0, compact-mantine@0.8.0, graphty-element@1.10.0, remote-logger@1.3.1 and graphty@0.7.0 (`npm view` confirms 0.2.0 for both format packages; the release commit is `dc08826b`). That run still shows RED: release.yml's explicit `pnpm exec nx release publish` safety-net step re-publishes every project after `nx release` already did and gets `409 Conflict - Cannot publish over previously staged version` for three of them -- an owner workflow item outside this plan; every Release run that publishes something will end red until it is changed, so Task M3-T7 checks npm, not the run's colour. | `git log 6fc56c1b..master`; `git status --short`; `gh run list`; `git show 6b4777df` |
| Every monorepo CI job is `ubuntu-latest`; no self-hosted runner, no `gpu` label, no `sudo apt-get` step, no `.gitattributes`, `*.log` ignored by `.gitignore`, `benchmarks/out/`, `browser-results.json`, `gpu-report.json` NOT ignored. | `.github/workflows/*.yml`, `.gitignore` |
| The consumer migration of the graph-format design has not started: no monorepo package imports `@graphty/graph-format` (A1 not begun, F2 = 1.0.0 not cut, `layout/src/simulation/` and `algorithms/src/indexed/` do not exist). `@graphty/layout` is 1.6.2, synchronous, positional-argument, zero-dependency; `@graphty/algorithms` 1.7.2; graphty-element 1.9.4. | `grep -rn graph-format */package.json` |
| The monorepo's shared root files are byte-identical to the staging copies (`tsconfig.base.json`, `eslint.config.js`, `vitest.shared.config.ts`, `vite.shared.config.ts`, `.prettierrc`, `.prettierignore`, `.npmrc`), the lockfile already resolves `webgpu@0.4.0`, `@webgpu/types@0.1.72`, `vitest@3.2.7`, `@vitest/browser@3.2.7`, `vite@7.3.6`; the package's `project.json` already uses monorepo paths (`cwd: webgpu-graph-algorithms`). | `diff -u`; `pnpm-lock.yaml` |
| `git filter-repo` rehearsal (scratch clone, 2026-09-16): the four path rules of Task M1-T3 produce 30 commits (8 dropped as empty: the 2025 scaffold, the graph-format/graph-io staging commit, the `.claudehistory` ignore, five CI-only commits), a tree of exactly `design/webgpu/**` and `webgpu-graph-algorithms/**`, 3.10 MiB packed, `git log --follow` reaching the original commits, zero non-ASCII, zero trailers. The monorepo HEAD has no `webgpu-graph-algorithms/` or `design/webgpu/` path, so the unrelated-history merge has no add/add conflict. | scratch clone under the session scratchpad; not committed anywhere |

### 0.2 Decisions (defaults stand unless the owner says otherwise before Phase M1 starts)

| Id | Decision | Default and reason | Alternative |
| --- | --- | --- | --- |
| D-1 | History strategy | Carry the history: `git filter-repo` on a fresh clone (four `--path` keeps, four `--path-rename` rules), fetched into a landing branch and merged with `--allow-unrelated-histories --no-ff`. Meets the owner's criterion ("not damaging, no risk to the monorepo"): it is the monorepo's own documented and four-times-used procedure; it changes no existing commit, tag or file; it adds one root and one merge to a history that has 5 and 85; the cost is 3.1 MiB (a plain copy would add 2.9 MiB); `nx release` is unaffected (0.2.0 either way, see D-4); commitlint is not retroactive. Costs stated plainly: filter-repo re-creates every commit, so the 21 signed source commits arrive UNSIGNED (the monorepo already carries 818 unsigned imported commits); the merge and the five wiring commits are signed by the owner. Rollback before the push: `git worktree remove --force .worktrees/land-webgpu-graph-algorithms && git branch -D land/webgpu-graph-algorithms` (owner). After `land` the import is permanent: `git revert -m 1 <merge>` plus reverting the wiring commits removes the tree but keeps the 31 commits in history, so the PR review (Task M3-T6) is the last checkpoint. | Plain move as one `feat(webgpu-graph-algorithms):` commit (the graph-format precedent). Choose it only if the owner rejects a sixth root or the unsigned imported commits; every later task is identical. |
| D-2 | What travels in the rewrite | `packages/webgpu-graph-algorithms/` -> `webgpu-graph-algorithms/`; `design/webgpu-acceleration-plan.md` -> `design/webgpu/webgpu-acceleration-plan.md`; `docs/superpowers/plans/` -> `design/webgpu/plans/`; the historical root `HEADLESS_GPU_REPORT.md` -> `webgpu-graph-algorithms/docs/HEADLESS_GPU_REPORT.md` (so its creation commit is kept). NOT carried: `.github/` (`ci.yml` collides; all three workflows are ported by hand in Phase M3), `.gitattributes` (a monorepo-wide decision, D-12), `packages/{README,STATUS,CONFORMANCE,MIGRATION_PROMPT}.md`, `packages/move/`, the staging root configs, the root `README.md`/`CLAUDE.md`. | -- |
| D-3 | Merge commit message | `feat: merge the webgpu-graph-algorithms package history into the monorepo` (unscoped, like the four precedent merges `feat: merge <pkg> history into monorepo`); passes commitlint (type-enum `feat`, no scope so scope-enum does not apply, 73 chars); contributes no files to `nx release` (a merge commit has no `--name-status` entries). The commit that adds the scope `webgpu-graph-algorithms` to `commitlint.config.js` comes AFTER the merge (Task M2-T1), which is why the merge message must not use that scope. | -- |
| D-4 | First published version | The imported history holds 10 `feat(webgpu-graph-algorithms)` and 4 `fix` commits, so the first `nx release` after landing takes the package from the 0.1.0 on disk to **0.2.0** (identical to what a single `feat` landing commit would do, and to what graph-format/graph-io get: compact-mantine landed at 0.1.0 and its first tag was 0.2.0). The design's "first release 0.1.0" (P10) is therefore 0.2.0; accepted. | Set `version` to `0.0.1` in the package fix-up commit if the owner wants `0.1.0` to be the first published version. |
| D-5 | graph-format 13.5 rule 5 | The package (0.1.0) depends on `@graphty/graph-format` `workspace:^` (peer `^0.2.0` after Task M1-T5, the minor graph-io states since `6b4777df`; see D-18) exactly as graph-io (0.1.0) already does on master; the rule's stated purpose is that 1.x consumers never pin a 0.x format, and the graph-io landing established that a 0.x package may. Bump the peer to `^1.0.0` in the F2 PR. The 1.x consumers (layout, algorithms, graphty-element) still wait for F2 (Phase M5 entry criterion). | Land the package commits as `chore` (no publish) until F2 -- not needed given the precedent. |
| D-6 | Workspace protocol | Keep the package's `workspace:^` (design Q-31: publishes a caret range); do not touch graph-io's `workspace:*` in this plan (note it for the F2 PR: an exact pin beside a `^1` peer gives an app two format copies). | -- |
| D-7 | Design document location | `design/webgpu/webgpu-acceleration-plan.md` (basename kept: 27 files cite it) and `design/webgpu/plans/<same basenames>`, per the design's own owner decision Q-18; package-internal docs stay under `webgpu-graph-algorithms/docs/`. Records move VERBATIM with a LANDED note APPENDED after the Review log explaining the path forms (the graph-format precedent, `design/graph-format/STATUS.md:1-11`, prepended; the WebGPU records are cited by line number, so here the note goes last); only the LIVE documents (package `README.md`, `CLAUDE.md`) get their citations re-pointed. Q-18's "on acceptance" stripping of Review notes into `docs/research/review-log.md` is NOT done (the contract, plans, gate records and verdicts cite the spec by line number). A `design/webgpu/README.md` index is added because GitHub does not render the two >1 MB phase plans. | `design/webgpu-graph-algorithms/` (mirrors `design/graph-format/`) -- rejected because Q-18 is an accepted owner decision. |
| D-8 | Open gate records G0/G2 | Committed in the SOURCE repository first (Task M0-T1) so the rewrite carries them with their own commit; each gets a two-line "OPEN -- the GPU runner is not provisioned" header note. Phase M4 closes them. | Add them in the monorepo package fix-up commit. |
| D-9 | What of P10 (W1) lands now | The physical move, the root touch points, the two software shards, `gpu.yml`, `hosts.yml`, the trusted publisher, the first release. NOT now (Phase M5b, gated on L1 and A2): deleting the D27 mirrors; CREATING `test/types/conformance.test-d.ts` (it does not exist today; design 9.8's W1 row says it is "retired" while G10 requires its `expectTypeOf` cross-compile -- the plan follows G10 and `src/types/accelerator.ts`'s own header: the file is created with the real imports and stays); `indexed.*` as a second oracle; the `seedPositions` cross-test; the design 10.3/14.5/14.6/16.2/16.7 amendments (Task M5b-T4). The mirrors, `CpuAlgorithmOptions` and `test/types/accelerator.test-d.ts` survive the move unchanged. The README performance table stays as regenerated from the dev-box baseline (the only baseline until M4); Task M4-T3 regenerates it with the T4 row. | -- |
| D-10 | CI shard shape | Two shards in `ci.yml`'s matrix: `webgpu-graph-algorithms-node` (lavapipe, `--coverage`, thresholds active, uploads `coverage-webgpu-graph-algorithms-node`) and `webgpu-graph-algorithms-browser` (SwiftShader smoke through `scripts/run-browser-project.js`, no coverage). The node shard calls vitest DIRECTLY (like the algorithms shards) rather than through `nx run <pkg>:coverage`: the nx -> npm -> vitest pipe chain starved the worker RPC for graph-format/graph-io (`6fc56c1b`). A new matrix key `needs-lavapipe` gates two steps (apt install; ICD discovery exporting `VK_DRIVER_FILES` and the three lane variables to `$GITHUB_ENV`). The `find`-based ICD step replaces the design's hard-coded `lvp_icd.x86_64.json` (ubuntu-24.04 ships `lvp_icd.json`). | -- |
| D-11 | Playwright version | Accept the monorepo's `playwright@1.57.0` (root devDependency `^1.54.1`); the package's flag facts were last re-measured on 1.63.0. The browser shard's first run is the check; `node scripts/probe-browser-flags.mjs` is the diagnostic if it regresses. Bumping the root devDependency is a separate `build(deps)` change. | -- |
| D-12 | Line endings | Add a root `.gitattributes` (`* text=auto eol=lf`, `-text` for the two CRLF corpora `graph-io/test/corpus/**` and `graphty-element/test/helpers/corpus/**`, `*.gsnp` / `*.png` / `*.zip` binary) in its own commit with a read-only `git ls-files --eol` check proving that no tracked file outside the two corpora is CRLF (so the attribute renormalises nothing). Needed by the Windows host lane (LF checkouts) and by `test/fixtures/rich-v1.gsnp`. | Skip the file and drop the Windows host lane. |
| D-13 | Pre-push hook | Add `(cd webgpu-graph-algorithms && GRAPHTY_GPU_REQUIRE=any npm run test:run)` after graph-io's line, as design 12.5 prescribes (`test:node` only, "matching CI": `any` makes a missing adapter FAIL up front in `test/setup/global.ts` instead of skipping every GPU test silently). On the dev box this runs on NVIDIA when `LD_LIBRARY_PATH` carries the libEGL tree, else on lavapipe (~5 min); the browser project runs only in CI. | Leave the package out of the hook (no precedent: "no package is skipped"). |
| D-14 | Dev box libEGL tree | Re-extract it under `graphty-monorepo/tmp/egl/` (gitignored) per `HEADLESS_GPU_REPORT.md` appendix D and re-point the three absolute paths in the package `CLAUDE.md`; recommend adding `libegl1` to the container image so the workaround disappears. | -- |
| D-15 | GPU runner | GitHub Team plan + hosted `gpu-linux-t4` in runner group `gpu` (design Q-3, decided 2026-09-14). `gpu.yml` lands with push/schedule commented out and is never dispatched before the runner exists (a dispatch would queue a phantom job for 24 h). Phase M4 is the owner's provisioning gate. | machine.dev `runs-on: machine/gpu=t4` (only `runs-on` and the provisioning steps differ). |
| D-16 | Layout seam scope | Phase M5 implements design 9.3 (the simulation barrel, the CPU steppable FA2 and FR, `createSimulation`, `resolveNodeVector` / `resolveWeights` / `seedPositions`) and the minimal `toLayoutSnapshot`; it does NOT port the other 13 layouts to `indexed.*` (that is graph-format design 14.3's own L1 work and can proceed in parallel). The CPU `ForceAtlas2Simulation` is a PORT (copy) of the GPU package's `test/oracle/forceatlas2.ts` formulas, not a move: the GPU package keeps its independent reference (design 9.8 W1 row: "the independent references stay"; two transcriptions of table 7.2 are the R-1 mitigation). Design 9.3's one-shot `indexed.forceAtlas2(s, options): LayoutResult`, `toPositionMap` and the `LayoutResult` type stay with graph-format 14.3's L1 proper; here the legacy `forceatlas2Layout` calls `ForceAtlas2Simulation` directly and keeps its `rescaleLayout` output (Task M5-T8). | Move the oracle (design P3 row's "moved -- not copied -- if L1 wants it"). |
| D-17 | Legacy `forceatlas2Layout` | Rewritten over the new simulation in Phase M5's last task (design 9.3: one code path), with the Chromatic re-baseline commit; separable if the owner wants the seam without the behaviour change. | Keep the dense legacy implementation beside the simulation. |
| D-18 | The graph-format peer range during 0.x | DECIDED BY THE OWNER on 2026-09-16 (`6b4777df`, before Phase M1 started): the minor pin `^0.2.0` -- "a 0.x minor is a breaking change, so graph-io states compatibility with exactly the minor it was built against"; every future format minor stops `nx release` again and asks for the same one-line edit, which the owner accepts as the designed prompt. The GPU package mirrors it: its peer becomes `^0.2.0` in the `package` commit (Task M1-T5); its `dependencies` entry stays `workspace:^`; the build-output test that pins the caret form stays as it is; the F2 PR turns both into `^1.0.0`. The plan's original default (`>=0.1.0 <1.0.0` in both packages, a `peerfix` commit on graph-io) is withdrawn; the landing script has no `peerfix` step. Background, verified by an `nx release --dry-run` in a scratch clone of master: `nx release` keeps a dependent's range only while the new version satisfies it (`preserveMatchingDependencyRanges` defaults to every dependency type) and otherwise ABORTS -- graph-format 0.1.0 -> 0.2.0 did not satisfy `^0.1.0`, so until `6b4777df` master could not release ANY package. | `>=0.1.0 <1.0.0` (the whole 0.x major, no re-statement per minor) -- the plan's original default; rejected by the owner's commit. `release.version.preserveMatchingDependencyRanges: false` in `nx.json` -- rejected: nx would then also narrow the GPU package's optional `^1.0.0` peers on algorithms/layout to the released patch on every release. |
| D-19 | nx graph edges from the optional peers | `implicitDependencies: ["!algorithms", "!layout"]` in `webgpu-graph-algorithms/project.json` until Phase M5b. Reason (verified in a scratch project graph): nx builds project-graph edges from the OPTIONAL peer ranges `^1.0.0` because they match the workspace versions 1.7.2 / 1.6.2, so without the negation `nx run webgpu-graph-algorithms:build` also builds algorithms and layout (on the paid T4 lane too), `updateDependents: auto` patch-bumps and publishes the package on their every release, and `nx affected` marks it affected by their every change. nx removes an edge whose implicit dependency starts with `!` (`nx/dist/src/project-graph/utils/implicit-project-dependencies.js`). Phase M5b deletes the negation when the real `workspace:^` devDependency on layout arrives (design Q-29 accepts the coupling then). | Accept the coupling from day one. |
| D-20 | The strict-consumer compile inside `lint` | The package's `lint` script runs `tsc -p tsconfig.strict-consumer.json` against `dist/*.d.ts` shims that only `build:bundle` writes, and the pre-push hook builds with `pnpm -r run build` (tsc only) before `pnpm -r run lint` -- reproduced: `TS2307 Cannot find module '@graphty/webgpu-graph-algorithms'`. Fix in two places: `project.json` `lint` gets `dependsOn: ["build"]` (the nx path; cached), and `tools/prepush.sh` runs `npm run build:bundle` in the package between its Build and Lint steps (the hook path; ~10 s). The compile stays inside `lint` because `test/build-output.test.ts` pins that. | A tsc root entry per subpath (the graph-format pattern) -- rejected: the package's bundle tests reason about the vite output, and three stubs would shadow them locally. |
| D-21 | knip 5.77.4 findings | The monorepo resolves knip 5.77.4 (root `^5.61.3`), staging had 5.88.1, and 5.77.4 reports two things on the package: `Referenced optional peerDependencies: webgpu` and `Unused exported types: ArenaPlan, PerArrayPlan, WindowedPlan` (`src/memory/upload-plan.ts`). Fix: `ignoreDependencies: ["webgpu"]` in the package's knip workspace entry (it is an optional peer AND an exact devDependency, imported inside `await import("webgpu")` in `src/node/index.ts`), and `@public` JSDoc on the three interfaces (they are members of the exported `UploadPlan` union; declaration emit needs them exported; the monorepo convention is `@public` with a clause, never `ignoreExportsUsedInFile`). | Bump root knip to `^5.88` -- rejected: 5.88 stops reporting exports used only in their own file, which the monorepo relies on. |

### 0.3 Phase map

| Phase | Where | Entry criteria | Deliverable | Gate | Size |
| --- | --- | --- | --- | --- | --- |
| M0 Preconditions | source repo + monorepo (owner) | -- | G0/G2 and this plan committed and pushed in the source; the monorepo's pending CI repair landed or consciously deferred; the decisions of 0.2 confirmed | `git status` clean in the source at the tip that will be rewritten; monorepo master CI state known | 0.5 d + owner |
| M1 Move with history | monorepo, fresh worktree | M0 | the rewritten history merged into `land/webgpu-graph-algorithms`; package path fix-ups; docs LANDED notes and index; `.gitignore` / `.gitattributes` | the tree builds with `pnpm install`; every file of the package present with `git log --follow` history; `git status` clean after the landing script's steps | 1-2 d |
| M2 Build-system wiring | monorepo worktree | M1 | root touch points (commitlint, knip, coverage preview, prepush, merge-coverage, root `CLAUDE.md` / `README.md`; the workspace entry and the lockfile are already in from M1-T4) | `nx run-many -t lint,build,test --projects=webgpu-graph-algorithms` green locally; `pnpm exec knip` no new findings; `tools/merge-coverage.sh` merges the package; `pnpm install --frozen-lockfile` passes | 1 d |
| M3 CI/CD port and landing | monorepo worktree, then GitHub | M2 | `ci.yml` shards + artifacts, `release.yml` download, `gpu.yml`, `hosts.yml`; the PR green; the branch on master; trusted publisher; first release `0.2.0`; staging repository archived | `all-checks` green on the PR with both new shards; `coverage.yml` merges the package on master; `release.yml` publishes `@graphty/webgpu-graph-algorithms@0.2.0` with provenance; Hosts green on master | 1-2 d + CI time |
| M4 GPU runner | GitHub org (owner) + monorepo | M3; Team plan | the `gpu-linux-t4` runner, group, label, spending limit; `gpu.yml` triggers restored; first green GPU run; T4 baseline committed; G0/G2 closed | `test-gpu` green on a labelled PR and on master; `benchmarks/results/gpu-linux-t4.json` committed; nightly guard verified | 0.5 d + owner |
| M5 Layout seam (L1-sim) | monorepo, `layout/` | F2 (`@graphty/graph-format >= 1.0.0` on master, which by graph-format design 14.6 follows the A1 merge) and Phase M3 | `layout/src/simulation/` per design 9.3; layout depends on graph-format; legacy FA2 over the simulation; Chromatic re-baseline | layout tests green incl. the fake-accelerator dispatch tests and the LCG cross-test; graphty-element tests green unchanged; `chromatic-layout` re-baselined | 5-8 d |
| M5b GPU package adopts the real layout types (W1b) | monorepo, `webgpu-graph-algorithms/` | M5 | layout mirrors -> `import type`; `conformance.test-d.ts` (layout half); layout's `ForceAtlas2Simulation` as the second FA2 oracle; the seedPositions cross-test | both CI shards green; the FA2 parity suite passes against BOTH oracles within the 11.4 tolerances | 1-2 d |
| M6 Element (E0 + E1) | `graphty-element/` | M5; A2 first commit (Phase M8a) | the graph-format 14.4 `DataManager` refactor (E0, unsized by the design), then design 9.4 items 1-10 | design G6 | own plan |
| M7 App (W2) | `graphty/` | M6 | design 9.5 `attachAccelerator`, the on/off indicator, real-GPU stories under a `gpu` tag | design G12 (W2 subset) | own plan |
| M8 Algorithms (A2 first commit, then P7) | `algorithms/`, GPU package | F2 for A2; M3 for P7 | design 9.2 `indexed/accelerator.ts` + `accelerated()`; then the GPU SpMV/WCC phase P7 inside the monorepo | design G7 | own plans |

Critical path: M0 -> M1 -> M2 -> M3 (the package is in the monorepo, releasing) ; M4 in parallel with anything after M3 ; M5 waits for F2 (owner-side graph-format work: A1 branch, then the 1.0.0 cut) ; M5b right after M5 ; M6-M8 each get their own writing-plans pass when their entry criteria are met, from the design sections named in section 6.

### 0.4 Departures from the design (all of them)

| Id | Departure | Reason |
| --- | --- | --- |
| DEP-A | The move (P10 / W1) happens after P3 instead of after A2/L1/E1 and P4/P5/P7. | Owner request (this plan). Split into "move now" (M1-M3) and "W1-proper" (M5b) so the design's W1 deliverables that need L1/A2 are not dropped. |
| DEP-B | `ci.yml`'s node shard runs vitest directly and discovers the lavapipe ICD with `find`; the design's 12.5 diff used `nx run` and a literal `VK_DRIVER_FILES`. The node shard uploads its coverage from its own step (`if-no-files-found: error`) instead of joining the shared step's `if:`. knip runs in `tools/prepush.sh` and Task M2-T8, not in `ci.yml` (design 12.1 lists knip in the default lane; the monorepo has no knip job). | `6fc56c1b` (reporter starvation through the nx pipe chain), the Mesa 25.2 path change (`lvp_icd.json`), the shared step's `if-no-files-found: ignore`, the monorepo's CI shape. |
| DEP-C | The T4 baseline path is `benchmarks/results/gpu-linux-t4.json` (what `bench-compare.js`, `gpu.yml` and `CLAUDE.md` use), not the design 12.1/12.4 `benchmarks/baselines/`. | The code is the record; the design text is amended in Phase M4. |
| DEP-D | `hosts.yml` is ported (design 12.5 predates it). | It is green and catches Metal/WebKit/WARP defects the default lane cannot. Filtered by `paths:` so other packages' changes do not spend two ~9-minute macOS/Windows jobs. |
| DEP-E | The CPU `ForceAtlas2Simulation` is a copy of the oracle's formulas, not a move of `test/oracle/forceatlas2.ts`. | D-16. |
| DEP-F | First release is 0.2.0, not 0.1.0. | D-4; `nx release` mechanics. |
| DEP-G | The graph-format peer range is the minor pin `^0.2.0` in the GPU package and in graph-io (design 2.5 / 3.1 say `^0.1.0`; graph-format 13.5 rule 3 says `^<major>`), re-stated at every format minor until F2. | D-18 as decided by the owner (`6b4777df`): `nx release` aborts otherwise; `^1.0.0` at F2; the rule-3 amendment is Task M5-T9. |
| DEP-H | The package merges to master and publishes (0.2.0, depending on graph-format `^0.2.0`) while the format is 0.x, which graph-format 13.5 rule 5 forbids by its letter; the design's own W1 sat after F2 (9.8: "peer range `^1.0.0` from F2"). | D-5: a consequence of DEP-A; the package is itself 0.x, and graph-io set the precedent on 2026-09-16. |
| DEP-I | Design Q-31's `workspace:*` -> `workspace:^` correction to graph-io and to rule 3 ("at W1") is not made here. | D-6: done in the F2 PR, where the peer ranges change anyway; Task M5-T9 records `workspace:^` as the intended protocol in the rule-3 amendment. |
| DEP-J | Q-18's post-acceptance move of the Review notes, `(graft: ...)` annotations and the Review log into `docs/research/review-log.md` is not performed. | D-7: the contract, phase plans, gate records and verdicts cite the design by line number; deferred until those records are closed. |

---
## Phase M0: Preconditions

### Task M0-T1: Commit the open gate records and this plan in the source repository

**Repository:** `/home/apowers/Projects/webgpu-graph-algorithms` (the staging repository).

**Files:**
- Modify: `packages/webgpu-graph-algorithms/docs/decisions/G0.md:1-2` (prepend a status note)
- Modify: `packages/webgpu-graph-algorithms/docs/decisions/G2.md:1-2` (prepend a status note)
- Add (already written): `docs/superpowers/plans/2026-09-16-graphty-monorepo-integration.md` (this file)
- Create: `tmp/ci-round11.sh` (the owner's commit script, using the existing `tmp/ci-push.sh`)

**Interfaces:**
- Consumes: `tmp/ci-push.sh <files...> <<'MSG'` (commits unsigned with `-c commit.gpgsign=false`, pushes `HEAD:master`, refuses trailers and non-ASCII messages).
- Produces: the source tip `T0` that Task M1-T3 rewrites; `git status --porcelain` empty at `T0` except ignored paths.

- [ ] **Step 1: Prepend the OPEN note to G0.md**

Insert as lines 1-2 of `packages/webgpu-graph-algorithms/docs/decisions/G0.md` (the note line and a blank line; the current line 1, the H1, becomes line 3):

```markdown
OPEN RECORD (2026-09-16): the org / runner / image rows below are unfilled because the hosted `gpu-linux-t4` runner was never provisioned (design 12.4 is an owner item). Committed as-is so the record travels with the package into graphty-monorepo; Phase M4 of `design/webgpu/plans/2026-09-16-graphty-monorepo-integration.md` fills and closes it there.

```

- [ ] **Step 2: Prepend the same note to G2.md**

Insert as lines 1-2 of `packages/webgpu-graph-algorithms/docs/decisions/G2.md` (the note line and a blank line; the H1 becomes line 3):

```markdown
OPEN RECORD (2026-09-16): the `{{...}}` slots below wait on the first run of the hosted `gpu-linux-t4` lane, which was never provisioned. Committed as-is so the record travels with the package into graphty-monorepo; Phase M4 of `design/webgpu/plans/2026-09-16-graphty-monorepo-integration.md` fills and closes it there.

```

- [ ] **Step 3: Check both files stay ASCII**

Run: `cd /home/apowers/Projects/webgpu-graph-algorithms && LC_ALL=C grep -nP '[^\x00-\x7F]' packages/webgpu-graph-algorithms/docs/decisions/G0.md packages/webgpu-graph-algorithms/docs/decisions/G2.md docs/superpowers/plans/2026-09-16-graphty-monorepo-integration.md | head`
Expected: no output.

- [ ] **Step 4: Write the owner's commit script**

Create `tmp/ci-round11.sh`:

```bash
#!/usr/bin/env bash
# Round 11: commit the open gate records and the monorepo integration plan (the last source-repo commit before
# the history rewrite of Phase M1).   bash /home/apowers/Projects/webgpu-graph-algorithms/tmp/ci-round11.sh
set -euo pipefail
cd /home/apowers/Projects/webgpu-graph-algorithms
PKG=packages/webgpu-graph-algorithms

bash tmp/ci-push.sh "$PKG/docs/decisions/G0.md" "$PKG/docs/decisions/G2.md" \
    docs/superpowers/plans/2026-09-16-graphty-monorepo-integration.md <<'MSG'
docs: record the open G0 and G2 gate records and the monorepo integration plan

G0.md and G2.md were left untracked because their commit scripts refuse
unfilled slots, and the slots wait on the hosted gpu-linux-t4 runner that
was never provisioned. They are gate evidence the package CLAUDE.md cites,
so they travel with the package: committed as OPEN records with a header
note, to be closed in the monorepo (Phase M4 of the plan below).

The plan moves the package and the whole design corpus into
graphty-monorepo with this repository's history (git filter-repo on a
fresh clone, merged with --allow-unrelated-histories, the monorepo's own
procedure), wires the build, coverage, hooks and release, ports the three
workflows, and then integrates the GPU layout through design section 9.3.
MSG
```

- [ ] **Step 5: Hand the command to the owner**

Tell the owner: `! bash /home/apowers/Projects/webgpu-graph-algorithms/tmp/ci-round11.sh`
Expected output: one commit pushed to `master`; the CI and Hosts workflows run on it. Record the new tip: `git rev-parse HEAD` is `T0`.

- [ ] **Step 6: Wait for CI and Hosts on T0**

Run: `bash /home/apowers/Projects/webgpu-graph-algorithms/tmp/watch-workflows.sh <T0> CI Hosts`
Expected: both green (a docs-only commit changes no test).

### Task M0-T2: Land or park the monorepo's pending CI repair (owner)

**Repository:** `/home/apowers/Projects/graphty-monorepo`, main worktree.

- [ ] **Step 1: Read the state**

Run: `cd /home/apowers/Projects/graphty-monorepo && git status --short && gh run list --workflow CI --limit 12 --json databaseId,headBranch,conclusion,status --jq '.[] | select(.headBranch=="master") | [.databaseId,.status,.conclusion] | @tsv' | head -3`
Expected: an empty status and the newest CI run on master `completed success` (on 2026-09-16 master was `6b4777df`; run 35181978899 on `169a1edf` was green and run 35188079599 on `6b4777df` in progress). If that run is red, the owner repairs first.

- [ ] **Step 2: Owner decision**

The plan assumes a GREEN master before the landing branch is cut: Phase M3's gate compares the PR against master, and `release.yml` only publishes after a green CI. If master is still red for a reason the owner chooses not to fix now, the landing PR's `all-checks` shows the SAME failures, Task M3-T6's gate becomes "no NEW failing job", and the first release waits until master is green.

- [ ] **Step 3: The release blocker that exists today (D-18)**

A dry run of `nx release` on master aborted: `"preserveMatchingDependencyRanges" is enabled for "peerDependencies" and the new version "0.2.0" is outside the current range for "@graphty/graph-format" in manifest "graph-io/package.json"`. DONE BY THE OWNER as `6b4777df fix(graph-io): admit the format's new minor into the peer range` (peer `^0.2.0`, the test at `graph-io/test/build-output.test.ts:65` unchanged because it pins the caret FORM). The landing branch therefore carries no graph-io change and no `peerfix` commit; Task M2-T2 keeps only its release dry run.

- [ ] **Step 4: Record the choice**

Write the choices and the master SHA (`6b4777df` or later) into the owner's notes; Task M1-T1 branches from that SHA.

### Task M0-T3: Confirm the decisions of section 0.2 (owner)

- [ ] **Step 1:** The owner reads section 0.2 and answers any D-n they want changed. Silence means the defaults stand (the design's convention). The answers that change tasks: D-1 (plain move instead of history: Task M1-T3 and M1-T4 become "copy the HEAD tree", everything else is unchanged), D-4 (first version), D-13 (pre-push line), D-17 (legacy FA2 rewrite).

### Task M0-T4: Verify the toolchain on the dev box

**Repository:** `/home/apowers/Projects/graphty-monorepo`.

- [ ] **Step 1: Check the tools**

Run:

```bash
git --version                       # 2.34.1 (has --allow-unrelated-histories, worktree, subtree)
git filter-repo --version           # a40bce548d2c (installed at /home/apowers/.local/bin/git-filter-repo)
node --version                      # v22.x
pnpm --version                      # 10.0.0 (root package.json packageManager)
cd /home/apowers/Projects/graphty-monorepo && pnpm exec nx --version   # 22.7.x
ls /usr/share/vulkan/icd.d/lvp_icd.x86_64.json                          # the dev box's lavapipe ICD (Ubuntu 22.04, Mesa 23.2)
gh --version                        # 2.4.0 (the Ubuntu package): every gh command in this plan is written for 2.4.0 --
                                    # no `gh label`, no `gh run list --branch/--status/--event`, no `gh pr checks --watch`;
                                    # filters go through --json/--jq, which 2.4.0 has
```

Expected: every command prints a version / the file exists. If `git filter-repo` is missing: `pip install --user git-filter-repo` (no sudo).

---

## Phase M1: Move with history

### Task M1-T1: Create the landing worktree

**Repository:** `/home/apowers/Projects/graphty-monorepo` (owner runs the git command; the agent verifies).

- [ ] **Step 1: The owner creates the worktree and branch**

The owner runs (from the main worktree; the fast-forward brings the local master up to origin/master -- `dc08826b`, the release commit, when execution started -- so the landing branch starts from the released 0.2.0 tree and the PR carries no stale-base merge later):

```bash
cd /home/apowers/Projects/graphty-monorepo && git fetch origin && git merge --ff-only origin/master && git worktree add .worktrees/land-webgpu-graph-algorithms -b land/webgpu-graph-algorithms master
```

`.worktrees/` is the repository's worktree convention (`.worktree-config.json`, gitignored at `.gitignore:126`).

- [ ] **Step 2: Verify**

Run: `cd /home/apowers/Projects/graphty-monorepo/.worktrees/land-webgpu-graph-algorithms && git status --short && git log --oneline -1 && git rev-parse origin/master && git ls-tree --name-only HEAD webgpu-graph-algorithms design/webgpu`
Expected: an empty status, the master tip equal to origin/master, and NO output from the `ls-tree` (the target paths do not exist -- the precondition of the unrelated-history merge, `design/monorepo/nx-monorepo-implementation-plan.md:1483-1491`).

- [ ] **Step 3: Install in the worktree**

Run: `cd /home/apowers/Projects/graphty-monorepo/.worktrees/land-webgpu-graph-algorithms && HUSKY=0 pnpm install --frozen-lockfile`
Expected: exit 0 (the lockfile is unchanged at this point; the message "dependencies have build scripts that were ignored: webgpu" is expected and harmless).

Every later task of Phases M1-M3 runs in this worktree; `WT` below means `/home/apowers/Projects/graphty-monorepo/.worktrees/land-webgpu-graph-algorithms`.

### Task M1-T2: Write the landing script

**Repository:** `WT`.

**Files:**
- Create: `tools/land-webgpu-graph-algorithms.sh`

**Interfaces:**
- Produces: `tools/land-webgpu-graph-algorithms.sh {prepare|merge|commit <step>|push|land} [--dry-run] [--no-sign] [--skip-gate]`; steps `workspace`, `package`, `docs`, `ignore`, `ci` (committed in that order) with the subjects and path claims below (Tasks M1-T5 through M3-T4 fill the working tree that each step commits).
- Consumes: the source tip `T0` (M0-T1); the worktree (M1-T1).

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
#
# Land @graphty/webgpu-graph-algorithms in graphty-monorepo with its git history.
#
# Modelled on tools/commit-graph-format-landing.sh (the graph-format landing): the same
# narrowed core.hooksPath (commitlint runs, Commitizen's tty wizard does not), the same
# "every changed path is claimed by exactly one step" guard, the same --dry-run. It differs
# in that the package arrives through a history merge, so the FIRST commit on the landing
# branch is the merge and the wiring commits come after it (design/monorepo/
# nx-monorepo-implementation-plan.md:1483-1491: history must be merged BEFORE the files exist).
#
# Signing: commits are GPG-signed by default (~/.gitconfig commit.gpgsign=true); run this
# script from an interactive terminal so the pinentry prompt can appear, or pass --no-sign
# (what the graph-format landing did).
#
# Usage, in order (all from the landing worktree):
#   ./tools/land-webgpu-graph-algorithms.sh prepare          # fresh clone + filter-repo rewrite + fetch
#   ./tools/land-webgpu-graph-algorithms.sh merge            # the unrelated-history merge (first commit)
#   ./tools/land-webgpu-graph-algorithms.sh commit workspace # then package, docs, ignore, ci -- one at a time,
#                                                            #   after the plan's tasks have filled the tree
#   ./tools/land-webgpu-graph-algorithms.sh push             # push the branch (pre-push gate runs)
#   ./tools/land-webgpu-graph-algorithms.sh land             # after the PR is green: ff master, push master
#                                                            #   (run through the worktree copy; it switches to the
#                                                            #   main worktree itself; pass --skip-gate: the pre-push
#                                                            #   gate already ran on the branch push, and the main
#                                                            #   worktree's node_modules predate the new importer)
# Options: --dry-run (stage/commit nothing), --no-sign, --skip-gate (push --no-verify).
#
# Every message below is plain ASCII and carries no Co-Authored-By / Claude-Session trailer;
# the script refuses to commit a message that has either.

set -u

cd "$(dirname "$0")/.." || exit 1

SOURCE_REPO=/home/apowers/Projects/webgpu-graph-algorithms
REWRITE_DIR=tmp/land/wga-rewrite            # inside the monorepo's gitignored tmp/
REMOTE_NAME=wga-rewrite
BRANCH=land/webgpu-graph-algorithms
MERGE_SUBJECT="feat: merge the webgpu-graph-algorithms package history into the monorepo"
EXPECTED_COMMITS=31                         # 30 from the rehearsal of 2026-09-16 + the M0-T1 docs commit

DRY_RUN=0
SIGN=1
SKIP_GATE=0
COMMAND=""
STEP=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --no-sign) SIGN=0 ;;
        --skip-gate) SKIP_GATE=1 ;;
        prepare|merge|commit|push|land) COMMAND=$arg ;;
        workspace|package|docs|ignore|ci) STEP=$arg ;;
        *) echo "unknown argument: $arg"; exit 2 ;;
    esac
done
[ -n "$COMMAND" ] || { echo "usage: $0 {prepare|merge|commit <step>|push|land} [--dry-run] [--no-sign] [--skip-gate]"; exit 2; }

fail() { echo "REFUSING: $1"; exit 1; }

# ---------------------------------------------------------------- the commit plan

STEP_ORDER="workspace package docs ignore ci"

declare -A SUBJECTS=(
    [workspace]="build(workspace): wire webgpu-graph-algorithms into the workspace, hooks, coverage and docs index"
    [package]="build(webgpu-graph-algorithms): point the manifest, project graph and live docs at the monorepo"
    [docs]="docs(webgpu-graph-algorithms): record the landing beside the design and index the design corpus"
    [ignore]="build(workspace): ignore the package's run output and normalise line endings"
    [ci]="ci: add the webgpu-graph-algorithms shards, the GPU lane and the host matrix"
)

declare -A PATHS=(
    [workspace]="pnpm-workspace.yaml commitlint.config.js knip.config.ts package.json pnpm-lock.yaml tools/prepush.sh tools/merge-coverage.sh tools/apply-root-claude-md-webgpu.py tools/land-webgpu-graph-algorithms.sh CLAUDE.md README.md"
    [package]="webgpu-graph-algorithms/package.json webgpu-graph-algorithms/project.json webgpu-graph-algorithms/README.md webgpu-graph-algorithms/CLAUDE.md webgpu-graph-algorithms/vitest.config.ts webgpu-graph-algorithms/test/build-output.test.ts webgpu-graph-algorithms/test/fixtures/networkx/generate.py webgpu-graph-algorithms/src/memory/upload-plan.ts webgpu-graph-algorithms/benchmarks/run.ts webgpu-graph-algorithms/benchmarks/layout-run.ts webgpu-graph-algorithms/scripts/bench-compare.js"
    [docs]="design/webgpu design/README.md graph-format/CLAUDE.md graph-format/test/audit/gpu-upload.test.ts"
    [ignore]=".gitignore .gitattributes"
    [ci]=".github/workflows/ci.yml .github/workflows/release.yml .github/workflows/gpu.yml .github/workflows/hosts.yml"
)

body_workspace() {
    cat <<'MSG'
The root touch points of design section 13.3 (graph-format design) and 12.5
(WebGPU design), the same set commit 0069386c applied for graph-format and
graph-io: the workspace entry, the commitlint scope, the knip workspace with
the two subpath barrels as extra entries, the coverage-preview script on port
9058, the pre-push fast-test line (the node project only; the browser project
runs in CI), the merge-coverage package list, and the root CLAUDE.md and
README.md sections. The lockfile gains the package's importer; every external
version it needs was already resolved by graph-format's devDependencies.

tools/apply-root-claude-md-webgpu.py edits the two CLAUDE.md lines that carry
non-ASCII characters (the tree and the build-order arrows), anchored on lines
that must occur exactly once; tools/land-webgpu-graph-algorithms.sh is the
landing script that made these commits.
MSG
}

body_package() {
    cat <<'MSG'
The package.json repository, bugs and homepage fields name this repository
and the directory webgpu-graph-algorithms (the test that pinned the staging
directory string follows). The graph-format peer range becomes "^0.2.0",
the minor graph-io states since 6b4777df: nx release keeps a dependent's
range only while the new version satisfies it and refuses to version
otherwise, so "^0.1.0" would make the format's 0.1.0 -> 0.2.0 bump abort the
whole release; below 1.0.0 a caret pins the minor, the honest claim for a
0.x format, and the F2 PR turns it into "^1.0.0".

project.json: implicitDependencies "!algorithms" and "!layout" remove the
project-graph edges nx builds from the OPTIONAL peer ranges (^1.0.0 matches
the workspace versions), which would otherwise build both packages before
this one, patch-bump it on their every release and mark it affected by their
every change; the edges return with the real devDependencies at W1-proper.
The lint target depends on build (the strict-consumer compile reads the d.ts
shims only build:bundle writes) and the test targets are uncached (their
outcome depends on the adapter and the GRAPHTY_* policy, which nx does not
hash).

README.md and CLAUDE.md cite the design and the interface contract at
design/webgpu/ instead of the staging root, and the NVIDIA recipes name the
libEGL tree under this repository's tmp/ (the dev container still lacks
libegl1; HEADLESS_GPU_REPORT.md appendix D). The vitest reporter is "default"
under CI, as graph-format and graph-io set it after the worker RPC starved
behind the verbose reporter (6fc56c1b). The three upload-plan interfaces carry
@public (members of the exported UploadPlan union; knip 5.77 reported them),
and the usage comments of the benchmark runners and bench-compare name the
new directory.
MSG
}

body_docs() {
    cat <<'MSG'
The accepted WebGPU design (design/webgpu/webgpu-acceleration-plan.md, owner
decision Q-18 chose this directory) and the P0-P3 interface contract and phase
plans under design/webgpu/plans/ arrived with the package's history. Each of
the two normative documents gains a LANDED entry appended at its end (never
prepended: the contract, the phase plans and the gate records cite the design
by line number) explaining the path forms of the staging era. design/webgpu/
README.md indexes the corpus, since the p1 and p3 phase plans exceed a
megabyte and GitHub does not render them; design/README.md gains the webgpu
row and the graph-format row the F1 landing forgot; graph-format/CLAUDE.md
and the GPU upload audit stop pointing at staging-only paths.
MSG
}

body_ignore() {
    cat <<'MSG'
The package writes benchmarks/out/, browser-results.json and gpu-report.json
when its benchmarks, browser smoke and adapter report run (contract 2.8); the
staging repository ignored all three and this one did not. The four review
probe logs under docs/research/review/probes/ are committed on purpose
(contract 1.1); the root "*.log" rule would hide any NEW probe log from git
status, so the directory is un-ignored by name (the tracked four are
unaffected either way).

.gitattributes: every text file is committed and checked out with LF (the
Windows host lane runs the package's tests on a checkout; the source-entry
test and prettier assume LF), the two import corpora keep their deliberate
CRLF files (-text), and .gsnp / .png / .zip fixtures are marked binary.
`git ls-files --eol` shows no CRLF or mixed file outside the two corpora, so
the attribute describes the tree as it already is.
MSG
}

body_ci() {
    cat <<'MSG'
Two software shards join the test matrix (design 12.5): webgpu-graph-
algorithms-node runs the node project on Mesa lavapipe with coverage and the
no-subgroups pass, and uploads coverage-webgpu-graph-algorithms-node; webgpu-
graph-algorithms-browser runs the Chromium SwiftShader smoke through
scripts/run-browser-project.js. A needs-lavapipe matrix key gates the apt
install and the ICD discovery (ubuntu-24.04 ships lvp_icd.json, not the
lvp_icd.x86_64.json the design's table names). The build job builds the
package explicitly on PRs (nothing depends on it yet), uploads build-webgpu-
graph-algorithms, every shard downloads it, and release.yml downloads it for
nx release publish.

gpu.yml is the staging GPU lane with the three changes of design 12.5 (paths
without the packages/ prefix, its own install and nx build on the runner,
pnpm/action-setup without package_json_file); its push and schedule triggers
stay commented out until the hosted gpu-linux-t4 runner exists, and it is
never a job of CI (release.yml and coverage.yml are workflow_run on CI).
hosts.yml is the informational macOS / Windows matrix, filtered to changes
under webgpu-graph-algorithms/ and graph-format/.
MSG
}

# ---------------------------------------------------------------- shared guards

hooks_dir() {
    # commitlint runs; the Commitizen tty wizard (prepare-commit-msg) is absent
    HOOKS_DIR=$(mktemp -d)
    cp .husky/commit-msg "$HOOKS_DIR/commit-msg"
    chmod +x "$HOOKS_DIR/commit-msg"
    echo "$HOOKS_DIR"
}

check_message() {
    local file=$1
    if LC_ALL=C grep -qP '[^\x00-\x7F]' "$file"; then fail "non-ASCII byte in the commit message $file"; fi
    if grep -qiE '^(Co-Authored-By|Claude-Session):' "$file"; then fail "attribution trailer in the commit message $file"; fi
    local subject
    subject=$(head -1 "$file")
    [ "${#subject}" -le 100 ] || fail "subject longer than 100 characters: $subject"
    # commitlint's body-max-line-length (config-conventional: 100) applies to the merge message too
    awk 'length($0) > 100 { exit 1 }' "$file" || fail "a line of the commit message $file exceeds 100 characters"
}

sign_args() {
    if [ "$SIGN" = "1" ]; then echo ""; else echo "-c commit.gpgsign=false"; fi
}

on_landing_branch() {
    [ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || fail "run this from the landing worktree on $BRANCH"
    # in a linked worktree .git is a file; rebase / merge state lives under the worktree's git dir
    local gd
    gd=$(git rev-parse --git-dir)
    { [ -d "$gd/rebase-merge" ] || [ -d "$gd/rebase-apply" ]; } && fail "a rebase is in progress"
    [ -f "$gd/MERGE_HEAD" ] && fail "a merge is in progress"
    [ -z "$(git diff --cached --name-only)" ] || fail "the index is not empty"
}

# ---------------------------------------------------------------- prepare

do_prepare() {
    on_landing_branch
    [ -d "$SOURCE_REPO/.git" ] || fail "source repository not found at $SOURCE_REPO"
    (cd "$SOURCE_REPO" && [ -z "$(git status --porcelain)" ]) || fail "the source repository has uncommitted changes; finish Task M0-T1 first"
    local tip
    tip=$(cd "$SOURCE_REPO" && git rev-parse HEAD)
    echo "source tip: $tip"
    [ "$DRY_RUN" = "1" ] && { echo "DRY RUN: would clone, rewrite and fetch"; return; }
    mkdir -p tmp/land
    if [ -d "$REWRITE_DIR" ]; then
        echo "removing the previous rewrite at $REWRITE_DIR"
        rm -rf "$REWRITE_DIR"
    fi
    git clone --no-local --quiet "$SOURCE_REPO" "$REWRITE_DIR" || fail "clone failed"
    (
        cd "$REWRITE_DIR" || exit 1
        git log --format='%h %s' > ../log-before.txt
        # --force is not needed for a --no-local clone (it passes filter-repo's fresh-clone check); kept so a
        # re-run over a partially rewritten clone cannot stop the script -- the clone is disposable
        git filter-repo --force \
            --path packages/webgpu-graph-algorithms/ \
            --path design/webgpu-acceleration-plan.md \
            --path docs/superpowers/plans/ \
            --path HEADLESS_GPU_REPORT.md \
            --path-rename packages/webgpu-graph-algorithms/:webgpu-graph-algorithms/ \
            --path-rename design/webgpu-acceleration-plan.md:design/webgpu/webgpu-acceleration-plan.md \
            --path-rename docs/superpowers/plans/:design/webgpu/plans/ \
            --path-rename HEADLESS_GPU_REPORT.md:webgpu-graph-algorithms/docs/HEADLESS_GPU_REPORT.md \
            || exit 1
        git log --format='%h %s' > ../log-after.txt
        git gc --quiet --prune=now
    ) || fail "filter-repo failed"
    # verification of the rewrite
    local count top stray nonascii trailers
    count=$(git -C "$REWRITE_DIR" rev-list --count HEAD)
    [ "$count" = "$EXPECTED_COMMITS" ] || echo "NOTE: $count commits survived the rewrite (expected $EXPECTED_COMMITS); compare tmp/land/log-before.txt and log-after.txt"
    top=$(git -C "$REWRITE_DIR" ls-tree --name-only HEAD | sort | tr '\n' ' ')
    [ "$top" = "design webgpu-graph-algorithms " ] || fail "unexpected top-level paths after the rewrite: $top"
    stray=$(git -C "$REWRITE_DIR" log --name-only --format= | grep -v '^$' | grep -v '^webgpu-graph-algorithms/\|^design/webgpu/' | head -3)
    [ -z "$stray" ] || fail "the rewrite still touches paths outside the two prefixes: $stray"
    nonascii=$(git -C "$REWRITE_DIR" log --format='%B' | LC_ALL=C grep -cP '[^\x00-\x7F]' || true)
    [ "$nonascii" = "0" ] || fail "$nonascii non-ASCII line(s) in the imported commit messages"
    trailers=$(git -C "$REWRITE_DIR" log --format='%B' | grep -ciE '^(Co-Authored-By|Claude-Session):' || true)
    [ "$trailers" = "0" ] || fail "$trailers attribution trailer(s) in the imported commit messages"
    echo "rewrite ok: $count commits, $(du -sh "$REWRITE_DIR/.git" | cut -f1) packed"
    git remote remove "$REMOTE_NAME" 2>/dev/null || true
    git remote add "$REMOTE_NAME" "$REWRITE_DIR" || fail "remote add failed"
    git fetch --quiet "$REMOTE_NAME" master || fail "fetch failed"
    echo "fetched $REMOTE_NAME/master = $(git rev-parse --short "$REMOTE_NAME/master")"
    echo "next: $0 merge"
}

# ---------------------------------------------------------------- merge

do_merge() {
    on_landing_branch
    # tracked modifications block the merge; untracked files (this script before its own commit, tmp/land/) do not
    [ -z "$(git status --porcelain --untracked-files=no)" ] || fail "the worktree has uncommitted tracked changes"
    git rev-parse --verify --quiet "$REMOTE_NAME/master" > /dev/null || fail "run prepare first ($REMOTE_NAME/master is missing)"
    [ -z "$(git ls-tree --name-only HEAD webgpu-graph-algorithms design/webgpu)" ] || fail "webgpu-graph-algorithms/ or design/webgpu/ already exists on this branch: the history merge must come first"
    [ ! -e webgpu-graph-algorithms ] && [ ! -e design/webgpu ] || fail "webgpu-graph-algorithms/ or design/webgpu/ exists on disk (untracked); remove it, the merge creates both"
    [ "$(git rev-list --count master..HEAD)" = "0" ] || fail "the branch already has commits on top of master; the merge must be the first"
    local msg
    msg=$(mktemp)
    {
        echo "$MERGE_SUBJECT"
        echo
        cat <<'MSG'
The package, its design corpus and the P0-P3 phase plans arrive with the
history of graphty-org/webgpu-graph-algorithms (31 of its 39 commits survive
the path filter), rewritten by git filter-repo to this repository's layout
(packages/webgpu-graph-algorithms/ -> webgpu-graph-algorithms/, design/ and
docs/superpowers/plans/ -> design/webgpu/); the monorepo's own history-import
procedure (design/monorepo/nx-monorepo-implementation-plan.md). The wiring
follows in separate commits.
MSG
    } > "$msg"
    check_message "$msg"
    if [ "$DRY_RUN" = "1" ]; then echo "DRY RUN: would merge $REMOTE_NAME/master with:"; cat "$msg"; rm -f "$msg"; return; fi
    local hooks
    hooks=$(hooks_dir)
    # shellcheck disable=SC2046
    git -c core.hooksPath="$hooks" $(sign_args) merge --allow-unrelated-histories --no-ff -F "$msg" "$REMOTE_NAME/master"
    local status=$?
    rm -rf "$hooks" "$msg"
    [ "$status" = "0" ] || fail "merge failed (exit $status); git merge --abort and inspect"
    [ -z "$(git diff --name-only --diff-filter=U)" ] || fail "conflicts after the merge"
    echo "merged: $(git log -1 --format='%h signed=%G? %s')"
    echo "package files: $(git ls-files webgpu-graph-algorithms | wc -l); design files: $(git ls-files design/webgpu | wc -l)"
    echo "next: run the plan's tasks, then $0 commit workspace (then package, docs, ignore, ci)"
}

# ---------------------------------------------------------------- commit <step>

do_commit() {
    on_landing_branch
    [ -n "$STEP" ] || fail "commit needs a step: workspace | package | docs | ignore | ci"
    for path in ${PATHS[$STEP]}; do
        [ -e "$path" ] || fail "planned path does not exist: $path"
    done
    # every changed path must belong to THIS step or to a LATER step; nothing may be unclaimed
    local changed unclaimed=""
    changed=$(git status --porcelain --untracked-files=normal | sed 's/^...//' | sed 's/ -> .*//' | sed 's:/$::')
    for c in $changed; do
        local claimed=0
        for s in $STEP_ORDER; do
            for p in ${PATHS[$s]}; do
                case "$c" in "$p" | "$p"/*) claimed=1 ;; esac
            done
        done
        [ "$claimed" = "1" ] || unclaimed="$unclaimed $c"
    done
    [ -z "$unclaimed" ] || fail "changed paths no step claims:$unclaimed"
    local msg
    msg=$(mktemp)
    { echo "${SUBJECTS[$STEP]}"; echo; "body_$STEP"; } > "$msg"
    check_message "$msg"
    echo "-- ${SUBJECTS[$STEP]}"
    for path in ${PATHS[$STEP]}; do
        printf '     %-60s %s file(s)\n' "$path" "$(git status --porcelain --untracked-files=all -- "$path" | wc -l)"
    done
    if [ "$DRY_RUN" = "1" ]; then echo "DRY RUN: nothing staged"; rm -f "$msg"; return; fi
    # shellcheck disable=SC2086
    git add -- ${PATHS[$STEP]} || fail "git add failed"
    [ -n "$(git diff --cached --name-only)" ] || { echo "nothing staged for $STEP; no commit made"; rm -f "$msg"; return; }
    local hooks
    hooks=$(hooks_dir)
    # shellcheck disable=SC2046
    git -c core.hooksPath="$hooks" $(sign_args) commit -F "$msg"
    local status=$?
    rm -rf "$hooks" "$msg"
    [ "$status" = "0" ] || fail "commit failed for $STEP"
    echo "     $(git log -1 --format='%h signed=%G? %s')"
}

# ---------------------------------------------------------------- push / land

do_push() {
    on_landing_branch
    [ -z "$(git status --porcelain --untracked-files=normal)" ] || fail "uncommitted changes; commit every step first"
    git remote remove "$REMOTE_NAME" 2>/dev/null || true
    [ "$DRY_RUN" = "1" ] && { echo "DRY RUN: would push $BRANCH to origin"; return; }
    if [ "$SKIP_GATE" = "1" ]; then git push --no-verify -u origin "$BRANCH"; else git push -u origin "$BRANCH"; fi
}

do_land() {
    # run through the landing worktree's copy of this script (master has no copy until the fast-forward); the
    # step moves itself to the MAIN worktree through the shared git dir and operates there
    cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." || fail "cannot find the main worktree"
    [ "$(git rev-parse --abbrev-ref HEAD)" = "master" ] || fail "the main worktree is not on master"
    git fetch --quiet origin
    [ "$(git rev-parse master)" = "$(git rev-parse origin/master)" ] || fail "local master differs from origin/master"
    git merge-base --is-ancestor master "origin/$BRANCH" || fail "master moved under the branch; merge master into $BRANCH in the landing worktree, re-run CI, then land again"
    [ "$DRY_RUN" = "1" ] && { echo "DRY RUN: would fast-forward master to origin/$BRANCH and push"; return; }
    git merge --ff-only "origin/$BRANCH" || fail "fast-forward failed"
    if [ "$SKIP_GATE" = "1" ]; then git push --no-verify origin master; else git push origin master; fi
}

case "$COMMAND" in
    prepare) do_prepare ;;
    merge) do_merge ;;
    commit) do_commit ;;
    push) do_push ;;
    land) do_land ;;
esac
```

- [ ] **Step 2: Make it executable and lint it**

Run: `chmod +x tools/land-webgpu-graph-algorithms.sh && bash -n tools/land-webgpu-graph-algorithms.sh && (command -v shellcheck >/dev/null && shellcheck -S warning tools/land-webgpu-graph-algorithms.sh || echo "shellcheck not installed; skipped")`
Expected: `bash -n` prints nothing; shellcheck reports nothing at warning level (the two `SC2046` sites are annotated; shellcheck is not installed on the dev box today, so the fallback message is the expected output).

Then check every subject against commitlint's 100-character header limit and the scope that the `workspace` commit adds (the merge subject has no scope; a scope-enum failure on the OTHER subjects is expected until M2-T1 adds the scope, a header-max-length failure is not):

```bash
cd WT && for s in "feat: merge the webgpu-graph-algorithms package history into the monorepo" \
  "build(workspace): wire webgpu-graph-algorithms into the workspace, hooks, coverage and docs index" \
  "build(webgpu-graph-algorithms): point the manifest, project graph and live docs at the monorepo" \
  "docs(webgpu-graph-algorithms): record the landing beside the design and index the design corpus" \
  "build(workspace): ignore the package's run output and normalise line endings" \
  "ci: add the webgpu-graph-algorithms shards, the GPU lane and the host matrix"; do printf '%3d %s\n' "${#s}" "$s"; done
```

Expected: every length at most 100 (73, 97, 95, 95, 76, 76 on 2026-09-16).

- [ ] **Step 3: Dry-run the guards**

Run: `cd WT && ./tools/land-webgpu-graph-algorithms.sh prepare --dry-run`
Expected: `source tip: <T0>` then `DRY RUN: would clone, rewrite and fetch` (the script refuses if the source has uncommitted changes -- then finish M0-T1 first).

- [ ] **Step 4: Commit**

The script is committed by its own `workspace` step (Task M2-T9). Nothing to do now.

### Task M1-T3: Rewrite and fetch the history (owner runs `prepare`)

**Repository:** `WT`.

- [ ] **Step 1: The owner runs the rewrite**

Tell the owner: `! cd /home/apowers/Projects/graphty-monorepo/.worktrees/land-webgpu-graph-algorithms && ./tools/land-webgpu-graph-algorithms.sh prepare`
Expected output ends with `rewrite ok: 31 commits, <about 3.5M> packed` (the script prints `du -sh .git`; the rehearsal's 30 commits packed to 3.4M, and T0 adds the 220 KB plan), `fetched wga-rewrite/master = <sha>`, `next: ./tools/land-webgpu-graph-algorithms.sh merge`. The rehearsal of 2026-09-16 produced 30 commits from tip `a66c889`; T0 adds one.

- [ ] **Step 2: Verify the rewritten history (agent, read-only)**

Run:

```bash
cd WT/tmp/land/wga-rewrite
git rev-list --count HEAD                                                     # 31
git ls-tree --name-only HEAD                                                  # design  webgpu-graph-algorithms
git ls-tree --name-only HEAD:design/webgpu                                    # plans  webgpu-acceleration-plan.md
git ls-tree --name-only HEAD:design/webgpu/plans | wc -l                      # 6 (the contract, p0-p3, this plan)
git ls-files webgpu-graph-algorithms | wc -l                                  # 437 (435 tracked in staging + G0.md + G2.md)
git log --oneline --follow -- webgpu-graph-algorithms/src/kernel/batch.ts    # the original commits (2 or more)
git log --oneline -- webgpu-graph-algorithms/docs/HEADLESS_GPU_REPORT.md      # 1 line: its creation commit
git log --format='%G?' | sort | uniq -c                                       # 31 N: filter-repo re-creates every commit, so the 21 source signatures are gone (D-1)
diff <(cut -d' ' -f2- ../log-before.txt | sort) <(cut -d' ' -f2- ../log-after.txt | sort) | grep '^<'
```

Expected: the numbers as commented; the last command lists exactly the 8 dropped subjects: `chore: import the 2025 scaffold`, `feat(packages): stage graph-format and graph-io for the monorepo`, `chore: ignore the session history archive`, and the five `ci:` commits (`implement the GPU nightly tracking-issue job`, `locate the lavapipe ICD instead of hardcoding its path`, `find the lavapipe ICD with find, not ls ...`, `run the GPU lane on dispatch and the gpu label only until the runner exists`, `the Windows host lane prints the Application log's crash events`). Any other dropped subject means a path rule is wrong: stop and fix the script.

### Task M1-T4: Merge the history (owner runs `merge`)

**Repository:** `WT`.

- [ ] **Step 1: The owner runs the merge**

Tell the owner: `! cd /home/apowers/Projects/graphty-monorepo/.worktrees/land-webgpu-graph-algorithms && ./tools/land-webgpu-graph-algorithms.sh merge`
Expected: `merged: <sha> signed=G feat: merge the webgpu-graph-algorithms package history into the monorepo`, `package files: 437; design files: 7`.

- [ ] **Step 2: Verify the merge (agent, read-only)**

Run:

```bash
cd WT
git log --oneline --topo-order -3                       # the merge, then master's tip, then the newest imported commit (plain date order would put T0 first)
git log --first-parent --oneline -2                     # the merge and master's tip only
git rev-list --max-parents=0 HEAD | wc -l               # 6 (five monorepo roots + the imported root)
git log --oneline -- webgpu-graph-algorithms/package.json | tail -1   # the oldest commit touching the file: feat(webgpu-graph-algorithms): reserve the package name on npm (2026-09-14), reached without --follow
git status --short                                      # empty
ls webgpu-graph-algorithms/src/index.ts design/webgpu/webgpu-acceleration-plan.md design/webgpu/plans/2026-09-14-webgpu-p0-p3-interfaces.md
```

Expected: as commented. If `git log -- webgpu-graph-algorithms/package.json` shows only the merge, the files existed before the merge -- the branch must be recreated from master (Task M1-T1) and M1-T3/M1-T4 repeated.

- [ ] **Step 3: Make the package a workspace member, then install**

`pnpm-workspace.yaml` lists packages by name (no glob), so pnpm ignores the directory until it is listed. Insert `    - "webgpu-graph-algorithms"` directly after `    - "graph-io"` in `WT/pnpm-workspace.yaml` (this is the workspace entry of the design's root touch points, pulled ahead of the install; the file is committed by the `workspace` step). Then:

Run: `cd WT && HUSKY=0 pnpm install`
Expected: exit 0; `git status --short` shows `M pnpm-workspace.yaml` and `M pnpm-lock.yaml`, and `grep -n '^  webgpu-graph-algorithms:$' pnpm-lock.yaml` finds the new importer block (no new external package versions -- `webgpu@0.4.0` and `@webgpu/types@0.1.72` were already resolved by graph-format's devDependencies; unrelated lockfile drift such as `@babel/parser` lines is expected). Then `HUSKY=0 pnpm install --frozen-lockfile` exits 0.

- [ ] **Step 4: Build once to prove the tree is a workspace member**

Run: `cd WT && pnpm exec nx run webgpu-graph-algorithms:build`
Expected: builds graph-format first (`dependsOn ^build`), then `tsc -p tsconfig.build.json` and `node scripts/build-bundle.js`; `ls webgpu-graph-algorithms/dist/` lists `webgpu-graph-algorithms.js`, `browser.js`, `node.js` and their `.d.ts`. The nx project graph accepts the package because `webgpu-graph-algorithms/project.json` already names `cwd: webgpu-graph-algorithms` (it was written for this layout in P0 and never exercised).

### Task M1-T5: Package fix-ups for the new location

**Repository:** `WT`.

**Files:**
- Modify: `webgpu-graph-algorithms/package.json:63-67,80-83` (repository, bugs, homepage) and `:90` (the graph-format peer range)
- Modify: `webgpu-graph-algorithms/test/build-output.test.ts:114` (line 132 pins the caret FORM and stays: `^0.2.0` satisfies it)
- Modify: `webgpu-graph-algorithms/project.json` (implicitDependencies, lint dependsOn, uncached test targets)
- Modify: `webgpu-graph-algorithms/README.md:16-17,321`
- Modify: `webgpu-graph-algorithms/CLAUDE.md:12,16,18,151,261,280,285`
- Modify: `webgpu-graph-algorithms/vitest.config.ts:173` (the root `test.reporters`)
- Modify: `webgpu-graph-algorithms/src/memory/upload-plan.ts:37,45,51` (`@public` tags)
- Modify: `webgpu-graph-algorithms/test/fixtures/networkx/generate.py:7-12` (comment paths)
- Modify: `webgpu-graph-algorithms/benchmarks/run.ts:7`, `webgpu-graph-algorithms/benchmarks/layout-run.ts:8`, `webgpu-graph-algorithms/scripts/bench-compare.js:6` (usage comments naming the staging path)

- [ ] **Step 1: Write the failing test changes first**

In `webgpu-graph-algorithms/test/build-output.test.ts` line 114 replace

```ts
        expect(packageJson.repository.directory).toBe("packages/webgpu-graph-algorithms");
```

with

```ts
        expect(packageJson.repository.directory).toBe("webgpu-graph-algorithms");
```

Line 132, where the test pins the graph-format peer to a caret range (`expect(packageJson.peerDependencies["@graphty/graph-format"]).toMatch(/^\^\d+\.\d+\.\d+$/);`), stays: the owner's D-18 form `^0.2.0` satisfies it.

- [ ] **Step 2: Run it to see it fail**

Run: `cd WT/webgpu-graph-algorithms && pnpm exec vitest run --project=node test/build-output.test.ts`
Expected: FAIL on the `repository.directory` expectation (`packages/webgpu-graph-algorithms` !== `webgpu-graph-algorithms`).

- [ ] **Step 3: Re-point the manifest and re-state the peer range at the format's minor**

In `webgpu-graph-algorithms/package.json` replace the three blocks with the graph-format spelling (`graph-format/package.json:46-63` is the precedent):

```json
    "repository": {
        "type": "git",
        "url": "git+https://github.com/graphty-org/graphty-monorepo.git",
        "directory": "webgpu-graph-algorithms"
    },
```

```json
    "bugs": {
        "url": "https://github.com/graphty-org/graphty-monorepo/issues"
    },
    "homepage": "https://github.com/graphty-org/graphty-monorepo/tree/master/webgpu-graph-algorithms#readme",
```

and in `peerDependencies` replace `"@graphty/graph-format": "^0.1.0",` with `"@graphty/graph-format": "^0.2.0",` (D-18 as the owner decided it in `6b4777df`: the minor the package is built against; graph-format is 0.2.0 on disk since the release commit `dc08826b`, so `workspace:^` publishes as `^0.2.0` and the peer says the same). `dependencies` keeps `"@graphty/graph-format": "workspace:^"`.

- [ ] **Step 4: Run the test to see it pass**

Run: `cd WT/webgpu-graph-algorithms && pnpm exec vitest run --project=node test/build-output.test.ts`
Expected: PASS (the bundle assertions run against the dist of M1-T4 step 4; `CI` is unset so the bundle checks skip when dist is absent).

- [ ] **Step 4a: The project graph, the lint order and the uncached tests**

Rewrite `webgpu-graph-algorithms/project.json` so that it reads (the existing targets keep their commands; three things change: `implicitDependencies`, `lint.dependsOn`, and `cache: false` on the six targets that run tests or benchmarks -- test, test:node, test:browser, test:limits, coverage, benchmark):

```json
{
    "name": "webgpu-graph-algorithms",
    "$schema": "../node_modules/nx/schemas/project-schema.json",
    "sourceRoot": "webgpu-graph-algorithms/src",
    "projectType": "library",
    "tags": [],
    "implicitDependencies": ["!algorithms", "!layout"],
    "targets": {
        "build": {
            "executor": "nx:run-commands",
            "outputs": ["{projectRoot}/dist"],
            "options": { "command": "npm run build:all", "cwd": "webgpu-graph-algorithms" },
            "dependsOn": ["^build"]
        },
        "test": {
            "executor": "nx:run-commands",
            "cache": false,
            "options": { "command": "npm run test:run", "cwd": "webgpu-graph-algorithms" }
        },
        "test:node": {
            "executor": "nx:run-commands",
            "cache": false,
            "options": { "command": "npm run test:node", "cwd": "webgpu-graph-algorithms" }
        },
        "test:browser": {
            "executor": "nx:run-commands",
            "cache": false,
            "options": { "command": "npm run test:browser:ci", "cwd": "webgpu-graph-algorithms" }
        },
        "test:limits": {
            "executor": "nx:run-commands",
            "cache": false,
            "options": { "command": "npm run test:limits", "cwd": "webgpu-graph-algorithms" }
        },
        "test:ui": {
            "executor": "nx:run-commands",
            "options": { "command": "vitest --ui", "cwd": "webgpu-graph-algorithms" }
        },
        "coverage": {
            "executor": "nx:run-commands",
            "cache": false,
            "options": { "command": "npm run coverage", "cwd": "webgpu-graph-algorithms" }
        },
        "lint": {
            "executor": "nx:run-commands",
            "options": { "command": "npm run lint", "cwd": "webgpu-graph-algorithms" },
            "dependsOn": ["build"]
        },
        "typecheck": {
            "executor": "nx:run-commands",
            "options": { "command": "npm run typecheck", "cwd": "webgpu-graph-algorithms" }
        },
        "benchmark": {
            "executor": "nx:run-commands",
            "cache": false,
            "options": { "command": "npm run bench", "cwd": "webgpu-graph-algorithms" }
        }
    }
}
```

Why: the optional peers `^1.0.0` on algorithms and layout match the workspace versions, so nx would make both packages build dependencies, release dependents and affected-triggers of this package (D-19); the strict-consumer compile inside `lint` needs the d.ts shims that only `build:bundle` writes (D-20); a cached test result would replay a run made under another adapter policy (`GRAPHTY_GPU_REQUIRE` unset makes every GPU test skip), and nx does not hash the adapter.

Check: `cd WT && pnpm exec nx show project webgpu-graph-algorithms --json | node -e 'const p=JSON.parse(require("fs").readFileSync(0,"utf8")); console.log(p.implicitDependencies, p.targets.lint.dependsOn, p.targets.test.cache)'`
Expected: `[ '!algorithms', '!layout' ] [ 'build' ] false`. Then `cd WT && NX_DAEMON=false pnpm exec nx graph --file=tmp/land/graph.json && node -e 'const g=JSON.parse(require("fs").readFileSync("tmp/land/graph.json","utf8")).graph; console.log(g.dependencies["webgpu-graph-algorithms"].map(d=>d.target))'`
Expected: `[ 'graph-format' ]` only (no `algorithms`, no `layout`).

- [ ] **Step 4b: knip's two findings (D-21)**

In `webgpu-graph-algorithms/src/memory/upload-plan.ts` each of the three interfaces (`ArenaPlan` line 37, `PerArrayPlan` 45, `WindowedPlan` 51) has a one-line JSDoc. Turn each into a block that keeps the existing sentence, adds the clause, and ends with `@public` ALONE on its tag line -- the shared eslint config's `jsdoc/empty-tags` rule rejects any text after `@public` (the `PlannedArray` block at lines 24-29 of the same file is the accepted form). For `ArenaPlan`:

```ts
/**
 * The arena path: ONE buffer of `bytes`, one writeBuffer, per-segment bindings at `segment.byteOffset - arena.byteOffset`.
 * A member of the exported UploadPlan union (contract 3.8 exports the three plan shapes by name); nothing imports it
 * by name, which knip 5.77 reports.
 * @public
 */
export interface ArenaPlan {
```

(keep each interface's own first sentence; only the clause and the tag line are new). The `ignoreDependencies: ["webgpu"]` half of D-21 is the knip config edit of Task M2-T3.

- [ ] **Step 4c: The fixture generator's comment**

In `webgpu-graph-algorithms/test/fixtures/networkx/generate.py` lines 7-12: replace all three `../../tmp/nx-venv` (lines 10-12: the venv creation, the `pip --python`, the interpreter) with `../tmp/nx-venv` (the package is one level below the repository root now), and on line 7 `from packages/webgpu-graph-algorithms` with `from webgpu-graph-algorithms`. A comment only; the 76 committed JSON fixtures are unchanged.

- [ ] **Step 5: Re-point the live documents**

`webgpu-graph-algorithms/README.md` lines 16-17: replace `` `design/webgpu-acceleration-plan.md` `` with `` `design/webgpu/webgpu-acceleration-plan.md` `` and `` `docs/superpowers/plans/2026-09-14-webgpu-p0-p3-interfaces.md` `` with `` `design/webgpu/plans/2026-09-14-webgpu-p0-p3-interfaces.md` ``; both paths are relative to the monorepo root, so add the words `(monorepo root)` after the first one. Line 321: replace `cd packages && pnpm install                       # the pnpm workspace root` with `pnpm install                                      # at the monorepo root` and the following `cd webgpu-graph-algorithms` stays.

`webgpu-graph-algorithms/CLAUDE.md`: line 16 `docs/superpowers/plans/2026-09-14-webgpu-p0-p3-interfaces.md` -> `design/webgpu/plans/2026-09-14-webgpu-p0-p3-interfaces.md`; line 18 `design/webgpu-acceleration-plan.md` -> `design/webgpu/webgpu-acceleration-plan.md`; add after line 18's bullet the sentence `Both paths are relative to the monorepo root.`; line 12: the "repository rule in the root CLAUDE.md" citation stays valid once Task M2-T7 adds the rule to the monorepo root `CLAUDE.md`; lines 151, 280, 285: replace `/home/apowers/Projects/webgpu-graph-algorithms/tmp/egl/root/usr/lib/x86_64-linux-gnu` with `/home/apowers/Projects/graphty-monorepo/tmp/egl/root/usr/lib/x86_64-linux-gnu` (three occurrences) and add one sentence to the paragraph around line 151: `The tree is extracted per docs/HEADLESS_GPU_REPORT.md appendix D into the monorepo's gitignored tmp/egl/; Task M1-T5 of the integration plan re-extracts it.`; line 261: `packages/node_modules` -> `node_modules` (the monorepo root). Usage comments: `benchmarks/run.ts:7`, `benchmarks/layout-run.ts:8`, `scripts/bench-compare.js:6`: `from packages/webgpu-graph-algorithms` -> `from webgpu-graph-algorithms`.

Check: `grep -n 'webgpu-graph-algorithms/tmp/egl\|docs/superpowers\|^.*design/webgpu-acceleration-plan.md\|packages/webgpu-graph-algorithms\|packages/node_modules' webgpu-graph-algorithms/README.md webgpu-graph-algorithms/CLAUDE.md webgpu-graph-algorithms/benchmarks/*.ts webgpu-graph-algorithms/scripts/*.js` prints nothing (the spec citation now always carries `design/webgpu/`).

- [ ] **Step 6: Re-extract the libEGL tree on the dev box (environment, not a repo change)**

Run: `mkdir -p /home/apowers/Projects/graphty-monorepo/tmp/egl && cp -a /home/apowers/Projects/webgpu-graph-algorithms/tmp/egl/root /home/apowers/Projects/graphty-monorepo/tmp/egl/ && ls /home/apowers/Projects/graphty-monorepo/tmp/egl/root/usr/lib/x86_64-linux-gnu/libEGL.so.1`
Expected: the file is listed (`tmp/` is gitignored in the monorepo). If the source tree is gone, follow `webgpu-graph-algorithms/docs/HEADLESS_GPU_REPORT.md` appendix D (apt-get download + dpkg-deb -x, no sudo).

- [ ] **Step 7: CI-aware reporter**

In `webgpu-graph-algorithms/vitest.config.ts` line 173 -- the root `test` block, above `projects:`; `reporters` is a root-only option in vitest 3, so this is the only `reporters` line -- replace `reporters: ["verbose"],` with:

```ts
            // verbose prints a line per test: useful locally, needless noise in CI (and 6fc56c1b: the nx -> npm ->
            // vitest pipe chain starved the worker RPC behind it in the sibling packages)
            reporters: process.env.CI ? ["default"] : ["verbose"],
```

(`graph-format/vitest.config.ts:11` uses the same expression at the same level.)

- [ ] **Step 8: Run the package's own gate**

Run: `cd WT/webgpu-graph-algorithms && pnpm run lint && GRAPHTY_GPU_ADAPTER=llvmpipe GRAPHTY_GPU_REQUIRE=any VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json XDG_RUNTIME_DIR=/tmp pnpm exec vitest run --project=node`
Expected: lint (eslint + `tsc --noEmit` + the strict-consumer compile) clean; the node project green on lavapipe (about 5 minutes; the CI shard runs exactly this).

- [ ] **Step 9: Commit**

The `package` step of the landing script (Task M2-T9 runs the steps in order after all of M1 and M2 are in the tree).

### Task M1-T6: LANDED notes, the design index and the pointer fixes

**Repository:** `WT`.

**Files:**
- Modify: `design/webgpu/webgpu-acceleration-plan.md` (append at the end, after the Review log)
- Modify: `design/webgpu/plans/2026-09-14-webgpu-p0-p3-interfaces.md` (append at the end)
- Create: `design/webgpu/README.md`
- Modify: `design/README.md` (the directory table)
- Modify: `graph-format/CLAUDE.md:20`
- Modify: `graph-format/test/audit/gpu-upload.test.ts:13`

- [ ] **Step 1: Append the Review-log entry to the design**

Append to `design/webgpu/webgpu-acceleration-plan.md` (after its last line; never insert above it -- the contract, the phase plans, the gate records and the review verdicts cite this file by line number):

```markdown

Landed in graphty-monorepo (2026-09-DD, phase M1 of
`design/webgpu/plans/2026-09-16-graphty-monorepo-integration.md`): this
document moved from `design/webgpu-acceleration-plan.md` in
graphty-org/webgpu-graph-algorithms to `design/webgpu/` here, with the
package at `webgpu-graph-algorithms/` and the P0-P3 contract and phase plans
under `design/webgpu/plans/`. Path forms in the body above keep their staging
meaning: `packages/webgpu-graph-algorithms/...` means
`webgpu-graph-algorithms/...`; `docs/superpowers/plans/...` means
`design/webgpu/plans/...`; `tmp/webgpu-plan/...` means the notes, drafts and
probes now under `webgpu-graph-algorithms/docs/research/...`; the absolute
`/home/apowers/Projects/graphty-monorepo/design/graph-format/...` citations
are `design/graph-format/...` in this repository; `HEADLESS_GPU_REPORT.md` is
`webgpu-graph-algorithms/docs/HEADLESS_GPU_REPORT.md`. Nothing above was
edited. The section 12.5 line references into ci.yml are dated 2026-09-14;
the shards as landed are in `.github/workflows/ci.yml` and the integration
plan's Phase M3. DEP-C of the integration plan: the T4 baseline path is
`benchmarks/results/gpu-linux-t4.json`, not `benchmarks/baselines/`.
```

Replace `2026-09-DD` with the landing date (the merge commit's date, `git log -1 --format=%cs <merge>`); Steps 2 and 3 carry the same placeholder -- after Step 3, `git grep -n 2026-09-DD design/webgpu` must print nothing.

- [ ] **Step 2: Append the same kind of entry to the contract**

Append to `design/webgpu/plans/2026-09-14-webgpu-p0-p3-interfaces.md`:

```markdown

Landed in graphty-monorepo (2026-09-DD): this contract moved from
`docs/superpowers/plans/` in the staging repository to `design/webgpu/plans/`;
the package it specifies is at `webgpu-graph-algorithms/`; `design/webgpu-acceleration-plan.md`
means `design/webgpu/webgpu-acceleration-plan.md`. The body above is unchanged.
```

- [ ] **Step 3: Write the design index**

Create `design/webgpu/README.md`:

```markdown
# WebGPU graph algorithms and layouts: the design corpus

The package is `webgpu-graph-algorithms/` (`@graphty/webgpu-graph-algorithms`). It was designed and built
(phases P0-P3) in the staging repository graphty-org/webgpu-graph-algorithms and moved here with its git
history on 2026-09-DD (phase M1 of the integration plan below).

| Document | What it is | Status |
| --- | --- | --- |
| `webgpu-acceleration-plan.md` (449 KB) | The accepted design (owner-approved 2026-09-14, plan of record; later changes are appended to its Review log): runtime model, package architecture, memory, kernels, primitives, force-directed layouts, algorithms, integration with the CPU packages, targets, testing, CI, phases and gates, risks, references | live spec |
| `plans/2026-09-14-webgpu-p0-p3-interfaces.md` (325 KB) | The interface contract of phases P0-P3: every file, signature, WGSL body and test; the package CLAUDE.md ranks it first | live contract |
| `plans/2026-09-15-webgpu-p0.md`, `-p1.md` (1.3 MB), `-p2.md`, `-p3.md` (1.2 MB) | The executed phase plans, which embed the files, scripts and gate templates of their time; records, not specs (GitHub does not render the two largest) | records |
| `plans/2026-09-16-graphty-monorepo-integration.md` | This move, the build / CI / release wiring, the GPU runner, and the layout integration (design section 9.3) | live plan |
| `../../webgpu-graph-algorithms/docs/decisions/G0.md` .. `G3.md` | The gate records (G1, G3 closed on the dev box; G0, G2 open until the hosted GPU runner exists) | records |
| `../../webgpu-graph-algorithms/docs/research/` | The seven research notes, the three drafts, the review reports and probes the design was synthesised from | records |
| `../../webgpu-graph-algorithms/docs/HEADLESS_GPU_REPORT.md` | The verified recipe for the real GPU under headless Chromium and Dawn-node on the dev box | record |

Path forms: the design and the contract were written in the staging repository, where the package lived at
`packages/webgpu-graph-algorithms/` and the plans at `docs/superpowers/plans/`; those forms in the bodies mean
`webgpu-graph-algorithms/` and `design/webgpu/plans/` here. The related graph-format design is
`design/graph-format/graph-format-design.md` (sections 10, 13.3, 14.3-14.6 are the ones the WebGPU design cites).
```

- [ ] **Step 4: Index rows in design/README.md**

In `design/README.md`'s directory table add, in alphabetical position among the existing rows (the row format is `| [`dir/`](dir/) | description | N |`):

```markdown
| [`graph-format/`](graph-format/) | The shared frozen CSR graph format: design, landing status, conformance record | 3 |
| [`webgpu/`](webgpu/) | WebGPU-accelerated graph algorithms and layouts: the accepted design, the P0-P3 contract and phase plans, the monorepo integration plan | 8 |
```

(`graph-format/` was not indexed when it landed; the numbers count files recursively like the other rows: `find design/graph-format -type f | wc -l` -> 3, `find design/webgpu -type f | wc -l` -> 8 with this README included. Edit the table in the file's existing unpadded style: `design/README.md` is not prettier-formatted on master and prettier is not enforced on markdown, so do not run prettier on it.)

- [ ] **Step 5: Fix the two dangling precedent pointers**

`graph-format/CLAUDE.md` line 20: replace `` `packages/STATUS.md` (staging) `` with `` `design/graph-format/STATUS.md` `` (the file that landed in `b7ed9b16`).
`graph-format/test/audit/gpu-upload.test.ts` line 13: replace `HEADLESS_GPU_REPORT.md appendix D` with `webgpu-graph-algorithms/docs/HEADLESS_GPU_REPORT.md appendix D` (a comment; no test changes).

- [ ] **Step 6: Check ASCII and prettier on the new files**

Run: `cd WT && LC_ALL=C grep -nP '[^\x00-\x7F]' design/webgpu/README.md && echo NONASCII || echo ascii-ok; pnpm exec prettier --write design/webgpu/README.md && pnpm exec prettier --check design/webgpu/README.md`
Expected: `ascii-ok`; the NEW file is formatted in place (prettier pads its table columns) and then checks clean. `design/README.md` and the two big documents are not prettier-formatted on master and are not checked.

- [ ] **Step 7: Commit**

The `docs` step of the landing script (Task M2-T9).

### Task M1-T7: Ignore rules and line endings

**Repository:** `WT`.

**Files:**
- Modify: `.gitignore` (after the `*.log` line, and a new block)
- Create: `.gitattributes`

- [ ] **Step 1: Ignore the run output, un-ignore the probe logs**

In `.gitignore`, directly after the line `*.log` (in the `# Logs` block, line 50 today) add:

```gitignore
# webgpu-graph-algorithms review probe logs are committed on purpose (its docs/research/review/probes)
!webgpu-graph-algorithms/docs/research/review/probes/*.log
```

and append a new block at the end of the file:

```gitignore
# webgpu-graph-algorithms run output (contract 2.8): benchmark sessions, the browser smoke's JSON report, the adapter report
**/benchmarks/out/
browser-results.json
gpu-report.json
```

Check: `cd WT && git check-ignore -v webgpu-graph-algorithms/benchmarks/out/x.json webgpu-graph-algorithms/browser-results.json webgpu-graph-algorithms/gpu-report.json; git check-ignore --no-index -v webgpu-graph-algorithms/docs/research/review/probes/design-probe-nvidia.log webgpu-graph-algorithms/docs/research/review/probes/new-probe.log`
Expected: the first three paths are matched by the new rules; the last two lines each name the negation rule `.gitignore:<n>:!webgpu-graph-algorithms/docs/research/review/probes/*.log` (a `!` pattern winning means NOT ignored; `--no-index` is needed because `check-ignore` never reports tracked files, and the four committed logs are tracked -- gitignore never affects tracked files, so the negation matters for NEW probe logs only).

- [ ] **Step 2: Write .gitattributes**

Create `.gitattributes` at the monorepo root:

```gitattributes
# Every text file is committed with LF and checked out with LF on every host (the webgpu-graph-algorithms host
# lane runs its tests on a windows-latest checkout; its source-entry test and prettier assume LF). The two import
# corpora keep their own line endings (seven of their files are CRLF on purpose), and binary fixtures are marked so.
* text=auto eol=lf
graph-io/test/corpus/** -text
graphty-element/test/helpers/corpus/** -text
*.gsnp binary
*.png binary
*.zip binary
```

- [ ] **Step 3: Prove the attribute changes nothing (read-only)**

Run: `cd WT && git ls-files --eol | awk '$1 ~ /^i\/(crlf|mixed)/ && $3 !~ /-text/'`
Expected: no output. `git ls-files --eol` reports every tracked file's index line endings and its effective attribute; the only index-CRLF files in the monorepo are the seven corpus files (`git ls-files --eol | grep -c 'i/crlf'` prints `7`, all under the two `-text` paths), and `text=auto eol=lf` renormalises nothing else. If a path prints, add a `-text` exemption for it rather than rewriting it, and note it in the commit body. Then `git check-attr -a webgpu-graph-algorithms/test/fixtures/rich-v1.gsnp graph-io/test/corpus/csv/got-edges.csv webgpu-graph-algorithms/src/index.ts` -- Expected: `binary`, `text: unset`, and `text: auto` + `eol: lf` respectively. (The index-level proof, `git add --renormalize .` followed by `git reset -q`, is the owner's to run if they want it: the executor never runs `git add`.)

- [ ] **Step 4: Commit**

The `ignore` step of the landing script (Task M2-T9).

---
## Phase M2: Build-system wiring

All tasks run in `WT` on the landing branch after Phase M1. Commit `0069386c` (`git show 0069386c`) is the precedent for every root file below; read it once before starting.

### Task M2-T1: Workspace entry and commit scope

**Files:**
- Modify: `pnpm-workspace.yaml:1-9`
- Modify: `commitlint.config.js:4-25`

- [ ] **Step 1: Workspace entry (already added in Task M1-T4 step 3)**

Verify: `cd WT && grep -n 'webgpu-graph-algorithms' pnpm-workspace.yaml` prints the line after `graph-io` (pnpm orders by the dependency graph; the list order is cosmetic and follows the build order). If it is missing, add `    - "webgpu-graph-algorithms"` after `    - "graph-io"` and re-run `HUSKY=0 pnpm install` so the lockfile gains the importer.

- [ ] **Step 2: Commit scope**

In `commitlint.config.js`, in the `scope-enum` array, insert `                "webgpu-graph-algorithms",` directly after `                "graph-io",`. Leave the tab-indented duplicate `compact-mantine` / `remote-logger` entries and the stale `gpu-3d-force-layout` scope alone (out of scope; a separate `chore(tools)` can clean them).

- [ ] **Step 3: Verify**

Run: `cd WT && printf 'build(webgpu-graph-algorithms): x\n' | pnpm exec commitlint && pnpm ls --depth -1 --filter @graphty/webgpu-graph-algorithms`
Expected: commitlint exits 0 silently; pnpm lists the package as a workspace project.

### Task M2-T2: The release dry run (D-18)

**Files:** none. The graph-io peer range was landed by the owner as `6b4777df` (`^0.2.0`; see D-18), so this task only proves that the landing branch, with the GPU package's `^0.2.0` peer from Task M1-T5, still versions.

- [ ] **Step 1: Check the graph-io range**

Run: `cd WT && grep -n '"@graphty/graph-format"' graph-io/package.json webgpu-graph-algorithms/package.json`
Expected: graph-io `"workspace:*"` and `"^0.2.0"`; the GPU package `"workspace:^"` and `"^0.2.0"`.

- [ ] **Step 2: Prove the release would version (dry run only)**

Run: `cd WT && NX_DAEMON=false pnpm exec nx release --dry-run --skip-publish 2>&1 | tail -40`

NEVER run `nx release` without `--dry-run`: without it, nx commits, tags and PUSHES (`nx.json` `release.git`). Expected: the dry run lists `webgpu-graph-algorithms 0.1.0 -> 0.2.0` (its imported `feat` commits are already reachable through the merge; the other projects were released as `dc08826b` and report no change unless master gained commits since), and NO `preserveMatchingDependencyRanges` error. (The dry run writes nothing; `git status --short` is unchanged afterwards -- check it.)

### Task M2-T3: knip workspace entry

**Files:**
- Modify: `knip.config.ts` (after the `graph-io` workspace entry, before `// Algorithms package`)

- [ ] **Step 1: Add the entry**

Insert after the `"graph-io": { ... },` block:

```ts
        // webgpu-graph-algorithms package: the root barrel re-exports neither subpath, so both are entries; the test
        // setup files, the fixture generators and the layout driver are standalone entries. @vitest/browser and
        // playwright are resolved by knip's vitest plugin from vitest.config.ts. `webgpu` is an optional peer AND an
        // exact devDependency, imported inside `await import("webgpu")` in src/node/index.ts (design 2.5); knip 5.77
        // reports referenced optional peers, so it is ignored by name.
        "webgpu-graph-algorithms": {
            entry: [
                "src/index.ts",
                "src/browser/index.ts",
                "src/node/index.ts",
                "test/**/*.test.ts",
                "test/types/**/*.test-d.ts",
                "test/setup/*.ts",
                "test/fixtures/**/*.ts",
                "benchmarks/layout-run.ts",
                "scripts/**/*.{ts,js}",
            ],
            project: ["src/**/*.ts", "test/**/*.ts", "benchmarks/**/*.ts", "scripts/**/*.{ts,js}"],
            ignore: ["dist/**", "coverage/**", "node_modules/**"],
            ignoreDependencies: ["webgpu"],
        },

```

- [ ] **Step 2: Run knip**

Run: `cd WT && pnpm exec knip --workspace webgpu-graph-algorithms; echo "exit=$?"`
Expected: `exit=0` with no findings for the package. If knip reports `benchmarks/run.ts` as an unused file, that is the vitest/npm-script inference: confirm `webgpu-graph-algorithms/package.json` still has the `bench` script (`tsx benchmarks/run.ts`) and that `knip` was run from the worktree root with its own `node_modules` (a scratch install once showed that false positive). Then the whole repository: `pnpm exec knip; echo "exit=$?"` -- expected: the same findings as `master` has today and nothing under `webgpu-graph-algorithms/`.

### Task M2-T4: Root scripts

**Files:**
- Modify: `package.json` (root; the `scripts` block)

- [ ] **Step 1: Coverage preview and the demo server**

After `"coverage:preview:graph-io": "npx serve graph-io/coverage -p 9057"` (today the LAST entry of `scripts`, so it has no trailing comma: add one to it) add, without a trailing comma because the new line becomes the last entry:

```json
        "coverage:preview:webgpu-graph-algorithms": "npx serve webgpu-graph-algorithms/coverage -p 9058"
```

and after `"dev:graphty": "pnpm --filter @graphty/graphty run dev",` add:

```json
        "dev:webgpu-graph-algorithms": "cd webgpu-graph-algorithms && pnpm exec vite demo --host --port 9030",
```

(The demo is a standalone Vite page, `webgpu-graph-algorithms/demo/`, outside the package's build, lint, knip and tests, never published by `deploy-pages.yml`; 9030 is unused anywhere in the monorepo.)

- [ ] **Step 2: Verify**

Run: `cd WT && node -e 'const s=require("./package.json").scripts; console.log(s["coverage:preview:webgpu-graph-algorithms"], "|", s["dev:webgpu-graph-algorithms"])'`
Expected: both strings printed.

### Task M2-T5: The pre-push hook

**Files:**
- Modify: `tools/prepush.sh` (after the Build step, and after the graph-io fast-test block)

- [ ] **Step 1: The bundle step (D-20)**

After the line `run_step "Build" "pnpm -r run build"` insert:

```bash

# webgpu-graph-algorithms: its lint runs the strict-consumer compile against the d.ts shims that only
# build:bundle writes (tsc emits none; the package has no root entry file), so bundle it before Lint
run_step "Bundle webgpu-graph-algorithms" "(cd webgpu-graph-algorithms && npm run build:bundle)"
```

- [ ] **Step 2: The fast-test line (D-13)**

After the graph-io block (`(cd graph-io && npm run test:run) || { FAILED=1; TESTS_FAILED=1; }`) and its blank line insert:

```bash
# webgpu-graph-algorithms - the node project only (design 12.5): Dawn on the local adapter -- NVIDIA when
# LD_LIBRARY_PATH carries the libEGL tree (package CLAUDE.md), else Mesa lavapipe (about 5 minutes); the
# browser project and the no-subgroups pass run in CI. GRAPHTY_GPU_REQUIRE=any: a machine with no adapter
# fails up front instead of skipping every GPU test and reporting a vacuous pass
echo "  Testing webgpu-graph-algorithms..."
(cd webgpu-graph-algorithms && GRAPHTY_GPU_REQUIRE=any npm run test:run) || { FAILED=1; TESTS_FAILED=1; }

```

- [ ] **Step 3: Run the two new lines by hand**

Run: `cd WT && (cd webgpu-graph-algorithms && npm run build:bundle) && (cd webgpu-graph-algorithms && GRAPHTY_GPU_REQUIRE=any npm run test:run); echo "exit=$?"`
Expected: `exit=0` (the node project on whatever adapter Dawn finds: lavapipe on the dev box unless `LD_LIBRARY_PATH` carries the libEGL tree; with `any` a machine that exposes no adapter fails in the global setup rather than skipping -- then check `ls /usr/share/vulkan/icd.d`).

### Task M2-T6: Coverage merge

**Files:**
- Modify: `tools/merge-coverage.sh:29,76-83,221-228`

- [ ] **Step 1: The package list and the two hint lists**

Line 29: `PACKAGES=("graph-format" "graph-io" "webgpu-graph-algorithms" "algorithms" "layout" "graphty" "graphty-element")`. In both echo lists insert `  cd webgpu-graph-algorithms && pnpm run coverage` after the graph-io line (the `echo "..."` form at 76-83 and the `log_error "..."` form at 221-228). The file's `THIS FILE IS AUTO GENERATED` header is wrong (no generator exists; `0069386c` edited it directly) -- edit it directly and leave the header for a separate cleanup.

- [ ] **Step 2: Verify locally**

Run: `cd WT/webgpu-graph-algorithms && GRAPHTY_GPU_ADAPTER=llvmpipe GRAPHTY_GPU_REQUIRE=any VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json XDG_RUNTIME_DIR=/tmp pnpm run coverage && cd WT && (cd graph-format && pnpm run coverage) && (cd graph-io && pnpm run coverage) && ./tools/merge-coverage.sh 2>&1 | tail -15`
Expected: the package's coverage run passes its 80/80/75/80 thresholds (94/94/98/94 on the last CI run); the merge script lists `webgpu-graph-algorithms` among the collected packages and writes `coverage/lcov.info` (the other packages' missing coverage is only a warning outside `--ci`).

### Task M2-T7: Root CLAUDE.md and README.md

**Files:**
- Create: `tools/apply-root-claude-md-webgpu.py`
- Modify: `CLAUDE.md` (root, through the script, plus two manual edits)
- Modify: `README.md` (root, a package section)

- [ ] **Step 1: Write the apply script**

The root `CLAUDE.md` tree and build-order lines carry non-ASCII glyphs, so the script spells them as `chr()` code points (the precedent `packages/move/apply-root-claude-md.py` of the staging repository; this file is plain ASCII). Every edit anchors on a line that must occur exactly once; the script refuses to write when an anchor is missing or ambiguous and is idempotent.

```python
#!/usr/bin/env python3
"""Apply the webgpu-graph-algorithms edits to the monorepo root CLAUDE.md.

Two of the edited lines carry non-ASCII characters (the box-drawing tree and the arrows of the
build-order line); the script itself is plain ASCII and spells them as chr() code points. Every
edit anchors on a line that must occur exactly once; nothing is written if an anchor is missing
or ambiguous; a second run changes nothing.

Usage (from the monorepo root):   python3 tools/apply-root-claude-md-webgpu.py CLAUDE.md
"""

import sys

BOX_BRANCH = chr(0x251C) + chr(0x2500) + chr(0x2500)  # the tree prefix of the Monorepo Structure block
ARROW = chr(0x2192)  # the arrow of the build-order line

# (anchor line, lines inserted BEFORE the anchor, lines inserted AFTER the anchor)
INSERTIONS = [
    (
        '| `@graphty/graph-io` (and `@graphty/graph-io/<format>` subpaths: gexf, graphml, gml, dot, pajek, csv, json, neo4j)'
        ' | **graph-io** | "io", "importers" |',
        [],
        [
            '| `@graphty/webgpu-graph-algorithms` (and `@graphty/webgpu-graph-algorithms/browser`, `/node` subpaths)'
            ' | **webgpu-graph-algorithms** | "webgpu", "the GPU package", "the GPU layout" |',
        ],
    ),
    (
        "| `@graphty/graph-io` | `graph-io/` | 0.1.0 | Importers and exporters (GEXF, GraphML, GML, DOT, Pajek, CSV, "
        "JSON, Neo4j) for the graph-format snapshot; subpath exports per format |",
        [],
        [
            "| `@graphty/webgpu-graph-algorithms` | `webgpu-graph-algorithms/` | 0.1.0 | WebGPU-accelerated graph "
            "algorithms and layouts (ForceAtlas2 first) over the graph-format snapshot, for Node (Dawn) and browsers; "
            "never falls back to the CPU |",
        ],
    ),
    (
        BOX_BRANCH + " graph-io/             # @graphty/graph-io package (depends on graph-format)",
        [],
        [BOX_BRANCH + " webgpu-graph-algorithms/  # @graphty/webgpu-graph-algorithms package (depends on graph-format)"],
    ),
    (
        "pnpm run coverage:preview:graph-io         # Port 9057",
        [],
        ["pnpm run coverage:preview:webgpu-graph-algorithms  # Port 9058"],
    ),
    (
        "- Coverage previews: 9051-9054, graph-format 9056, graph-io 9057, webgpu-graph-algorithms 9058",
        ["- webgpu-graph-algorithms demo (vite): 9030"],
        [],
    ),
    (
        "**graphty:**",
        [
            "**webgpu-graph-algorithms:**",
            "- `node` - Node.js on Dawn (`GRAPHTY_GPU_REQUIRE` unset skips without an adapter; CI sets `any` on lavapipe)",
            "- `node-limits` - the GPU lane only (real device limits)",
            "- `browser` - Playwright Chromium with the `GRAPHTY_BROWSER_GPU` flag set (swiftshader in CI) through "
            "`scripts/run-browser-project.js`",
            "",
        ],
        [],
    ),
    (
        "- `graph-io`",
        [],
        ["- `webgpu-graph-algorithms-node`, `webgpu-graph-algorithms-browser`"],
    ),
    (
        "| `deploy-pages.yml` | After CI | Deploy docs to GitHub Pages |",
        [],
        [
            "| `gpu.yml` | Dispatch and labelled same-repo PRs; push and nightly once the `gpu-linux-t4` runner exists "
            "| The webgpu-graph-algorithms NVIDIA T4 lane; never a job of CI, never required |",
            "| `hosts.yml` | Push/PR touching `webgpu-graph-algorithms/` or `graph-format/`, dispatch "
            "| Informational host matrix: Dawn on Metal + WebKit (macOS), Dawn on D3D12 WARP + Chromium (Windows) |",
        ],
    ),
    (
        "- `graph-io/CLAUDE.md` - Importer / exporter contract, adding a format",
        [],
        [
            "- `webgpu-graph-algorithms/CLAUDE.md` - The GPU context and adapter policy, the kernel layers, the lanes "
            "and their environment variables, verified platform facts",
        ],
    ),
    (
        "### TypeScript",
        [
            "### WebGPU",
            "",
            "- Never create fallbacks if WebGPU isn't supported. The GPU package throws (`E_NO_WEBGPU`, `E_NO_ADAPTER`, "
            "`E_TOO_LARGE`, ...) and never runs a CPU path; the CPU packages' dispatchers choose the CPU only when no "
            "accelerator was injected (`design/webgpu/webgpu-acceleration-plan.md` section 2.4).",
            "",
        ],
        [],
    ),
    (
        "- `ci-parity-plan.md` - CI/CD alignment plan",
        [],
        [
            "- `graph-format/graph-format-design.md` - The shared graph data format and the consumer migration",
            "- `webgpu/webgpu-acceleration-plan.md` - WebGPU acceleration: the design, `webgpu/plans/` the contract, "
            "the phase plans and the monorepo integration plan",
        ],
    ),
]

# (anchor line, replacement line)
REPLACEMENTS = [
    (
        "- Coverage previews: 9051-9054, graph-format 9056, graph-io 9057",
        "- Coverage previews: 9051-9054, graph-format 9056, graph-io 9057, webgpu-graph-algorithms 9058",
    ),
    (
        "- gpu-3d-force-layout: 9060",
        "- compact-mantine Storybook: 9060",
    ),
    (
        "| `ci.yml` | Push/PR | Build, lint, sharded tests (18 parallel jobs) |",
        "| `ci.yml` | Push/PR | Build, lint, sharded tests (20 parallel jobs) |",
    ),
    (
        "The CI runs 18 parallel test jobs:",
        "The CI runs 20 parallel test jobs:",
    ),
    (
        "  - Build order enforced by TypeScript: `graph-format` " + ARROW + " `graph-io` " + ARROW + " `algorithms` "
        + ARROW + " `layout` " + ARROW + " `graphty-element` " + ARROW + " `graphty`",
        "  - Build order enforced by TypeScript: `graph-format` " + ARROW + " `graph-io` " + ARROW
        + " `webgpu-graph-algorithms` " + ARROW + " `algorithms` " + ARROW + " `layout` " + ARROW
        + " `graphty-element` " + ARROW + " `graphty`",
    ),
]


def index_of(lines, anchor):
    """The index of the one line equal to `anchor`, or an error naming the problem."""
    hits = [i for i, line in enumerate(lines) if line == anchor]
    if len(hits) != 1:
        raise SystemExit(f"anchor found {len(hits)} times (need exactly 1): {anchor!r}")
    return hits[0]


def apply(text):
    lines = text.split("\n")
    # replacements first: one insertion anchors on a replaced line
    for anchor, replacement in REPLACEMENTS:
        if replacement in lines and anchor not in lines:
            continue  # already applied
        lines[index_of(lines, anchor)] = replacement
    for anchor, before, after in INSERTIONS:
        new_lines = [line for line in before + after if line != ""]
        if new_lines and all(line in lines for line in new_lines):
            continue  # already applied
        i = index_of(lines, anchor)
        lines[i:i + 1] = before + [anchor] + after
    return "\n".join(lines)


def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: apply-root-claude-md-webgpu.py <path to the monorepo root CLAUDE.md>")
    path = argv[1]
    with open(path, encoding="utf-8") as handle:
        original = handle.read()
    updated = apply(original)
    if updated == original:
        print("CLAUDE.md already up to date")
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(updated)
    print(f"CLAUDE.md updated ({updated.count(chr(10)) - original.count(chr(10))} lines added)")


if __name__ == "__main__":
    main(sys.argv)
```

- [ ] **Step 2: Run it twice**

Run: `cd WT && python3 tools/apply-root-claude-md-webgpu.py CLAUDE.md && python3 tools/apply-root-claude-md-webgpu.py CLAUDE.md`
Expected: `CLAUDE.md updated (N lines added)` then `CLAUDE.md already up to date`. If it refuses with `anchor found 0 times`, the root file drifted since 2026-09-16: `grep -n` for the nearest text, fix the anchor string in the script, re-run.

- [ ] **Step 3: The README section**

In the root `README.md` insert after the `[View package](./graph-io)` block's `---` line (before `### @graphty/remote-logger`):

```markdown
### @graphty/webgpu-graph-algorithms

[![npm version](https://img.shields.io/npm/v/@graphty/webgpu-graph-algorithms.svg)](https://www.npmjs.com/package/@graphty/webgpu-graph-algorithms)

WebGPU-accelerated graph algorithms and force-directed layouts over the graph-format snapshot, for Node (Google Dawn) and browsers: a GPU context with an explicit adapter policy, upload planning and readback over the CSR arena, a kernel layer with WGSL composition, and ForceAtlas2 as a steppable layout simulation. Throws when no WebGPU device exists; never falls back to the CPU.

[View package](./webgpu-graph-algorithms)

---

```

- [ ] **Step 4: Check ASCII of everything this task wrote**

Run: `cd WT && LC_ALL=C grep -nP '[^\x00-\x7F]' tools/apply-root-claude-md-webgpu.py README.md | head; git diff -U0 CLAUDE.md | grep '^+' | LC_ALL=C grep -nP '[^\x00-\x7F]'`
Expected: nothing from the first command; the second prints ONLY the two lines that legitimately carry the tree glyph and the arrows (the inserted tree line and the replaced build-order line) -- no other added line is non-ASCII.

### Task M2-T8: Local verification of the whole wiring

- [ ] **Step 1: The nx targets**

Run: `cd WT && pnpm exec nx run-many -t lint,build --projects=webgpu-graph-algorithms --parallel=1 && pnpm exec nx run webgpu-graph-algorithms:test`
Expected: lint (after its build dependency) and build succeed; `test` runs the node project on the local adapter with the `node` project's own reporter (verbose locally).

- [ ] **Step 2: The frozen lockfile**

Run: `cd WT && HUSKY=0 pnpm install --frozen-lockfile && git status --short pnpm-lock.yaml`
Expected: exit 0 and `M pnpm-lock.yaml` (the file regenerated in M1-T4 step 3, unchanged by the frozen install). The regeneration also touches unrelated entries (`@babel/parser`, `@babel/types`, `tinyglobby` drift on 2026-09-16); expected, do not revert.

- [ ] **Step 3: knip, prettier, audit**

Run: `cd WT && pnpm exec knip; echo "knip=$?"; pnpm exec prettier --check knip.config.ts package.json README.md pnpm-workspace.yaml webgpu-graph-algorithms/project.json .github/workflows/release.yml && bash -n tools/prepush.sh tools/merge-coverage.sh; pnpm audit --audit-level=high; echo "audit=$?"`
Expected: `knip=0` with no findings (master is clean today), prettier clean on the listed files, `bash -n` silent, `audit=0` (38 low/moderate today, none high). Not listed on purpose: `commitlint.config.js` (its tab-indented duplicate entries are not prettier-clean on master and M2-T1 leaves them alone), `.github/workflows/ci.yml` (one pre-existing `all-checks` `needs:` line exceeds the print width; M3-T1 step 7 covers the file), and shell scripts (prettier has no parser for them).

- [ ] **Step 4: The pre-push gate itself**

Run: `cd WT && ./tools/prepush.sh; echo "prepush=$?"`
Expected: `prepush=0` in about 15-25 minutes (build, lint, knip, every package's fast tests including the new line, graphty's browser suite). A failure in `remote-logger`'s random-port tests is the known ECONNRESET flake -- re-run once. Any failure under `webgpu-graph-algorithms` is this plan's to fix before the commit.

### Task M2-T9: Commit Phases M1-M2 (owner)

- [ ] **Step 1: Dry-run every step**

Tell the owner: `! cd /home/apowers/Projects/graphty-monorepo/.worktrees/land-webgpu-graph-algorithms && for s in workspace package docs ignore; do ./tools/land-webgpu-graph-algorithms.sh commit $s --dry-run || break; done`
Expected: each step lists its files with non-zero counts and `DRY RUN: nothing staged`; no `REFUSING:` line. `.github/workflows/*` changes, if already present from Phase M3, are claimed by the `ci` step and do not block.

- [ ] **Step 2: Commit the four steps**

Tell the owner: `! cd /home/apowers/Projects/graphty-monorepo/.worktrees/land-webgpu-graph-algorithms && for s in workspace package docs ignore; do ./tools/land-webgpu-graph-algorithms.sh commit $s || break; done`
Expected: four `<sha> signed=G <subject>` lines; commitlint accepts every message (the `workspace` commit adds the scope before the `package` commit uses it). Then `git log --oneline -6` shows them on top of the merge.

---

## Phase M3: CI/CD port, the landing, the first release

All edits in `WT`; the design's 12.5 diff is re-derived against the file as it reads today (its line references are from 2026-09-14).

### Task M3-T1: The two software shards in ci.yml

**Files:**
- Modify: `.github/workflows/ci.yml` (build job PR step, build upload, matrix shard list, matrix include, test-job download, two lavapipe steps, a coverage upload step)

- [ ] **Step 1: Build the package on PRs**

Replace the step

```yaml
            - name: Build graph-format and graph-io (PR)
              if: github.event_name == 'pull_request'
              run: pnpm exec nx run-many -t build --projects=graph-format,graph-io --parallel=2
```

with

```yaml
            - name: Build graph-format, graph-io and webgpu-graph-algorithms (PR)
              if: github.event_name == 'pull_request'
              run: pnpm exec nx run-many -t build --projects=graph-format,graph-io,webgpu-graph-algorithms --parallel=3
```

and in the comment block above it change `graph-format and graph-io are not yet in graphty's dependency closure` to `graph-format, graph-io and webgpu-graph-algorithms are not yet in graphty's dependency closure`.

- [ ] **Step 2: Upload the build**

After the `Upload graph-io build` step add:

```yaml
            - name: Upload webgpu-graph-algorithms build
              uses: actions/upload-artifact@v4
              with:
                  name: build-webgpu-graph-algorithms
                  path: webgpu-graph-algorithms/dist/
                  retention-days: 1
```

- [ ] **Step 3: The matrix**

In `strategy.matrix.shard` insert after `                    - graph-io`:

```yaml
                    - webgpu-graph-algorithms-node
                    - webgpu-graph-algorithms-browser
```

In `strategy.matrix.include` insert after the `graph-io` include entry (before `# algorithms - two shards`):

```yaml
                    # webgpu-graph-algorithms - two shards (design/webgpu/webgpu-acceleration-plan.md 12.5): the node
                    # project on Dawn over Mesa lavapipe with coverage (thresholds active) plus the no-subgroups twins
                    # pass, and the Chromium SwiftShader browser smoke. The node shard calls vitest directly: the
                    # nx -> npm -> vitest pipe chain starved the worker RPC for graph-format and graph-io (6fc56c1b).
                    # needs-lavapipe gates the apt install and the ICD lookup that export the lane's environment.
                    - shard: webgpu-graph-algorithms-node
                      package: webgpu-graph-algorithms
                      test-command: cd webgpu-graph-algorithms && pnpm exec vitest run --project=node --coverage && GRAPHTY_GPU_NO_SUBGROUPS=1 pnpm exec vitest run --project=node test/primitives test/layouts --passWithNoTests
                      needs-browser: false
                      needs-storybook: false
                      needs-lavapipe: true
                    # the browser smoke: scripts/run-browser-project.js wraps vitest in `timeout -k 10 600` and passes a
                    # timeout iff every test passed (browser.close() can hang after GPU work); SwiftShader flags come
                    # from the package's vitest.config.ts, not from CI
                    - shard: webgpu-graph-algorithms-browser
                      package: webgpu-graph-algorithms
                      test-command: cd webgpu-graph-algorithms && GRAPHTY_BROWSER_GPU=swiftshader GRAPHTY_GPU_REQUIRE=any node scripts/run-browser-project.js
                      needs-browser: true
                      needs-storybook: false
```

- [ ] **Step 4: Download the build in every shard**

After the `Download graph-io build` step of the test job add:

```yaml
            - name: Download webgpu-graph-algorithms build
              uses: actions/download-artifact@v4
              with:
                  name: build-webgpu-graph-algorithms
                  path: webgpu-graph-algorithms/dist/
```

- [ ] **Step 5: The lavapipe steps**

Immediately before `- name: Run tests` (that is, after the `Start Storybook server` step, which sits between the Playwright steps and the test step) insert:

```yaml
            # webgpu-graph-algorithms node shard: Dawn needs a Vulkan ICD; ubuntu-24.04 has none preinstalled
            - name: Install Mesa lavapipe
              if: matrix.needs-lavapipe
              run: sudo apt-get update && sudo apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1 vulkan-tools

            - name: Locate the lavapipe ICD and export the lane's environment
              if: matrix.needs-lavapipe
              run: |
                  # find, not ls with globs: the ICD's file name moved between Mesa releases (lvp_icd.json on noble,
                  # lvp_icd.x86_64.json on jammy); an unmatched glob makes ls exit 2 under bash -e
                  icd=$(find /usr/share/vulkan/icd.d /etc/vulkan/icd.d /usr/lib/x86_64-linux-gnu/vulkan/icd.d -name 'lvp_icd*.json' 2>/dev/null | sort | head -1 || true)
                  if [ -z "$icd" ]; then
                      echo "::error::no lavapipe ICD json after installing mesa-vulkan-drivers"
                      dpkg -L mesa-vulkan-drivers | grep -i 'json\|lvp\|lavapipe' || true
                      exit 1
                  fi
                  echo "lavapipe ICD: $icd"
                  VK_DRIVER_FILES="$icd" vulkaninfo --summary 2>/dev/null | head -40 || true
                  {
                      echo "VK_DRIVER_FILES=$icd"
                      echo "GRAPHTY_GPU_ADAPTER=llvmpipe"
                      echo "GRAPHTY_GPU_REQUIRE=any"
                      echo "XDG_RUNTIME_DIR=/tmp"
                  } >> "$GITHUB_ENV"
```

(`matrix.needs-lavapipe` is unset for every other shard, which the `if:` treats as false, so no other shard runs `apt-get`. The environment reaches the `Run tests` step through `$GITHUB_ENV`; `test-command` can therefore stay a single string.)

- [ ] **Step 6: The coverage artifact**

After the shared `Upload coverage (graph-format/graph-io/algorithms/layout/graphty/remote-logger/compact-mantine)` step add a dedicated step (the shared one has `if-no-files-found: ignore`; the node shard's coverage must never be silently missing, because Task M2-T6 made it mandatory for `merge-coverage.sh --ci`):

```yaml
            - name: Upload coverage (webgpu-graph-algorithms)
              if: ${{ !cancelled() && matrix.shard == 'webgpu-graph-algorithms-node' }}
              uses: actions/upload-artifact@v4
              with:
                  name: coverage-${{ matrix.shard }}
                  path: webgpu-graph-algorithms/coverage/lcov.info
                  retention-days: 1
                  if-no-files-found: error
                  overwrite: true
```

`coverage-webgpu-graph-algorithms-node` matches `tools/merge-coverage.sh`'s `coverage-<pkg>-*` rule for the package `webgpu-graph-algorithms` and cannot match `algorithms` (whose rule needs the prefix `coverage-algorithms-`).

- [ ] **Step 7: Syntax check**

Run: `cd WT && node -e 'const fs=require("fs"); const y=fs.readFileSync(".github/workflows/ci.yml","utf8"); const d=require("yaml").parse(y); const m=d.jobs.test.strategy.matrix; for (const s of ["webgpu-graph-algorithms-node","webgpu-graph-algorithms-browser"]) if (!m.shard.includes(s) || !m.include.some(e=>e.shard===s)) throw new Error("missing shard "+s); if (m.include.find(e=>e.shard==="webgpu-graph-algorithms-node")["needs-lavapipe"]!==true) throw new Error("needs-lavapipe"); for (const s of ["build-webgpu-graph-algorithms","Upload coverage (webgpu-graph-algorithms)","Install Mesa lavapipe"]) if (!y.includes(s)) throw new Error("missing "+s); console.log("ok", m.shard.length, "shards")'`
Expected: `ok 20 shards` (the `yaml` package is resolvable from the worktree root). Do NOT use `prettier --check` here: `ci.yml` carries one pre-existing warning on master (the `all-checks` job's `needs:` line exceeds the print width), and a `--write` would drag that unrelated reflow into the `ci` commit.

### Task M3-T2: The release download

**Files:**
- Modify: `.github/workflows/release.yml` (after `Download graph-io build`)

- [ ] **Step 1: Add the step**

```yaml
            - name: Download webgpu-graph-algorithms build
              uses: actions/download-artifact@v4
              with:
                  name: build-webgpu-graph-algorithms
                  path: webgpu-graph-algorithms/dist/
                  run-id: ${{ github.event.workflow_run.id }}
                  github-token: ${{ secrets.GITHUB_TOKEN }}
```

(`release.yml` publishes from the downloaded `dist/` without rebuilding; without this step the tarball's `dist/` would be empty.)

- [ ] **Step 2: Check**

Run: `cd WT && pnpm exec prettier --check .github/workflows/release.yml && grep -c build-webgpu-graph-algorithms .github/workflows/release.yml`
Expected: prettier clean (the file is clean on master), `1`.

### Task M3-T3: gpu.yml

**Files:**
- Create: `.github/workflows/gpu.yml`

- [ ] **Step 1: Write the workflow**

The staging `gpu.yml` with the three changes of design 12.5 (paths without `packages/`, its own install and nx build on the runner, `pnpm/action-setup@v4` without `package_json_file`), the push and schedule triggers still commented out (D-15), and the tracking-issue text citing the design at its new path:

```yaml
# .github/workflows/gpu.yml -- the webgpu-graph-algorithms GPU lane: never required, never a job of CI
# (release.yml and coverage.yml are workflow_run on CI's conclusion, and a job whose runner is offline queues for
# 24 hours). design/webgpu/webgpu-acceleration-plan.md sections 12.1-12.6; the staging lane ported with three
# changes: paths without the packages/ prefix, its own install and nx build on the runner (no artifact download,
# no needs: on CI), and pnpm/action-setup without package_json_file (the root package.json has packageManager).
name: GPU
on:
    # push and the nightly schedule are OFF until the hosted gpu-linux-t4 runner exists (design 12.4, an owner item;
    # integration plan Phase M4): a job whose runner does not exist queues for 24 h and is then cancelled, on every
    # push. Restore both lines when the runner is registered and the monorepo is in runner group `gpu`:
    #   push: { branches: [master] }
    #   schedule: [{ cron: "17 6 * * *" }]          # nightly
    pull_request: { types: [labeled, synchronize] }
    workflow_dispatch:
permissions: { contents: read }
concurrency:
    group: gpu-lane-${{ github.event.pull_request.number || github.ref }}
    cancel-in-progress: ${{ github.event_name == 'pull_request' }}

jobs:
    changed: # nightly cost guard: skip when master has not moved since the last green GPU run (design 12.6)
        runs-on: ubuntu-latest
        outputs:
            run: ${{ steps.check.outputs.run }}
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
        runs-on: gpu-linux-t4 # the hosted runner of design 12.4 (runner group `gpu`); machine.dev alternative: `machine/gpu=t4`
        timeout-minutes: 45
        env:
            GRAPHTY_GPU_REQUIRE: nvidia # a software adapter fails the job
            GRAPHTY_BROWSER_GPU: nvidia # --enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan --disable-vulkan-surface
            GRAPHTY_RUNNER_CLASS: gpu-linux-t4 # the benchmark runner class of the T4 lane is the runner name
            XDG_RUNTIME_DIR: /tmp
        defaults: { run: { working-directory: webgpu-graph-algorithms, shell: bash } }
        steps:
            - uses: actions/checkout@v4
            - name: Driver up (the partner image may need the modules loaded; G0 records whether this step is a no-op)
              run: sudo modprobe nvidia nvidia_uvm || true; nvidia-smi
            - uses: pnpm/action-setup@v4
            - uses: actions/setup-node@v4
              with: { node-version: 22.x, cache: pnpm }
            - run: pnpm install --frozen-lockfile
              working-directory: .
              env: { HUSKY: "0" }
            - name: Build the package and what it depends on (nx follows ^build; the negated implicit dependencies keep algorithms and layout out)
              run: pnpm exec nx run-many -t build --projects=graph-format,webgpu-graph-algorithms
              working-directory: .
            - name: Adapter report (fails loudly on a software adapter; samples GPU utilisation for 10 s)
              run: node scripts/gpu-report.js > gpu-report.json && cat gpu-report.json # no pipe into tee: the report's non-zero exit must reach the step
            - name: graph-format GPU audit on NVIDIA (canary)
              working-directory: graph-format
              run: pnpm exec vitest run test/audit/gpu-upload.test.ts 2>&1 | tee canary.log && grep -q "adapter vendor=nvidia" canary.log # the audit skips on acquisition failure; the grep turns a skip or a lavapipe pick into a red step
            - run: pnpm exec vitest run --project=node --project=node-limits
            - run: GRAPHTY_GPU_NO_SUBGROUPS=1 pnpm exec vitest run --project=node # the whole node project on the twins
            - name: Cache Playwright browsers
              id: pw
              uses: actions/cache@v4
              with: { path: ~/.cache/ms-playwright, key: "playwright-${{ runner.os }}-${{ hashFiles('pnpm-lock.yaml') }}" }
            - if: steps.pw.outputs.cache-hit != 'true'
              run: pnpm exec playwright install chromium --with-deps # a fresh VM each job
            - name: Browser smoke on NVIDIA
              run: node scripts/run-browser-project.js # G0 decides whether this needs `xvfb-run -a` on the T4 image (design 12.2)
            - run: pnpm run bench # tsx benchmarks/run.ts -> benchmarks/out/gpu-linux-t4.json
            - run: node scripts/bench-compare.js # > 3x the checked-in baseline for THIS runner class (benchmarks/results/gpu-linux-t4.json) fails; no baseline -> every row "new", exit 0
            - uses: actions/upload-artifact@v4
              if: ${{ !cancelled() }}
              with: { name: "gpu-results-${{ github.run_id }}", path: "webgpu-graph-algorithms/gpu-report.json\nwebgpu-graph-algorithms/benchmarks/out/", retention-days: 90, overwrite: true }

    gpu-nightly-report: # the only job with a write permission; runs on a standard runner, never on the paid one
        needs: test-gpu
        if: always() && github.event_name == 'schedule' && needs.test-gpu.result != 'success'
        runs-on: ubuntu-latest
        permissions: { issues: write, actions: read } # actions: read for listWorkflowRuns (the previous night's conclusion)
        steps:
            - uses: actions/github-script@v7
              with:
                  script: |
                      // Open or refresh the "GPU lane nightly" tracking issue, only after two consecutive
                      // nightly failures (design 12.6). The previous night's result is read from the workflow
                      // runs themselves rather than from the issue body: it needs no state to be kept in an
                      // issue and it is right even when the issue was closed by hand in between.
                      const { owner, repo } = context.repo;
                      const title = "GPU lane nightly";
                      const runUrl = `${context.serverUrl}/${owner}/${repo}/actions/runs/${context.runId}`;
                      const runs = await github.rest.actions.listWorkflowRuns({
                          owner, repo, workflow_id: "gpu.yml", event: "schedule", per_page: 5,
                      });
                      const previous = runs.data.workflow_runs
                          .filter((r) => r.id !== context.runId && r.status === "completed")
                          .sort((a, b) => (a.created_at < b.created_at ? 1 : -1))[0];
                      const twoInARow = previous !== undefined && previous.conclusion !== "success";
                      const { data: open } = await github.rest.issues.listForRepo({ owner, repo, state: "open", per_page: 100 });
                      const existing = open.find((i) => i.title === title);
                      if (!twoInARow && existing === undefined) {
                          core.info(`first nightly failure (${runUrl}); no issue yet -- a second consecutive failure opens one`);
                          return;
                      }
                      const line = `- ${new Date().toISOString()}: nightly GPU lane failed: ${runUrl}` +
                          (previous === undefined ? "" : ` (previous night: ${previous.conclusion}, ${previous.html_url})`);
                      if (existing !== undefined) {
                          await github.rest.issues.createComment({ owner, repo, issue_number: existing.number, body: line });
                          core.info(`refreshed #${existing.number}`);
                      } else {
                          const body = "The nightly GPU lane (gpu.yml, the gpu-linux-t4 runner) has failed on two consecutive " +
                              "nights. Close this issue once a nightly run is green again; the job comments here on " +
                              "every further failure while it stays open (design/webgpu/webgpu-acceleration-plan.md 12.6).\n\n" + line;
                          const created = await github.rest.issues.create({ owner, repo, title, body });
                          core.info(`opened #${created.data.number}`);
                      }
```

Do NOT dispatch this workflow before Phase M4: with no `gpu-linux-t4` runner the `test-gpu` job queues for 24 hours. The `gpu` label does not exist in the monorepo yet, so the labelled-PR path cannot fire either.

- [ ] **Step 2: Syntax check**

Run: `cd WT && pnpm exec prettier --check .github/workflows/gpu.yml || pnpm exec prettier --write .github/workflows/gpu.yml`
Expected: clean (prettier may reflow the flow-style mappings; that is fine).

### Task M3-T4: hosts.yml

**Files:**
- Create: `.github/workflows/hosts.yml`

- [ ] **Step 1: Write the workflow**

The staging `hosts.yml` (green on `a66c889`) with the paths rewritten, the root install, an nx build, and `paths:` filters so other packages' changes do not spend two ~9-minute macOS/Windows jobs:

```yaml
# .github/workflows/hosts.yml -- the webgpu-graph-algorithms host matrix: the other shader compilers (Dawn on Metal
# and on D3D12, WebKit). Informational, never a required check (design/webgpu/webgpu-acceleration-plan.md Review
# log, 2026-09-16): the default lane covers Tint on Vulkan only, and a real WebKit / Metal defect (unread pipeline
# constants) was first seen on an iPad. macos-latest is an Apple Silicon VM with a Metal device (Dawn-node and
# Chromium on Metal; Playwright's WebKit 26 exposes WebGPU on it too); windows-latest has no GPU but Dawn's D3D12
# backend enumerates WARP, Microsoft's software rasterizer, for Dawn-node and for the full Chromium build.
name: Hosts
on:
    push:
        branches: [master]
        paths: ["webgpu-graph-algorithms/**", "graph-format/**", ".github/workflows/hosts.yml", "pnpm-lock.yaml"]
    pull_request:
        paths: ["webgpu-graph-algorithms/**", "graph-format/**", ".github/workflows/hosts.yml", "pnpm-lock.yaml"]
    workflow_dispatch:
permissions: { contents: read }
concurrency:
    group: hosts-${{ github.ref }}
    cancel-in-progress: ${{ github.event_name == 'pull_request' }}

jobs:
    test:
        name: Test (${{ matrix.backend }} on ${{ matrix.os }})
        strategy:
            fail-fast: false
            matrix:
                include:
                    - os: macos-latest
                      backend: metal
                      node_adapter: "" # Dawn picks the runner's Metal device
                      browser_gpu: metal
                      webkit: "true"
                    - os: windows-latest
                      backend: d3d12
                      node_adapter: "Microsoft Basic Render Driver" # WARP, the software D3D12 adapter
                      browser_gpu: warp # the full Chromium build on Dawn's D3D12 backend over WARP (the headless shell grants nothing here)
                      webkit: "false"
        runs-on: ${{ matrix.os }}
        timeout-minutes: 45
        env:
            GRAPHTY_GPU_REQUIRE: any # an adapter must exist; a missing one fails the run up front (test/setup/global.ts)
            GRAPHTY_GPU_ADAPTER: ${{ matrix.node_adapter }}
            GRAPHTY_BROWSER_GPU: ${{ matrix.browser_gpu }}
        defaults: { run: { working-directory: webgpu-graph-algorithms, shell: bash } }
        steps:
            - uses: actions/checkout@v4
            - uses: pnpm/action-setup@v4
            - uses: actions/setup-node@v4
              with: { node-version: 22.x, cache: pnpm }
            - run: pnpm install --frozen-lockfile
              working-directory: .
              env: { HUSKY: "0" }
            - run: pnpm exec nx run-many -t build --projects=graph-format,webgpu-graph-algorithms
              working-directory: .
            - name: Adapter report (informational on this lane; the JSON is uploaded)
              run: node scripts/gpu-report.js > gpu-report.json || echo "gpu-report.js exited $?"; cat gpu-report.json
            # Every step from here on runs whatever the earlier ones did (`!cancelled()`), so one push reports every
            # lane's node suite, twins pass, flag probe and browser smokes at once instead of one finding per round.
            - name: Node suite (Dawn on ${{ matrix.backend }})
              run: pnpm exec vitest run --project=node
            - name: Node suite, no-subgroups twins
              if: ${{ !cancelled() }}
              run: GRAPHTY_GPU_NO_SUBGROUPS=1 pnpm exec vitest run --project=node test/primitives test/layouts --passWithNoTests
            - name: Crash reports (macOS -- a vitest worker that vanished with "Channel closed" left its native stack here)
              if: ${{ !cancelled() && runner.os == 'macOS' }}
              run: node scripts/print-crash-reports.mjs
            - name: Crash reports (Windows -- the Application log's faulting module and exception code of a vanished worker)
              if: ${{ !cancelled() && runner.os == 'Windows' }}
              shell: pwsh
              run: |
                  $since = (Get-Date).AddMinutes(-40)
                  $events = Get-WinEvent -FilterHashtable @{ LogName = 'Application'; StartTime = $since } -ErrorAction SilentlyContinue |
                      Where-Object { $_.ProviderName -in @('Application Error', 'Windows Error Reporting', '.NET Runtime') }
                  if (-not $events) { Write-Output "no Application Error / Windows Error Reporting event since $since"; exit 0 }
                  foreach ($e in $events) {
                      Write-Output "==> $($e.TimeCreated) $($e.ProviderName) id $($e.Id)"
                      Write-Output $e.Message
                  }
            - name: Cache Playwright browsers
              if: ${{ !cancelled() }}
              id: pw
              uses: actions/cache@v4
              with:
                  path: |
                      ~/.cache/ms-playwright
                      ~/AppData/Local/ms-playwright
                  key: "playwright-hosts-${{ runner.os }}-${{ hashFiles('pnpm-lock.yaml') }}"
            - if: ${{ !cancelled() && steps.pw.outputs.cache-hit != 'true' }}
              run: pnpm exec playwright install chromium
            - name: Probe Chromium flag sets (informational -- which flags grant which adapter on this host)
              if: ${{ !cancelled() }}
              continue-on-error: true
              run: node scripts/probe-browser-flags.mjs
            - name: Browser smoke (Chromium on ${{ matrix.backend }})
              if: ${{ !cancelled() }}
              run: node scripts/run-browser-project.js
            # WebKit 26 exposes WebGPU on the runner's Metal device and passes the whole browser project, so the step
            # gates the lane like the Chromium smoke.
            - name: Browser smoke (WebKit, the Safari proxy)
              if: ${{ !cancelled() && matrix.webkit == 'true' }}
              env: { GRAPHTY_BROWSER: webkit }
              run: pnpm exec playwright install webkit && node scripts/run-browser-project.js
            - uses: actions/upload-artifact@v4
              if: ${{ !cancelled() }}
              with:
                  name: "hosts-${{ matrix.backend }}-${{ github.run_id }}"
                  path: |
                      webgpu-graph-algorithms/gpu-report.json
                      webgpu-graph-algorithms/browser-results.json
                  if-no-files-found: ignore
                  retention-days: 14
                  overwrite: true
```

The Windows job depends on the `.gitattributes` of Task M1-T7 (LF checkouts: without it the source-entry test of `test/build-output.test.ts` failed on every entry in the staging repository, commit `11c39e9`).

- [ ] **Step 2: Syntax check**

Run: `cd WT && pnpm exec prettier --check .github/workflows/hosts.yml || pnpm exec prettier --write .github/workflows/hosts.yml`
Expected: clean.

### Task M3-T5: Commit the CI step and push (owner)

- [ ] **Step 1: Commit and push**

Tell the owner: `! cd /home/apowers/Projects/graphty-monorepo/.worktrees/land-webgpu-graph-algorithms && ./tools/land-webgpu-graph-algorithms.sh commit ci && ./tools/land-webgpu-graph-algorithms.sh push`
Expected: the `ci` commit; then the push runs the pre-push gate (`tools/prepush.sh`, 15-25 minutes) and pushes `land/webgpu-graph-algorithms`. On the known remote-logger ECONNRESET flake, re-run with `push --skip-gate` (the gate already passed in M2-T8).

- [ ] **Step 2: Open the PR (owner)**

Tell the owner: `! cd /home/apowers/Projects/graphty-monorepo && gh pr create --base master --head land/webgpu-graph-algorithms --title "feat(webgpu-graph-algorithms): land the WebGPU package, its design and its lanes" --body "Phases M1-M3 of design/webgpu/plans/2026-09-16-graphty-monorepo-integration.md: the package and its design corpus arrive with the staging repository's history; the root touch points, the two software shards, gpu.yml (triggers off until the runner exists) and hosts.yml follow. Merge by fast-forward from the landing worktree (tools/land-webgpu-graph-algorithms.sh land), never by squash or rebase: the branch's first commit is the history merge."`
Expected: the PR URL. The `lint-pr` job lints that title with the scope the branch adds.

### Task M3-T6: Make the PR green

- [ ] **Step 1: Watch the run**

Run: `cd /home/apowers/Projects/graphty-monorepo && gh run list --workflow CI --limit 20 --json databaseId,headBranch,status --jq '[.[] | select(.headBranch=="land/webgpu-graph-algorithms")][0].databaseId'` then `gh run watch <ci-run-id> --interval 60 --exit-status`; the same with `--workflow Hosts` for the host matrix; `gh pr checks land/webgpu-graph-algorithms` once at the end for the summary (gh 2.4.0 has no `--watch`).
Expected, in order: `Build` (about 10 minutes; builds the affected closure plus the three explicit packages, lints), `Test (webgpu-graph-algorithms-node)` (about 5 minutes: apt install 10-20 seconds, coverage run about 3.5 minutes with 94/94/98/94 coverage, the twins pass about 1.5 minutes), `Test (webgpu-graph-algorithms-browser)` (about 1.5 minutes: Playwright cache, 39 tests / 38 passed / 1 pending on SwiftShader), the 18 pre-existing shards, the five Chromatic jobs, `All Checks Pass`. `Hosts` runs too (the PR touches `webgpu-graph-algorithms/**`): both hosts green in about 9 minutes each. `GPU` runs only after a `synchronize` push (never for `opened`), and then only its `changed` job executes; `test-gpu` is skipped by its label guard, so nothing queues on the absent runner.

- [ ] **Step 2: Triage a failure**

For a red webgpu shard: `gh run view <run-id> --job <job-id> --log > tmp/land/<shard>.log` and read the `[gpu]` / `[run-browser-project]` lines (the adapter line comes first in every lane). The two expected new-environment differences: (a) lavapipe on ubuntu-24.04 is Mesa 25.2 (`vendor=mesa architecture=software ... subgroups=8-8`) -- the same as the staging lane, so the package's noise fixtures apply; (b) Playwright 1.57.0 instead of 1.63.0 (D-11): if `requestAdapter()` returns null on SwiftShader, run `node scripts/probe-browser-flags.mjs` in a dispatch of `hosts.yml` or locally, and adjust `BROWSER_FLAGS.swiftshader` in `vitest.config.ts` (a `package`-scoped follow-up commit through `tools/commit-changes.sh` on the branch). Pre-existing failures (M0-T2 option b) are recorded in the PR body, not fixed here.

- [ ] **Step 3: Gate**

`All Checks Pass` green (or, under M0-T2 option b, no job red that is green on master), both `Hosts` jobs green.

### Task M3-T7: Trusted publisher, landing, first release (owner)

- [ ] **Step 1: The trusted publisher, BEFORE landing**

`release.yml` publishes every releasable project on the first green master; a package without a trusted publisher fails `nx release publish` AFTER the tag exists. The owner runs (design Q-28; npm >= 11.15):

```bash
npx -y -p npm@11 npm trust github @graphty/webgpu-graph-algorithms --repo graphty-org/graphty-monorepo --file release.yml --allow-publish
```

and the same for `@graphty/graph-format` and `@graphty/graph-io` if not yet done (the memory note of 2026-09-15 says the three placeholders `0.0.0` exist and the trust command was pending). Verify: `npm view @graphty/webgpu-graph-algorithms --json | node -e 'const p=JSON.parse(require("fs").readFileSync(0,"utf8")); console.log(p.version, p.time && Object.keys(p.time))'` prints `0.0.0` (the placeholder) -- the trust relationship itself is visible on npmjs.com under the package's Settings > Trusted publishers.

- [ ] **Step 2: Land by fast-forward**

Tell the owner: `! cd /home/apowers/Projects/graphty-monorepo/.worktrees/land-webgpu-graph-algorithms && ./tools/land-webgpu-graph-algorithms.sh land --skip-gate`
(master has no copy of the script until this fast-forward; the `land` step switches itself to the main worktree. `--skip-gate` because the pre-push gate already ran on the branch push and the main worktree's `node_modules` predate the new importer.)
Expected: `master` fast-forwards to the branch tip and pushes. If master moved since the branch was cut, the script refuses: the owner merges `master` into the branch in the landing worktree (`git merge master`, a normal merge; conflicts are unlikely outside `CLAUDE.md` / `ci.yml`), the PR re-runs, then `land` again. Afterwards the owner runs `cd /home/apowers/Projects/graphty-monorepo && HUSKY=0 pnpm install --frozen-lockfile` in the main worktree.

- [ ] **Step 3: Watch master**

Run: `cd /home/apowers/Projects/graphty-monorepo && gh run list --limit 12 --json databaseId,name,headBranch,status,conclusion --jq '.[] | select(.headBranch=="master")'` then `gh run watch <ci-run-id> --exit-status`.
Expected: `CI` green on master (the full build, 20 shards, Chromatic), then `Coverage` (merges `coverage-webgpu-graph-algorithms-node` into the Coveralls upload; `merge-coverage.sh --ci` lists the package), `Release` (nx versions and publishes `webgpu-graph-algorithms@0.2.0` -- plus whatever master gained since `dc08826b` -- and pushes the `chore(release): publish ... [skip ci]` commit and the tag; the run's own colour is not the check: its explicit `nx release publish` safety-net step gets `409 Conflict` on every already-published package, as run 35189211639 did on 2026-09-17, until the owner changes release.yml), `Deploy` (unchanged), `Hosts` (both hosts green on master).

- [ ] **Step 4: Verify the publication**

Run: `npm view @graphty/webgpu-graph-algorithms version dependencies peerDependencies --json && git -C /home/apowers/Projects/graphty-monorepo fetch --tags && git -C /home/apowers/Projects/graphty-monorepo tag -l 'webgpu-graph-algorithms@*'`
Expected: `0.2.0`, `dependencies["@graphty/graph-format"]` `^0.2.0` (pnpm replaced `workspace:^`), `peerDependencies["@graphty/graph-format"]` `^0.2.0`, the optional peers unchanged; the tag `webgpu-graph-algorithms@0.2.0`; the npm page shows the provenance badge.

### Task M3-T8: Retire the staging repository (owner)

- [ ] **Step 1: The staging README**

In `/home/apowers/Projects/webgpu-graph-algorithms/README.md` (the staging root) add under the title: `LANDED: @graphty/webgpu-graph-algorithms, its design and the P0-P3 plans moved to graphty-org/graphty-monorepo on 2026-09-DD with this repository's history (webgpu-graph-algorithms/ and design/webgpu/ there); @graphty/graph-format and @graphty/graph-io moved on 2026-09-16. This repository is archived as the record of the staging.` Replace `2026-09-DD` with the landing date (`git -C /home/apowers/Projects/graphty-monorepo log -1 --format=%cs master`). Write `tmp/ci-round12.sh` on the M0-T1 pattern (`bash tmp/ci-push.sh README.md <<'MSG' ... MSG`, subject `docs: record the landing in graphty-monorepo and archive this repository`) and tell the owner: `! bash /home/apowers/Projects/webgpu-graph-algorithms/tmp/ci-round12.sh` (the last commit here).

- [ ] **Step 2: Cancel the phantom GPU run and archive**

Run (owner): `gh run list --repo graphty-org/webgpu-graph-algorithms --limit 50 --json databaseId,status --jq '.[] | select(.status=="queued") | .databaseId' | xargs -r -n1 gh run cancel --repo graphty-org/webgpu-graph-algorithms` then `gh repo archive graphty-org/webgpu-graph-algorithms --confirm` (gh 2.4.0 spells the flag `--confirm`).
Expected: no queued runs remain; the repository is read-only. The dev box checkout stays (it holds the `tmp/` trees the docs cite until D-14 is done); the npm name is now published from the monorepo.

- [ ] **Step 3: The memory note**

Update `/home/apowers/.claude/projects/-home-apowers-Projects-webgpu-graph-algorithms/memory/ci-cd-repair-loop.md` and `MEMORY.md`: the package lives in the monorepo; the source repository is archived; the runner is still owner-side (Phase M4).

---
## Phase M4: The GPU runner (owner-gated)

Entry: Phase M3 landed. Everything here follows design 12.4 ("Provisioning the hosted GPU runner", `design/webgpu/webgpu-acceleration-plan.md:4031-4062`) and G0's open rows. The org is on the GitHub Free plan today (`gh api orgs/graphty-org --jq .plan.name` prints `free`); hosted GPU runners need Team (design Q-3, decided 2026-09-14: "I don't want to self-host the runner, I'm happy to pay for a hosted runner"). Work in the main worktree on a short-lived branch, committed through `tools/commit-changes.sh` (the owner's usual script).

### Task M4-T1: Provision (owner, GitHub UI)

- [ ] **Step 1: Team plan** -- graphty-org > Settings > Billing: upgrade to Team ($4 per seat per month; 2 seats). Record the date in G0 row O3.
- [ ] **Step 2: The runner** -- Organization > Settings > Actions > Runners > New runner > New GitHub-hosted runner: Linux x64, size "4-core GPU" (NVIDIA T4), image "NVIDIA GPU-Optimized Image for AI and HPC", name `gpu-linux-t4`, runner group `gpu` with repository access limited to `graphty-monorepo` (the archived staging repository needs none), "Allow public repositories" ON, maximum concurrency 1. Record O4/O5.
- [ ] **Step 3: The spending limit** -- Settings > Billing > Spending limits: Actions $50 per month, alert at 75 percent. Record O6. The T4 bills $0.052 per minute even on public repositories; a 20-minute run is about $1.
- [ ] **Step 4: Repository settings on graphty-monorepo** -- Settings > Actions > General: default workflow permissions "Read repository contents and packages permissions" (the repo's default is "write" today; every workflow that needs more declares it: `release.yml` `contents: write`, `gpu.yml`'s report job `issues: write`); "Require approval for all external contributors" (already set on the staging repository, check it here). Create the label `gpu` (Issues > Labels in the UI, or `gh api -X POST repos/graphty-org/graphty-monorepo/labels -f name=gpu -f color=76B900 -f description="run the NVIDIA T4 lane on this PR (same-repo PRs only)"` -- gh 2.4.0 has no `gh label`). Record O7-O9.

### Task M4-T2: Enable the triggers and run the lane once

**Files:**
- Modify: `.github/workflows/gpu.yml:7-14` (the `on:` block: drop the three prose comment lines, un-comment the two trigger lines)

- [ ] **Step 1: Restore the triggers**

Replace the `on:` block of `.github/workflows/gpu.yml` with:

```yaml
on:
    push: { branches: [master] }
    pull_request: { types: [labeled, synchronize] }
    schedule: [{ cron: "17 6 * * *" }] # nightly, skipped by the `changed` job when master has not moved (design 12.6)
    workflow_dispatch:
```

- [ ] **Step 2: Commit and push (owner)**

Through `tools/commit-changes.sh` with the message `ci: enable the GPU lane's push and nightly triggers now that gpu-linux-t4 exists` and a body naming the runner, the group and the spending limit. The push to master itself triggers the first run.

- [ ] **Step 3: Watch the first run**

Run: `cd /home/apowers/Projects/graphty-monorepo && gh run list --workflow GPU --limit 1 --json databaseId,status --jq '.[0]' && gh run watch <run-id> --exit-status; gh run view <run-id> --log > tmp/gpu-first-run.log`
Expected: the job starts within minutes (no 24-hour queue); `Driver up` prints the T4 in `nvidia-smi`; the adapter report prints `vendor=nvidia architecture=turing ... software=false`; the canary greps `adapter vendor=nvidia`; the node and node-limits projects, the twins pass, the browser smoke on NVIDIA, the bench and `bench-compare` (`new (no baseline)`, exit 0) all green; the artifact `gpu-results-<run-id>` holds `gpu-report.json` and `benchmarks/out/gpu-linux-t4.json`. About 15-20 minutes.

- [ ] **Step 4: Record the image facts**

From the log, fill G0 rows F1-F21 (`webgpu-graph-algorithms/docs/decisions/G0.md`): the image's Ubuntu release and glibc (`ldd --version` is not run by the workflow; read `/etc/os-release` lines from the `Driver up` output or add a one-line `cat /etc/os-release; ldd --version | head -1` to that step in a follow-up commit), the driver version from `nvidia-smi`, whether `modprobe` was needed (its exit status), whether the browser smoke found the T4 without `xvfb-run` (the `[browser]` adapter line: `vendor=nvidia` means yes; `swiftshader` means G0 follow-up U1 applies: wrap the smoke in `xvfb-run -a` and add `sudo apt-get install -y xvfb` before it), whether `libegl1` is present (`dpkg -s libegl1` in the same diagnostic step).

- [ ] **Step 5: The deliberate red run (design 12.4 item 5, G0 row M2)**

The policy proof: on a software adapter the report must FAIL under `GRAPHTY_GPU_REQUIRE=nvidia`. Run on the dev box: `cd /home/apowers/Projects/graphty-monorepo/webgpu-graph-algorithms && pnpm run build:all >/dev/null && GRAPHTY_GPU_ADAPTER=llvmpipe GRAPHTY_GPU_REQUIRE=nvidia VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json XDG_RUNTIME_DIR=/tmp node scripts/gpu-report.js; echo "exit=$?"`
Expected: a JSON report with `"ok": false`, `"reason"` naming the vendor mismatch, and `exit=2`. Paste the reason line and the exit code into G0.

### Task M4-T3: The T4 baseline

**Files:**
- Create: `webgpu-graph-algorithms/benchmarks/results/gpu-linux-t4.json`

- [ ] **Step 1: Download the first green run's benchmark session**

Run: `cd /home/apowers/Projects/graphty-monorepo && gh run download <run-id> --name gpu-results-<run-id> --dir tmp/gpu-first && ls tmp/gpu-first/webgpu-graph-algorithms/benchmarks/out/`
Expected: `gpu-linux-t4.json` (the session file `benchmarks/harness.ts` appends per runner class; `GRAPHTY_RUNNER_CLASS` fixed the name).

- [ ] **Step 2: Promote it to the baseline**

Run: `cp tmp/gpu-first/webgpu-graph-algorithms/benchmarks/out/gpu-linux-t4.json webgpu-graph-algorithms/benchmarks/results/gpu-linux-t4.json && node -e 'const b=require("./webgpu-graph-algorithms/benchmarks/results/gpu-linux-t4.json"); console.log(Object.keys(b))'`
Expected: the file parses and has the same top-level shape as `benchmarks/results/nvidia-lovelace-driver580.json` (the dev box baseline; `scripts/bench-compare.js` reads `benchmarks/results/<class>.json`). Design 12.1/12.4 say `benchmarks/baselines/`; the code says `results/` (DEPARTURE-C).

- [ ] **Step 2a: The README performance table** -- design P10 / GOAL-9: regenerate the package README's Performance section (`webgpu-graph-algorithms/README.md`, the table around line 281) from `benchmarks/results/` by the procedure the README states, so it carries both the dev-box (`nvidia-lovelace-driver580`) and the T4 rows; until this step it stays as regenerated from the dev-box baseline alone.

- [ ] **Step 3: Commit (owner)** -- `perf(webgpu-graph-algorithms): record the first T4 benchmark session as the gpu-linux-t4 baseline` through `tools/commit-changes.sh` (the README change rides in the same commit). The next GPU run's `bench-compare` then compares (a median above 3x the baseline fails the lane; `SKIPPED: GPU not quiet` when `nvidia-smi` shows another process).

### Task M4-T4: The nightly guard and the label path

- [ ] **Step 1: The labelled PR path** -- on any open same-repo PR, `gh pr edit <n> --add-label gpu`; expected: a `GPU` run starts for the PR (`pull_request: labeled`), and a push to the PR re-runs it (`synchronize`). Remove the label afterwards.
- [ ] **Step 2: The nightly skip** -- after the 06:17 UTC cron on a day master did not move, `gh run list --workflow GPU --limit 10 --json databaseId,event,status,conclusion --jq '[.[] | select(.event=="schedule")][0]'` shows the run with the `test-gpu` job skipped (`changed` printed `run=false`). Record in G0.
- [ ] **Step 3: The tracking issue** -- not provoked; the `gpu-nightly-report` job's logic is unchanged from the staging lane.

### Task M4-T5: Close G0 and G2, amend the design's stale rows

**Files:**
- Modify: `webgpu-graph-algorithms/docs/decisions/G0.md` (every `[[fill: ...]]` slot; replace the OPEN header note with a CLOSED line)
- Modify: `webgpu-graph-algorithms/docs/decisions/G2.md` (every `{{...}}` slot: the `gpu-run-line`, `gpu-verdict`, `gpu-report-summary` from the first green run's log)
- Modify: `design/webgpu/webgpu-acceleration-plan.md` (APPEND a Review-log entry; never edit the body)

- [ ] **Step 1: Fill the records** -- `grep -c '\[\[fill' webgpu-graph-algorithms/docs/decisions/G0.md` and `grep -c '{{' webgpu-graph-algorithms/docs/decisions/G2.md` must both print `0` afterwards; the header notes become `CLOSED (<the first green run's date, YYYY-MM-DD>): the hosted gpu-linux-t4 lane ran green on graphty-monorepo run <id>.` (every `<...>` and the date below are filled from that run; `git grep -n 'MM-DD' webgpu-graph-algorithms/docs/decisions design/webgpu` must print nothing before the commit).
- [ ] **Step 2: The Review-log entry** -- append to the design:

```markdown

GPU lane live in graphty-monorepo (<the first green run's date>, integration plan Phase M4): runner `gpu-linux-t4` (runner group
`gpu`, Team plan, $50/month limit), the `gpu` label, push + nightly triggers restored. Corrections to the body
above, recorded here rather than edited in place: the P10 row's "self-hosted runner registered for the monorepo"
and R-6's "dev-box runner now" mean the HOSTED runner of Q-3; the baseline path is
`benchmarks/results/gpu-linux-t4.json` (12.1, 12.4 say `benchmarks/baselines/`); the lavapipe ICD on
ubuntu-24.04 is `/usr/share/vulkan/icd.d/lvp_icd.json`, discovered by `find` in ci.yml (12.2, 12.5 name the
jammy path); the design's 12.5 diff is superseded by the shards as landed. Image facts: <the F1-F21 summary>.
```

- [ ] **Step 3: Commit (owner)** -- `docs(webgpu-graph-algorithms): close the G0 and G2 gate records with the first T4 run`.

---
## Phase M5: The layout seam (design 9.3, "L1-sim")

**Entry criteria (both):** F2 -- `@graphty/graph-format` is `>= 1.0.0` on master (graph-format design 14.6: cut after the A1 branch is green; owner-side work outside this plan) -- and Phase M3 is done. Until F2, this phase may be PREPARED on a branch and must not merge (graph-format design 13.5 rule 5: layout is a 1.x package). Work on branch `feat/layout-simulation` in a worktree (`git worktree add .worktrees/layout-simulation -b feat/layout-simulation master`, owner); every commit through `tools/commit-changes.sh` with scope `layout`; the phase lands as ONE PR whose last commit is the Chromatic re-baseline.

**Step 0 of the phase (a fresh worktree has no `node_modules` and no `dist/`, both gitignored):** `cd LT && HUSKY=0 pnpm install --frozen-lockfile && pnpm exec nx run graph-format:build`. Every later command reads `graph-format/dist/` (layout's `tsc` resolves `@graphty/graph-format` through graph-format's `types`, vitest through its `exports`).

**What this phase builds** (design 9.3 verbatim, `design/webgpu/webgpu-acceleration-plan.md:2989-3064`): `layout/src/simulation/` exporting `LayoutSimulation`, `SimulationOptions`, `CommonLayoutOptions`, `ForceAtlas2Options`, `FruchtermanReingoldOptions`, `SpringElectricalOptions`, `LayoutAccelerator`, `SimulationType`, `createSimulation`, `ForceAtlas2Simulation`, `FruchtermanReingoldSimulation`, `resolveNodeVector`, `resolveWeights`, `seedPositions`; plus the minimal `toLayoutSnapshot` the legacy wrappers need. It does NOT port the other thirteen layouts to `indexed.*` (graph-format design 14.3's own work, which can proceed in parallel: the two share only `toLayoutSnapshot` and `CommonLayoutOptions`).

**The source of truth for the CPU ForceAtlas2:** `webgpu-graph-algorithms/test/oracle/forceatlas2.ts` (1162 lines): an index-based f64 transcription of the design's 7.2 formula table, checked against NetworkX trajectories (`test/oracle/forceatlas2-networkx.test.ts`, 76 committed JSON fixtures under `test/fixtures/networkx/`) and hand-computed values (`test/oracle/swing-mode.test.ts`). The design's P3 row calls it "the SPEC of the L1 `ForceAtlas2Simulation`". Task M5-T5 ports it; the oracle stays where it is (D-16).

### Task M5-T1: layout depends on graph-format

**Repository:** `/home/apowers/Projects/graphty-monorepo/.worktrees/layout-simulation` (`LT` below).

**Files:**
- Modify: `layout/package.json` (dependencies, peerDependencies)
- Modify: `layout/scripts/build-bundle.js:22-43` (externalise the dependency)
- Modify: `layout/scripts/bundle-types.js:26-43` (the simulation barrel)
- Modify: `layout/typedoc.json:3-9` (entry point)
- Modify: `layout/scripts/build-gh-pages.js:72-74` (the examples bundle inlines graph-format)
- Create: `layout/src/simulation/index.ts` (the barrel)
- Modify: `layout/src/index.ts` (`export * from "./simulation"`)
- Modify: `layout/test/package-structure.test.ts` (the new exports)

**Interfaces:**
- Produces: `@graphty/layout` resolves `@graphty/graph-format` (types through `graph-format/dist/graph-format.d.ts` -- layout's `tsconfig.json` keeps `moduleResolution: "node"`, which reads graph-format's top-level `main`/`types`); `dist/layout.js` leaves `@graphty/graph-format` external; `dist/layout.d.ts` re-exports `./src/simulation/index`.

- [ ] **Step 1: Write the failing test**

In `layout/test/package-structure.test.ts`, next to the existing export assertions on `../dist/layout.js`, add:

```ts
    it("exports the simulation seam (design 9.3)", async () => {
        const layout = await import("../dist/layout.js");
        for (const name of ["createSimulation", "ForceAtlas2Simulation", "FruchtermanReingoldSimulation", "seedPositions", "resolveNodeVector", "resolveWeights", "toLayoutSnapshot"]) {
            assert.equal(typeof layout[name], "function", `${name} is exported`);
        }
    });

    it("leaves @graphty/graph-format external in the bundle", () => {
        const bundle = readFileSync(new URL("../dist/layout.js", import.meta.url), "utf8");
        assert.ok(/from\s+["']@graphty\/graph-format["']/.test(bundle), "the bundle imports @graphty/graph-format instead of inlining it");
        assert.ok(!/class GraphBuilder\b/.test(bundle), "no graph-format source inlined");
    });
```

(Change the file's vitest import to `import { assert, describe, expect, it } from "vitest"` -- layout's tests use vitest's `assert`/`expect`, never `node:assert` -- and add `import { readFileSync } from "node:fs";`.)

Run: `cd LT/layout && npm run build:all && pnpm exec vitest run test/package-structure.test.ts`
Expected: FAIL (`createSimulation is exported` is `undefined`; the bundle has no `@graphty/graph-format` import).

- [ ] **Step 2: The dependency**

In `layout/package.json` add (the package has neither key today):

```json
    "dependencies": {
        "@graphty/graph-format": "workspace:^"
    },
    "peerDependencies": {
        "@graphty/graph-format": "^1.0.0"
    },
```

(`^1.0.0`: F2 has cut 1.0.0 by the entry criterion; nx preserves the range on every 1.x minor.) Then `cd LT && HUSKY=0 pnpm install` (the lockfile gains the link) and `HUSKY=0 pnpm install --frozen-lockfile`.

- [ ] **Step 3: Externalise in the bundle**

In `layout/scripts/build-bundle.js` replace `external: [],` with `external: externalDependencies(),` and add above the `build(...)` call the graph-io helper (`graph-io/scripts/build-bundle.js:33-47`):

```js
function externalDependencies() {
    const packageJson = JSON.parse(readFileSync(path.resolve(__dirname, "../package.json"), "utf8"));
    const names = new Set([
        ...Object.keys(packageJson.dependencies ?? {}),
        ...Object.keys(packageJson.peerDependencies ?? {}),
    ]);
    return (id) => {
        for (const name of names) {
            if (id === name || id.startsWith(`${name}/`)) {
                return true;
            }
        }
        return false;
    };
}
```

with `import { readFileSync } from "node:fs";` at the top if absent. Reason: `external: []` would inline a second copy of graph-format into `dist/layout.js`; `isGraphSnapshot()` is a `Symbol.for` brand check so it would still interoperate, but the bundle would double and the element would ship two format copies.

- [ ] **Step 3a: The gh-pages examples keep a self-contained bundle**

`layout/scripts/build-gh-pages.js:72-74` copies `dist/layout.js` to `gh-pages/examples/layout.js`, and the example pages load it as a raw browser module; a bare `import ... from "@graphty/graph-format"` inside it would fail in every browser ("Failed to resolve module specifier"). Replace the copy with a second vite lib build that INLINES graph-format for the examples only: in `build-gh-pages.js`, in place of the `copyFile(layoutJsPath, ...)` call, run

```js
await build({
    configFile: false,
    build: {
        lib: { entry: path.resolve(__dirname, "../src/index.ts"), name: "GraphLayout", formats: ["es"], fileName: () => "layout.js" },
        outDir: ghPagesExamplesDir,
        emptyOutDir: false,
        rollupOptions: { external: [], output: { preserveModules: false, inlineDynamicImports: true } },
        minify: false,
        sourcemap: false,
    },
});
```

with `import { build } from "vite";` at the top (the same import `build-bundle.js` uses). Check after `npm run build:gh-pages`: `grep -c 'from "@graphty/graph-format"' gh-pages/examples/layout.js` prints `0` and `grep -c 'from "@graphty/graph-format"' dist/layout.js` prints `1`.

- [ ] **Step 4: The d.ts shim and typedoc**

In `layout/scripts/bundle-types.js` add after the `./src/layouts/index` re-export line of the template:

```js
// Re-export the simulation seam (design/webgpu/webgpu-acceleration-plan.md section 9.3)
export * from './src/simulation/index';
```

In `layout/typedoc.json` add `"src/simulation/index.ts"` to `entryPoints`.

- [ ] **Step 5: Create the barrel now so the build passes (filled by the later tasks)**

Create `layout/src/simulation/index.ts`:

```ts
/**
 * The layout seam of the WebGPU design (design/webgpu/webgpu-acceleration-plan.md section 9.3): steppable
 * simulations over a graph-format snapshot and the owner's stride-3 scene-unit position array, the accelerator
 * interface an injected GPU implements, and the dispatcher that chooses between the two. Nothing here imports the
 * GPU package: the dependency direction is graph-format <- layout <- graphty-element <- the app, and the app is the
 * only importer of @graphty/webgpu-graph-algorithms (design 9.1).
 */

export { createSimulation } from "./create-simulation";
export { ForceAtlas2Simulation } from "./forceatlas2";
export { FruchtermanReingoldSimulation } from "./fruchterman-reingold";
export { resolveNodeVector, resolveWeights } from "./inputs";
export { Lcg, seedPositions } from "./seed";
export { toLayoutSnapshot } from "./snapshot";
export type {
    CommonLayoutOptions,
    ForceAtlas2Options,
    FruchtermanReingoldOptions,
    LayoutAccelerator,
    LayoutSimulation,
    SimulationOptions,
    SimulationType,
    SpringElectricalOptions,
} from "./types";
```

(sorted by source path: the root eslint config's `simple-import-sort/exports` is an error) and add `export * from "./simulation";` to `layout/src/index.ts` after the generators line. The modules it names are written in M5-T2..T7; until then the build fails on the missing files, which is the point of writing the tests first.

### Task M5-T2: The types

**Files:**
- Create: `layout/src/simulation/types.ts`

- [ ] **Step 1: Write the file** -- the design's 9.3 declarations, spelled with `?: T | undefined` (exactOptionalPropertyTypes-compatible) exactly as the GPU package's mirrors `webgpu-graph-algorithms/src/types/options.ts:10-57` and `src/types/accelerator.ts:17-38` spell them, so an option object the element parses is accepted by both without a cast:

```ts
import type { F32, GraphSnapshot, NodeId, NodeMask } from "@graphty/graph-format";

/** graph-format design 14.3 CommonLayoutOptions. */
export interface CommonLayoutOptions {
    readonly dim?: 2 | 3 | undefined;
    readonly scale?: number | undefined;
    readonly center?: ArrayLike<number> | undefined;
    readonly seed?: number | null | undefined;
}

/** Design 9.3 SimulationOptions: shared by every simulation type; the CPU simulations ignore maxInFlight. */
export interface SimulationOptions {
    /** Settle when the mean per-node displacement stays below settleThreshold * rmsRadius for settleWindow iterations (design 7.17). */
    readonly settleThreshold?: number | undefined;
    readonly settleWindow?: number | undefined;
    /** Iterations per step() call; default 1; the element passes its stepMultiplier (design 9.4 item 4). */
    readonly iterationsPerStep?: number | undefined;
    /** GPU simulations only (design 7.19); default 2. */
    readonly maxInFlight?: number | undefined;
}

/** Design 9.3 ForceAtlas2Options: the names and defaults of the positional forceatlas2Layout (design 7.14). */
export interface ForceAtlas2Options extends CommonLayoutOptions, SimulationOptions {
    readonly maxIter?: number | undefined;
    readonly jitterTolerance?: number | undefined;
    readonly scalingRatio?: number | undefined;
    readonly gravity?: number | undefined;
    readonly strongGravity?: boolean | undefined;
    readonly distributedAction?: boolean | undefined;
    readonly linlog?: boolean | undefined;
    /** A per-node mass (n values), the name of a numeric node column, the legacy id-keyed record, or null (role-`mass` column, else outDegree + 1). */
    readonly nodeMass?: F32 | string | Readonly<Record<NodeId, number>> | null | undefined;
    readonly nodeSize?: F32 | string | Readonly<Record<NodeId, number>> | null | undefined;
    /** true: the snapshot's weights; a string: a numeric edge column; false / null: unweighted. */
    readonly weight?: boolean | string | null | undefined;
    readonly dissuadeHubs?: boolean | undefined;
}

/** Design 9.3 FruchtermanReingoldOptions. */
export interface FruchtermanReingoldOptions extends CommonLayoutOptions, SimulationOptions {
    readonly k?: number | null | undefined;
    readonly iterations?: number | undefined;
    /** A node mask (the bool-column bit layout) or the name of a bool node column with role "fixed". */
    readonly fixed?: NodeMask | string | null | undefined;
}

/** Design 9.3 SpringElectricalOptions (ngraph's names and defaults, design 7.20); no CPU simulation in v1. */
export interface SpringElectricalOptions extends CommonLayoutOptions, SimulationOptions {
    readonly springLength?: number | undefined;
    readonly springCoefficient?: number | undefined;
    readonly gravity?: number | undefined;
    readonly dragCoefficient?: number | undefined;
    readonly timeStep?: number | undefined;
}

/** graph-format design 14.3 LayoutSimulation, verbatim: steppable layouts over the owner's stride-3 scene-unit array. */
export interface LayoutSimulation {
    /** positions: the owner's stride-3 scene-unit array, read AND written in place. */
    load(snapshot: GraphSnapshot, positions: F32): void;
    /** GPU implementations are async (the readback); CPU implementations return void. */
    step(iterations?: number): void | Promise<void>;
    readonly settled: boolean;
    /** The same bitmap layout as a bool column with role "fixed". */
    setFixed(mask: NodeMask): void;
    /** A drag during the simulation: one node's scene-unit position. */
    setPosition(index: number, x: number, y: number, z: number): void;
    dispose(): void;
}

/** Design 9.3 LayoutAccelerator: what an injected GPU implements; every method optional (only implemented ones exist). */
export interface LayoutAccelerator {
    readonly kind: string;
    forceAtlas2?(options?: ForceAtlas2Options): LayoutSimulation;
    fruchtermanReingold?(options?: FruchtermanReingoldOptions): LayoutSimulation;
    springElectrical?(options?: SpringElectricalOptions): LayoutSimulation;
    release?(s: GraphSnapshot): void;
    dispose?(): void;
}

/** Design 9.3 SimulationType; "spring" is the element's name for Fruchterman-Reingold. */
export type SimulationType = "forceatlas2" | "fruchtermanReingold" | "spring" | "spring-electrical";
```

- [ ] **Step 2: Type-check** -- `cd LT/layout && pnpm exec tsc --noEmit` (fails only on the modules M5-T3..T7 have not written yet; the errors must name those files, nothing in `types.ts`).

### Task M5-T3: The LCG and seedPositions

**Files:**
- Create: `layout/src/simulation/seed.ts`
- Create: `layout/test/simulation/seed.test.ts`

- [ ] **Step 1: Write the failing tests**

```ts
import assert from "node:assert";
import { fromEdgeArrays } from "@graphty/graph-format";
import { describe, it } from "vitest";

import { Lcg, LCG_A, LCG_C, LCG_M, seedPositions } from "../../src/simulation/seed";
import { RandomNumberGenerator } from "../../src/utils/random";

function ring(n: number) {
    const src = new Uint32Array(n);
    const dst = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
        src[i] = i;
        dst[i] = (i + 1) % n;
    }
    return fromEdgeArrays({ directed: false, nodeCount: n, src, dst });
}

describe("Lcg", () => {
    it("carries the port's constants (m = 2^35 - 31, a = 185852, c = 1)", () => {
        assert.equal(LCG_M, 2 ** 35 - 31);
        assert.equal(LCG_A, 185852);
        assert.equal(LCG_C, 1);
    });

    it("is the package's RandomNumberGenerator bit for bit over 10,000 draws (the W1 cross-test of design 9.3)", () => {
        for (const seed of [1, 7, 42, 123456, 999999]) {
            const lcg = new Lcg(seed);
            const rng = new RandomNumberGenerator(seed);
            for (let k = 0; k < 10000; k++) {
                assert.equal(lcg.next(), rng.rand(), `seed ${seed}, draw ${k}`);
            }
        }
    });

    it("seed 0 and null are unseeded, as the port's `seed || random` quirk has it", () => {
        assert.ok(new Lcg(0).seed > 0 && new Lcg(null).seed > 0);
    });
});

describe("seedPositions", () => {
    it("draws every NaN row in [-1, 1) per axis in index order and writes z = center.z in 2D", () => {
        const s = ring(4);
        const positions = new Float32Array(12).fill(Number.NaN);
        seedPositions(s, positions, 7, 2, 1, null, "fa2");
        const rng = new RandomNumberGenerator(7);
        for (let i = 0; i < 4; i++) {
            assert.equal(positions[3 * i], Math.fround(rng.rand() * 2 - 1));
            assert.equal(positions[3 * i + 1], Math.fround(rng.rand() * 2 - 1));
            assert.equal(positions[3 * i + 2], 0);
        }
    });

    it("keeps finite rows, draws the rest inside the finite rows' box, applies scale and center", () => {
        const s = ring(3);
        const positions = new Float32Array([10, 20, 0, 30, 40, 0, Number.NaN, Number.NaN, Number.NaN]);
        seedPositions(s, positions, 42, 2, 5, [100, 100, 0], "fa2");
        assert.deepEqual(Array.from(positions.subarray(0, 6)), [10, 20, 0, 30, 40, 0]);
        assert.ok(positions[6] >= 10 && positions[6] <= 30 && positions[7] >= 20 && positions[7] <= 40);
        assert.equal(positions[8], 0);
    });

    it('range "fr" draws in [0, 1)', () => {
        const s = ring(2);
        const positions = new Float32Array(6).fill(Number.NaN);
        seedPositions(s, positions, 3, 2, 1, null, "fr");
        for (let i = 0; i < 4; i++) {
            const v = positions[3 * Math.floor(i / 2) + (i % 2)];
            assert.ok(v >= 0 && v < 1);
        }
    });

    it("rejects a wrong length, a bad dim, a non-positive scale and a non-finite center", () => {
        const s = ring(2);
        assert.throws(() => seedPositions(s, new Float32Array(5), 1, 2, 1, null, "fa2"), /positions has 5 entries/);
        assert.throws(() => seedPositions(s, new Float32Array(6), 1, 4 as 2, 1, null, "fa2"), /dim must be 2 or 3/);
        assert.throws(() => seedPositions(s, new Float32Array(6), 1, 2, 0, null, "fa2"), /scale must be/);
        assert.throws(() => seedPositions(s, new Float32Array(6), 1, 2, 1, [Number.NaN], "fa2"), /center\[0\]/);
    });
});
```

Run: `cd LT/layout && pnpm exec vitest run test/simulation/seed.test.ts` -- Expected: FAIL (module not found).

- [ ] **Step 2: Write the module**

Copy `webgpu-graph-algorithms/src/layouts/seed.ts` (198 lines) to `layout/src/simulation/seed.ts` with three changes: (1) the header comment says this is the CANONICAL copy that design 9.3 gives to @graphty/layout and that the GPU package keeps its own (D27), cross-tested at W1b; (2) `import { WebGpuGraphError } from "../errors.js"` is removed and every `throw new WebGpuGraphError("E_INVALID_ARGUMENT", message, details)` becomes `throw new RangeError(message)` (same message texts, so the tests above match); (3) the relative import style drops `.js` extensions to match layout's `moduleResolution: "node"` sources (`import type { F32, GraphSnapshot } from "@graphty/graph-format";` stays). Keep `LCG_M`, `LCG_A`, `LCG_C`, the `Lcg` class, `resolveCenter` and `seedPositions` verbatim otherwise.

Run: `cd LT/layout && pnpm exec vitest run test/simulation/seed.test.ts` -- Expected: PASS (the 10,000-draw comparison is the design's W1 cross-test, done here because both generators now live in one package).

### Task M5-T4: Node vectors and weights

**Files:**
- Create: `layout/src/simulation/inputs.ts`
- Create: `layout/test/simulation/inputs.test.ts`

- [ ] **Step 1: Write the failing tests**

```ts
import assert from "node:assert";
import { GraphBuilder } from "@graphty/graph-format";
import { describe, it } from "vitest";

import { resolveNodeVector, resolveWeights } from "../../src/simulation/inputs";

function triangle(weighted: boolean) {
    const b = new GraphBuilder({ directed: false, weighted: weighted ? true : "auto" });
    b.addNode("a"); b.addNode("b"); b.addNode("c");
    b.addEdge("a", "b", weighted ? 2 : undefined);
    b.addEdge("b", "c", weighted ? 3 : undefined);
    b.addEdge("c", "a", weighted ? 4 : undefined);
    return b.freeze();
}

describe("resolveNodeVector", () => {
    it("null: the fallback per index (outDegree + 1 for ForceAtlas2)", () => {
        const s = triangle(false);
        const deg = s.outDegree();
        const v = resolveNodeVector(null, s, (i) => deg[i] + 1);
        assert.deepEqual(Array.from(v), [3, 3, 3]);
    });
    it("a Float32Array of length n is returned as given; a wrong length throws", () => {
        const s = triangle(false);
        const given = new Float32Array([1, 2, 3]);
        assert.equal(resolveNodeVector(given, s, () => 1), given);
        assert.throws(() => resolveNodeVector(new Float32Array(2), s, () => 1), /2 values, expected 3/);
    });
    it("the legacy id-keyed record resolves through the id map; missing ids take the fallback", () => {
        const s = triangle(false);
        const v = resolveNodeVector({ b: 7 }, s, (i) => i + 10);
        assert.deepEqual(Array.from(v), [10, 7, 12]);
    });
    it("an unknown column name throws", () => {
        assert.throws(() => resolveNodeVector("mass", triangle(false), () => 1), /node column "mass"/);
    });
});

describe("resolveWeights", () => {
    it("true returns the snapshot's arc weights, or null when unweighted", () => {
        assert.equal(resolveWeights(true, triangle(false)), null);
        const w = resolveWeights(true, triangle(true));
        assert.ok(w instanceof Float32Array && w.length === 6);
    });
    it("false, null and undefined mean unweighted", () => {
        const s = triangle(true);
        assert.equal(resolveWeights(false, s), null);
        assert.equal(resolveWeights(null, s), null);
        assert.equal(resolveWeights(undefined, s), null);
    });
    it("a missing edge column name throws", () => {
        assert.throws(() => resolveWeights("capacity", triangle(true)), /edge column "capacity"/);
    });
});
```

Run: `cd LT/layout && pnpm exec vitest run test/simulation/inputs.test.ts` -- Expected: FAIL (module not found).

- [ ] **Step 2: Write the module**

```ts
/**
 * The per-node and per-arc inputs of the CPU simulations (design 9.3 resolveNodeVector / resolveWeights; D28:
 * inputs resolve by graph-format ROLE on both the CPU and the GPU path). The CPU path is the one that still accepts
 * the legacy id-keyed record (the element passes it today); the GPU package rejects that form and graphty-element
 * converts it into a role column at engine creation (design 9.4 item 10).
 */

import { type Column, expandEdges, type F32, type GraphSnapshot, type NodeId } from "@graphty/graph-format";

function numericValues(s: GraphSnapshot, name: string, column: Column): F32 {
    if (column.meta.components !== 1) {
        throw new RangeError(`node column "${name}" has ${column.meta.components} components; a node vector has one`);
    }
    switch (column.dtype) {
        case "f32":
        case "f64": {
            // gpuView is the f32 array itself for f32 and its cached f32 copy for f64
            const view = s.nodes.gpuView(name);
            return view instanceof Float32Array ? view : new Float32Array(view);
        }
        case "u32":
        case "i32":
            return new Float32Array(s.nodes.gpuView(name));
        case "u8":
            // gpuView packs u8 four to a word; the per-node bytes are column.data
            return new Float32Array(column.data);
        default:
            throw new TypeError(`node column "${name}" is ${column.dtype}, not numeric`);
    }
}

/**
 * Resolves a per-node vector: null -> the role-`mass` node column when present, else `fallback(i)` for every i;
 * a Float32Array of length n as given; a column name -> that numeric node column; the legacy record -> every id
 * through the id map, missing ids take the fallback.
 * @param spec - the option value
 * @param s - the snapshot
 * @param fallback - the default per index
 * @returns n values
 */
export function resolveNodeVector(
    spec: F32 | string | Readonly<Record<NodeId, number>> | null | undefined,
    s: GraphSnapshot,
    fallback: (i: number) => number,
): F32 {
    const n = s.nodeCount;
    if (spec === null || spec === undefined) {
        const byRole = s.nodes.byRole("mass");
        if (byRole !== null) {
            return numericValues(s, byRole.meta.name, byRole);
        }
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            out[i] = fallback(i);
        }
        return out;
    }
    if (spec instanceof Float32Array) {
        if (spec.length !== n) {
            throw new RangeError(`the node vector has ${spec.length} values, expected ${n}`);
        }
        return spec;
    }
    if (typeof spec === "string") {
        const column = s.nodes.get(spec);
        if (column === null) {
            throw new RangeError(`the option names node column "${spec}", which the snapshot does not hold`);
        }
        return numericValues(s, spec, column);
    }
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const value = spec[s.ids.idOf(i)];
        out[i] = value === undefined ? fallback(i) : value;
    }
    return out;
}

/**
 * Resolves per-arc weights: true -> `s.weights` (null when unweighted); a string -> a one-component numeric edge
 * column expanded to arcs; false / null / undefined -> null.
 * @param spec - the option value
 * @param s - the snapshot
 * @returns arcCount weights, or null for "every weight is 1"
 */
export function resolveWeights(spec: boolean | string | null | undefined, s: GraphSnapshot): F32 | null {
    if (spec === true) {
        return s.weights;
    }
    if (spec === false || spec === null || spec === undefined) {
        return null;
    }
    const column = s.edges.get(spec);
    if (column === null) {
        throw new RangeError(`weight names edge column "${spec}", which the snapshot does not hold`);
    }
    if (column.meta.components !== 1) {
        throw new RangeError(`edge column "${spec}" has ${column.meta.components} components; a weight column has one`);
    }
    const expanded = expandEdges(s, s.edges.gpuView(spec));
    return expanded instanceof Float32Array ? expanded : new Float32Array(expanded);
}
```

(The graph-format spellings, from `graph-format/dist/src/types/columns.d.ts`: `Column.dtype`, `Column.meta.name`, `Column.meta.components`, `Column.data` (the raw `Uint8Array` of a u8 column), `nodes.byRole(role)`, `nodes.get(name)`, `nodes.gpuView(name)` (a padded u32 view for u8 -- never per-node values), `edges.gpuView(name)`, `s.ids.idOf(i)`, `s.weights`, `expandEdges(s, perEdge)`; the GPU package's `src/layouts/inputs.ts:74-104` is the model and compiles against the same d.ts.)

Run: `cd LT/layout && pnpm exec vitest run test/simulation/inputs.test.ts` -- Expected: PASS.

### Task M5-T5: The steppable CPU ForceAtlas2

**Files:**
- Create: `layout/src/simulation/forceatlas2.ts`
- Create: `layout/src/simulation/constants.ts`
- Create: `layout/test/simulation/forceatlas2.test.ts`
- Copy: `webgpu-graph-algorithms/test/fixtures/networkx/{karate-base-iter1,karate-base-iter5,gnm200-base-iter1,gnm200-base-iter5}.json` -> `layout/test/simulation/fixtures/networkx/` (four of the 77 committed fixtures, copied verbatim; the karate ones are 34 nodes / 78 edges, the gnm200 ones 200 nodes)

**Interfaces:**
- Produces: `class ForceAtlas2Simulation implements LayoutSimulation { constructor(options?: ForceAtlas2Options & { readonly compat?: "paper" | "networkx" }); load(snapshot, positions): void; step(iterations?): void; readonly settled: boolean; readonly iterationsDone: number; setFixed(mask): void; setPosition(index, x, y, z): void; reheat(): void; dispose(): void; }` with `FA2_DEFAULTS` in `constants.ts`.

- [ ] **Step 1: The defaults**

`layout/src/simulation/constants.ts`:

```ts
/** ForceAtlas2 defaults (design 7.14, 7.17, 7.19); the same values as the GPU package's FA2_DEFAULTS. */
export const FA2_DEFAULTS = Object.freeze({
    maxIter: 100,
    jitterTolerance: 1,
    scalingRatio: 2,
    gravity: 1,
    strongGravity: false,
    distributedAction: false,
    linlog: false,
    dissuadeHubs: false,
    dim: 2 as 2 | 3,
    scale: 1,
    settleThreshold: 0.001,
    settleWindow: 10,
    iterationsPerStep: 1,
});
/** The distance floor `max(d, 0.01)` of design 7.2 (webgpu-graph-algorithms/src/constants.ts:110). */
export const FA2_DISTANCE_FLOOR = 0.01;
/** The square of FA2_DISTANCE_FLOOR. */
export const FA2_DISTANCE_FLOOR_SQ = 0.0001;
/** The coincident threshold `d^2 < 1e-8` of design 7.2 (webgpu-graph-algorithms/src/constants.ts:114). */
export const FA2_COINCIDENT_SQ = 1e-8;
/** The lane count of the oracle's fold order (webgpu-graph-algorithms/src/constants.ts:9 WORKGROUP_SIZE); kept so the two f64 transcriptions sum in the same order. */
export const FA2_FOLD_LANES = 256;
```

(The three thresholds are the GPU package's `src/constants.ts:110-114`, copied so both transcriptions of table 7.2 floor and kick at the same thresholds; the oracle's `treeReduce` / `groupTotals` / `foldGroups` helpers are written over `WORKGROUP_SIZE` lanes, which the port names `FA2_FOLD_LANES`.)

- [ ] **Step 2: Write the failing tests**

```ts
import assert from "node:assert";
import { readFileSync } from "node:fs";
import { fromEdgeArrays, makeMask, maskSet } from "@graphty/graph-format";
import { describe, it } from "vitest";

import { ForceAtlas2Simulation } from "../../src/simulation/forceatlas2";
import { seedPositions } from "../../src/simulation/seed";

function grid(w: number, h: number) {
    const src: number[] = [];
    const dst: number[] = [];
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const i = y * w + x;
            if (x + 1 < w) { src.push(i); dst.push(i + 1); }
            if (y + 1 < h) { src.push(i); dst.push(i + w); }
        }
    }
    return fromEdgeArrays({ directed: false, nodeCount: w * h, src: Uint32Array.from(src), dst: Uint32Array.from(dst) });
}

function seeded(s: ReturnType<typeof grid>, seed: number, dim: 2 | 3 = 2) {
    const positions = new Float32Array(3 * s.nodeCount).fill(Number.NaN);
    seedPositions(s, positions, seed, dim, 1, null, "fa2");
    return positions;
}

describe("ForceAtlas2Simulation", () => {
    it("steps the owner's array in place and settles within maxIter", () => {
        const s = grid(6, 6);
        const positions = seeded(s, 7);
        const sim = new ForceAtlas2Simulation({ maxIter: 200 });
        sim.load(s, positions);
        assert.equal(sim.settled, false);
        const before = Float32Array.from(positions);
        sim.step();
        assert.equal(sim.iterationsDone, 1);
        assert.notDeepEqual(Array.from(positions), Array.from(before), "positions moved");
        while (!sim.settled) {
            sim.step(10);
        }
        assert.ok(sim.iterationsDone <= 200);
        for (const v of positions) {
            assert.ok(Number.isFinite(v));
        }
    });

    it("is deterministic for a seed and independent of the step batching", () => {
        const s = grid(5, 5);
        const a = seeded(s, 42);
        const b = seeded(s, 42);
        const simA = new ForceAtlas2Simulation();
        const simB = new ForceAtlas2Simulation();
        simA.load(s, a);
        simB.load(s, b);
        for (let k = 0; k < 20; k++) { simA.step(); }
        simB.step(20);
        assert.deepEqual(Array.from(a), Array.from(b));
    });

    it("a fixed node never moves; unpinning reheats", () => {
        const s = grid(4, 4);
        const positions = seeded(s, 3);
        const sim = new ForceAtlas2Simulation({ maxIter: 50 });
        sim.load(s, positions);
        const mask = makeMask(s.nodeCount);
        maskSet(mask, 5, true);
        sim.setFixed(mask);
        const x = positions[15]; const y = positions[16];
        sim.step(30);
        assert.equal(positions[15], x);
        assert.equal(positions[16], y);
        while (!sim.settled) { sim.step(); }
        maskSet(mask, 5, false);
        sim.setFixed(mask);
        assert.equal(sim.settled, false, "unpinning reheats (design 7.12)");
    });

    it("setPosition writes the scene position immediately and reheats without resetting the speed controller", () => {
        const s = grid(3, 3);
        const positions = seeded(s, 9);
        const sim = new ForceAtlas2Simulation({ maxIter: 20 });
        sim.load(s, positions);
        while (!sim.settled) { sim.step(); }
        sim.setPosition(4, 100, -100, 0);
        assert.deepEqual(Array.from(positions.subarray(12, 15)), [100, -100, 0]);
        assert.equal(sim.settled, false);
    });

    it("2D keeps z at center.z; 3D moves z", () => {
        const s = grid(4, 3);
        const p2 = seeded(s, 5, 2);
        const sim2 = new ForceAtlas2Simulation({ dim: 2, center: [0, 0, 7] });
        sim2.load(s, p2);
        sim2.step(5);
        for (let i = 0; i < s.nodeCount; i++) { assert.equal(p2[3 * i + 2], 7); }
        const p3 = seeded(s, 5, 3);
        const sim3 = new ForceAtlas2Simulation({ dim: 3 });
        sim3.load(s, p3);
        sim3.step(5);
        assert.ok(Array.from(p3).some((_, k) => k % 3 === 2 && p3[k] !== 0));
    });

    it("matches the NetworkX trajectories the GPU package pins, in networkx compat (design 11.4 oracle independence)", () => {
        // The fixture format is the GPU package's (test/fixtures/networkx/generate.py, NetworkX 3.4.2, seed 7):
        // { graph: { name, directed, nodeCount, src, dst, weights }, options: { max_iter, jitter_tolerance,
        //   scaling_ratio, gravity, distributed_action, strong_gravity, linlog, dissuade_hubs, weight, dim },
        //   initialPositions: number[n][dim], positions: number[n][dim] (after max_iter iterations), rescaled: false }
        for (const name of ["karate-base-iter1", "karate-base-iter5", "gnm200-base-iter1", "gnm200-base-iter5"]) {
            const f = JSON.parse(readFileSync(new URL(`./fixtures/networkx/${name}.json`, import.meta.url), "utf8"));
            const s = fromEdgeArrays({ directed: false, nodeCount: f.graph.nodeCount, src: Uint32Array.from(f.graph.src), dst: Uint32Array.from(f.graph.dst) });
            const n = s.nodeCount;
            const dim: 2 | 3 = f.options.dim === 3 ? 3 : 2;
            const positions = new Float32Array(3 * n);
            for (let i = 0; i < n; i++) {
                for (let a = 0; a < dim; a++) {
                    positions[3 * i + a] = f.initialPositions[i][a];
                }
            }
            const sim = new ForceAtlas2Simulation({
                maxIter: f.options.max_iter, jitterTolerance: f.options.jitter_tolerance, scalingRatio: f.options.scaling_ratio,
                gravity: f.options.gravity, distributedAction: f.options.distributed_action, strongGravity: f.options.strong_gravity,
                linlog: f.options.linlog, dissuadeHubs: f.options.dissuade_hubs, weight: f.options.weight === null ? false : true,
                dim, settleThreshold: 0, compat: "networkx",
            });
            sim.load(s, positions);
            sim.step(f.options.max_iter);
            // the GPU package's oracle matches these fixtures within 1e-9 at 1 and 5 iterations from an f64 start; the
            // simulation starts from the f32-rounded positions of the owner's array and writes f32 back, which the
            // oracle itself measured at 8.0e-5 relative on gnm200-base-iter5 (7.3e-7 karate-iter1, 1.3e-5 karate-iter5,
            // 2.0e-6 gnm200-iter1) -- so 3e-4 is the bound: a real port bug is orders of magnitude larger
            for (let i = 0; i < n; i++) {
                for (let a = 0; a < dim; a++) {
                    const expected = f.positions[i][a];
                    assert.ok(Math.abs(positions[3 * i + a] - expected) <= 3e-4 * Math.max(1, Math.abs(expected)), `${name} node ${i} axis ${a}`);
                }
            }
        }
    });

    it("rejects a directed snapshot and a wrong array length; dispose() ends stepping", () => {
        const s = grid(2, 2);
        const sim = new ForceAtlas2Simulation();
        const directed = fromEdgeArrays({ directed: true, nodeCount: 2, src: Uint32Array.from([0]), dst: Uint32Array.from([1]) });
        assert.throws(() => sim.load(directed, new Float32Array(6)), /undirected/);
        assert.throws(() => sim.load(s, new Float32Array(5)), /positions/);
        sim.load(s, seeded(s, 1));
        sim.dispose();
        assert.throws(() => sim.step(), /disposed/);
    });
});
```

Adjust the fixture-reading lines to the JSON shape `generate.py` writes (open one file; the test comment says where the shape is defined). Run: `cd LT/layout && pnpm exec vitest run test/simulation/forceatlas2.test.ts` -- Expected: FAIL (module not found).

- [ ] **Step 3: Port the oracle**

Create `layout/src/simulation/forceatlas2.ts` as a transcription of `webgpu-graph-algorithms/test/oracle/forceatlas2.ts` with this mapping (the formulas, the K1-K5 stage order, `estimateFactor`, `kickDir` and the settle rule are copied verbatim; what changes is the shell around them):

| Oracle | Simulation | Note |
| --- | --- | --- |
| `constructor(s, positions, options: OracleOptions)` (layout units, resolved inputs) | `constructor(options?)` stores resolved options; `load(snapshot, positions)` validates (`Error("ForceAtlas2Simulation: the snapshot must be undirected (pass toUndirected().snapshot)")` on a directed one; `positions.length === 3 * n`), resolves mass (`resolveNodeVector(options.nodeMass, s, (i) => deg[i] + 1)` with `deg = s.outDegree()`) and weights (`resolveWeights(options.weight, s)`), converts the owner's SCENE array into the internal f64 LAYOUT array `(v - center[a]) / scale` (z forced to 0 in 2D), resets the controller (`speed = speedEfficiency = swing = traction = 1`), the settle window, `iterationsDone = 0`, clears `fixed` when `n` changed, and calls `reheat()` | design 7.18 units, 7.19 load |
| `precision: "f64" \| "f32"`, `round` | dropped: f64 scratch always; the OUTPUT is written back with `Math.fround` semantics by the Float32Array itself | graph-format design 14.3: f64 scratch, f32 output |
| `stages`, `trace`, `traceRecords`, `resync`, `peekFold`, `OracleStages` | dropped (test instrumentation of the GPU) | -- |
| `step(): OracleTraceRecord` (one iteration) | `step(iterations = options.iterationsPerStep): void`: throws `Error("ForceAtlas2Simulation: disposed")` after `dispose()`, `Error("... not loaded")` before `load()`; returns at once when `settled`; runs the oracle's one-iteration body `iterations` times; after the batch writes every FREE node's layout position back to the owner's array as `layout * scale + center[a]` (fixed rows are never written: their scene value is the owner's), `iterationsDone += k`, updates `settled` | design 7.19 step, 7.12 |
| `settledCount`, `meanDisplacement`, `rmsRadius` | kept; `settled = iterationsDone >= maxIter || settledCount >= settleWindow`, `settledCount` incremented by the K1 fold when `meanDisplacement <= settleThreshold * rmsRadius` | design 7.17 |
| `setFixed(mask \| null)` | `setFixed(mask: NodeMask)`: validates `mask.length >= ceil(n / 32)`; copies; reheats ONLY when a bit went 1 -> 0 | design 7.12 |
| `setPosition(index, x, y, z)` (layout units) | `setPosition(index, x, y, z)` in SCENE units: writes the owner's array and the internal layout position `(v - center) / scale`, then `reheat()` (settle window and iteration budget only; the speed controller is untouched -- D8) | design 7.12, D8 |
| `reheat()` | `reheat()`: `iterationsDone = 0`, `settledCount = 0` (and mode-1's swing/traction reset as the oracle has it) | D8 |
| `compat` via `OracleOptions` | constructor option `compat?: "paper" \| "networkx"` (default `"paper"`; D5) beside `ForceAtlas2Options`; `gravityCenter` follows compat as in the oracle | design 7.2 |
| -- | `dispose()`: releases the scratch arrays and marks the simulation disposed | -- |

Keep the oracle's file-level helpers (`kickDir`, `estimateFactor`, the fold/partials code) as module-private functions with the same names. Every formula line keeps the oracle's citation comment (`7.2 row ...` / `layout.py line ...`) so a reviewer can diff the two files by formula.

Run: `cd LT/layout && pnpm exec vitest run test/simulation/forceatlas2.test.ts` -- Expected: PASS. The NetworkX case is the independence proof: the port reproduces the trajectories the GPU package's oracle was checked against.

- [ ] **Step 4: Coverage and lint**

Run: `cd LT/layout && pnpm run lint && pnpm run coverage 2>&1 | tail -20`
Expected: eslint and `tsc --noEmit` clean (layout's `test/` is outside its tsconfig and eslint set, so the new test files are checked only by vitest; the `src/simulation/` files are); coverage still above layout's thresholds (65/60/85/65 -- branches 85 is the tight one: the option-validation branches need the tests above to hit them; add cases to `forceatlas2.test.ts` for `linlog`, `strongGravity`, `distributedAction` and a weighted graph if branch coverage drops below 85).

### Task M5-T6: The steppable Fruchterman-Reingold ("spring")

**Files:**
- Create: `layout/src/simulation/fruchterman-reingold.ts`
- Create: `layout/test/simulation/fruchterman-reingold.test.ts`

- [ ] **Step 1: Write the failing tests** -- the same shape as M5-T5's first four cases (`load` / `step` moves positions and settles by `iterations`; deterministic for a seed; a fixed node never moves; `setPosition` reheats), plus: `it("one step equals one iteration of fruchtermanReingoldLayout's loop on the same start", ...)` comparing the simulation after `k` steps with the legacy function's positions after `k` iterations from the same `pos` (`toBeCloseTo` 1e-6 relative; the legacy function rescales at the end only when `fixed` is null, so pass a `fixed` list of one node to both to disable the rescale).

- [ ] **Step 2: Write the module** -- `class FruchtermanReingoldSimulation implements LayoutSimulation` over the loop body of `layout/src/layouts/force-directed/fruchterman-reingold.ts:70-151` (the `k` default `1 / sqrt(n)`, the temperature `t = 0.1` -- a CONSTANT start, line 81; design 7.20 -- decreasing by `dt = 0.1 / (iterations + 1)` per iteration, the `|| 0.1` zero-distance guard of lines 103 and 123, the repulsion `k*k / d`, the attraction `d*d / k`, the displacement capped at `t`, fixed nodes skipped), index-based over the snapshot's CSR (`for (a = rowPtr[u]; a < rowPtr[u + 1]; a++)` for the attraction, all pairs for the repulsion, no per-pair allocation), f64 scratch, scene-unit write-back like M5-T5. Settle rule as design 9.3 / 7.17 has it for every CPU simulation: `settled = iterationsDone >= iterations || settledCount >= settleWindow`, where `settledCount` increments when the mean per-free-node displacement of the iteration is `<= settleThreshold * rmsRadius` (the same fold M5-T5 uses; defaults `FA2_DEFAULTS.settleThreshold` / `settleWindow`) and resets otherwise; `iterationsPerStep` honoured; `maxInFlight` ignored. `fixed` from the options (a mask or a bool column name with role "fixed") is applied at `load()`; `setFixed` replaces it.

Run: `cd LT/layout && pnpm exec vitest run test/simulation/fruchterman-reingold.test.ts` -- Expected: PASS.

### Task M5-T7: The dispatcher, the snapshot adapter and the exports

**Files:**
- Create: `layout/src/simulation/create-simulation.ts`
- Create: `layout/src/simulation/snapshot.ts`
- Create: `layout/test/simulation/create-simulation.test.ts`
- Create: `layout/test/simulation/snapshot.test.ts`

- [ ] **Step 1: Write the failing dispatcher tests** (the design's fake-accelerator tests, 9.2 applied to 9.3):

```ts
import assert from "node:assert";
import { describe, it } from "vitest";

import { createSimulation, ForceAtlas2Simulation, FruchtermanReingoldSimulation } from "../../src/simulation";
import type { LayoutAccelerator, LayoutSimulation } from "../../src/simulation";

const fakeSim: LayoutSimulation = { load() {}, step() {}, settled: true, setFixed() {}, setPosition() {}, dispose() {} };

describe("createSimulation", () => {
    it("delegates to the accelerator when it has the method", () => {
        const calls: string[] = [];
        const acc: LayoutAccelerator = { kind: "fake", forceAtlas2: () => { calls.push("fa2"); return fakeSim; } };
        assert.equal(createSimulation("forceatlas2", {}, acc), fakeSim);
        assert.deepEqual(calls, ["fa2"]);
    });
    it("runs the CPU simulation when no accelerator is injected or the method is missing", () => {
        assert.ok(createSimulation("forceatlas2") instanceof ForceAtlas2Simulation);
        assert.ok(createSimulation("forceatlas2", {}, null) instanceof ForceAtlas2Simulation);
        assert.ok(createSimulation("forceatlas2", {}, { kind: "fake" }) instanceof ForceAtlas2Simulation);
        assert.ok(createSimulation("fruchtermanReingold") instanceof FruchtermanReingoldSimulation);
        assert.ok(createSimulation("spring") instanceof FruchtermanReingoldSimulation, "the element's name for FR");
    });
    it("routes spring to the accelerator's fruchtermanReingold", () => {
        const acc: LayoutAccelerator = { kind: "fake", fruchtermanReingold: () => fakeSim };
        assert.equal(createSimulation("spring", {}, acc), fakeSim);
    });
    it("a throwing accelerator method propagates (no fallback)", () => {
        const acc: LayoutAccelerator = { kind: "fake", forceAtlas2: () => { throw new Error("E_DEVICE_LOST"); } };
        assert.throws(() => createSimulation("forceatlas2", {}, acc), /E_DEVICE_LOST/);
    });
    it("spring-electrical has no CPU simulation in v1", () => {
        assert.throws(() => createSimulation("spring-electrical"), /spring-electrical/);
        const acc: LayoutAccelerator = { kind: "fake", springElectrical: () => fakeSim };
        assert.equal(createSimulation("spring-electrical", {}, acc), fakeSim);
    });
});
```

- [ ] **Step 2: Write the dispatcher**

```ts
import { ForceAtlas2Simulation } from "./forceatlas2";
import { FruchtermanReingoldSimulation } from "./fruchterman-reingold";
import type {
    ForceAtlas2Options,
    FruchtermanReingoldOptions,
    LayoutAccelerator,
    LayoutSimulation,
    SimulationType,
    SpringElectricalOptions,
} from "./types";

/**
 * The layout-side dispatcher (design 9.3): the accelerator's method when it has one, else the CPU simulation;
 * evaluated BEFORE any GPU work and never after it (design 2.4: the only branch that chooses the CPU). A thrown
 * accelerator error propagates.
 * @param type - the simulation type ("spring" is Fruchterman-Reingold)
 * @param options - the type's options
 * @param accelerator - the injected accelerator, or null / undefined for the CPU
 * @returns a simulation ready for load()
 */
export function createSimulation(
    type: SimulationType,
    options?: ForceAtlas2Options | FruchtermanReingoldOptions | SpringElectricalOptions,
    accelerator?: LayoutAccelerator | null,
): LayoutSimulation {
    switch (type) {
        case "forceatlas2":
            return accelerator?.forceAtlas2 !== undefined
                ? accelerator.forceAtlas2(options as ForceAtlas2Options | undefined)
                : new ForceAtlas2Simulation(options as ForceAtlas2Options | undefined);
        case "fruchtermanReingold":
        case "spring":
            return accelerator?.fruchtermanReingold !== undefined
                ? accelerator.fruchtermanReingold(options as FruchtermanReingoldOptions | undefined)
                : new FruchtermanReingoldSimulation(options as FruchtermanReingoldOptions | undefined);
        case "spring-electrical":
            if (accelerator?.springElectrical === undefined) {
                throw new Error('createSimulation: "spring-electrical" has no CPU simulation; inject an accelerator that implements springElectrical');
            }
            return accelerator.springElectrical(options as SpringElectricalOptions | undefined);
        default: {
            const never: never = type;
            throw new Error(`createSimulation: unknown simulation type ${String(never)}`);
        }
    }
}
```

- [ ] **Step 3: The snapshot adapter and its test**

`layout/src/simulation/snapshot.ts` -- the minimal `toLayoutSnapshot` of graph-format design 14.3 (an undirected snapshot passes through unchanged; a directed one yields its `toUndirected().snapshot`, cached; the legacy duck type is walked once through a `GraphBuilder({ directed: false, weighted: "auto" })`, ids preserved in `nodes()` order, `getEdgeData(s, t, weightAttr)` read when `weightAttr` is given; a `Node[]` becomes an edgeless snapshot; every result cached in a `WeakMap<object, GraphSnapshot>`):

```ts
import { GraphBuilder, type GraphSnapshot, isGraphSnapshot } from "@graphty/graph-format";

import type { Graph, Node } from "../types";

const cache = new WeakMap<object, GraphSnapshot>();

/**
 * The undirected snapshot of a layout input (graph-format design 14.3): a snapshot is returned as its undirected
 * derived graph (the input itself when already undirected); a legacy duck-typed graph is walked once and cached;
 * a node list becomes an edgeless snapshot.
 * @param G - the input
 * @param weightAttr - the edge attribute read through getEdgeData, or null for unweighted
 * @returns the undirected snapshot
 */
export function toLayoutSnapshot(G: Graph | Node[] | GraphSnapshot, weightAttr: string | null = null): GraphSnapshot {
    if (isGraphSnapshot(G)) {
        if (!G.directed) {
            return G;
        }
        // one undirected copy per directed source, never two (graph-format design 14.3; the format never caches
        // derived graphs itself)
        const cachedUndirected = cache.get(G);
        if (cachedUndirected !== undefined) {
            return cachedUndirected;
        }
        const undirected = G.toUndirected().snapshot;
        cache.set(G, undirected);
        return undirected;
    }
    const hit = cache.get(G);
    if (hit !== undefined) {
        return hit;
    }
    // weighted "auto": the snapshot carries weights only when getEdgeData supplied some (an explicit 1 would
    // mark every edge weighted); addMissingNodes defaults to true
    const builder = new GraphBuilder({ directed: false, weighted: "auto" });
    if (Array.isArray(G)) {
        builder.addNodes(G);
    } else {
        builder.addNodes(G.nodes());
        for (const [source, target] of G.edges()) {
            const weight = weightAttr !== null && G.getEdgeData !== undefined ? G.getEdgeData(source, target, weightAttr) : undefined;
            builder.addEdge(source, target, weight);
        }
    }
    const snapshot = builder.freeze();
    cache.set(G, snapshot);
    return snapshot;
}
```

Test (`layout/test/simulation/snapshot.test.ts`): a duck graph of 3 nodes / 3 edges yields `nodeCount 3`, `arcCount 6`, `ids.idOf(i)` in `nodes()` order and `weights === null`; the same object returns the cached snapshot (`===`); a weighted duck graph with `getEdgeData` and `weightAttr "w"` yields `weights !== null`; a `Node[]` yields `arcCount 0`; a directed snapshot input yields an undirected one, and the SAME one on a second call (`===`); an undirected snapshot is returned as-is (`===`).

- [ ] **Step 4: Run everything**

Run: `cd LT/layout && pnpm exec vitest run test/simulation && npm run build:all && pnpm exec vitest run test/package-structure.test.ts && pnpm run lint`
Expected: all green, including M5-T1's two tests. `cd LT && pnpm exec knip --workspace layout` -- expected: no findings (every new export is reached from `src/index.ts`).

### Task M5-T8: The legacy forceatlas2Layout over the simulation (D-17) and the Chromatic re-baseline

**Files:**
- Modify: `layout/src/layouts/force-directed/forceatlas2.ts` (the body becomes a wrapper; the 15 positional parameters and the return type are unchanged)
- Modify: `layout/test/forceatlas2-layout.test.ts` (only if an assertion fails for a documented reason)

- [ ] **Step 1: Run the existing FA2 suite first** -- `cd LT/layout && pnpm exec vitest run test/forceatlas2-layout.test.ts test/forceatlas2-3d-edge-cases.test.ts test/forceatlas2-npm-bug.test.ts test/bug-report-2-forceatlas2-3d.test.ts` -- Expected: PASS (the baseline before the rewrite; these tests pin no exact coordinate, design 9.3).

- [ ] **Step 2: Rewrite the body**

Replace the implementation of `forceatlas2Layout` (keep the signature at lines 26-42) with:

```ts
    const s = toLayoutSnapshot(G, weight);
    const n = s.nodeCount;
    const dimension: 2 | 3 = dim === 3 ? 3 : 2;
    const positions = new Float32Array(3 * n).fill(Number.NaN);
    if (pos !== null) {
        for (let i = 0; i < n; i++) {
            const given = pos[s.ids.idOf(i)];
            if (given !== undefined) {
                positions[3 * i] = given[0];
                positions[3 * i + 1] = given[1];
                positions[3 * i + 2] = dimension === 3 ? (given[2] ?? 0) : 0;
            }
        }
    }
    seedPositions(s, positions, seed, dimension, 1, null, "fa2");
    const sim = new ForceAtlas2Simulation({
        maxIter, jitterTolerance, scalingRatio, gravity, distributedAction, strongGravity,
        nodeMass, nodeSize,
        // the legacy attribute NAME was consumed by toLayoutSnapshot (getEdgeData -> the snapshot's arc weights);
        // the simulation's `weight` is therefore a boolean here, never the attribute name
        weight: weight !== null,
        dissuadeHubs: _dissuadeHubs, linlog,
        dim: dimension, settleThreshold: 0,   // the one-shot function runs every iteration (design 9.3)
    });
    sim.load(s, positions);
    sim.step(maxIter);
    sim.dispose();
    const result: PositionMap = {};
    for (let i = 0; i < n; i++) {
        const row = [positions[3 * i], positions[3 * i + 1]];
        if (dimension === 3) {
            row.push(positions[3 * i + 2]);
        }
        result[s.ids.idOf(i)] = row;
    }
    return rescaleLayout(result) as PositionMap;   // rescaleLayout is typed PositionMap | number[][]; the legacy body casts the same way
```

Replace the file's import block with `import { ForceAtlas2Simulation, seedPositions, toLayoutSnapshot } from "../../simulation";`, `import type { Graph, Node, PositionMap } from "../../types";` and `import { rescaleLayout } from "../../utils/rescale";` (sorted per `simple-import-sort`), and delete the now-unused module-private helpers of the old body (`getNodeDegree` at the bottom of the file, the `RandomNumberGenerator` import, `_processParams` / `getNodesFromGraph` if present): `@typescript-eslint/no-unused-vars` is an error in the root config. Documented behaviour changes (design 9.3, 7.2, graph-format design 14.3 "Chromatic re-baselining"): the published FA2 laws replace the port's `1/d^2` repulsion; force-based swing / traction; the attraction sums over parallel arcs (the dense matrix collapsed them); `weight` now reaches the layout from the element; mass defaults to `outDegree() + 1` (self-loop once).

- [ ] **Step 3: Run the suite again** -- the four files above; Expected: PASS (determinism, separation, bounds, aspect ratio, the mass test, and the "should handle edge weights" case that passes `"weight"` as the attribute name). Add one case to `test/forceatlas2-layout.test.ts`: a duck graph with `getEdgeData` returning distinct weights laid out with `weight: "w"` runs without throwing and differs from the unweighted result of the same seed. If an assertion fails for one of the documented reasons, relax it to the documented behaviour with a comment citing design 9.3; anything else is a port bug in M5-T5.

- [ ] **Step 4: The element and the stories** -- `cd LT && pnpm exec nx run-many -t build --projects=layout,graphty-element && (cd graphty-element && npm run test:shard:default:run)`; Expected: green (graphty-element calls the same positional signature). Then `pnpm exec nx run layout:build-storybook` and open the ForceAtlas2 stories locally (`cd layout && PORT=9061 npm run storybook` -- the script defaults to 6006, outside the allowed 9000-9099 range); take a screenshot of the 2D ForceAtlas2 story through the Playwright MCP and ask the Nanobanana MCP two objective yes/no questions: "Are the nodes spread out rather than collapsed onto one point?" and "Do connected nodes appear closer to each other than unconnected ones?" (both must be yes; the owner's visual-verification rule).

- [ ] **Step 5: Commit and re-baseline (owner)** -- two commits through `tools/commit-changes.sh`: `feat(layout): the simulation seam of the WebGPU design and the steppable ForceAtlas2` for M5-T1..T7, then `feat(layout): forceatlas2Layout runs on the steppable simulation with the published laws` for this task. No `!`: the signatures are unchanged and a major bump would cascade into graphty-element and the app; the behaviour change is documented in the body, as graph-format design 14.3 prescribes for the re-baseline commit ("one commit per package re-baselines stories whose output changes for the documented reasons; the commit message lists the reasons"). The `Chromatic (layout)` job of the PR shows the changed ForceAtlas2 snapshots; the owner accepts them in Chromatic; `all-checks` then passes.

### Task M5-T9: Documentation and the design amendments

**Files:**
- Modify: `layout/CLAUDE.md` (a "Simulations" section: the seam, the units, the two classes, the dispatcher, the tests)
- Modify: `layout/README.md` (a short "Steppable simulations" section with a 10-line example: `toLayoutSnapshot`, `seedPositions`, `createSimulation("forceatlas2")`, `load`, `step` until `settled`)
- Modify: `design/graph-format/graph-format-design.md` (APPEND to section 17's decision log, never edit 13.5 / 14.3 in place): "17.x (<the re-baseline commit's date, `git log -1 --format=%cs`>): rule 3's peer range during 0.x is the minor pin (`^0.<minor>.0`, re-stated at every format minor: `6b4777df`) and the intended workspace protocol is `workspace:^` (D-18 and Q-31 via the integration plan); 14.3's ForceAtlas2 adopts the WebGPU design's 7.2 reference formulas (DEPARTURE-3) and the GPU keeps its own vec4f device positions (DEPARTURE-7)"
- Modify: `design/webgpu/webgpu-acceleration-plan.md` (APPEND a Review-log entry: "L1-sim landed <the same date>: layout/src/simulation/ per 9.3; the CPU ForceAtlas2Simulation is a copy of the oracle's formulas (DEP-E); both CPU simulations settle by the 7.17 rule")

- [ ] **Step 1: Write the four edits.** Commit (owner): `docs(layout): describe the simulation seam and record the L1 amendments`.

---

## Phase M5b: The GPU package adopts the real layout interfaces (W1b, layout half)

**Entry:** Phase M5 on master (`@graphty/layout` exports the simulation seam). Branch `feat/webgpu-layout-types`, scope `webgpu-graph-algorithms`, work in `webgpu-graph-algorithms/` of a fresh worktree (`GT`).

### Task M5b-T1: The devDependency and the graph edge

**Files:**
- Modify: `webgpu-graph-algorithms/package.json` (devDependencies: `"@graphty/layout": "workspace:^"`)
- Modify: `webgpu-graph-algorithms/project.json` (`implicitDependencies: ["!algorithms"]` -- the `!layout` negation goes: the edge is real now, D-19)
- Modify: `webgpu-graph-algorithms/tsconfig.json` (`paths`: `"@graphty/layout": ["../layout/dist/layout.d.ts"]` -- the BUILT declarations, not layout's sources: layout compiles with `noUnusedLocals` / `noUnusedParameters` off and its sources would fail the GPU package's stricter `tsc --noEmit`; D-20's `lint -> build -> ^build` chain builds layout first)
- Modify: `webgpu-graph-algorithms/tsconfig.strict-consumer.json` (`paths`: `"@graphty/layout": ["../layout/dist/layout.d.ts"]`)
- Modify: `webgpu-graph-algorithms/test/build-output.test.ts` (the devDependency table, if pinned)

**Step 0 of the phase (fresh worktree):** `cd GT && HUSKY=0 pnpm install --frozen-lockfile && pnpm exec nx run-many -t build --projects=graph-format,layout` (both `dist/` directories are gitignored; the strict-consumer compile, the type tests and the seed cross-test read them).

- [ ] **Step 1:** Make the five edits; `cd GT && HUSKY=0 pnpm install && HUSKY=0 pnpm install --frozen-lockfile`. Check: `pnpm exec nx show project webgpu-graph-algorithms --json | node -e 'const p=JSON.parse(require("fs").readFileSync(0,"utf8")); console.log(p.implicitDependencies)'` prints `[ '!algorithms' ]`, and the project graph now lists `layout` among the package's dependencies (`nx graph --file`). `updateDependents: auto` will patch-bump this package on every layout release from now on (design Q-29, accepted).

### Task M5b-T2: The mirrors become imports

**Files:**
- Create: `webgpu-graph-algorithms/test/types/conformance.test-d.ts` (design 9.8's W1 row says the file is "retired"; G10 and `src/types/accelerator.ts`'s own header require its `expectTypeOf` cross-compile, so it is CREATED here and stays -- D-9)
- Modify: `webgpu-graph-algorithms/src/types/accelerator.ts:17-38` (delete the two mirrored interfaces; `import type { LayoutAccelerator, LayoutSimulation } from "@graphty/layout"; export type { LayoutAccelerator, LayoutSimulation };`; prune the imports only the deleted mirrors used -- `NodeMask`, `FruchtermanReingoldOptions`, `SpringElectricalOptions` -- `noUnusedLocals` is on)
- Modify: `webgpu-graph-algorithms/src/types/options.ts:10-57` (delete the five mirrored option interfaces; `import type { CommonLayoutOptions, ForceAtlas2Options, FruchtermanReingoldOptions, SimulationOptions, SpringElectricalOptions } from "@graphty/layout";` and re-export them; keep `ResolvedForceAtlas2Options`)
- Modify: `webgpu-graph-algorithms/eslint.config.js` (add `src/types/options.ts` to the file set allowed to `import type` from the CPU packages -- `CPU_PATHS_TYPES_ALLOWED` around line 24-28 / the `src/types/**` override around 156-166)
- Modify: `webgpu-graph-algorithms/src/layouts/seed.ts` (header comment: the layout copy is canonical; this one stays for the package's own use, cross-tested below)

- [ ] **Step 1: Write the failing type test first** -- create `webgpu-graph-algorithms/test/types/conformance.test-d.ts`:

```ts
// Compiled by both tsconfigs: tsconfig.json (paths to the sources) and tsconfig.strict-consumer.json (paths to
// dist/*.d.ts under exactOptionalPropertyTypes + noUncheckedIndexedAccess). Import through the PACKAGE NAMES, as the
// three sibling test-d files do: a relative `../../src/...` import would pull src/ into the strict-consumer program,
// which src/ does not satisfy (1481 errors), and the root eslint config sorts and de-duplicates imports.
import type { LayoutAccelerator, LayoutSimulation } from "@graphty/layout";
import {
    createAccelerator,
    type ForceAtlas2Options,
    type ForceAtlas2Stats,
    type GpuContext,
    type GpuLayoutSimulation,
} from "@graphty/webgpu-graph-algorithms";
import { expectTypeOf, test } from "vitest";

declare const ctx: GpuContext;

test("the accelerator satisfies @graphty/layout's LayoutAccelerator (design 9.8 W1)", () => {
    expectTypeOf(createAccelerator(ctx)).toMatchTypeOf<LayoutAccelerator>();
});

test("the GPU simulation satisfies LayoutSimulation and a LayoutSimulation is what the element sees", () => {
    expectTypeOf<GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>>().toMatchTypeOf<LayoutSimulation>();
    expectTypeOf<ReturnType<NonNullable<LayoutAccelerator["forceAtlas2"]>>>().toEqualTypeOf<LayoutSimulation>();
});
```

Run: `cd GT/webgpu-graph-algorithms && pnpm exec tsc --noEmit -p tsconfig.json` -- Expected: after M5b-T1 the file compiles (the mirrors and the real interfaces are field-for-field identical, so it passes even before Step 2; Step 2 is what makes the assertion meaningful). The strict-consumer pass (`tsc -p tsconfig.strict-consumer.json`, part of `lint`) needs `build:bundle` first (D-20).

- [ ] **Step 2: Replace the mirrors** as listed in Files; `test/types/accelerator.test-d.ts`'s keyof pins keep compiling because the real interfaces have exactly the mirrored members (design 9.3 verbatim on both sides). Run `pnpm run lint` -- Expected: clean, including the strict-consumer compile (the published d.ts now references `@graphty/layout`, D27's accepted consequence).

### Task M5b-T3: The second oracle and the seed cross-test

**Files:**
- Create: `webgpu-graph-algorithms/test/layouts/fa2-layout-oracle.test.ts`
- Create: `webgpu-graph-algorithms/test/layouts/seed-cross.test.ts`

- [ ] **Step 1: seed cross-test** -- for seeds `[1, 7, 42, 123456]`, dims 2 and 3, `scale` 1 and 5, a null and a non-null center, an all-NaN array and a half-finite array: `seedPositions` from `../../src/layouts/seed.js` and from `@graphty/layout` write bit-identical Float32Arrays (`expect(Array.from(a)).toEqual(Array.from(b))`). Node-only, no GPU.
- [ ] **Step 2: the second oracle** -- reuse `test/helpers/fa2-parity.ts`'s fixtures (`karate` / `grid10` / `random1k` at `gpuScale()`): after `k = 1` and `k = 5` iterations (and `k = 10` in `compat: "networkx"`), the GPU positions are within 5e-2 relative of layout's `ForceAtlas2Simulation` started from the same f32 seed in the same `compat` -- the tolerance `fa2-trace-parity.test.ts` asserts on its re-synchronised f64 leg. The 50-iteration leg is PRINTED, never asserted: the free-running f64 trajectory is chaotic (G3-F3; `fa2-trace-parity.test.ts:12-21` records that the f64 oracle misses 5e-2 against ITSELF under a one-ulp start perturbation by iteration 50). The existing oracle tests stay untouched (the independent references stay).

Run: `cd GT/webgpu-graph-algorithms && GRAPHTY_GPU_ADAPTER=llvmpipe GRAPHTY_GPU_REQUIRE=any VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json XDG_RUNTIME_DIR=/tmp pnpm exec vitest run --project=node test/layouts/fa2-layout-oracle.test.ts test/layouts/seed-cross.test.ts && pnpm run lint` -- Expected: PASS on lavapipe (the `*.test-d.ts` type tests run through `lint`'s two tsc passes, not through vitest); then the same with `LD_LIBRARY_PATH=/home/apowers/Projects/graphty-monorepo/tmp/egl/root/usr/lib/x86_64-linux-gnu GRAPHTY_GPU_REQUIRE=hardware` on the RTX 4070 SUPER.

- [ ] **Step 3: Commit (owner)** -- `feat(webgpu-graph-algorithms): adopt the layout simulation types and the second FA2 oracle` (90 characters; the body says: @graphty/layout's LayoutSimulation / LayoutAccelerator and option types replace the D27 mirrors; layout's ForceAtlas2Simulation is the second FA2 oracle; the seedPositions cross-test); the PR runs both shards; the design's G10 items for the layout half are met (`expectTypeOf(createAccelerator(ctx)).toMatchTypeOf<LayoutAccelerator>()` and the reverse; every differential test against BOTH oracles). Append the Review-log entry "W1b (layout half) landed" to the design. The algorithms half (`AlgorithmAccelerator` import, `!algorithms` negation removed, `CpuAlgorithmOptions` retired) repeats these three tasks after Phase M8a.

### Task M5b-T4: The W1 design amendments

**Files:**
- Modify: `design/graph-format/graph-format-design.md` (APPEND a section 17.6 table; never edit 10.3 / 14.5 / 14.6 / 16.2 / 16.7 in place)
- Modify: `design/webgpu/webgpu-acceleration-plan.md` (APPEND a Review-log entry)

Design 9.8 (W1 row) and 13 (P10 row) list the "design 10.3 / 14.5 / 14.6 / 16.2 / 16.7 amendments (DEPARTURE-1, -2, -4, -5, -6)" as a W1 deliverable, and DEPARTURE-1 says they are recorded "when this plan is accepted"; the graph-format design's decision log (17.5) has no WebGPU entry today. Docs only.

- [ ] **Step 1:** Append to `design/graph-format/graph-format-design.md` after its last section-17 entry a `### 17.6 Amendments from the WebGPU design (W1, <date>)` table with one row per departure, each citing the WebGPU design line: DEPARTURE-1 (node-first testing with a browser smoke: amends 14.5 "browser-only vitest project", 14.6's W1 gate "browser tests green" and 16.7), DEPARTURE-2 (the hot-prefix arena upload: amends 10.3), DEPARTURE-4 (the per-snapshot residency record: amends 14.5), DEPARTURE-5 (injection spelled `accelerated(acc).x(s, options)`: amends 14.5), DEPARTURE-6 (1e-4 for betweenness and one-iteration force parity: amends 16.2). The DEPARTURE texts are `design/webgpu/webgpu-acceleration-plan.md:225-237`.
- [ ] **Step 2:** Append to the WebGPU design's Review log: "W1 amendments recorded in graph-format-design.md 17.6 (<the commit date, YYYY-MM-DD>)".
- [ ] **Step 3: Commit (owner)** -- `docs(webgpu-graph-algorithms): record the W1 amendments in the graph-format design`.

---
## 6. The later phases: scoped, gated, planned when their entry criteria are met

Each phase below is defined by its entry criteria, its deliverable, its gate and the design sections it executes. Each gets its own writing-plans pass (a plan file under `design/webgpu/plans/`) when the criteria are met; the tasks are not decomposed here because they depend on code that does not exist yet (`indexed.*`, the element's `DataManager` refactor) and on decisions the owner takes at those gates. The order is what both designs fix (graph-format 14.6; WebGPU 9.8, 13).

### Phase M6: graphty-element (E0, then E1) -- design 9.4, graph-format design 14.4

| | |
| --- | --- |
| Entry | Phase M5 on master; the first A2 commit (Phase M8a) on master; graph-format `>= 1.0.0`. |
| E0 (a precondition the WebGPU design names and does not size) | graph-format design 14.4: `DataManager` owns one `GraphBuilder` for the graph's life; an element-owned `positions: Float32Array(3 * capacity)` attached by reference as the `position` column after every freeze; `getSnapshot()`, `dm.undirected(s)`, `snapshot-replaced { previous, next, report }`; `Node.index`; `LayoutManager` calls `engine.load(dm.undirected(getSnapshot()).snapshot, positions)` / `engine.reload(...)`; `toAlgorithmGraph` and `EdgeMap` retired. This is the element's own migration; it is the larger half of the phase. |
| E1 deliverables (design 9.4 items 1-10) | `Graph.accelerator: GraphAccelerator \| null`, `setAccelerator()`, the `accelerator-changed` event and its `LayoutManager` consumer (late injection engages a running layout; `setAccelerator(null)` moves it to the CPU with positions and pins kept); the `snapshot-replaced` release list; adapters through `accelerated(this.graph.accelerator).x(s)` with ONE result-writing loop for CPU and GPU; the `SimulationLayoutEngine` bridge (`load` / `reload` / `step` once per frame with `.catch` attached once per distinct promise / `pin` / `unpin` / `setNodePosition` / `beginDrag` / `endDrag` / `getNodePositionInto` / `dispose`); `ForceAtlas2LayoutEngine` and `SpringLayoutEngine` re-registered on it under the same type names and zod schemas (`gravity` loosened to nonnegative, `weightPath` live, `scalingFactor` becomes the `scale` option); `behavior.layout.iterationsPerStep`, `maxInFlight`, `gpuMinNodes`; `setRunning(running)` with `reheat()` on resume; `nodeMass` / `nodeSize` records resolved once into role columns (D28); the fake-accelerator stories `Layout/ForceAtlas2 (GPU)` and `Layout/Spring (GPU)` for Chromatic. graphty-element gains NO dependency on the GPU package, not even dev (design 9.1). |
| Gate (design G6, element part) | element tests + stories green with a fake accelerator: late injection, mid-run removal with positions and pins preserved, pin survival across a remap (`pin A, remove B < A, freeze, reload -> A still fixed`), `gpuMinNodes` above the node count keeps the CPU engine until a reload crosses it, `setRunning(false)` lands only the in-flight batches and `setRunning(true)` reheats a settled simulation; the CPU and GPU stories look statistically the same (design 11.4 distributional metrics); Chromatic re-baseline. |
| Frame-loop facts to design against | `UpdateManager.updateLayout()` calls `layoutManager.step()` `stepMultiplier` times per Babylon render frame synchronously; `Node.update()` pulls `layoutEngine.getNodePosition(this)` unless `node.dragging`; `NodeBehavior` writes `layoutEngine.setNodePosition` every pointer move and `node.pin()` on drag end when `pinOnDrag`; `LayoutManager.isSettled` and `Graph.update()`'s `graph-settled`. The GPU `step(k)` coalesces above `maxInFlight` (design 7.19), so the per-frame call is naturally throttled. |
| Size (design) | 8-10 ed for E1 across the three packages, on top of E0, which the design does not size. |

### Phase M7: the graphty app (W2) -- design 9.5, 12 (P12)

| | |
| --- | --- |
| Entry | Phase M6 on master. |
| Deliverables | `graphty/src/gpu/accelerator.ts` `attachAccelerator(element, { gpu: "auto" \| "off" \| "required", exactMaxNodes?, calibrate? })` with static imports from `@graphty/webgpu-graph-algorithms/browser` (`probeBrowserWebGpu`, `requestGpuContext`) and the root (`createAccelerator`, `calibrateLayout` -- the latter lands with the design's P4); `ctx.lost.then(() => element.setAccelerator(null) + toast)`; the "GPU acceleration: on (vendor arch) / off" indicator from `ctx.caps`; the real-GPU stories under a `gpu` tag in the APP's Storybook that render a deterministic no-GPU state on Chromatic (no TurboSnap there; `exitZeroOnChanges: false`); the `metricCost.ts` gate gains a per-metric accelerator constant once the element exposes the flag. The app is the only importer of the GPU package: `graphty/package.json` gains `"@graphty/webgpu-graph-algorithms": "workspace:^"`, which puts the package into graphty's dependency closure -- at that point the `Build graph-format, graph-io and webgpu-graph-algorithms (PR)` step of `ci.yml` becomes redundant for it. |
| Gate (design G12, W2 subset) | the app's stories green; the story on the real GPU on the dev box settles, drags and pins (Playwright MCP screenshot + Nanobanana yes/no questions, the owner's visual rule); `gpuMinNodes` default measured (design 7.21). |
| Size (design) | 4-6 ed. |

### Phase M8: algorithms -- M8a the first A2 commit (design 9.2), M8b the GPU SpMV family (design P7)

| | |
| --- | --- |
| Entry M8a | graph-format `>= 1.0.0` on master and the A1 branch merged (graph-format design 14.6); prepared on a branch after A1, merged after F2. |
| M8a deliverables | `algorithms/src/indexed/accelerator.ts` exported from the barrel: the `*ResultLike` shapes (scores as `NumericVector`), `AlgorithmAccelerator` (every method optional, `GraphSnapshot` in), `AcceleratedAlgorithms`, `accelerated(acc)` (`acc?.x !== undefined ? acc.x(s, ...) : Promise.resolve(indexed.x(s, ...))`, `pathTo` / `pathEdges` decoration for SSSP), `sources` / `k` on `BetweennessCentralityOptions`; fake-accelerator tests (delegation, CPU path, a throwing method propagates). Additive; it carries only the methods whose `indexed.*` port exists. Then the GPU package's W1b algorithms half: `AlgorithmAccelerator` by `import type`, `CpuAlgorithmOptions` retired, `implicitDependencies` negation removed, the conformance test's reverse compile (design G10 `expectTypeOf(createAccelerator(ctx)).toMatchTypeOf<AlgorithmAccelerator & LayoutAccelerator>()`). |
| Entry M8b | Phase M3 (the package in the monorepo) and the design's P2 gate (met); may run in parallel with M5-M7 (design: "P7 may start after P2"). |
| M8b deliverables (design P7) | grid-stride dispatch, `packViews`, `spmvPull`, PageRank (+ personalized) with `firstConvergedIteration`, HITS, eigenvector, Katz, Afforest WCC, `renumberPartition` on readback, `reverse()` residency, the `GpuAccelerator` algorithm members, oracles, `pagerank` / `wcc` benchmarks; result parity per design 9.7 (1e-5 relative; labels renumbered first-seen; `precision: "f32"`). |
| Gate | design G7. |
| Size (design) | 8-10 ed for P7; the first A2 commit is small. |

### The GPU package's own next phases inside the monorepo

The design's remaining phases run in `webgpu-graph-algorithms/` exactly as P0-P3 did, with the monorepo's lanes: P-ENV (Ubuntu 24.04 dev container and `webgpu` 0.6.x -- a cross-package change now: graph-format's devDependency and the shared lockfile; the T4 image facts of Phase M4 decide whether the pin can move), P4 (grid pyramid, `node-limits` on the T4), P5 (Fruchterman-Reingold and the spring-electrical preset; adds `fruchtermanReingold` / `springElectrical` to `createAccelerator`, which `createSimulation` then routes), P8-P9, P11, P12. Each phase plan is written into `design/webgpu/plans/` with the monorepo paths; the gate records go on under `webgpu-graph-algorithms/docs/decisions/`.

---

## 7. Appendices

### 7.1 The owner's command sheet (in order)

| When | Command (paste as an `!` command in this session, or run in a shell) |
| --- | --- |
| M0-T1 | `bash /home/apowers/Projects/webgpu-graph-algorithms/tmp/ci-round11.sh` |
| M0-T2 | nothing to run: the CI repair, the graph-io peer range (`6b4777df`, peer `^0.2.0`) and the 0.2.0 release (`dc08826b`) are on origin/master; M1-T1 fast-forwards the local master |
| M1-T1 | `cd /home/apowers/Projects/graphty-monorepo && git fetch origin && git merge --ff-only origin/master && git worktree add .worktrees/land-webgpu-graph-algorithms -b land/webgpu-graph-algorithms master` |
| M1-T3 | `cd /home/apowers/Projects/graphty-monorepo/.worktrees/land-webgpu-graph-algorithms && ./tools/land-webgpu-graph-algorithms.sh prepare` |
| M1-T4 | `... && ./tools/land-webgpu-graph-algorithms.sh merge` |
| M2-T9 | `... && for s in workspace package docs ignore; do ./tools/land-webgpu-graph-algorithms.sh commit $s || break; done` |
| M3-T5 | `... && ./tools/land-webgpu-graph-algorithms.sh commit ci && ./tools/land-webgpu-graph-algorithms.sh push` then `gh pr create ...` (Task M3-T5 step 2) |
| M3-T7 | `npx -y -p npm@11 npm trust github @graphty/webgpu-graph-algorithms --repo graphty-org/graphty-monorepo --file release.yml --allow-publish` then `cd /home/apowers/Projects/graphty-monorepo/.worktrees/land-webgpu-graph-algorithms && ./tools/land-webgpu-graph-algorithms.sh land --skip-gate` |
| M3-T8 | `bash /home/apowers/Projects/webgpu-graph-algorithms/tmp/ci-round12.sh`, cancel the queued GPU run, then `gh repo archive graphty-org/webgpu-graph-algorithms --confirm` |
| M4 | the GitHub UI steps of Task M4-T1; the `gh api ... labels` call; the `tools/commit-changes.sh` commits of M4-T2/T3/T5 |

The agent never runs any of these; it prepares the tree and verifies the results.

### 7.2 Verification matrix

| Check | Where | Command | Green means |
| --- | --- | --- | --- |
| The rewrite | M1-T3 | `git -C WT/tmp/land/wga-rewrite rev-list --count HEAD` = 31; `ls-tree` = `design webgpu-graph-algorithms` | the path rules are right |
| The merge | M1-T4 | `git log --first-parent --oneline -2`; `git rev-list --max-parents=0 HEAD \| wc -l` = 6 | history attached, one merge, one new root |
| Package tests on lavapipe | M1-T5 (without `--coverage`), M2-T6 and the CI node shard (with) | `GRAPHTY_GPU_ADAPTER=llvmpipe GRAPHTY_GPU_REQUIRE=any VK_DRIVER_FILES=... pnpm exec vitest run --project=node --coverage` | 94/94/98/94 coverage, 0 failures |
| Browser smoke on SwiftShader | CI browser shard | `GRAPHTY_BROWSER_GPU=swiftshader GRAPHTY_GPU_REQUIRE=any node scripts/run-browser-project.js` | `numTotalTests=39 numFailedTests=0` |
| Lint incl. strict-consumer | M2-T8 | `pnpm exec nx run webgpu-graph-algorithms:lint` (builds first) | eslint + 2 tsc runs clean |
| knip | M2-T3 | `pnpm exec knip` | no finding under the package |
| Release dry run | M2-T2 | `NX_DAEMON=false pnpm exec nx release --dry-run --skip-publish` | `webgpu-graph-algorithms 0.1.0 -> 0.2.0` listed, no range error |
| Frozen lockfile | M2-T8 | `HUSKY=0 pnpm install --frozen-lockfile` | exit 0 |
| Pre-push gate | M2-T8 | `./tools/prepush.sh` | exit 0 |
| PR | M3-T6 | `gh run watch <id> --exit-status` (CI and Hosts), then `gh pr checks` | `All Checks Pass`, both `Hosts` jobs |
| Publication | M3-T7 | `npm view @graphty/webgpu-graph-algorithms version` | `0.2.0` with provenance |
| GPU lane | M4-T2 | `gh run watch <id> --exit-status` | `vendor=nvidia` in the report; every step green |
| Layout seam | M5 | `pnpm exec vitest run test/simulation` in `layout/`; the NetworkX fixture case | the port reproduces NetworkX within 1e-4 |
| Two oracles | M5b | `test/layouts/fa2-layout-oracle.test.ts` on lavapipe and NVIDIA | GPU within 5e-2 of layout's CPU simulation |

### 7.3 Risk register for this plan

| Id | Risk | Mitigation |
| --- | --- | --- |
| R-M1 | The unrelated-history merge is done after package files exist on the branch, detaching the history. | The landing script's `merge` refuses when `webgpu-graph-algorithms/` or `design/webgpu/` exists or the branch has commits; Task M1-T4 step 2 checks `git log -- webgpu-graph-algorithms/package.json` reaches the imported commits. |
| R-M2 | Master CI is red for unrelated reasons and the landing PR's signal is muddied; `release.yml` never runs. | M0-T2 asks the owner to land the repair first; under option (b) the gate is "no new red job" and the release waits. |
| R-M3 | The first `nx release` publishes six packages at once (three first releases) and one trusted publisher is missing -> tags exist, publish fails. | M3-T7 step 1 configures the publisher BEFORE landing; `nx release publish` is idempotent and re-runnable (`release.yml` already runs it explicitly). |
| R-M4 | Playwright 1.57.0 (monorepo) vs 1.63.0 (staging) changes the bundled Chromium; SwiftShader grants no adapter. | D-11; the browser shard is the check; `scripts/probe-browser-flags.mjs` finds the flag set; a `package`-scoped follow-up. |
| R-M5 | The node shard passes 10 minutes on lavapipe as later phases add tests. | design 12.6: split with `--shard=1/2` into two matrix entries. |
| R-M6 | The T4 image differs from the assumptions (`modprobe`, xvfb, libegl1). | M4-T2 records the facts on the first run; G0 follow-up U1 (xvfb) is the known fix. |
| R-M7 | The `!algorithms` / `!layout` negation hides a real dependency later. | M5b-T1 and M8a remove each negation when the real devDependency arrives; the conformance tests fail to compile if the types are missing. |
| R-M8 | Layout's `moduleResolution: "node"` cannot see graph-format's `exports` map. | graph-format keeps top-level `main` / `types`; M5-T1 relies on them; if graph-format drops them, add `paths` to layout's tsconfig as graph-io does. |
| R-M9 | The FA2 port diverges from the oracle in a formula. | M5-T5's NetworkX fixture case (four fixtures, 1e-4) and M5b-T3's GPU-vs-layout parity (5e-2) catch it from two independent sides (design R-1). |
| R-M10 | The `git ls-files --eol` probe (M1-T7) reveals CRLF files outside the two corpora. | Add `-text` exemptions rather than rewriting; record in the `ignore` commit body. |
| R-M11 | A `git push` of the branch runs the full pre-push gate (15-25 min) and the remote-logger port flake trips it. | `push --skip-gate` after M2-T8 already ran the gate. |
