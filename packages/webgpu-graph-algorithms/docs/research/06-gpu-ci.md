# 06 -- GPU-based testing in GitHub Actions

Status: research note for the WebGPU plan (planning only, no implementation).
Date: 2026-09-14. Scope: how to run this package's GPU tests in CI now
(standalone repo) and later (graphty-monorepo), with one default lane that
needs no GPU and one GPU lane.

Every fact about existing code cites a path; every external fact cites a
URL (collected in "Sources" at the end). Measurements were taken on the
owner's dev box on 2026-09-14 with the probe scripts under
`tmp/webgpu-plan/probe/` (kept next to this note; not part of the package).

---

## 0. TL;DR

1. **The full correctness suite can run on every PR on `ubuntu-latest` with
   no GPU**, using Dawn-in-Node (`webgpu` npm package) on Mesa **lavapipe**
   (`apt-get install mesa-vulkan-drivers libvulkan1`, then
   `create(["adapter=llvmpipe"])`). This is exactly what the `webgpu`
   package's own CI does (`WEBGPU_USE_CI_AVAILABLE_RENDERER=1` ->
   `adapter=llvmpipe` on Linux, `adapter=Microsoft` (WARP) on Windows), and
   what wgpu and three.js do for their GPU suites. Verified locally: the
   graph-format GPU audit suites already run on llvmpipe when the NVIDIA ICD
   is unusable, and a 100k-node / 1M-arc CSR gather kernel runs 50
   iterations in ~170-270 ms on lavapipe pinned to 4 threads (a hosted
   runner's core count) vs ~5-36 ms on the RTX 4070 SUPER. Slow, but far
   inside any test budget.
2. **GitHub's own GPU-hosted runners are not an option for graphty-org
   today**: larger runners (which the GPU T4 runners are) are only available
   to organizations on GitHub Team or Enterprise Cloud; `gh api /orgs/graphty-org`
   reports `plan: free`. They also are not free for public repos
   ($0.052/min Linux). Revisit only if the org plan changes.
3. **The GPU lane should be a self-hosted ephemeral runner on the owner's
   dev box** (RTX 4070 SUPER, driver 580.173.02), which is what
   `atoms-org/cuda-ffi` does (`runs-on: cudaffi-gpu-runner`, a custom
   self-hosted label, with a `container:` that has `options: --gpus all`).
   Because both repos are public, the runner must be ephemeral (one job then
   deregister), fork PRs must never reach it (gate on `pull_request` from the
   same repo plus a label, or on `push` to master / schedule /
   `workflow_dispatch`), and the "Require approval for all external
   contributors" setting must be on.
4. **What runs where**: default lane = build, lint, typecheck, the whole
   Node correctness suite on lavapipe, plus a small Chromium browser smoke
   on SwiftShader (`--enable-unsafe-webgpu --use-angle=swiftshader
   --enable-unsafe-swiftshader`; verified locally: Chromium 139 SwiftShader
   adapter gives bit-identical results to the NVIDIA adapter on the probe
   kernel). GPU lane = the same suite on NVIDIA with `GRAPHTY_GPU_REQUIRE=nvidia`
   (so a silent software fallback fails), the limit-dependent tests
   (lavapipe caps `maxStorageBufferBindingSize` at 128 MiB, NVIDIA at 2 GiB
   in Dawn / 4 GiB in Chromium), benchmarks with baseline comparison, and the
   browser-on-real-GPU smoke with the four flags from
   `HEADLESS_GPU_REPORT.md`. The GPU lane runs on push to master, nightly,
   on `workflow_dispatch`, and on PRs labelled `gpu`. It is **not** a
   required check; the default lane is.
5. **Coverage** comes only from the default lane (deterministic, always
   runs). The GPU lane uploads benchmark JSON and a `gpu-report.json`
   (adapter info + limits), never lcov, so `tools/merge-coverage.sh --ci`
   never sees a missing artifact when the GPU lane is skipped.
6. **Two version pins to keep straight**: `webgpu@0.4.0` is the last release
   whose Linux binary links against glibc <= 2.34 (verified with `strings`:
   0.4.0 needs `GLIBC_2.34`, 0.6.1 needs `GLIBC_2.38`). The dev container is
   Ubuntu 22.04 / glibc 2.35, `ubuntu-24.04` is glibc 2.39. Keep 0.4.0 in
   both lanes until the dev container (and thus the self-hosted runner
   image) moves to 24.04; then bump once, everywhere.

---

## 1. Constraints taken from the project and the owner's account

| Fact | Evidence |
|---|---|
| graphty-org is an Organization on the **free** plan | `gh api /orgs/graphty-org --jq '{login,type,plan:.plan.name}'` -> `{"login":"graphty-org","plan":"free","type":"Organization"}` (run 2026-09-14) |
| graphty-monorepo is **public** | `gh repo view graphty-org/graphty-monorepo --json isPrivate` -> `false` |
| No repository-level self-hosted runners exist on the monorepo today | `gh api /repos/graphty-org/graphty-monorepo/actions/runners --jq .total_count` -> `0`; org-level listing needs `admin:org` (403 with the current token) |
| The WebGPU package is not yet on GitHub under graphty-org (package.json still points at `github.com/graphty/webgpu-graph-algorithms`) | `/home/apowers/Projects/webgpu-graph-algorithms/package.json` `repository.url`; `gh repo view graphty-org/webgpu-graph-algorithms` -> not found |
| The existing scaffold workflow is stale: Node 18/20 matrix, `npm ci`, Xvfb, and `--use-gl=swiftshader --use-vulkan=swiftshader` (forces software) | `/home/apowers/Projects/webgpu-graph-algorithms/.github/workflows/test.yml`; `vitest.config.ts` lines 13-20 (and the `launch` key that Vitest 2.1 rejects, per `HEADLESS_GPU_REPORT.md` line 107-109) |
| Dev box: Docker container, Ubuntu 22.04.5, glibc 2.35, RTX 4070 SUPER, driver 580.173.02, `NVIDIA_DRIVER_CAPABILITIES=all`, no Docker CLI or socket inside the container, no `libegl1` | `HEADLESS_GPU_REPORT.md` lines 25-43; `which docker` -> not found; `ls /var/run/docker.sock` -> missing (checked 2026-09-14) |
| Working recipe for headless Chromium on the NVIDIA GPU: `libEGL.so.1` on `LD_LIBRARY_PATH` + `--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan --disable-vulkan-surface` | `HEADLESS_GPU_REPORT.md` lines 19-23, 165-196 |
| Dawn-in-Node already used by graph-format's GPU audit tests; falls back to llvmpipe when the NVIDIA ICD cannot init | `packages/graph-format/test/audit/gpu-upload.test.ts` lines 11-16 and `acquire()` lines 37-64; `packages/graph-format/package.json` devDependency `"webgpu": "^0.4.0"` |
| Design doc 14.5 currently says the WebGPU package gets "a browser-only vitest project (Playwright Chromium on the real GPU)" | `/home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md` lines 4212-4217. The owner's new request ("tested primarily under nodejs with some light browser testing") supersedes this; the note below assumes Node-first and flags the amendment in section 10. |
| The GPU package never falls back to CPU | design doc line 4243-4244; `/home/apowers/Projects/webgpu-graph-algorithms/CLAUDE.md` |

---

## 2. What `atoms-org/cuda-ffi` actually does

Cloned to `tmp/webgpu-plan/repos/cuda-ffi` (HEAD 086d5d6, 2024-10-20).
Only two workflows exist: `.github/workflows/build.yml` and `labels.yml`.

`build.yml` (whole file read):

- Trigger: `on: [push, workflow_dispatch]` -- **no `pull_request`** trigger at
  all, which sidesteps the fork-PR problem for the self-hosted runner.
- Job `lint`: `runs-on: ubuntu-latest`, Python 3.12, poetry, safety/bandit,
  ruff/black. This is the "runs without a GPU" part; **no tests** run here.
- Job `build` (`needs: lint`):
  - `runs-on: cudaffi-gpu-runner` -- a **single custom label**, i.e. a
    self-hosted runner registered with that label (GitHub-hosted labels are
    `ubuntu-*`/`windows-*`/`macos-*` or larger-runner names configured in an
    org; `cudaffi-gpu-runner` is neither). The runner itself is not
    visible: `gh api /repos/atoms-org/cuda-ffi/actions/runners` -> 0
    repo-level runners; the org listing needs admin; and
    `/actions/runs` returns `total_count: 0` today, so the runs have aged
    out or the runner was org-level and since removed. The `atoms-org`
    org is on the **team** plan (`gh api /orgs/atoms-org`), so org-level
    runner groups were available to it.
  - `container: image: ghcr.io/apowers313/roc-dev:1.5.2`, `env: CUDA_HOME,
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:...`, `options: "--gpus all
    --user root"`. So the GPU reaches the job through **Docker on the
    runner host with the NVIDIA container toolkit** (`--gpus all`), the
    runner process itself being a plain self-hosted runner that has Docker.
  - Runs `make test` (pytest), `make coverage` -> Coveralls, `make docs` ->
    gh-pages deploy. **Every test needs the GPU; nothing is gated or
    skipped** -- there is no GPU-detection in `tests/conftest.py` (it only
    clears module state) and no marker.
- Nothing in the repo documents how the runner was registered (no runner
  scripts, no `docker/` directory despite the `docker-build` Makefile
  target, no README note on CI). `pyproject.toml` lists `gputil` as a
  dependency (line 68) but it is not used for gating.

Lessons to carry over: (a) a custom label on a self-hosted runner and a
`container:` with `--gpus all` is the simplest GPU plumbing when the runner
host has Docker; (b) not triggering on `pull_request` is a blunt but
effective fork-safety measure; (c) cuda-ffi never solved "what runs without
a GPU" -- the lint job is the only GPU-free work, so this note has to do
better because the WebGPU package wants its correctness suite on every PR.

---

## 3. Options

### 3.1 Options table

| Option | Availability for graphty-org | Cost | Security | Latency / capacity | Maintenance | Verdict |
|---|---|---|---|---|---|---|
| **GitHub GPU-hosted larger runner** (Linux 4 vCPU, 28 GB, 1x Tesla T4 16 GB, 176 GB SSD; Windows too) | **Not available**: larger runners require GitHub Team or Enterprise Cloud; graphty-org is on the free plan. Also "not free for public repositories". | $0.052/min Linux, $0.102/min Windows (2026 pricing), billed even for public repos; needs a credit card and spending limit | GitHub-managed ephemeral VMs (best) | Fast queue; T4 is a 2018 datacenter part, fine for correctness, weak for perf baselines | Low; image is the NVIDIA GPU-Optimized partner image (conda permissions gotcha per scikit-learn's write-up); `modprobe nvidia` was needed in one report | Rejected for now; re-evaluate if the org moves to Team ($4/user/month) |
| **Self-hosted runner on the dev box** (RTX 4070 SUPER in an Ubuntu 22.04 container) | Available today; org or repo level | Free for public repos (the March-2026 $0.002/min self-hosted platform charge does not apply: "Runner usage in public repositories will remain free"); electricity only | Weakest by default: GitHub says self-hosted runners "should almost never be used for public repositories" because fork PRs can run code on them. Mitigations: `--ephemeral`/JIT registration, never trigger from fork PRs, require approval for all external contributors, no secrets in the GPU job, dedicated container not the dev workspace | Zero queue when up; single runner, so one GPU job at a time; unavailable when the box is off | Medium: runner image (needs `libegl1`, `libvulkan1`, `mesa-vulkan-drivers`), driver updates, ephemeral restart loop, `--disableupdate` | **Recommended GPU lane** |
| **Third-party runner provider in your own cloud** (RunsOn, Cirun, machine.dev) | RunsOn: `runs-on=${{ github.run_id }}/family=g4dn.xlarge/image=ubuntu24-gpu-x64` in your AWS account, GPU AMIs with NVIDIA driver + container toolkit; Cirun: free for open source, runners on AWS/GCP/Azure/..., `.cirun.yml` with `gpu: nvidia-tesla-t4`; machine.dev: `runs-on: machine/gpu=t4`, spot from ~$0.003/min | RunsOn: flat licence (commercial tier quoted at EUR 300/year) + AWS spot; Cirun: $0 platform for public repos + cloud bill; machine.dev per-minute | Ephemeral cloud VMs (good); cloud account credentials to manage | Cold start 1-3 min (VM boot); scales to N | Medium-high: AWS/GCP account, quotas for GPU instances (often need a support ticket), image drift | Good fallback if the dev box is unreliable; needs a cloud account the project does not have today |
| **Software adapter on `ubuntu-latest`** -- Dawn-in-Node on Mesa **lavapipe**; Chromium on bundled **SwiftShader** (or lavapipe under xvfb) | Available today, free | $0 for public repos | GitHub-hosted ephemeral VM (best) | Slow per kernel (see 3.5) but correct; 4 vCPUs | Low: `apt-get install mesa-vulkan-drivers libvulkan1`; pin `webgpu` version | **Recommended default lane** |

### 3.2 GitHub GPU-hosted runners: details verified

- GA announcement (2024-07-08): "GPU hosted runners are now generally
  available for Windows and Linux ... T4 GPU access", images "managed by
  trusted partners on the Azure marketplace", configured "through your
  runner groups" and then `runs-on: <runner name>`.
- Spec (Larger runners reference): 4 vCPU, 28 GB RAM, 1x Tesla T4, 16 GB
  VRAM, 176 GB SSD, Ubuntu or Windows.
- Plan gate: "Larger runners are only available for organizations and
  enterprises using the GitHub Team or GitHub Enterprise Cloud plans."
- Public repos: "The larger runners are not free for public repositories."
- Price (Actions runner pricing, 2026): Linux 4-core GPU $0.052/min,
  Windows 4-core GPU $0.102/min; standard Linux 2-core $0.006/min. Minutes
  round up per job.
- Practitioner reports: scikit-learn uses a label ("CUDA CI") to route a
  PR to the `cuda-gpu-runner-group`, stayed under a $50/month spending
  limit, and notes fork PRs cannot get the write permissions the labelling
  flow needs; Dave Snider's Playwright-on-`gpu-linux-4` report found
  headless Chromium did **not** use the T4 until run headed under
  `xvfb-run` with `--use-angle=vulkan --enable-features=Vulkan ...`, that
  `sudo modprobe nvidia nvidia_uvm` was required, and that the GPU image
  ships Mesa/llvmpipe as a software fallback.

### 3.3 Self-hosted runner on the dev box: details verified

- Registration: `./config.sh --url https://github.com/<owner>/<repo>
  --token <registration token> --ephemeral [--labels gpu,nvidia
  --disableupdate --unattended]`; the registration token comes from
  `POST /repos/{owner}/{repo}/actions/runners/registration-token` (expires
  after one hour). With `--ephemeral` "the GitHub Actions service will
  automatically de-register the runner after it has processed one job".
- JIT alternative: `POST /repos/{owner}/{repo}/actions/runners/generate-jitconfig`
  with `{name, runner_group_id, labels, work_folder}` (token scope `repo`)
  returns `encoded_jit_config`, passed as `./run.sh --jitconfig <value>`.
  One config = one job; no `config.sh` step and no leftover registration.
- Labels: a runner carries `self-hosted`, `linux`, `x64` plus custom labels;
  `runs-on: [self-hosted, linux, x64, gpu]` requires all four; the object
  form `runs-on: { group: ..., labels: ... }` targets a runner group (org
  feature).
- Security guidance (GitHub): self-hosted runners "do not have guarantees
  around running in ephemeral clean virtual machines, and can be
  persistently compromised by untrusted code in a workflow"; "should almost
  never be used for public repositories"; the mitigation is ephemeral/JIT
  runners plus runner groups. Fork approval settings: default is "Require
  approval for first-time contributors"; the stricter "Require approval
  for all external contributors" exists at repo/org level.
- Docker on the runner host is only needed if jobs use `container:` (as
  cuda-ffi does); the runner application itself can run directly inside a
  GPU-enabled container. The `ghcr.io/actions/actions-runner` image is the
  official runner image (used by ARC), and `docker run --gpus all` needs
  the NVIDIA Container Toolkit on the host (community discussion #190443,
  GDeLaurentis/docker-gpu-runner-for-github-actions which builds on
  `nvidia/cuda:12.2.0-devel-ubuntu22.04` and runs with `--gpus all`).
- Note about ephemeral runners in containers: the runner may try to
  self-update on every start; GitHub documents `--disableupdate` for this.

### 3.4 Third-party providers: what was verified

- **RunsOn**: NVIDIA T4, A10G, L4, L40S, M60, V100, A100, H100, H200 and AMD
  via AWS instance families; label syntax
  `runs-on=${{ github.run_id }}/family=g4dn.xlarge/image=ubuntu24-gpu-x64`;
  GPU images `ubuntu22-gpu-x64`, `ubuntu24-gpu-x64`, `ubuntu24-gpu-arm64`
  with "NVIDIA drivers, CUDA toolkit, and container toolkit" rebuilt every
  15 days; deploys into your AWS account (spot billing by AWS, flat annual
  licence -- the search summary quoted EUR 300/year for the commercial
  tier; the pricing page itself was not fetched, treat as approximate).
- **Cirun.io**: runners created in your own cloud account (AWS, GCP, Azure,
  Oracle, OpenStack, DigitalOcean), terminated after each job, GPU
  supported (`gpu: nvidia-tesla-t4` example), **free for open source /
  public repos**, $29-$499/month tiers for private repos.
- **machine.dev**: `runs-on: machine/gpu=l4` style labels; T4G, T4, L4,
  A10G, L40S, RTX 6000 plus Inferentia/Trainium; spot from ~$0.003/min;
  image ships NVIDIA driver 580.126.20, CUDA 13.0, container toolkit.
- Not verified (could not confirm a GPU offering from the searches run):
  Namespace, Blacksmith, Depot (Depot's plans are CPU-minute based),
  Buildjet, Modal, Lambda as GitHub Actions runners. Do not plan on them
  without checking.

### 3.5 Software adapters: what was verified and measured

**Dawn-in-Node on lavapipe (default lane, Node).**

- The `webgpu` npm package's own CI on `ubuntu-24.04` installs
  `mesa-vulkan-drivers libvulkan1` and runs its tests with
  `WEBGPU_USE_CI_AVAILABLE_RENDERER=1`, which `test/webgpu.js` turns into
  `create(['adapter=llvmpipe'])` on Linux and `create(['adapter=Microsoft'])`
  on Windows (WARP). Its README documents `create()` options
  `enable-dawn-features=...`, `disable-dawn-features=...`, `backend=vulkan`,
  `adapter=<name>` (an unknown name lists the adapters).
- Verified locally with `webgpu@0.4.0`: `dawn.create(["adapter=llvmpipe"])`
  selects llvmpipe even when the NVIDIA adapter is available;
  `adapter=bogus` throws `no suitable backends found` and prints
  `Available adapters:`; with the NVIDIA ICD failing (no libEGL) Dawn picks
  llvmpipe on its own; `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`
  also isolates lavapipe. All three runs of the 1M-element compute probe
  returned correct results.
- Limits reported by llvmpipe (Mesa 23.2.1 / LLVM 15 on the dev box):
  `maxStorageBufferBindingSize = 134,217,728` (128 MiB),
  `maxComputeWorkgroupsPerDimension = 65,535`,
  `maxComputeInvocationsPerWorkgroup = 1,024`, features `subgroups`,
  `shader-f16`, `timestamp-query` all present. NVIDIA via Dawn:
  `maxStorageBufferBindingSize = 2,147,483,644`, no `shader-f16` on this
  driver. Chromium 139 on NVIDIA: `4,294,967,292`; on SwiftShader:
  `1,073,741,824`. So **any test above 128 MiB per binding, or that asserts
  `shader-f16` is absent, is a GPU-lane test**; conversely, the 100k/1M
  reference graph (rowPtr 400 KB, colIdx 4 MB) fits everywhere.
- `ubuntu-24.04` (current `ubuntu-latest`) has no Vulkan ICD preinstalled
  (the runner image README lists xvfb, Chrome 152, Chromium 152 but no
  mesa-vulkan-drivers); `apt-get install -y mesa-vulkan-drivers libvulkan1`
  pulls Mesa 25.2.8 from noble-updates (Launchpad), i.e. a newer lavapipe
  than the dev box's 23.2.1. wgpu goes further and downloads a pinned Mesa
  26.1.3 build from `gfx-rs/ci-build`, writes its own ICD JSON and sets
  `VK_DRIVER_FILES` -- worth copying only if the distro package misbehaves.
- Speed, probe kernel = 100k nodes / 1M arcs random CSR, per-node neighbour
  gather + scale (PageRank / attraction access pattern), 50 iterations,
  timed after warm-up (`tmp/webgpu-plan/probe/bench-node.mjs`):

  | Adapter | 50 iterations | per iteration |
  |---|---|---|
  | lavapipe, 32 threads (dev box) | 60-76 ms | 1.2-1.5 ms |
  | lavapipe, `LP_NUM_THREADS=4` (hosted-runner core count) | 166-273 ms | 3.3-5.5 ms |
  | NVIDIA RTX 4070 SUPER via Dawn | 4-36 ms | 0.1-0.7 ms |

  All runs produce the same checksum (999.712 over the first 1,000 nodes)
  on lavapipe, SwiftShader and NVIDIA, i.e. the software adapters are
  numerically faithful for this f32 gather.

  The 1M-invocation trivial kernel (20 dispatches + readback) took 8 ms on
  lavapipe vs 1 ms on NVIDIA. Device acquisition: 20-30 ms lavapipe,
  ~140 ms NVIDIA. Expect lavapipe at 4 threads to be roughly 10-50x slower
  than the 4070 per dispatch on memory-bound kernels; a suite whose
  kernels total a few seconds on the GPU will take tens of seconds to a few
  minutes on the hosted runner. wgpu's whole GPU test job on lavapipe
  "is normally 5-15 minutes" (comment in its ci.yml), which is the right
  order of magnitude to budget.

**Chromium on SwiftShader (default lane, browser smoke).**

- SwiftShader is Chromium's bundled CPU Vulkan; it "allows Chromium to
  exercise hardware only code paths on GPU-less bots" and is selected with
  `--use-angle=swiftshader`; `--enable-unsafe-swiftshader` opts in to
  SwiftShader-backed WebGL/WebGPU without the warning (Chromium
  docs/gpu/swiftshader.md and the "Intent to Remove: SwiftShader Fallback"
  thread).
- Verified locally (Playwright Chromium build 1181 = Chromium 139, full
  build, `tmp/webgpu-plan/probe/bench-browser.mjs`): with
  `--enable-unsafe-webgpu --use-angle=swiftshader --enable-unsafe-swiftshader`
  `requestAdapter({powerPreference:"high-performance"})` returns
  `vendor: google, architecture: swiftshader, isFallbackAdapter: true`, and
  the probe kernel produces the same checksum (999.712) as on NVIDIA.
  Speed: 80-111 ms per 50 iterations (1.6-2.2 ms/iter) on 32 cores. With
  **no flags** and `powerPreference: "high-performance"` the adapter was
  `null` in this run (HEADLESS_GPU_REPORT step 2 got SwiftShader with a
  default-preference request) -- pass the flags explicitly, do not rely on
  Chromium's implicit fallback.
- three.js (CI as of 2026-09-07) runs its WebGPU e2e on `ubuntu-latest`
  with `apt-get install -y mesa-vulkan-drivers xvfb`, `xvfb-run -a`, a
  **headed** Chromium (`headless: false` when `CI` is set), flags
  `--enable-unsafe-webgpu --enable-features=Vulkan --disable-vulkan-surface
  --ignore-gpu-blocklist --disable-gpu-driver-bug-workarounds
  --disable-gpu-watchdog --no-sandbox`, and
  `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` -- i.e.
  Chromium on lavapipe rather than SwiftShader, 5 shards, 30 min timeout,
  and an exception list of demos that time out or render black. Its
  `restart()` SIGKILLs Chrome because "browser.close() can hang after a
  wedged GPU process" -- the same hang was reproduced here on the NVIDIA
  path (the probe printed its result, then `browser.close()` hung until the
  outer `timeout` killed it). On the dev box (Mesa 23.2 lavapipe, headless)
  Chromium's GPU process failed Skia init (`HEADLESS_GPU_REPORT.md` lines
  125-134), and the same configuration in today's probe crashed the browser
  before the page loaded (`page.route: Target page, context or browser has
  been closed`), so Chromium-on-lavapipe is **not** used in this plan;
  SwiftShader is enough for a browser smoke and needs no apt packages. If a
  headed Chromium under `xvfb-run` on Mesa 25.x is ever wanted (three.js's
  setup), treat it as a separate experiment on the hosted runner.

---

## 4. Recommended two-lane design

### 4.1 Adapter policy in the test harness (one place, both lanes)

The package gets one device-acquisition helper for tests (the production
`GpuContext` throws when no adapter; the test helper is the only place
that reads environment variables). Two variables:

- `GRAPHTY_GPU_ADAPTER` -- passed through to `dawn.create(["adapter=<v>"])`
  in Node. Default lane sets `llvmpipe` (deterministic even on a machine
  that also has a GPU). GPU lane leaves it unset (Dawn picks the discrete
  GPU) or sets the NVIDIA name.
- `GRAPHTY_GPU_REQUIRE` -- `nvidia` on the GPU lane. After acquisition the
  helper asserts `adapter.info.vendor === GRAPHTY_GPU_REQUIRE` (Node and
  browser) and `isFallbackAdapter === false` in the browser; a mismatch is a
  hard failure, never a skip. This is recommendation 3 of
  `HEADLESS_GPU_REPORT.md` (lines 227-231) made lane-aware.

Test tiers, expressed as vitest projects (names are proposals):

| Project | Environment | Runs on default lane | Runs on GPU lane |
|---|---|---|---|
| `node` | Node + Dawn: every kernel, primitive and algorithm correctness test, differential tests vs `@graphty/algorithms` / `@graphty/layout` CPU results, upload-plan tests with **faked limits** (the DispatchPlanner and upload planner take limits as data, so the 128 MiB / 2 GiB / windowed cases are unit tests, not device tests) | yes (lavapipe) | yes (NVIDIA, `GRAPHTY_GPU_REQUIRE=nvidia`) |
| `node-limits` | Node + Dawn: tests that need real limits above lavapipe's (bindings > 128 MiB, `maxBufferSize` near 2 GiB, 2D dispatch above 16,776,960 invocations on real data), and vendor-specific feature assertions | skipped by project selection (not by runtime detection) | yes |
| `bench` | Node + Dawn: `vitest bench` for the primitives and the force-directed step at 10k/100k/1M; compares against a checked-in baseline per runner class the way the monorepo's `performance` job does for algorithms (`ci.yml` lines 656-703: download baseline artifact, run, upload with 90-day retention) | no (numbers on lavapipe are meaningless) | yes |
| `browser-smoke` | Vitest browser mode, Playwright Chromium: device acquisition, one upload + trivial kernel + readback, one small force-directed run, `LayoutSimulation` contract in a page. A handful of files, not a mirror of `node`. | yes, SwiftShader flags | yes, NVIDIA flags + `LD_LIBRARY_PATH` with libEGL (until the runner image has `libegl1`) |

Test timeouts: keep `testTimeout: 30000` from the scaffold but give the
lavapipe lane a per-file budget; if a single file exceeds ~2 min on
lavapipe it is a candidate for `node-limits` or for shrinking its graph.

### 4.2 Default lane (every PR and push; required check)

`runs-on: ubuntu-latest`. Steps: checkout, pnpm/npm install, build, lint,
typecheck, `apt-get install -y mesa-vulkan-drivers libvulkan1`, run
`vitest run --project=node --coverage` with `GRAPHTY_GPU_ADAPTER=llvmpipe`,
install Playwright Chromium (cached exactly like the monorepo, `ci.yml`
lines 413-427), run `vitest run --project=browser-smoke`, upload
`coverage/lcov.info`. `timeout-minutes: 30`. Print the adapter info at the
top of the log (wgpu prints its `.gpuconfig` for the same reason).

No Xvfb: Node needs no display; headless Chromium with SwiftShader needs
no display. Drop the scaffold's `xvfb`, `libgl1-mesa-glx`, `libegl1-mesa`
lines; they were for a WebGL-era setup.

### 4.3 GPU lane (self-hosted; not a required check)

`runs-on: [self-hosted, linux, x64, gpu, nvidia]`. Triggers:

- `push` to `master`/`main` (post-merge truth on real hardware),
- `schedule` nightly (catches driver/Playwright drift; the
  `HEADLESS_GPU_REPORT.md` open question about Playwright bumps is exactly
  this),
- `workflow_dispatch`,
- `pull_request` with `types: [opened, synchronize, reopened, labeled]`
  **and** `if: github.event.pull_request.head.repo.full_name ==
  github.repository && contains(github.event.pull_request.labels.*.name,
  'gpu')` -- same-repo branches only, and only when a maintainer applies the
  `gpu` label (the scikit-learn pattern; fork PRs cannot satisfy the first
  clause).

Steps: checkout; install; download the default lane's build artifact when
in the same workflow (or rebuild -- the dev box is fast); run
`vitest run --project=node --project=node-limits` with
`GRAPHTY_GPU_REQUIRE=nvidia`; `vitest bench --project=bench` writing JSON;
`vitest run --project=browser-smoke` with the four NVIDIA flags; upload
`bench/*.json` and `gpu-report.json`. `timeout-minutes: 45` and
`concurrency: { group: gpu-runner, cancel-in-progress: false }` so a single
GPU is never oversubscribed. `permissions: contents: read` and no secrets.

The GPU job is `continue-on-error: false` but is **not** listed in the
required-checks gate (the monorepo's `all-checks` job, `ci.yml` lines
705-738, only needs `test` and the Chromatic jobs; keep the GPU shard out
of `needs`). Skipped or failed GPU runs surface as a red badge on master
and a nightly issue, not as a blocked PR -- otherwise a powered-off dev box
blocks every merge.

### 4.4 Coverage when the GPU lane may be skipped

Coverage is produced by the default lane only. Rationale: (1) the GPU lane
runs the same `node` project, so it adds no lines; (2) `tools/merge-coverage.sh
--ci` fails on any missing package (`merge-coverage.sh` lines 228-232),
and `coverage.yml` only runs on a successful CI run (line 17, 51), so a
GPU shard that uploads `coverage-*` sometimes and not others would either
break the merge or hide it. The GPU shard therefore uploads no
`coverage-*` artifact at all. Vitest coverage in browser mode (SwiftShader
smoke) is optional; if enabled it goes into the same lcov as the Node run
via `--coverage.reportsDirectory` per project and the existing
multi-shard merge (`merge-coverage.sh` lines 146-176 already merges
several `coverage-<pkg>-*` artifacts per package).

---

## 5. Workflow YAML sketch for this repo now

Replaces `.github/workflows/test.yml`. Sketch, not final; package manager
follows whatever the repo standardises on (the scaffold uses npm; the
monorepo uses pnpm).

```yaml
name: CI

on:
    push:
        branches: [master]
    pull_request:
        types: [opened, synchronize, reopened, labeled]
    schedule:
        - cron: "17 6 * * *"   # nightly GPU lane
    workflow_dispatch:

permissions:
    contents: read

concurrency:
    group: ${{ github.workflow }}-${{ github.ref }}
    cancel-in-progress: ${{ github.event_name == 'pull_request' }}

env:
    WEBGPU_NPM_VERSION_NOTE: "webgpu@0.4.0 pinned: 0.5+ needs glibc 2.38 (dev box is 22.04 / 2.35)"

jobs:
    # ---------------------------------------------------------------- default lane
    test:
        name: Test (software adapter)
        if: github.event_name != 'schedule'
        runs-on: ubuntu-latest
        timeout-minutes: 30
        steps:
            - uses: actions/checkout@v4
            - uses: actions/setup-node@v4
              with:
                  node-version: 22.x
                  cache: npm
            - run: npm ci

            # Mesa lavapipe = the Vulkan ICD Dawn-in-Node will use. Not preinstalled on ubuntu-24.04.
            - name: Install Mesa lavapipe (software Vulkan)
              run: |
                  sudo apt-get update
                  sudo apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1

            - run: npm run build
            - run: npm run lint
            - run: npm run typecheck

            - name: Node correctness suite on lavapipe
              env:
                  GRAPHTY_GPU_ADAPTER: llvmpipe      # dawn.create(["adapter=llvmpipe"])
                  VK_DRIVER_FILES: /usr/share/vulkan/icd.d/lvp_icd.x86_64.json
              run: npx vitest run --project=node --coverage

            - name: Cache Playwright browsers
              id: playwright-cache
              uses: actions/cache@v4
              with:
                  path: ~/.cache/ms-playwright
                  key: playwright-${{ runner.os }}-${{ hashFiles('package-lock.json') }}
            - name: Install Playwright Chromium
              if: steps.playwright-cache.outputs.cache-hit != 'true'
              run: npx playwright install chromium --with-deps
            - name: Install Playwright deps (if cached)
              if: steps.playwright-cache.outputs.cache-hit == 'true'
              run: npx playwright install-deps chromium

            - name: Browser smoke on SwiftShader
              env:
                  GRAPHTY_BROWSER_GPU: swiftshader   # vitest.config picks the flag set
              run: npx vitest run --project=browser-smoke

            - uses: actions/upload-artifact@v4
              if: ${{ !cancelled() }}
              with:
                  name: coverage-webgpu-graph-algorithms
                  path: coverage/lcov.info
                  retention-days: 1
                  if-no-files-found: error

    # -------------------------------------------------------------------- GPU lane
    test-gpu:
        name: Test (NVIDIA, self-hosted)
        # Same-repo pushes to master, nightly, manual, or a same-repo PR labelled "gpu".
        # Fork PRs can never satisfy head.repo.full_name == github.repository.
        if: >-
            github.event_name == 'push' ||
            github.event_name == 'schedule' ||
            github.event_name == 'workflow_dispatch' ||
            (github.event_name == 'pull_request' &&
             github.event.pull_request.head.repo.full_name == github.repository &&
             contains(github.event.pull_request.labels.*.name, 'gpu'))
        runs-on: [self-hosted, linux, x64, gpu, nvidia]
        timeout-minutes: 45
        concurrency:
            group: gpu-runner            # one GPU, one job at a time
            cancel-in-progress: false
        env:
            GRAPHTY_GPU_REQUIRE: nvidia          # software fallback = failure
            # Until the runner image has libegl1 (HEADLESS_GPU_REPORT.md appendix D):
            LD_LIBRARY_PATH: /opt/egl/usr/lib/x86_64-linux-gnu
        steps:
            - uses: actions/checkout@v4
            - uses: actions/setup-node@v4
              with:
                  node-version: 22.x
            - run: npm ci
            - run: npm run build

            - name: Report adapter
              run: node scripts/gpu-report.mjs > gpu-report.json && cat gpu-report.json

            - name: Node suite + limit-dependent tests on NVIDIA
              run: npx vitest run --project=node --project=node-limits

            - name: Benchmarks
              run: npx vitest bench --project=bench --outputJson bench/results.json

            - name: Browser smoke on the real GPU (headless Chromium, Vulkan)
              env:
                  GRAPHTY_BROWSER_GPU: nvidia      # the four flags from HEADLESS_GPU_REPORT.md
              run: npx vitest run --project=browser-smoke

            - uses: actions/upload-artifact@v4
              if: ${{ !cancelled() }}
              with:
                  name: gpu-results-${{ github.run_id }}
                  path: |
                      gpu-report.json
                      bench/results.json
                  retention-days: 90
```

`vitest.config.ts` sketch for the browser project (the scaffold's top-level
`launch` key is rejected by Vitest 2.1; `providerOptions.launch` is the
supported key per `HEADLESS_GPU_REPORT.md` lines 205-227):

```ts
const browserFlags = {
    swiftshader: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    nvidia: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan", "--disable-vulkan-surface"],
}[process.env.GRAPHTY_BROWSER_GPU ?? "swiftshader"];
// browser: { enabled: true, name: "chromium", provider: "playwright", headless: true,
//            providerOptions: { launch: { args: browserFlags } } }
```

### 5.1 Self-hosted runner recipe for the dev box

The dev container has no Docker socket, so the runner must run either
inside the dev container (quick start) or, better, as a **sibling
container started on the host** with the GPU passed through. Sketch of the
sibling container (Dockerfile-level, not to be committed to the package):

```
FROM ghcr.io/actions/actions-runner:latest          # official runner image (Ubuntu-based)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libvulkan1 mesa-vulkan-drivers vulkan-tools \
        # Playwright Chromium deps: `npx playwright install-deps chromium` at build time
    && rm -rf /var/lib/apt/lists/*
USER runner
```

Run on the host with the NVIDIA Container Toolkit:

```
docker run -d --restart unless-stopped --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e RUNNER_REPO=graphty-org/webgpu-graph-algorithms \
  -e RUNNER_LABELS=gpu,nvidia,rtx4070 \
  -v /srv/gha-runner/token:/run/secrets/gh-token:ro \
  graphty-gpu-runner
```

Entry loop (ephemeral, one job per registration; `--disableupdate` because
GitHub notes ephemeral runners in containers otherwise re-update on every
start):

```
while true; do
  TOKEN=$(gh api -X POST /repos/$RUNNER_REPO/actions/runners/registration-token --jq .token)
  ./config.sh --unattended --ephemeral --disableupdate \
      --url https://github.com/$RUNNER_REPO --token "$TOKEN" \
      --name "devbox-$(hostname)-$RANDOM" --labels "$RUNNER_LABELS" --work _work
  ./run.sh
  rm -rf _work
done
```

(or `generate-jitconfig` + `./run.sh --jitconfig`, which skips `config.sh`
entirely). `NVIDIA_DRIVER_CAPABILITIES=all` is required for the Vulkan ICD
to be injected (HEADLESS_GPU_REPORT.md line 32 and the
nvidia-container-toolkit issue #1952 it cites). `libegl1` in the image
removes the `LD_LIBRARY_PATH` workaround entirely; keep the env var only
while running the runner inside the current dev container.

Repository settings to flip before registering: Settings -> Actions ->
General -> "Fork pull request workflows from outside collaborators" ->
**Require approval for all external contributors**; "Workflow permissions"
-> read-only. The `gpu` label is created once; only users with triage/write
can apply it. Org level (later): put the runner in a runner group scoped to
the two repos so no other org repo can target it.

---

## 6. Monorepo `ci.yml` diff sketch (for the W1 move-in)

Follows the pattern of `packages/move/root-touch-points.diff` (build once
in `build`, upload `build-<pkg>`, add a shard to the `test` matrix,
download in the shard, upload `coverage-<shard>`, add the package to
`merge-coverage.sh` PACKAGES and `prepush.sh`). New parts are the apt
step, the adapter env vars, and a separate GPU job.

```diff
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ on:
     pull_request:
+        types: [opened, synchronize, reopened, labeled]
+    schedule:
+        - cron: "17 6 * * *"
     workflow_dispatch:
@@ jobs.build (after "Build graph-format and graph-io (PR)")
+            - name: Build webgpu-graph-algorithms (PR)
+              if: github.event_name == 'pull_request'
+              run: pnpm exec nx run webgpu-graph-algorithms:build
@@ jobs.build uploads
+            - name: Upload webgpu-graph-algorithms build
+              uses: actions/upload-artifact@v4
+              with:
+                  name: build-webgpu-graph-algorithms
+                  path: webgpu-graph-algorithms/dist/
+                  retention-days: 1
@@ jobs.test.strategy.matrix.shard
+                    - webgpu-graph-algorithms-node
+                    - webgpu-graph-algorithms-browser
@@ jobs.test.strategy.matrix.include
+                    # webgpu-graph-algorithms -- Node correctness suite on Dawn + Mesa lavapipe (no GPU)
+                    - shard: webgpu-graph-algorithms-node
+                      package: webgpu-graph-algorithms
+                      test-command: cd webgpu-graph-algorithms && pnpm exec vitest run --project=node --coverage
+                      needs-browser: false
+                      needs-storybook: false
+                      needs-vulkan: true
+                    # webgpu-graph-algorithms -- browser smoke on Chromium SwiftShader
+                    - shard: webgpu-graph-algorithms-browser
+                      package: webgpu-graph-algorithms
+                      test-command: cd webgpu-graph-algorithms && pnpm exec vitest run --project=browser-smoke --coverage
+                      needs-browser: true
+                      needs-storybook: false
+                      needs-vulkan: false
@@ jobs.test.steps (before "Run tests")
+            - name: Install Mesa lavapipe (software Vulkan for Dawn-in-Node)
+              if: matrix.needs-vulkan
+              run: |
+                  sudo apt-get update
+                  sudo apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1
+
+            - name: Download webgpu-graph-algorithms build
+              uses: actions/download-artifact@v4
+              with:
+                  name: build-webgpu-graph-algorithms
+                  path: webgpu-graph-algorithms/dist/
@@ jobs.test.steps "Run tests"
             - name: Run tests
               run: ${{ matrix.test-command }}
+              env:
+                  GRAPHTY_GPU_ADAPTER: llvmpipe
+                  GRAPHTY_BROWSER_GPU: swiftshader
+                  VK_DRIVER_FILES: /usr/share/vulkan/icd.d/lvp_icd.x86_64.json
@@ jobs.test.steps coverage upload condition
-              if: ${{ !cancelled() && (matrix.shard == 'graph-format' || matrix.shard == 'graph-io' || startsWith(matrix.shard, 'algorithms-') || ...) }}
+              if: ${{ !cancelled() && (matrix.shard == 'graph-format' || matrix.shard == 'graph-io' || startsWith(matrix.shard, 'webgpu-graph-algorithms-') || startsWith(matrix.shard, 'algorithms-') || ...) }}
@@ new job after "performance"
+    # GPU lane -- self-hosted RTX 4070 SUPER. Not in all-checks; master, nightly, manual, or same-repo PR labelled "gpu".
+    test-gpu:
+        name: Test (NVIDIA, self-hosted)
+        needs: build
+        if: >-
+            github.event_name == 'push' || github.event_name == 'schedule' || github.event_name == 'workflow_dispatch' ||
+            (github.event_name == 'pull_request' &&
+             github.event.pull_request.head.repo.full_name == github.repository &&
+             contains(github.event.pull_request.labels.*.name, 'gpu'))
+        runs-on: [self-hosted, linux, x64, gpu, nvidia]
+        timeout-minutes: 45
+        concurrency:
+            group: gpu-runner
+            cancel-in-progress: false
+        env:
+            GRAPHTY_GPU_REQUIRE: nvidia
+            GRAPHTY_BROWSER_GPU: nvidia
+        steps:
+            - uses: actions/checkout@v4
+            - uses: pnpm/action-setup@v4
+            - uses: actions/setup-node@v4
+              with: { node-version: 22.x, cache: pnpm }
+            - run: pnpm install --frozen-lockfile
+              env: { HUSKY: "0" }
+            - uses: actions/download-artifact@v4
+              with: { name: build-graph-format, path: graph-format/dist/ }
+            - uses: actions/download-artifact@v4
+              with: { name: build-webgpu-graph-algorithms, path: webgpu-graph-algorithms/dist/ }
+            - run: cd webgpu-graph-algorithms && node scripts/gpu-report.mjs | tee gpu-report.json
+            - run: cd webgpu-graph-algorithms && pnpm exec vitest run --project=node --project=node-limits
+            - run: cd webgpu-graph-algorithms && pnpm exec vitest bench --project=bench --outputJson bench/results.json
+            - run: cd webgpu-graph-algorithms && pnpm exec vitest run --project=browser-smoke
+            - uses: actions/upload-artifact@v4
+              if: ${{ !cancelled() }}
+              with:
+                  name: gpu-results-webgpu-graph-algorithms
+                  path: |
+                      webgpu-graph-algorithms/gpu-report.json
+                      webgpu-graph-algorithms/bench/results.json
+                  retention-days: 90
```

Also, mirroring `root-touch-points.diff`: add `webgpu-graph-algorithms` to
`pnpm-workspace.yaml`, `commitlint.config.js` scopes, `knip.config.ts`,
`tools/merge-coverage.sh` PACKAGES (so the default-lane lcov is required
and merged), `tools/prepush.sh` (run `--project=node` only; the pre-push
hook on the dev box will use the NVIDIA adapter, which is fine, or set
`GRAPHTY_GPU_ADAPTER=llvmpipe` there to match CI exactly), and
`release.yml` (download `build-webgpu-graph-algorithms`).

Gating in the monorepo: the `test` matrix already runs every shard on every
PR regardless of `nx affected` (`ci.yml` lines 235-351 have no affected
filter on shards), so the two software shards simply join the matrix. The
GPU job is gated by event + label, not by paths; if per-path gating is
wanted later, `dorny/paths-filter@v4` with a filter on
`webgpu-graph-algorithms/**` and `graph-format/**` is the standard tool.
Do not use `nx affected` for the GPU job: a change to graph-format is
exactly the case where the GPU lane should run, and `affected` would need
the full dependency graph to know that -- the label is explicit.

---

## 7. Exact packages and flags for the software-adapter path (ubuntu-latest)

| Need | Exactly | Evidence |
|---|---|---|
| Vulkan loader + lavapipe ICD for Dawn-in-Node | `sudo apt-get update && sudo apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1` (installs `/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`; Mesa 25.2.8 on noble-updates) | node-webgpu `build.yml` installs `mesa-vulkan-drivers libvulkan1`; three.js `ci.yml` installs `mesa-vulkan-drivers xvfb`; Launchpad noble mesa 25.2.8-0ubuntu0.24.04.2; local ICD path |
| Deterministic adapter choice in Node | `dawn.create(["adapter=llvmpipe"])`; optionally `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` (or the older `VK_ICD_FILENAMES`) | node-webgpu `test/webgpu.js`; verified locally with `webgpu@0.4.0` |
| `webgpu` npm version | `0.4.0` (Linux binary needs `GLIBC_2.34`; `0.6.1` needs `GLIBC_2.38`). ubuntu-24.04 (glibc 2.39) could run 0.6.1, the 22.04 dev box cannot; keep one version. | `strings dist/linux-x64.dawn.node \| grep GLIBC_` on both tarballs (2026-09-14); Launchpad noble glibc 2.39; `HEADLESS_GPU_REPORT.md` line 31 |
| Browser smoke on SwiftShader | Playwright Chromium (`npx playwright install chromium --with-deps`), headless, args `--enable-unsafe-webgpu --use-angle=swiftshader --enable-unsafe-swiftshader`; request the adapter with `powerPreference: "high-performance"` or default and accept `isFallbackAdapter: true` on this lane only | Chromium `docs/gpu/swiftshader.md`; local probe (adapter `google/swiftshader`, checksum equal to NVIDIA) |
| Browser on real GPU (GPU lane) | headless Chromium args `--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan --disable-vulkan-surface`; `libEGL.so.1` reachable (package `libegl1`); assert `adapter.info.vendor === "nvidia"` | `HEADLESS_GPU_REPORT.md` lines 19-23, 165-196, 227-231; jasonmayes/headless-chrome-nvidia-t4-gpu-support flags (adds `--headless=new --no-sandbox`; warns `--disable-vulkan-surface` breaks canvas presentation, irrelevant for compute) |
| Not needed | `xvfb`, `libgl1-mesa-glx`, `libegl1-mesa`, `vulkan-tools` (handy for `vulkaninfo --summary` debugging only), `--use-gl=swiftshader`, `--use-vulkan=swiftshader`, `--ignore-gpu-blocklist`, Dawn blocklist toggles | scaffold `test.yml` lines 27-44 are the WebGL-era list; `HEADLESS_GPU_REPORT.md` lines 240-241 |
| Thread count on lavapipe | leave default (uses all vCPUs); `LP_NUM_THREADS` exists to pin it (used above to emulate 4 vCPUs) | measured locally |
| Chromium hang on close after GPU work | wrap browser runs in a hard timeout; three.js SIGKILLs the process tree because `browser.close()` can hang | three.js `test/e2e/puppeteer.js` lines 262-273; reproduced locally on the NVIDIA path |

---

## 8. Timeouts, artifacts, retention -- summary

| Job | timeout-minutes | Artifacts | Retention |
|---|---|---|---|
| default `test` (Node on lavapipe + SwiftShader smoke) | 30 (wgpu budgets 30 for a 5-15 min lavapipe suite; three.js 30 per shard) | `coverage-webgpu-graph-algorithms` (lcov) | 1 day (matches monorepo) |
| `test-gpu` | 45 | `gpu-results-*` (`gpu-report.json`, `bench/results.json`); no lcov | 90 days (matches `performance-baseline`) |
| nightly `test-gpu` | 45 | same + open/refresh a tracking issue on failure (optional, `actions/github-script`) | 90 days |

Split the default Node suite into two shards (`--shard=1/2`) if it passes
~10 min on lavapipe; the monorepo already shards graphty-element five ways
with `--reporter=blob` and merges in `coverage.yml`.

---

## 9. Risks and open questions

1. **Runner availability**: a single self-hosted runner on a dev box will
   be offline sometimes. Mitigated by not making the GPU job required and
   by the nightly schedule. If this bites, Cirun (free for public repos) on
   a cloud account is the cheapest managed escape hatch; GitHub's T4 runners
   need a Team plan.
2. **Security of a self-hosted runner on public repos**: the `if` guard
   plus "require approval for all external contributors" plus ephemeral
   registration plus no secrets in the job is the full mitigation set
   GitHub documents; it is still a persistent machine. Keep the runner in
   its own container, not the dev workspace, and give the registration
   token (fine-grained PAT, `Administration: write` on the repo, or a
   GitHub App) only to the host-side loop.
3. **lavapipe version skew**: dev box Mesa 23.2 (LLVM 15) vs
   hosted-runner Mesa 25.2. Either is fine for correctness, but a WGSL
   feature that lavapipe 23.2 lacks may pass in CI and fail in the local
   pre-push (or vice versa). Recording the adapter description at the top
   of every run makes this diagnosable; upgrading the dev container to
   24.04 removes it (and unblocks `webgpu@0.6.x`).
4. **Chromium-on-NVIDIA `browser.close()` hang**: budget a hard kill in
   the browser-smoke runner (Vitest's Playwright provider owns the browser;
   the job-level `timeout-minutes` is the backstop).
5. **Design doc 14.5 says "browser-only vitest project"**; this plan makes
   Node the primary project with a small browser project, per the owner's
   request. Amend 14.5 and 16.7 (add the WebGPU package's CI budget) when
   the plan is accepted.
6. **Coverage thresholds**: the Node suite on lavapipe exercises every
   kernel, so 80/80/75/80 (graph-format's thresholds) are reachable without
   the GPU lane; device-limit branches (windowed uploads, 2D dispatch)
   must be unit-tested with faked limits or they will be uncovered on the
   default lane.
7. **Not verified**: the RunsOn licence price and calculator, whether
   Blacksmith/Namespace/Depot/Buildjet offer GPUs, GitHub's larger-runner
   concurrency limits for GPU SKUs, and the exact behaviour of Chromium 152
   (the version on the current runner image) with SwiftShader WebGPU --
   Playwright pins its own Chromium (139 in this project), so the runner's
   system Chrome is irrelevant unless the config switches to
   `channel: "chromium"`.

---

## Sources

Local files (all paths absolute):

- `/home/apowers/Projects/webgpu-graph-algorithms/HEADLESS_GPU_REPORT.md` (lines 19-23, 25-43, 107-134, 165-196, 198-241)
- `/home/apowers/Projects/webgpu-graph-algorithms/.github/workflows/test.yml` (stale scaffold workflow)
- `/home/apowers/Projects/webgpu-graph-algorithms/vitest.config.ts`, `package.json`
- `/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/test/audit/gpu-upload.test.ts` (lines 11-16, 37-64), `packages/graph-format/package.json`, `packages/graph-format/vitest.config.ts`
- `/home/apowers/Projects/webgpu-graph-algorithms/packages/move/root-touch-points.diff`
- `/home/apowers/Projects/graphty-monorepo/.github/workflows/ci.yml` (lines 3-12, 42-102, 234-351, 413-464, 656-738), `coverage.yml` (lines 13-89), `release.yml` (lines 61-109)
- `/home/apowers/Projects/graphty-monorepo/tools/prepush.sh`, `tools/merge-coverage.sh` (lines 28-29, 120-178, 228-232)
- `/home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md` (lines 4212-4244, 4436-4446, 4643-4655)
- `/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/repos/cuda-ffi/.github/workflows/build.yml`, `labels.yml`, `Makefile`, `tests/conftest.py`, `pyproject.toml`
- Probe scripts and raw results: `/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/probe/{dawn-probe.mjs,kernel.mjs,bench-node.mjs,bench-browser.mjs}`
- Commands run 2026-09-14: `gh api /orgs/graphty-org`, `gh api /orgs/atoms-org`, `gh repo view graphty-org/graphty-monorepo --json isPrivate`, `gh api /repos/graphty-org/graphty-monorepo/actions/runners`, `gh api /repos/atoms-org/cuda-ffi/actions/runners`, `gh api /repos/atoms-org/cuda-ffi/actions/runs`, `strings` on `webgpu@0.4.0` and `webgpu@0.6.1` Linux binaries

GitHub documentation and changelog:

- GPU hosted runners GA: https://github.blog/changelog/2024-07-08-github-actions-gpu-hosted-runners-are-now-generally-available/
- Larger runners reference (GPU spec 4 vCPU / 28 GB / T4 16 GB / 176 GB): https://docs.github.com/en/actions/reference/runners/larger-runners
- Larger runners concept ("only available for organizations and enterprises using the GitHub Team or GitHub Enterprise Cloud plans"): https://docs.github.com/en/enterprise-cloud@latest/actions/concepts/runners/larger-runners
- Actions runner pricing (GPU Linux $0.052/min, Windows $0.102/min; "larger runners are not free for public repositories"): https://docs.github.com/en/billing/reference/actions-runner-pricing
- 2026 pricing change (self-hosted $0.002/min from 2026-03-01; "Runner usage in public repositories will remain free"): https://github.blog/changelog/2025-12-16-coming-soon-simpler-pricing-and-a-better-experience-for-github-actions/ and https://github.blog/changelog/2026-01-01-reduced-pricing-for-github-hosted-runners-usage/
- Security hardening for self-hosted runners: https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions#hardening-for-self-hosted-runners
- Self-hosted runners reference (`--ephemeral`, `--disableupdate`, supported OS): https://docs.github.com/en/actions/reference/runners/self-hosted-runners
- Using self-hosted runners in a workflow (`runs-on` label arrays and `group`/`labels`): https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow
- REST: registration token and `generate-jitconfig`: https://docs.github.com/en/rest/actions/self-hosted-runners?apiVersion=2022-11-28#create-configuration-for-a-just-in-time-runner-for-a-repository
- Fork PR approval settings: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository#configuring-required-approval-for-workflows-from-public-forks
- Ubuntu 24.04 runner image contents: https://raw.githubusercontent.com/actions/runner-images/main/images/ubuntu/Ubuntu2404-Readme.md
- Community: NVIDIA GPU on self-hosted runners (Docker needs `--gpus` + container toolkit): https://github.com/orgs/community/discussions/190443
- Example GPU runner container: https://github.com/GDeLaurentis/docker-gpu-runner-for-github-actions

Practitioner reports on GitHub GPU runners:

- Tim Head (scikit-learn), label-gated `cuda-gpu-runner-group`, $50/month cap: https://betatim.github.io/posts/github-action-with-gpu/
- Dave Snider, Playwright on `gpu-linux-4`, xvfb + Vulkan flags + `modprobe nvidia`: https://davesnider.com/posts/gputests

Third-party runners:

- RunsOn GPU runners: https://runs-on.com/runners/gpu/ (pricing page https://runs-on.com/pricing/ not fetched; licence figure from search summary only)
- Cirun.io (free for open source, own-cloud GPU runners): https://cirun.io/
- machine.dev GPU runners: https://machine.dev/docs/platform-specifications/gpu-runners/

Software adapters and other projects' CI:

- node-webgpu (the `webgpu` npm package) CI: https://raw.githubusercontent.com/dawn-gpu/node-webgpu/main/.github/workflows/build.yml (installs `mesa-vulkan-drivers libvulkan1`, `WEBGPU_USE_CI_AVAILABLE_RENDERER=1`) and https://raw.githubusercontent.com/dawn-gpu/node-webgpu/main/test/webgpu.js (`adapter=llvmpipe` / `adapter=Microsoft`); README `create()` options: https://github.com/dawn-gpu/node-webgpu/blob/main/README.md; releases: https://github.com/dawn-gpu/node-webgpu/releases (v0.4.0 2026-03-27, v0.6.1 2026-09-12)
- wgpu CI (`gpu-test` on `ubuntu-24.04`, "runtime is normally 5-15 minutes", 30 min timeout): https://raw.githubusercontent.com/gfx-rs/wgpu/trunk/.github/workflows/ci.yml; Mesa install action (pinned Mesa 26.1.3 from gfx-rs/ci-build, custom ICD JSON, `VK_DRIVER_FILES`): https://raw.githubusercontent.com/gfx-rs/wgpu/trunk/.github/actions/install-mesa/action.yml; testing docs: https://raw.githubusercontent.com/gfx-rs/wgpu/trunk/docs/testing.md
- three.js CI (`mesa-vulkan-drivers xvfb`, `xvfb-run -a`, 5 shards, 30 min): https://raw.githubusercontent.com/mrdoob/three.js/dev/.github/workflows/ci.yml; Chromium flags and `VK_DRIVER_FILES` lavapipe, SIGKILL on hang: https://raw.githubusercontent.com/mrdoob/three.js/dev/test/e2e/puppeteer.js (last commit 2026-09-07)
- Chromium SwiftShader docs (`--use-angle=swiftshader`, GPU-less bots): https://github.com/chromium/chromium/blob/main/docs/gpu/swiftshader.md; `--enable-unsafe-swiftshader` intent: https://groups.google.com/a/chromium.org/g/blink-dev/c/yhFguWS_3pM
- Chrome WebGPU troubleshooting (unsafe-webgpu flag, Linux Vulkan flag): https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips
- Headless Chrome on NVIDIA T4 (flags, `--disable-vulkan-surface` caveat): https://github.com/jasonmayes/headless-chrome-nvidia-t4-gpu-support
- Ubuntu package versions: https://launchpad.net/ubuntu/noble/+source/mesa (25.2.8-0ubuntu0.24.04.2), https://launchpad.net/ubuntu/jammy/+source/mesa (23.2.1-1ubuntu3.1~22.04.4), https://launchpad.net/ubuntu/noble/+source/glibc (2.39)
- dorny/paths-filter (v4) for optional path gating: https://github.com/dorny/paths-filter

Owner-supplied links checked for CI content (none has GitHub workflows):
https://github.com/harp-lab/GraphWaGu, https://github.com/cosmosgl/cosmos,
https://github.com/jaredmcqueen/analytics (GitHub contents API returned no
`.github/workflows` directory for any of the three on 2026-09-14).
