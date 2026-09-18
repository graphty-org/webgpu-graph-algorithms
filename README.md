# webgpu-graph-algorithms (graphty-org staging)

LANDED: @graphty/webgpu-graph-algorithms, its design and the P0-P3 plans moved to graphty-org/graphty-monorepo on 2026-09-18 with this repository's history (webgpu-graph-algorithms/ and design/webgpu/ there); @graphty/graph-format and @graphty/graph-io moved on 2026-09-16. This repository is archived as the record of the staging.

The public staging repository of graphty-org for packages that move into the graphty monorepo
(`graphty-org/graphty-monorepo`) once they are done. The repository root is not a package: the pnpm
workspace root is `packages/`, laid out like the monorepo root so that each package directory moves
with a plain `mv`.

## Packages

- `packages/graph-format/` -- `@graphty/graph-format`, the CSR graph snapshot (arena, views, columns).
  Done and audited; waiting for the move.
- `packages/graph-io/` -- `@graphty/graph-io`, importers and exporters for the snapshot. Done and
  audited; waiting for the move.
- `packages/webgpu-graph-algorithms/` -- `@graphty/webgpu-graph-algorithms`, WebGPU graph algorithms
  and force-directed layouts over the snapshot, for Node (Dawn) and browsers. In progress: phases
  P0-P3 of the plan below.

## Where to look

- `packages/README.md` -- working in the workspace (`cd packages && pnpm install && pnpm -r run build:all`),
  the tool versions, the rehearsed move checklist.
- `design/webgpu-acceleration-plan.md` -- the WebGPU acceleration plan (the spec: runtime model, memory,
  kernels, layouts, integration, testing, CI, phases).
- `docs/superpowers/plans/` -- the normative P0-P3 interface contract
  (`2026-09-14-webgpu-p0-p3-interfaces.md`) and the per-phase implementation plans.
- `packages/webgpu-graph-algorithms/docs/HEADLESS_GPU_REPORT.md` -- how headless Chromium reaches the
  NVIDIA GPU on the dev box (the `libEGL.so.1` root cause, the four flags, appendix D).
- `packages/webgpu-graph-algorithms/docs/research/` -- the seven research notes, the three drafts and
  the review record (`review/`, with the probe scripts and their logs) that the plan cites as
  "note NN", "draft A/B/C" and "[M]".
- `packages/webgpu-graph-algorithms/docs/decisions/` -- the phase-gate records (G0, G1, ...): measured
  numbers and sign-offs.
- `.github/workflows/` -- `ci.yml` (the default lane: Dawn on Mesa lavapipe plus Chromium on SwiftShader,
  GitHub-hosted runners) and `gpu.yml` (the NVIDIA T4 lane, gated by the `gpu` label).
- `tmp/` -- gitignored scratch: cloned repositories, papers, probes, the NetworkX venv, the owner's
  commit scripts.

## Rules that apply everywhere in this repository

- WebGPU is mandatory: no CPU, WebGL or software-adapter fallback ever lands in `src/` (`CLAUDE.md`).
  The test layer may skip with a printed reason only while `GRAPHTY_GPU_REQUIRE` is unset.
- Plain ASCII in every file. Formatting is Prettier's (`packages/.prettierrc`); the lint rules are the
  monorepo's (`packages/eslint.config.js`).
- Agent sessions never `git add`, `git commit` or `git push`; the owner commits from `tmp/commit-p<N>.sh`.

License: MIT (`packages/*/LICENSE`).
