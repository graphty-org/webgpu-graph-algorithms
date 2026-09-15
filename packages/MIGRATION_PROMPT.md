# Prompt: land @graphty/graph-format and @graphty/graph-io in graphty-monorepo

Paste everything below this line into the agent running in
`/home/apowers/Projects/graphty-monorepo`, or tell it to read this file by
its absolute path. Fill in the "Owner decisions" table first; blank rows
mean "keep what is implemented".

---

## Mission

Two packages were implemented and audited outside this monorepo, in a
staging workspace that mirrors this repo's root:

- `@graphty/graph-format` -- the shared graph data format (frozen CSR over
  typed arrays, mutable builder, columnar attributes, external id map, wire
  container). 1309 tests, lint and strict-consumer typecheck clean.
- `@graphty/graph-io` -- importers and exporters for GEXF, GraphML, GML, DOT,
  Pajek, CSV, JSON dialects and Neo4j CSV, plus format sniffing and a
  registry, with per-format subpath exports. 4271 tests (34 gated
  benchmarks skipped by default), lint and strict-consumer typecheck clean.

Your job, in order:

1. Move both packages into this monorepo and make them first-class
   workspace members: installed, linted, built, tested, covered, wired into
   CI, the pre-push hook, knip, commitlint and the root docs. This is phase
   F1 (plus the IO1 package landing) of the design's landing order.
2. Apply the owner decisions listed below and close the known gaps that are
   marked "before 1.0".
3. Stop and report. Do not cut `1.0.0` and do not start the consumer
   migration (algorithms, layout, graphty-element, webgpu) until the owner
   says so; those phases are described at the end so you can plan, not so
   you can start them.

## Read these first, in this order

1. `/home/apowers/Projects/graphty-monorepo/design/graph-format/graph-format-design.md`
   -- the accepted design (status: Accepted). Section 3.2 has the
   invariants, 12.2 the normative type surface, 13 the package layout and
   monorepo conventions, 14.6 the landing order, 17 the decision log,
   18 the owner decisions already confirmed on 2026-09-13.
2. `/home/apowers/Projects/webgpu-graph-algorithms/packages/README.md`
   -- the move checklist. It was REHEARSED on 2026-09-14 in a scratch copy
   of this monorepo with this repo's own toolchain, and every command in it
   passed there. Follow it literally; it explains every root edit and why.
3. `/home/apowers/Projects/webgpu-graph-algorithms/packages/move/`
   -- `root-touch-points.diff` (the root edits, applies cleanly with
   `git apply --check` against the tree of 2026-09-14),
   `apply-root-claude-md.py` (idempotent edit of the root CLAUDE.md; it
   exists because those lines carry non-ASCII characters), and
   `pnpm-lock.yaml.diff` (reference only; regenerate, never apply).
4. `/home/apowers/Projects/webgpu-graph-algorithms/packages/STATUS.md`
   -- what is implemented, every deviation from the design by section,
   benchmark numbers, verification results, known gaps, and the
   consolidated "Owner decisions needed" list.
5. `/home/apowers/Projects/webgpu-graph-algorithms/packages/CONFORMANCE.md`
   -- a mechanical symbol-by-symbol comparison of the built `.d.ts` files
   against design sections 12.2 and 12.4.
6. The two package `README.md` and `CLAUDE.md` files inside
   `/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/`
   and `.../graph-io/`.
7. `/home/apowers/Projects/webgpu-graph-algorithms/HEADLESS_GPU_REPORT.md`
   -- only if the GPU upload audit test does not skip cleanly (see
   Environment notes).

## Rules that override anything else

- Never run `git add`, `git commit` or `git push`. Prepare the tree; the
  owner commits. Work on a branch the owner names, or ask for one before
  touching the tree. Start from a clean `git status`.
- Plain ASCII in every file you write. Use `--` for dashes and straight
  quotes. The only exception is data: corpus fixtures and the golden
  `.gsnp` container, which already exist and must not be edited.
- No `eslint-disable`, `ts-expect-error` or `ts-ignore` to silence errors.
  Fix the code. Never lower a coverage threshold or relax a lint rule.
- No fallbacks that hide a missing capability. An unsupported input is a
  reported issue or an error, never a silent default.
- The design document is the specification. Where the code and the design
  disagree, the design wins unless the decision log (section 17) or the
  owner decisions (section 18 and the table below) say otherwise. Every
  intentional departure must be recorded in STATUS.md with the section
  reference; never silently "fix" the design.
- Temporary files go under `./tmp/` (gitignored). Servers, if any, use a
  port in 9000-9099. No `sudo`.
- Do not modify `/home/apowers/Projects/webgpu-graph-algorithms/` except to
  `mv` the two package directories out of it, as the checklist says.

## Phase 1: move and land

Follow `packages/README.md` sections 1 through 4 exactly. Summary of what
that means, with the gates you must hit:

1. Clean tree on a branch. Remove `node_modules`, `dist`, `coverage`, `tmp`
   and `*.tsbuildinfo` from both staged packages, then `mv` them to
   `graph-format/` and `graph-io/` at this repo's root. Move both in one
   go: graph-io depends on graph-format via `workspace:*` and a peer range.
2. `git apply --check` then `git apply` `move/root-touch-points.diff`, then
   run `move/apply-root-claude-md.py CLAUDE.md`. If the diff no longer
   applies because the root files changed since 2026-09-14, apply each hunk
   by hand using README section 2's file-by-file description as the spec.
   Every touch point is there for a reason that was found by a failing
   rehearsal run; do not skip the ci.yml items (e), the PR build step, or
   (d), the coverage upload shard names -- both break CI silently.
3. `pnpm install` (not frozen; this is what adds the two importers to the
   lockfile), then `pnpm install --frozen-lockfile` must pass. The expected
   lockfile change is described in README section 3. pnpm's note about
   ignored build scripts for `webgpu` is expected on Linux.
4. Verify from the repo root and record the exact output of each:

   ```bash
   pnpm exec nx run-many --target=lint  --projects=graph-format,graph-io
   pnpm exec nx run-many --target=build --projects=graph-format,graph-io
   pnpm exec nx run-many --target=test  --projects=graph-format,graph-io
   pnpm exec nx run graph-format:coverage && pnpm exec nx run graph-io:coverage && ./tools/merge-coverage.sh
   (cd graph-format && pnpm exec tsc --noEmit -p tsconfig.strict-consumer.json)
   (cd graph-io     && pnpm exec tsc --noEmit -p tsconfig.strict-consumer.json)
   pnpm exec knip
   rm -rf graph-format/dist graph-io/dist && pnpm -r --filter graph-format --filter graph-io run build && (cd graph-io && npm run test:run)
   ```

   Expected: lint clean; build in dependency order; 59 files / 1309 tests
   and 75 files / 4271 tests (34 skipped without `IO_BENCH=1`); coverage
   about 96 and 97 percent lines, merged; both strict compiles pass. knip
   is ALREADY red in this monorepo on two unused exports in
   `graphty/src/components/shell/readings/nodeMetricReading.ts`; the move
   must add nothing to that list. Fix those two only if the owner asks.
5. Read the resulting `.github/workflows/ci.yml` and `release.yml` end to
   end and confirm by reasoning (you cannot run Actions here) that: every
   test shard downloads both new builds; the PR build step guarantees the
   two `dist/` directories exist on PRs that do not touch them; the
   coverage upload step's `if:` lists both shard names; release.yml
   downloads both builds before `nx release publish`.
6. Run `pnpm exec nx show projects --with-target=nx-release-publish` and
   confirm both projects are listed. Run `pnpm exec prettier --check
   graph-format graph-io` and note the six files this monorepo's prettier
   version formats differently (README section 5); reformat them with
   `pnpm exec prettier --write` only if the owner wants the monorepo's
   formatting -- it is not enforced by CI.
7. Delete the verbatim corpus copy's ORIGINAL only when the owner says so:
   `graphty-element/test/helpers/corpus/` was copied into
   `graph-io/test/corpus/`, but the element's own corpus tests still import
   the original. Leave both in place for now.

Gate to phase 2: every command in step 4 green (knip modulo the two
pre-existing findings), and a written report of steps 5 and 6.

## Phase 2: owner decisions and pre-1.0 gaps

STATUS.md ends with "Owner decisions needed (consolidated)", eleven items.
The owner's answers are in this table. Implement each answer; for blank
rows keep what is implemented and leave the item on the list. Every
implemented answer needs: the code change, tests that pin it, a row in
the design's decision log (section 17, append a "post-implementation"
subsection rather than editing existing rows), and the STATUS.md entry
moved from "needed" to "decided".

| # | Topic (see STATUS.md for the full text) | Owner's answer |
| --- | --- | --- |
| 1 | `AttributeTable.set()` on a snapshot: range-check same-space `refersTo` and refuse cross-space declarations, or leave to `validate()` | |
| 2 | Weight flags describe the f32 arc array (as implemented) or the exact f64 staged weights when a shadow exists | |
| 3 | Confirm the doc-silent choices: `weighted: false` semantics, `graphty.weight` shadow column name, carried-views policy under "full", last-wins rule of `toRecord` / `toStringMap` | |
| 4 | `GraphBuilder.setMeta()` merges `extra` instead of replacing it | |
| 5 | `runFreeze` and five other builder functions above the length threshold: split, or accept as one dtype switch each | |
| 6 | The replaced fixture `test/corpus/malformed/csv/binary-content.csv` (now invalid UTF-8): keep, or restore the legacy 14-byte file | |
| 7 | The issue-code renames (about 90 codes, one `<FORMAT>_ISSUE` / `<FORMAT>_LOSS` scheme): confirm before anything ships | |
| 8 | Loss notes (`W_ID_RENUMBERED`, `W_WEIGHT_KEY_CLASH`) as the answer to Pajek labelled ids and plain `weight` / `value` columns, or rename the written column; and whether Pajek's default `sanitizeIds: "error"` should refuse instead of renumbering | |
| 9 | JSON keys colliding with a caller-declared column of another dtype are per-element `E_COLUMN_TYPE` issues (no rename for inferred attributes) | |
| 10 | GEXF `idtype` is not honoured on import (canonical id rule; `ids: "string"` opts out) | |
| 11 | `E_EMPTY_ID` dropped: an empty-string id round-trips through CSV and Neo4j as a quoted empty cell | |

Known gaps to close before `1.0.0` (from STATUS.md "known gaps"; the
others can wait):

- GEXF import still parses the whole document into memory instead of
  streaming through the shared XML tokenizer that GraphML uses. Design 8.4
  allows whole-document parsing for XML, so this is a performance gap, not
  a correctness one: fix it if the 1M-edge GEXF benchmark in
  `graph-io/benchmarks/` shows more than about 3x GraphML's cost per edge,
  otherwise record the numbers and defer.
- Hot-loop allocations noted by the streaming audit (GEXF node frame, DOT
  per-edge `Map` copy, Pajek per-line token array): remove them only where
  the change is local; re-run `pnpm run benchmark -- --quick` in graph-io
  before and after and keep the numbers in STATUS.md.
- Compaction after removals costs about 32 ms of a 43 ms re-freeze at 1M
  edges against a design estimate of 10-15 ms. The design's estimate was
  wrong, not the code; record it as a design correction in section 15 with
  the measured breakdown rather than optimising further now.

Gate to reporting: all of phase 1's commands green again after every
change; STATUS.md and CONFORMANCE.md updated; the design doc's new
decision-log rows written.

## Stop here and report

Report, in this order: the exact output of the phase 1 verification
commands; the list of root files changed and any hunk you applied by hand;
the lockfile delta; which owner decisions were implemented and how; the
benchmark numbers before and after any hot-path change; everything still
open. Then wait. The next steps below are the owner's call.

## Next phases (do not start without an explicit go)

These are design section 14.6. They are listed so you can size them and
so that nothing you do in phases 1 and 2 forecloses them.

- **F2: cut `1.0.0`.** The design's rule (13.5 item 5): no consumer PR that
  adds `@graphty/graph-format` to its `dependencies` may merge until the
  format on `master` is `>= 1.0.0`, and a CI check must compare the two
  `package.json` files. The A1 algorithms branch must be green against the
  0.x format first. `nx release` handles the version; trusted publishing on
  npmjs.com must be configured for both packages before the first publish
  (placeholder `0.0.0` packages exist for that purpose; configure them
  against `.github/workflows/release.yml` in `graphty-org/graphty-monorepo`).
- **A1 (branch) then A2: algorithms.** `toSnapshot(Graph)` with
  `mutationCount` memoisation, the differential harness, then the `indexed`
  namespace porting all 95+ functions with every public first parameter
  widened to `Graph | GraphSnapshot`. Design sections 14.1-14.2 have the
  result-shape table and six worked ports; research note
  `tmp/graph-format-design/01-algorithms-needs-core.md` and `02-...` (if
  still present; they are gitignored) catalogue what every algorithm needs.
- **L1: layout.** `toLayoutSnapshot`, indexed layouts, positional
  wrappers, the `LayoutGraph` generator wrapper. Section 14.3. Chromatic
  re-baseline is accepted (section 18 decision 6).
- **E1 and IO1: graphty-element.** `DataManager` owns the builder; the
  adapters and layout engines use `indexed.*`; `DataSource` subclasses
  become wrappers over the graph-io importers and `format-detection.ts` is
  replaced by `sniff()`; the element's corpus tests move to graph-io and
  the original corpus directory is deleted. Sections 14.4 and 8.2. The
  behaviour changes users will see are enumerated in section 18 decision
  5 and were accepted as unflagged improvements.
- **W1: webgpu-graph-algorithms.** Move-in of
  `/home/apowers/Projects/webgpu-graph-algorithms` as a package consuming
  snapshots directly; repeat the root touch points of section 13.3. Its
  current `src/types/index.ts` predates the design and is replaced.
- **D1 and 2.0.** Deprecation tags only after no in-repo caller remains
  (the root `no-deprecated` lint rule is an error in `src`), then removal
  of the legacy `Graph` input paths and facades at the consumer majors.

## Environment notes

- pnpm 10.0.0, Node 22. The staging workspace used a newer knip and
  prettier than this monorepo pins; the README's section 5 lists the
  consequences already handled.
- `graph-format` declares `webgpu@^0.4.0` (Google Dawn for Node) as a
  devDependency for one audit test that uploads real buffers to a GPU.
  0.4.0 is pinned because newer builds need a glibc this machine lacks.
  The test skips with a printed `E_NO_ADAPTER` reason when no adapter is
  available, and it needs `LD_LIBRARY_PATH` to include a directory
  containing `libEGL.so.1` to reach the NVIDIA card in this container
  (HEADLESS_GPU_REPORT.md appendix D shows how to obtain it without root).
  A skip is acceptable; a wrong result is not.
- `IO_BENCH=1 pnpm exec vitest run test/audit` in graph-io runs the gated
  streaming and memory checks (about five minutes). Run it once after the
  move to confirm the environment, not on every change.
- The design doc lives at `design/graph-format/`. Research notes and
  drafts were under `tmp/graph-format-design/`, which is gitignored and
  may be gone; the design doc is self-contained without them.
