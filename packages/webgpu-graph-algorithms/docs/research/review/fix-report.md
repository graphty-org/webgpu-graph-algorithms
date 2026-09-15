# Fix report -- webgpu-acceleration-plan.md, adversarial review of 2026-09-14

Document: /home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md
Before: 3,329 lines. After: 4,406 lines. Status unchanged: "Draft for owner review".

## Counts

| | |
| --- | --- |
| Reviewer findings across the six lenses | 123 |
| Confirmed by the verifiers | 94 |
| Downgraded / narrowed (applied in the narrowed form) | 27 |
| Refuted (not applied): PERF-14, MAINT-12 | 2 |
| Verifier-added findings (MISSED-* / M-*) | 28 |
| Surviving findings handed to the fixer | 149 |
| Applied | 149 |
| Rejected by the fixer | 0 |

The document's new "Review log" section (after section 15) carries the same
counts and a table mapping every applied finding id to the sections changed.
Finding ids that collide across lenses (MISSED-1 / MISSED-2 in PERF, DESIGN and
VERIFY; M-1..M-5 in INTEG) are prefixed by lens in the log and below.

## Conflicts between findings, and how they were resolved

1. COMPLETE-5 (move the indirect `finalize` kernel and grid-stride out of P2
   to P8 / P7) versus PERF-11 and DESIGN-12 (the grid build's hub-cell tier
   needs an indirect dispatch, i.e. in P4). Resolution: the finalize kernel and
   `planIndirect` land in P4 -- out of P2 as COMPLETE-5 asked, before P8 as
   the grid needs; grid-stride and `packViews` go to P7. Recorded in P2's
   "NOT in P2" list and in P4's deliverables.
2. MAINT-14 (replace `GRAPHTY_GPU_NO_SUBGROUPS` with a `GpuContextOptions`
   feature override) versus D19 / 12.2 (the test setup reads one policy
   variable and `src/` reads no env vars) and VERIFY-12 / COMPLETE-M4 (the
   no-subgroups CI pass must cover layouts). Resolution: the mechanism inside
   `src/` is the existing `optionalFeatures` option (`[]` forces the twins);
   the env variable stays as a test-setup knob that maps onto it; every kernel
   with a twin is tested against both variants in-process (a `variants` axis in
   11.3, via `acquire({ subgroups: false })` with a fresh adapter), and the CI
   pass runs over `test/primitives test/layouts` on the default lane and the
   whole node project on the GPU lane.
3. PERF-1's fix as written ("clamp the CELL KEY of nodes outside the extent to
   the boundary cell and exclude them from that cell's Horvitz-Thompson count")
   needs per-node flags inside real cells; the plan instead keys every outside
   node to ONE pseudo-cell (index `G^dim`), which the sort groups, G4 / G4b
   centroid like any cell, the far field adds as one extra term for inside
   nodes and the near field samples for outside nodes. Same effect (no real
   cell's count includes a stray, strays still feel the core), fewer special
   cases; `stats.outsideGrid` is `cellHist[G^dim]`.
4. PERF-3 offered two options for the resampling-noise concern; the plan
   records it as risk R-24 with adaptive `nearMax` as the measured mitigation
   (the "free nodes not in over-capacity cells" variant would need K5 to know
   cell occupancy).
5. MAINT-18's fix (a CPU-side helper owned by @graphty/algorithms inside the
   GPU result) is impossible under D3; the verifier's alternative (the
   `accelerated()` dispatcher decorates `SsspResultLike` with `pathTo` /
   `pathEdges`) was applied, which also resolves INTEG-12.
6. INTEG-16's main claim was refuted by its verifier; only the wording change
   ("a package-local eslint.config.js extending the root") was applied.
7. VERIFY-11's tolerances were applied as the verifier revised them (1e-9 at
   maxIter 1 and 5, 1e-6 at 50 after rescaleLayout; NetworkX at iteration 0
   only).

## Decisions turned into owner questions (section 14.2, each with a default)

- Q-26 (INTEG-2 / MAINT-7): published accelerator contract = structural
  mirrors (D27) versus optional peer dependencies with `import type`. Default:
  mirrors.
- Q-27 (DESIGN-10 / INTEG-6): `withColumns()` siblings as one residency unit
  versus per-object counting. Default: one unit.
- Q-28 (COMPLETE-7): the public repository `graphty-org/webgpu-graph-algorithms`
  created and pushed in P0. Default: yes.
- Q-29 (INTEG-7): accept nx side-effect patch releases versus
  `updateDependents: "never"`. Default: accept.
- Q-30 (INTEG-4): who resolves `nodeMass` / named-column `weight` for an
  element-created simulation. Default: the GPU simulation, from graph-format
  primitives (a documented structural duplicate of the layout helper).
- Q-31 (INTEG-18): `workspace:^` and the correction to design 13.5 rule 3 /
  graph-io. Default: `workspace:^` here; propose the correction at W1.
- Q-32 (PERF-1 / PERF-6): `extentFactor = 6` and `gridMax2D = 512` re-checked
  at G4. Default: keep until the P4 decision record says otherwise.

New decisions recorded in 1.4: D23 (vec4f positions), D24 (sorted-order grid
dispatch), D25 (no displacement clamp), D26 (layering / composition root),
D27 (mirrors are the published contract). New departures in 1.5: DEPARTURE-5
(injection spelling), DEPARTURE-6 (the two 1e-4 tolerances). New risks: R-23
(useWebGPU + injected accelerator), R-24 (near-field resampling vs settle),
R-25 (busy dev GPU vs the 3x benchmark rule).

## Cross-section reconciliation performed after the edits

- `exactMaxNodes` 16,384 labelled conservative everywhere (D7, 3.1, 7.8, 7.14,
  R-2, Q-6, Summary); `calibrateLayout` ladder 8k-65k; T-4's exact ladder and
  the grid ladder named once and cited from 7.8, 11.7, T-4, P3, P4.
- Tolerances: 1e-5 everywhere except BC and the one-iteration force parity at
  1e-4 (1.3 row 16.2, DEPARTURE-6, 9.7, 11.4, Q-24); the trace's 1e-4 leg is
  against an f32 oracle.
- Binding counts: K2 6, K3 6, K5 6, G7 8 (fixed replaces mass), PageRank pull
  exactly the 8 named in 8.2 / 3.5 / 11.3 / G7; a new 8.10 table for every
  algorithm kernel.
- Bytes per node: exact ~53, grid 65 resident / 81 peak, used identically in
  4.7, 7.3 and 10.1; arena figures 164,000,256 / 284,000,256 /
  1,640,000,256 / 2,840,000,256 in 4.2, 4 review notes, 10.1 and DEPARTURE-2.
- Lane budgets: 15 min default / 20 min GPU in T-12, G2 and 12.6; the 1M
  200-iteration exact-vs-grid run moved to the nightly benchmark job.
- Names: `createAccelerator(ctx, options?)` and `calibrateLayout(ctx,
  options?)` replace `ctx.accelerator()` / `ctx.calibrate()` in 2.2, 3.2,
  3.3, 7.8, 9.5, 9.6, 9.8, P10, P12, R-2, Q-6, Q-21; `test/oracle/<name>.ts`
  replaces `test/helpers/oracle.ts`; scripts are `.js`; `bench` is a tsx
  harness, not a vitest project (11.1, 12.3, 3.1); one results directory.
- Error codes: `E_IN_FLIGHT` removed; `E_NO_DEVICE { reason: "consumed" }`
  and `E_SHADER_COMPILE { stage: "compose" }` added; pass-through
  `GraphFormatError` codes are `E_GPU_INELIGIBLE` and `E_UNKNOWN_NODE` (the
  accepted finding wrote `E_UNKNOWN_ID`; graph-format's `requireIndex` throws
  `E_UNKNOWN_NODE`, `src/ids/node-id-map.ts` line 529 -- corrected).
- Option names: `SimulationOptions` (layout-owned: settleThreshold,
  settleWindow, iterationsPerStep, maxInFlight) versus `GpuLayoutTuning`
  (repulsion, exactMaxNodes, nearMax, deterministic, gridMax2D, gridMax3D,
  extentFactor, compat) consistent across 3.3, 7.14, 9.3, 9.4 item 7.
- Gate sign-off: the 7.2 table is signed off at G0 (D21, 7.2, R-1, Q-1, P0),
  not G3.
- Every Q-id is now cited from at least one section outside 14.2.
- Markdown tables checked programmatically: every table has a consistent
  column count.

## Verification commands

    grep -nP '[^\x00-\x7F]' design/webgpu-acceleration-plan.md | head   -> (empty)
    grep -n -i 'TBD\|TODO\|placeholder' design/webgpu-acceleration-plan.md | head   -> (empty)
    wc -l design/webgpu-acceleration-plan.md   -> 4406

## Applied finding ids (149)

PERF-1, PERF-2, PERF-3, PERF-4, PERF-5, PERF-6, PERF-7, PERF-8, PERF-9,
PERF-10, PERF-11, PERF-12, PERF-13, PERF-15, PERF-16, PERF-MISSED-1,
PERF-MISSED-2, DESIGN-1 .. DESIGN-21, DESIGN-MISSED-1 .. DESIGN-MISSED-4,
MAINT-1 .. MAINT-11, MAINT-13 .. MAINT-20, MAINT-M1 .. MAINT-M6, INTEG-1 ..
INTEG-23, INTEG-M-1 .. INTEG-M-5, VERIFY-1 .. VERIFY-21, VERIFY-MISSED-1 ..
VERIFY-MISSED-5, COMPLETE-1 .. COMPLETE-22, COMPLETE-M1 .. COMPLETE-M6.

Rejected: none.
