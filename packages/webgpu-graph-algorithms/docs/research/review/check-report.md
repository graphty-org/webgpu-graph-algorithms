# Check report -- fix verification for webgpu-acceleration-plan.md

Document: /home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md
Inputs: fix-report.md and the six *-verdicts.md files under
/home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/review/.
Date: 2026-09-14.

Method: the plan was read in full (4,406 lines as handed over); for every
one of the 149 applied ids the changed text was located and compared with
the verdict's fix (as narrowed by the verifier); the fixer rejected nothing,
so no rejection had to be checked. A whole-document consistency pass then
covered ids (D, R, T, G, GOAL, Q, DEPARTURE, phases), section
cross-references, inline URLs against section 15, markdown table widths,
repeated numbers / defaults / names, the upstream design (sections 10,
14.3-14.6) for undeclared contradictions, and the Review log counts.
Claims about graph-format were checked against the staged source
(`node-id-map.ts` line 529 `E_UNKNOWN_NODE`, `column.ts` 1994
`E_GPU_INELIGIBLE`, `index.ts` 27 `expandEdges`, `derived.ts` 1155
`renumberPartition`, `views.ts` 568-604 `segmentOffsets`, `columns.ts` 80
`NumericVector`); every probe file cited as [M] exists under
`review/probes/`.

## Result

- Applied and verified: 149 / 149.
- Not actually applied: 0.
- Wrong rejections: 0 (nothing was rejected).
- Regressions introduced by the fixes: 4, all small and repaired here.
- Other consistency defects found in the pass: 2 repaired here, 6 reported
  for the owner (no edit).
- Non-ASCII characters: 0. TBD / TODO / placeholder: 0. Lines: 4,409.

## Edits made by the verifier

1. Twelve table rows carried unescaped `|` inside backtick spans (math
   absolute-value bars, `F32 | F64 | U32 | I32`, `||`), which GitHub
   markdown treats as cell separators; 35 pipes escaped as `\|` on lines
   206, 411, 1665, 1666, 1712, 1921, 1954, 2382, 2385, 2388, 3414, 4035
   (pre-edit numbering). The fix report's "every table has a consistent
   column count" did not hold under GFM rules; it does now (checked with a
   script that ignores escaped pipes and fenced blocks).
2. Section 15 lacked two references the body cites: `pnpm.io/workspaces`
   (1.3 row 13.5, 2.5) and `docs.github.com/en/actions/reference/limits`
   (12.1). Both added to the external list (the plan's own rule in "How to
   read": section 15 collects every URL).
3. Summary, 3.5 and 8.10 called the PageRank pull "the one kernel that uses
   exactly 8" while 7.7's G7 and the new 8.10 table list six kernels at 8
   (G7, BFS fused, BFS bottom-up, SSSP near-far, BC forward, Louvain move).
   Reworded in the three places; the 8.10 table itself was already right.
4. 1.3 row 14.5 said "Two DEPARTURES" (1 and 5); DEPARTURE-4 also departs
   from 14.5 (line 4231). Now "Three DEPARTURES (1, 4, 5)".
5. 9.8 W1 row and P10 scheduled design amendments for DEPARTURE-1, -5, -6
   only; 1.5 declares six. Both now read "10.3 / 14.5 / 14.6 / 16.2 / 16.7
   amendments (DEPARTURE-1, -2, -4, -5, -6; DEPARTURE-3 amends 14.3 in the
   L1 PR)".

Items 3-5 were introduced by the fixes (8.10, DEPARTURE-5 / -6 and the
"one kernel at 8" sentences are new text) and are listed as regressions
below; items 1-2 are pre-existing consistency gaps the fixer's checks
should have caught.

## Applied ids: where the fix lives (spot references, post-edit line numbers approximate)

PERF-1 7.7 extent table (min(bbox, extentFactor * rmsRadius), outside
pseudo-cell), 7.3 partials `.w`, 3.3 `outsideGrid`, 11.4 isolated-node
fixture, R-3. PERF-2 D24, 7.7 sorted-order paragraph and G6 / G7 rows,
7.21 "conservative". PERF-3 7.17 RMS normaliser, R-24. PERF-4 8.2 eight
named bindings via pre-scaled `xNorm`, 6 row 9, 3.5, G7. PERF-5 6 row 7
overflow rule, 8.4, 10.1 "A entries: no overflow", G8 faked 4,096
capacity. PERF-6 7.7 "raise gridMax2D (FINER cells)", Q-32. PERF-7 D23,
7.3 vec4f, 7.5 / 7.6 / 7.7 loads. PERF-8 7.6 measured curve, 7.8, 7.21,
10.3, T-4. PERF-9 2.2 `CalibrateOptions` default [8k, 16k, 32k, 65k],
min(4 ms, gridMs), first-call cost. PERF-10 11.4 1M scope, G4, T-12.
PERF-11 7.7 G4 / G4a / G4b, 6 row 12, 5.4. PERF-12 5.4 selector, 6 row 8,
8.4 device-side switch, `switches` counter. PERF-13 8.4 per-source model,
10.3 BFS 10-30 ms, BC 2-20 s, Q-13. PERF-15 7.2 and 7.7 parenthesised
hashes, 3.5 rule 2. PERF-16 7.21 basis paragraph (10-17x / 5-10x / equal
readback), 10.3 integrated row. PERF-MISSED-1 / DESIGN-1 D25, 7.7 clamp
paragraph, 7.11 K5 (no clamp), 11.4 expansion parity, G4. PERF-MISSED-2
7.3 partials A `.w` lane, stride 64 B.

DESIGN-2 7.2 distance-floor row, 7.7 `eps = 0.25 * cellSize`, 7.3 state.
DESIGN-3 7.2 global sums row, 7.4 K3 / G7 bind `fixed`, 7.6 epilogue,
7.10, 7.11. DESIGN-4 / INTEG-8 7.10 snippet conditional form, 7.2
`estimateFactor` row. DESIGN-5 3.5 rule 1, 7.6 / 7.11 sketches, 6 rows 3
and 8. DESIGN-6 / DESIGN-MISSED-2 D16, 2.6 subgroups row, 3.5 elected-lane
counter, 6 subgroup paragraph, R-8. DESIGN-7 8.2, 8.10 table. DESIGN-8 5.4,
8.4, G8. DESIGN-9 8.4 `n x k` deltas + gather, 8.10 BC rows, P9.
DESIGN-10 / INTEG-6 4.1 (one residency unit; no `refs: Set`),
DEPARTURE-4, Q-27. DESIGN-11 6 row 3 `[midEnd, n)`, 7.4 K2, 7.5, G4
isolated-node force. DESIGN-12 6 row 12 histogram + scan, 7.7 G3, 7.3
`cellHist`. DESIGN-13 7.2 coincident row, 7.6 kick, 7.7 G7 own-cell
weight. DESIGN-14 8.4 `atomicMin` claim, 6 row 4 two-dispatch dedupe.
DESIGN-15 2.2 steps 1 / 4-5, 2.6 rows, 5.3, 5.4, 5.5, 5.7, R-22, Q-16.
DESIGN-16 11.3 variants row, 11.5. DESIGN-17 8.7 (5,792 / 23,170 /
32,767), 8.1. DESIGN-18 7.7 cell-size floor, 11.4 floored denominator.
DESIGN-19 as PERF-5. DESIGN-20 6 row 6 digit-major, 7.4 33-42 / 31-40
(recounted: correct). DESIGN-21 7.1 item 6, 7.13, 7.18, 11.3.
DESIGN-MISSED-1 6 row 8, 8.4. DESIGN-MISSED-3 D8, 7.12, 7.17.
DESIGN-MISSED-4 7.4 K1 `select`, 7.17, 11.3.

MAINT-1 3.5 `BindingDecl` / `OverrideDecl`, 5.1, 7.5 / 7.6 bodies only.
MAINT-2 3.5 compose-stage `E_SHADER_COMPILE`, functor parameter lists in
6 rows 3 / 8. MAINT-3 5.3 storage mode, 7.3 state row, 11.3 round trip.
MAINT-4 D26, 2.2 header comment and step 6, 3.1 tree, 3.2 layer rule and
lint zones, 3.3, 9.5, 9.6. MAINT-5 7.19 `ForceModel`, 3.2 class list.
MAINT-6 3.1 `kernels.ts`, 3.5, 5.1, 11.3, CLAUDE.md recipes. MAINT-7 /
INTEG-2 D27, 2.4, 3.3, 9.1, 9.8, P10. MAINT-8 / INTEG-17 / VERIFY-6 /
COMPLETE-M3 11.8, 11.1 node `include`, G1. MAINT-9 4.4 slot lifecycle,
5.8. MAINT-10 3.5 prelude interpolation and grep test, 5.1 override list
(no `DIM`), 2.2 `RaisableLimit` comment, 5.2. MAINT-11 2.2 declarations,
3.2 `Profiler` row, 3.3 `ForceAtlas2Options`, 4.1 `Extract<ViewName,...>`
(`ForceAtlas2Params` gone). MAINT-13 2.4 two readers, 5.2. MAINT-14 3.1
`.js` scripts and one results directory, 11.3 variants axis, 12.2 / 12.3
no-subgroups passes. MAINT-15 3.1 / 6 / 11.3 `test/oracle/<name>.ts`, 9.8
/ P10 second oracle, P3 spec-of-L1. MAINT-16 3.3 `GpuLayoutTuning`
comment, 7.14, 9.3 `SimulationOptions`. MAINT-17 3.3 `precision` on every
score result, `@internal` residency + `stripInternal` (3.1), `degree`
diagnostic, Q-24. MAINT-18 1.3 row 14.5, DEPARTURE-5, 9.2 decoration.
MAINT-19 3.1 CLAUDE.md sections, Q-18. MAINT-20 2.2 step 1, 2.6, 5.7 /
5.8 batch-local rejection, 11.2 fresh adapter. MAINT-M1 D12, 3.3, 4.3,
5.7 (`E_UNKNOWN_NODE` verified in graph-format). MAINT-M2 3.3
`AcceleratorOptions`, 7.14, 9.4 item 7, 9.5, R-2. MAINT-M3 2.3
`scripts/gpu-policy.js`, 12.2, 12.3. MAINT-M4 2.4 scoped lint rule.
MAINT-M5 3.1 tree (no root shim), 2.5 `entries.js`. MAINT-M6 3.3 generic
`GpuLayoutSimulation<Options, Stats>`, 7.19, 7.20.

INTEG-1 / VERIFY-3 D10, 12.1, 12.3 `gpu.yml`, 12.4, 12.5, R-6. INTEG-3 3.3
`AcceleratorOptions` / `setParams(Partial<Options>)`, 8.4, 9.2 `sources` /
`k`, 9.3, 9.4, 9.5, Q-13. INTEG-4 7.3, 7.5, 7.14 `resolve.ts`, 9.3, Q-30.
INTEG-5 4.5 diagram and rationale, 9.4 item 2, 11.3 property. INTEG-7 9.8,
Q-29. INTEG-9 / VERIFY-1 2.5 item 1 (specifier test, hard-fail under CI,
bundle-only subpaths), 12.3 `pnpm run build`, 12.5, G0, G1. INTEG-10 /
INTEG-11 / COMPLETE-17 1.3 rows 14.5 and 16.2, DEPARTURE-5, DEPARTURE-6,
Q-24. INTEG-12 3.3 `GpuSsspResult` comment, 9.2, 9.7. INTEG-13 D13, 2.6
row, 9.1 `Algorithm.ts:217`, 9.4 item 4. INTEG-14 9.3 `SimulationType`
table, 3.3, 7.20, 9.4 item 6. INTEG-15 9.2 growing method list, 9.8.
INTEG-16 2.4 package-local eslint config, 12.5. INTEG-18 1.3 row 13.5,
2.5, 3.1, 9.8, Q-31. INTEG-19 2.2 `ProbeResult.adapter: GPUAdapter`,
`rejectSoftware` in create, 3.4, 9.5. INTEG-20 1.2, R-23, 7.19. INTEG-21
9.1, 9.4 item 8, 9.8 W2. INTEG-22 9.4 preamble, 9.8 E1 row, P6. INTEG-23
2.5 peer range, P-ENV. INTEG-M-1 9.4 item 1, 9.5, 7.19, 5.7. INTEG-M-2 /
VERIFY-MISSED-3 12.5. INTEG-M-3 / VERIFY-15 12.1, 12.3 canary grep.
INTEG-M-4 3.1 `lint` script, 12.5, G10. INTEG-M-5 7.1 item 6, 7.12, 7.19,
9.4 `reload`, 11.3 property.

VERIFY-2 12.5 monorepo `gpu.yml` and the "never gains schedule / labeled"
paragraph. VERIFY-4 12.2 per-job container, 12.4 host loop. VERIFY-5 11.5,
G1, section 11 review notes. VERIFY-7 12.3 `shell: bash`, no `tee`.
VERIFY-8 11.6, 12.3 `run-browser-project.js`, R-13, 13 rule (a). VERIFY-9
/ COMPLETE-14 12.3 Playwright steps in the GPU job, 12.4. VERIFY-10 D21,
7.2, G0. VERIFY-11 7.2, 11.4 (1e-9 at 1 / 5, 1e-6 at 50 after
`rescaleLayout`; NetworkX iteration 0 only), G3, P3. VERIFY-12 /
COMPLETE-M4 12.2, 12.3, 12.5, G3, G7. VERIFY-13 / VERIFY-MISSED-5 11.1,
11.7 async `bench()`, 12.3, 10.4 runner class, 3.1. VERIFY-14 /
COMPLETE-13 12.3 `gpu-nightly-report`, 12.6. VERIFY-16 2.3, 11.1, 3.1.
VERIFY-17 11.4, 11.1. VERIFY-18 5.1 `override-matrix.ts`, 11.3, 11.6, 3.1.
VERIFY-19 12.3 `overwrite: true`. VERIFY-20 T-13, 11.7, 12.3, 12.6, R-25.
VERIFY-21 10.4, 11.4 f32-oracle leg. VERIFY-MISSED-1 12.3
`package_json_file`, P0. VERIFY-MISSED-2 7.2, 9.3, 11.4. VERIFY-MISSED-4
12.2, 12.4.

COMPLETE-1 as PERF-4 plus the Summary. COMPLETE-2 as MAINT-M2 plus P12.
COMPLETE-3 9.4 items 1 / 4, 2.4, 7.19 `.catch` once. COMPLETE-4 D7, 3.1,
7.8, 7.21 32k row, Q-6. COMPLETE-5 13 P2 "NOT in P2" list, P4 (finalize +
`planIndirect`), P7 (grid-stride, `packViews`), 7.5, 6 row 3, critical
path. COMPLETE-6 7.20, Q-9, P5 (4-5 ed). COMPLETE-7 3.1, 12.1, P0, Q-28.
COMPLETE-8 7.3 state COPY_DST, 7.17. COMPLETE-9 2.5, 3.1, 11.3.
COMPLETE-10 4.7, 7.3, 10.1 (53 / 65 / 81 B per node, checked
arithmetically against the 10.1 rows). COMPLETE-11 4.2, section 4 review
notes, 10.1, DEPARTURE-2 (`...,256` figures, matching arena-bytes.mjs).
COMPLETE-12 10.4, 11.1, 11.7, T-4, P4 (one baseline name, one exact and
one grid ladder). COMPLETE-15 1.1 `GOAL-n`, preamble. COMPLETE-16 every
Q-id is cited outside 14.2 (checked by grep for Q-1..Q-32), D11 / P-ENV
disambiguate draft C's Q-10. COMPLETE-18 2.2, 2.5 ("0.5.x assumed"), 5.1,
7.4 ([M] per-dispatch overhead from far-field-order.mjs), 8.4 ([P]),
D13, 9.1, 1.2, R-22. COMPLETE-19 3.3 `step()` contract, 7.14 default 1,
7.19, D12, 9.3 `spring` alias; `E_IN_FLIGHT` appears only in the
sentence that removes it. COMPLETE-20 Q-18, 3.1. COMPLETE-21 13
critical-path paragraph. COMPLETE-22 G2, T-12, 12.6, 10.3 x100 column
(recomputed: 20-140 ms / 0.2-1.4 s / 2-14 s), T-8. COMPLETE-M1 8.2
`firstConvergedIteration`, 3.3, 9.7. COMPLETE-M2 T-5, 11.6 item (8),
11.7. COMPLETE-M5 7.21 / 10.3 one exact figure, 13 total 83-113 ed
(re-summed from the fourteen phase sizes: 83 / 113), DEPARTURE-2.
COMPLETE-M6 7.3 trace record, 7.10, 3.3.

## Consistency pass

- Ids: D1-D27, R-1..R-25, T-1..T-15, G0-G12 + G-ENV, GOAL-1..9, Q-1..Q-32,
  DEPARTURE-1..6, phases P0-P12 + P-ENV are each defined once and every
  defined id is cited; every cited id is defined.
- Section references: every `N.N` reference that is not a numeric literal
  resolves to a plan heading or is an explicit design / note section.
- URLs: every inline URL or bare-domain citation in the body is in
  section 15 (after edit 2).
- Tables: consistent widths after edit 1.
- Numbers repeated across sections agree: `exactMaxNodes` 16,384
  (conservative; 32,768 predicted); `nearMax` 64; `gridMax2D` 512 /
  `gridMax3D` 128; `extentFactor` 6; `settleThreshold` 1e-3 /
  `settleWindow` 10; `maxInFlight` 2; `iterationsPerStep` 1;
  `warnUnreleasedSnapshots` 2; port 9058; lane budgets 15 / 20 min with
  `timeout-minutes` 30 / 45; arena bytes 164,000,256 / 284,000,256 /
  1,640,000,256 / 2,840,000,256; per-node bytes 53 / 65 / 81; exact tile
  1.13 ms at 16k, ~18 ms at 100k, ~1.8 s at 1M; grid 35-95 ms at 1M;
  binding counts K2 6 / K3 6 / K5 6 / G7 8 / PageRank pull 8; tolerances
  1e-5 except BC and one-iteration force parity at 1e-4; oracle
  cross-check 1e-9 / 1e-9 / 1e-6; APSP 5,792 / 23,170 / 32,767; dispatch
  boundary 16,776,960; grid dispatch count 33-42 (2D) / 31-40 (3D)
  recounted from the kernel table.
- Upstream design: sections 10 and 14.3-14.6 were re-read. The six
  DEPARTURE entries cover every contradiction found, with one soft
  exception reported below (14.3's "stride as a uniform" sentence).
- Review log: the per-lens counts (16/12/3/1/2, 21/19/2/0/4, 20/10/9/1/6,
  23/20/3/0/5, 21/15/6/0/5, 22/18/4/0/6; totals 123 / 94 / 27 / 2 / 28;
  149 applied) match the PER-FINDING verdicts in the six files; the
  mapping table has exactly one row per applied id. Note that three
  verdict files carry a summary line that disagrees with their own
  per-finding tallies (DESIGN "17 confirmed" is 19; VERIFY "12 confirmed,
  8 downgraded" is 15 / 6; COMPLETE "15 confirmed, 5 downgraded" is 18 /
  4) -- the Review log follows the per-finding verdicts, which is correct.

## Reported, not edited (owner decisions or nits)

1. Design 14.3 (lines ~3995-3997) says `LayoutSimulation` "kernels take the
   position STRIDE (3) as a uniform and operate on the owner's stride-3
   column directly". D23 makes device positions the simulation's own
   `array<vec4f>` with a `toScene` kernel; the sentence's OUTCOME (the
   owner's stride-3 array read at `load()` and written by every readback,
   no per-frame copy) is honoured, but the mechanism differs and 1.5 does
   not list it. Owner call: add a DEPARTURE-7 row, or a clause in D23
   stating that 14.3's sentence describes the CPU simulations' in-place
   loop.
2. D24's "measured 4-60x faster" is a loose bracket: the near-field pairs
   in 7.7 give 4.7x / 16x / 43x for sorted order alone and 9x / 18x / 70x
   with the vec4f load; the far field 2.3-2.8x. 7.7 states the exact
   numbers, so this is a summary-line nit.
3. The 5.2 heading and 1.3 row 10.6 still say "DispatchPlanner" while 3.1
   / 3.2 name the pure functions `plan1d` / `plan2d` / `planGridStride` /
   `planIndirect` (MAINT-11 asked for one name per thing); a label, not an
   API name.
4. 9.5 sketch: `requestGpuContext({ adapter: probe.adapter, ... })` passes
   a `GPUAdapter | null` into an option typed `GPUAdapter | undefined`
   (non-null after the `if (!probe.ok) return` guard at runtime, but a
   strict TS consumer would need `?? undefined`). Sketch-level.
5. Design 10.1 sizes the low tier as `[midEnd, lowEnd)`; the plan's 6 row
   3 uses `[midEnd, n)` and says why (views.ts 568-604 puts degree-0 rows
   in `[lowEnd, n)`; design 10.6 itself names the zero-degree segment).
   Explicit, not silent; no action.
6. The three verdict-file summary lines noted above are wrong against
   their own tallies (a defect in the review artefacts, not the plan).

## Verification commands (post-edit)

    grep -nP '[^\x00-\x7F]' design/webgpu-acceleration-plan.md | wc -l   -> 0
    grep -c -i 'TBD\|TODO\|placeholder' design/webgpu-acceleration-plan.md   -> 0
    wc -l design/webgpu-acceleration-plan.md   -> 4409
    python3 tablecheck.py (escaped-pipe-aware, fences ignored)   -> none inconsistent

Ready for the owner: yes, with item 1 above as the one open design-text
question the owner should settle when reading 1.5.
