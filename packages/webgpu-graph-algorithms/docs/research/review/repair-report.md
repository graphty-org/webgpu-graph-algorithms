# Repair report -- webgpu-acceleration-plan.md

Document: /home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md
Input: the fix verifier's item list (4 regressions, 7 consistency observations,
0 not-applied). Date: 2026-09-14.

## 1. Regressions (all four were already repaired by the check pass; re-verified here)

| Item | Where | State found | Evidence |
| --- | --- | --- | --- |
| "the one kernel that uses exactly 8" | Summary (line ~90), 3.5 (~1038), 8.10 table note (~2781) | applied | Summary: "the PageRank pull kernel of 8.2, the grid near field of 7.7 and five frontier / community kernels of 8.10 use all 8"; 3.5: "the first kernel that uses exactly 8 ... G7 of 7.7 and the five 8.10 kernels marked (8) are the others"; 8.10: "at exactly 8, with G7 (7.7) and the five kernels below marked (8)" |
| 1.3 row 14.5 "Two DEPARTURES" | line 184 | applied | "Three DEPARTURES: ... (DEPARTURE-1) ... (DEPARTURE-4) ... (DEPARTURE-5)" |
| W1 row and P10 amendment lists | lines 3221, 4047 | applied | both read "design 10.3 / 14.5 / 14.6 / 16.2 / 16.7 amendments (DEPARTURE-1, -2, -4, -5, -6; ...)" -- the trailing clause now names DEPARTURE-3 and -7 (see REPAIR-1) |
| unescaped pipes in backtick spans | 12 rows | applied | `probes/table-check.mjs` (GFM rules: fenced blocks skipped, backslash-pipe is text, a pipe inside backticks splits): 47 tables, 0 bad rows |

Not-actually-applied list from the verifier: empty; nothing to do.

## 2. Consistency observations

| Item | Action | Edit |
| --- | --- | --- |
| D23 vs design 14.3 lines 3997-3999 (stride as a uniform, kernels on the stride-3 column) not in 1.5 | EDITED (REPAIR-1) | DEPARTURE-7 row added to 1.5 (design text quoted, plan mechanism, why); 1.3 row 14.3 now "Two DEPARTURES ... (DEPARTURE-3) ... (DEPARTURE-7)"; D23 row points at the departure; 7.3 paragraph states the outcome holds and the mechanism is declared; W1 row and P10: "DEPARTURE-3 and -7 amend 14.3 in the L1 PR" (2 occurrences). The verifier called this an owner call; the plan's own rule ("1.5 ... all of them"; "Everything else ... is honoured as written") makes the undeclared mechanism a defect, so it was declared. The owner can downgrade it to a D23 clause by deleting the row and the six references (grep DEPARTURE-7: 6 hits). |
| D24 "measured 4-60x faster" loose bracket | EDITED (REPAIR-2) | D24 now carries the 7.7 numbers (8.4 / 32.6 / 125.3 ms vs 1.8 / 2.0 / 2.9 ms = 4.7x / 16x / 43x; far field 2.3-2.8 ms vs 1.0 ms) and the vec4f figures (9x / 18x / 70x). 7.3 gained the clustered vec4f number "2.01 -> 1.78 ms" so the 18x has an in-document source; its evidence is the verifier's re-run recorded in PERF-verdicts.md line 17 ("sorted + vec4 0.92 / 1.78 / 1.79 ms"). |
| "DispatchPlanner" label in 5.2 heading and 1.3 row 10.6 | EDITED (REPAIR-3) | heading: "5.2 Dispatch planning (`plan1d` / `plan2d` / `planGridStride` / `planIndirect`): ..."; 1.3 row 10.6: "5.2 (`plan1d` / `plan2d` / `planGridStride`)". No other occurrence of the label remains (grep). |
| 9.5 sketch passes `GPUAdapter \| null` into `GPUAdapter \| undefined` | EDITED (REPAIR-4) | `adapter: probe.adapter ?? undefined` with a comment; `ProbeResult.ok` is a plain `boolean` (line ~366), so no narrowing is available. `GpuContextOptions.adapter` is declared `GPUAdapter \| undefined` (line ~293), so the expression is valid under `exactOptionalPropertyTypes`, which the strict-consumer compile of 11.3 enables. |
| Low-degree tier `[midEnd, n)` vs design 10.1 `[midEnd, lowEnd)` | no edit | explicit in the plan (6 row 3 cites views.ts 568-604; design 10.6 names the zero-degree segment); recorded in the Review log paragraph. |
| Section 15 lacked pnpm.io/workspaces and actions/reference/limits | already applied by the checker | lines ~4198 and ~4206 carry both. |
| Verdict-file summary lines disagree with their per-finding verdicts | no edit (review artefacts, not the plan) | the plan's Review log table follows the per-finding verdicts; noted in the Review log paragraph. Not edited in the verdict files either -- they are the reviewers' record. |

## 3. Review log

- New paragraph "Fix verification and repair" after the reconciliation
  paragraph: 149 / 149 applied, 0 wrong rejections, 4 regressions + 1
  pre-existing gap repaired by the checker, 6 observations to the repair pass
  (4 acted on, 2 no-edit), departure count now seven, the two no-edit items
  named.
- Mapping table gained one row per edit: CHECK-R1..R4, CHECK-C1, REPAIR-1..4.
- The lens table totals (123 / 94 / 27 / 2 / 28; 149 applied, 0 rejected)
  are unchanged: no finding was added or removed.

## 4. Checks after the edits

- Non-ASCII characters: 0 (`grep -P '[^\x00-\x7F]'`).
- TBD / TODO / FIXME / placeholder / XXX / lorem: 0.
- GFM table column counts: 47 tables, 0 bad rows (`probes/table-check.mjs`).
- DEPARTURE ids referenced: 1-7, every one has a row in 1.5.
- Stale phrases: "4-60x", "one kernel that", "DispatchPlanner" -- 0 hits outside the Review log rows that describe their removal.
- Lines: 4,439 (was 4,409 at hand-over).

## 5. Files

- Edited: /home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md
- Added: /home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/review/probes/table-check.mjs (the GFM width check)
- This report: /home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/review/repair-report.md
