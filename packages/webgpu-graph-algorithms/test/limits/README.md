# test/limits -- the node-limits project (P4+)

Tests that need limits or time above lavapipe's run here (spec 11.1), selected by
`pnpm exec vitest run --project=node-limits` on the GPU lane only; the default lane
never selects this project. No test file exists before P4 (spec 13 rule (b): P2
adds only what P3 needs). Planned files and the gate that lands each:

- `binding-2gib.test.ts` (G4): a real 2 GiB `maxStorageBufferBindingSize` request
  succeeds on the RTX 4070 SUPER and a binding above 128 MiB is created and read.
- `windowed-200mb.test.ts` (G4): a 200 MB per-array upload is bound windowed at the
  default limits and the windowed `degree` equals `outDegree()`.
- `dispatch-2d-100m.test.ts` (G4): a real 2D dispatch on 100M items (above
  16,776,960) through `linear_id`.
- `oom-scope.test.ts` (G4): the "out-of-memory" error scope on a deliberately
  oversized allocation yields `E_OUT_OF_MEMORY` with `requested` / `resident`.
- `vendor-features.test.ts` (G4): the NVIDIA adapter's feature and limit
  assertions (`subgroups` 32 / 32, `timestamp-query`, the raised limits).
- `layout-1m.test.ts` (G4): the 262k and 1M ForceAtlas2 fixtures of spec 11.4 --
  the one-iteration and unbiasedness checks at 1M.
- `apsp-bound.test.ts` (G9): APSP exact / weighted inside the binding-size bound
  of spec 8.7 and `E_TOO_LARGE` above it.

Every file follows `test/setup/gpu.ts` (`requireGpu`, a fresh adapter per device)
with `GRAPHTY_GPU_REQUIRE=nvidia`; a wrong result is never a skip.
