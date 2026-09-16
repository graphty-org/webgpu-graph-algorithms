# @graphty/webgpu-graph-algorithms

WebGPU-accelerated graph algorithms and layouts over the `@graphty/graph-format` snapshot, for Node
(Dawn, through the `webgpu` npm package) and browsers (Chromium). One code base, three entry points:

| Entry                                      | Import        | What it gives you                                                                                                                                                                                                                                       |
| ------------------------------------------ | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `@graphty/webgpu-graph-algorithms`         | the core      | `createForceAtlas2`, `createAccelerator`, `GpuContext`, `degree`, `seedPositions`, `WebGpuGraphError`, `isSoftwareAdapter`, the constants (`EXACT_MAX_NODES`, `FA2_DEFAULTS`, `LAYOUT_TUNING_DEFAULTS`, ...) and the option / stats / accelerator types |
| `@graphty/webgpu-graph-algorithms/node`    | Node only     | `createNodeGpuContext`, `probeNodeWebGpu`, `createNodeGpu` (Dawn), `dawnFlags`                                                                                                                                                                          |
| `@graphty/webgpu-graph-algorithms/browser` | browsers only | `probeBrowserWebGpu`, `requestGpuContext`                                                                                                                                                                                                               |

**Status: phase P3 (ForceAtlas2, exact tier).** The GPU ForceAtlas2 is usable from Node (`run()`) and from a
browser frame loop (`step()` once per frame) up to `exactMaxNodes` = 32768 nodes with the default
`repulsion: "auto"`, and at any size with `repulsion: "exact"` (all pairs, O(n^2) per iteration: 18.971 ms per
iteration at 100k nodes / 1M edges on an RTX 4070 SUPER). The grid tier for 10^5-10^6 nodes (P4),
Fruchterman-Reingold (P5) and the algorithms (P7+) follow the phase plan of `design/webgpu-acceleration-plan.md`
section 13 and the interface contract `docs/superpowers/plans/2026-09-14-webgpu-p0-p3-interfaces.md`; the gate
record of this phase is `docs/decisions/G3.md`. There is no CPU fallback anywhere in this package: when no
adapter or device exists it throws `WebGpuGraphError`.

## Install

```bash
npm install @graphty/webgpu-graph-algorithms @graphty/graph-format
# Node only: Dawn is an OPTIONAL peer dependency (browser consumers never install it)
npm install webgpu@0.4.0
```

`webgpu@0.4.0` is the last Linux binary linking against glibc <= 2.34 (Ubuntu 22.04 ships 2.35;
`webgpu@0.6.x` needs glibc 2.38). The peer range `>=0.4.0 <1.0.0` admits the newer builds on a newer
glibc; a Node consumer that forgets the package gets `E_NO_WEBGPU` with the message
"install the optional peer dependency webgpu@0.4.0". `@graphty/algorithms` and `@graphty/layout` are
optional peer dependencies too: the accelerator types mirror their interfaces (structurally until the
package moves into the monorepo), so a consumer that type-checks against this package installs them; a
consumer that never touches the accelerator types does not need them.

## ForceAtlas2 from Node

```ts
import { fromEdgeArrays } from "@graphty/graph-format";
import { createForceAtlas2 } from "@graphty/webgpu-graph-algorithms";
import { createNodeGpuContext } from "@graphty/webgpu-graph-algorithms/node";

const ctx = await createNodeGpuContext(); // Dawn; { adapter: "llvmpipe" } selects Mesa's software adapter
const snapshot = fromEdgeArrays({ directed: false, nodeCount, src, dst, weights }); // undirected: pass toUndirected().snapshot otherwise

// The owner's stride-3 array: x, y, z per node in scene units. NaN rows are seeded by the CPU port's LCG
// (seed 42 here) inside [-1, 1) x scale + center; finite rows are kept as the starting layout.
const positions = new Float32Array(3 * snapshot.nodeCount).fill(Number.NaN);

const sim = createForceAtlas2(ctx, { seed: 42, dim: 2, maxIter: 200, gravity: 1, scalingRatio: 2 });
sim.load(snapshot, positions); // uploads the CSR core, seeds the NaN rows, compiles the kernels
const stats = await sim.run({ batch: 8 }); // 8 iterations per submit until settled or maxIter
console.log(sim.iterationsDone, sim.settled, stats.speed, stats.rmsRadius, stats.layoutRadius);
// positions now holds the layout; stats.trace holds the last batch's per-iteration controller values

sim.dispose(); // destroys the simulation's buffers
ctx.release(snapshot); // destroys the snapshot's buffers (nothing is freed by GC)
ctx.dispose(); // destroys the device and lets the process exit
```

Sizes above `exactMaxNodes` need `repulsion: "exact"` until the grid tier lands (P4); with the default
`"auto"` the simulation rejects `load()` with `E_UNSUPPORTED { feature: "repulsion.grid" }`. The same run
from the command line, with a verification of the result:

```bash
pnpm exec tsx benchmarks/layout-run.ts --nodes 100000 --edges 1000000 --iterations 100 --batch 8
```

## ForceAtlas2 in a browser frame loop

The simulation never blocks a frame: `step()` submits one batch and returns a promise that resolves when
that batch's positions have been copied into your array; while `maxInFlight` batches are in flight the call
coalesces (nothing is queued, the oldest pending batch's promise is returned). Attach `.catch` once per
distinct promise and draw whatever the array holds -- it lags the GPU by at most one batch.

```ts
import { createAccelerator } from "@graphty/webgpu-graph-algorithms";
import { probeBrowserWebGpu, requestGpuContext } from "@graphty/webgpu-graph-algorithms/browser";

const probe = await probeBrowserWebGpu(); // never throws: { ok, code, reason, adapter, summary }
if (!probe.ok) {
    throw new Error(`${probe.code}: ${probe.reason ?? ""}`); // E_NO_WEBGPU, E_NO_ADAPTER or E_SOFTWARE_ONLY
}
const ctx = await requestGpuContext({ adapter: probe.adapter ?? undefined });
const acc = createAccelerator(ctx, { layout: { exactMaxNodes: 32768 } }); // the tuning every simulation inherits
const sim = acc.forceAtlas2({ seed: 1, iterationsPerStep: 1, maxInFlight: 2 });
sim.load(snapshot, positions); // positions: your stride-3 Float32Array; NaN rows are seeded

let last: Promise<void> | null = null;
function frame(): void {
    if (!sim.settled) {
        const p = sim.step(); // never awaited
        if (p !== last) {
            last = p;
            p.catch((error: unknown) => {
                console.error(error); // E_DEVICE_LOST, E_VALIDATION, ...; the simulation is disposed on device loss
            });
        }
    }
    draw(positions);
    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// interaction
sim.setPosition(i, x, y, 0); // a drag: written to the device now, visible in the next readback, reheats
sim.setFixed(mask); // pins: a graph-format NodeMask (ceil(n / 32) words); an unpin reheats, a pin does not
sim.reheat(); // "play again" after the layout settled (nothing is reset but the counters)
await sim.flush(); // pause: resolves when every submitted batch has landed in `positions`
sim.dispose(); // stop; then ctx.release(snapshot) and ctx.dispose() when the graph goes away
```

`stats` and `settled` describe the last completed batch. Topology changes go through `load(next,
remappedPositions)` on the same simulation: in-flight readbacks of the old graph are discarded, new (NaN)
rows are seeded inside the current bounding box, the fixed mask and the position overrides are cleared (the
caller re-issues its pins with a mask over the new index space), and the simulation reheats.

## Options

`createForceAtlas2(ctx, options)` and `accelerator.forceAtlas2(options)` take `ForceAtlas2Options` (the same
names and defaults as the CPU port in `@graphty/layout`) plus the GPU tuning:

| Option                            | Default          | Meaning                                                                                                                                                                   |
| --------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dim`                             | `2`              | `2` or `3`; in 2D every readback writes `z = center[2]` whatever was uploaded                                                                                             |
| `scale`, `center`                 | `1`, `[0, 0, 0]` | scene units = layout units x `scale` + `center`                                                                                                                           |
| `seed`                            | `null`           | the LCG seed of the NaN rows (the CPU port's generator, bit for bit); `0` / `null` draws a random seed                                                                    |
| `maxIter`                         | `100`            | `run()` stops and `settled` turns true after this many iterations since `load()` / `reheat()`                                                                             |
| `jitterTolerance`                 | `1`              | the speed controller's tolerance                                                                                                                                          |
| `scalingRatio`                    | `2`              | the repulsion constant                                                                                                                                                    |
| `gravity`                         | `1`              | toward the centroid (`compat: "paper"`) or the origin (`compat: "networkx"`)                                                                                              |
| `strongGravity`                   | `false`          | gravity proportional to the distance                                                                                                                                      |
| `distributedAction`               | `false`          | attraction divided by the source's mass                                                                                                                                   |
| `linlog`                          | `false`          | logarithmic attraction                                                                                                                                                    |
| `nodeMass`                        | `null`           | `null`: the node column with the role `mass` when present, else `outDegree + 1`; a `Float32Array` of length n; a numeric node column name (a `Record` is `E_UNSUPPORTED`) |
| `nodeSize`                        | `null`           | `E_UNSUPPORTED` when non-null (no overlap prevention)                                                                                                                     |
| `weight`                          | unset            | `true`: the snapshot's arc weights; an edge column name: that numeric column; unset / `false` / `null`: every edge weighs 1                                               |
| `dissuadeHubs`                    | `false`          | accepted and ignored                                                                                                                                                      |
| `settleThreshold`, `settleWindow` | `0.001`, `10`    | settled when the mean displacement stayed below the threshold for `settleWindow` iterations                                                                               |
| `iterationsPerStep`               | `1`              | iterations per `step()` (the Node `run()` uses its own `batch`)                                                                                                           |
| `maxInFlight`                     | `2`              | batches in flight before `step()` coalesces; `1` for the strictest freshness                                                                                              |

GPU tuning (`GpuLayoutTuning`; also the `layout` field of `createAccelerator`'s options, inherited by every
simulation the accelerator creates):

| Option                                              | Default                 | Meaning                                                                                                                                    |
| --------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `repulsion`                                         | `"auto"`                | `"exact"` at any n; `"auto"` = exact iff n <= `exactMaxNodes`; `"grid"` is `E_UNSUPPORTED` until P4                                        |
| `exactMaxNodes`                                     | `32768`                 | the crossover, measured on the RTX 4070 SUPER (`docs/decisions/G3.md`); pass your own for another GPU (P4's `calibrateLayout` measures it) |
| `deterministic`                                     | `true`                  | fixed summation order (the exact tier is always deterministic)                                                                             |
| `compat`                                            | `"paper"`               | `"networkx"` reproduces NetworkX 3.4's `forceatlas2_layout` (gravity toward the origin, its accumulated swing / traction)                  |
| `nearMax`, `gridMax2D`, `gridMax3D`, `extentFactor` | `64`, `512`, `128`, `6` | stored for the grid tier (P4)                                                                                                              |

Every range error is `E_INVALID_ARGUMENT` (`gravity < 0`, `scalingRatio <= 0`, `jitterTolerance <= 0`,
`maxIter < 1`, `settleWindow < 1`, `maxInFlight < 1`, `iterationsPerStep < 1`, `dim` not 2 or 3,
`scale <= 0`). `setParams(patch)` changes the options of a live simulation; a force-law change (`linlog`,
`strongGravity`, `distributedAction`) recompiles and resets the speed controller, a numeric tweak only changes
the next batch's parameters; `dim` and `maxInFlight` cannot change after creation.

## Stats

`sim.stats` (`ForceAtlas2Stats`) after every completed batch: `iteration`, `swing`, `traction`, `speed`,
`speedEfficiency`, `meanDisplacement`, `rmsRadius`, `layoutRadius` (max |p - centroid|), `centroid`,
`repulsionTier` (`"exact"`), `msPerIteration` (the GPU time per iteration when `timestamp-query` was granted,
else the batch's wall time divided by its iteration count), the grid fields (`null` on the exact tier) and
`trace`: one `{ swing, traction, speed, speedEfficiency, meanDisplacement, settledCount }` record per
iteration of the last batch.

## Acquisition

### Node

```ts
import { hasErrorCode } from "@graphty/webgpu-graph-algorithms";
import { createNodeGpu, createNodeGpuContext, probeNodeWebGpu } from "@graphty/webgpu-graph-algorithms/node";

const probe = await probeNodeWebGpu(); // never throws; probe.summary has vendor / architecture / software / limits
console.log(probe.ok, probe.code, probe.summary?.software);
try {
    const ctx = await createNodeGpuContext({ rejectSoftware: true }); // Dawn + GpuContext.create
    console.log(ctx.caps.vendor, ctx.caps.architecture, ctx.caps.software ? "software" : "hardware");
    ctx.dispose();
} catch (error) {
    if (hasErrorCode(error, "E_NO_WEBGPU") || hasErrorCode(error, "E_SOFTWARE_ONLY")) {
        console.error((error as Error).message);
        process.exit(1);
    }
    throw error;
}
const handle = await createNodeGpu({ software: false }); // the bare Dawn handle: import("webgpu"), globals, create(flags)
handle.dispose();
```

Options of `createNodeGpu` / `createNodeGpuContext`: `adapter` (Dawn `adapter=<substring>`, e.g. `"llvmpipe"`
or `"4070"`), `backend` (`"vulkan"` | `"null"` | ...), `dawnFeatures` (Dawn toggles), `software` (shorthand
for `adapter=llvmpipe`, Linux / Mesa specific), `installGlobals` (default `true`: `GPUBufferUsage` and friends
on `globalThis`), plus the `GpuContext.create` options (`powerPreference`, `rejectSoftware`, `limits`
(`"raise"` by default), `optionalFeatures` (`["subgroups", "timestamp-query"]` by default), `label`,
`onError`).

On a machine where the NVIDIA Vulkan driver cannot find `libEGL.so.1` (a container without `libegl1`),
Dawn silently lists only llvmpipe; see `docs/HEADLESS_GPU_REPORT.md` appendix D for the `LD_LIBRARY_PATH`
recipe.

### Browser

```ts
import { probeBrowserWebGpu, requestGpuContext } from "@graphty/webgpu-graph-algorithms/browser";

const probe = await probeBrowserWebGpu(); // never throws: { ok, code, reason, adapter, summary }
if (probe.ok) {
    const ctx = await requestGpuContext({ adapter: probe.adapter ?? undefined });
    // ... createAccelerator(ctx), createForceAtlas2(ctx, ...), degree(ctx, snapshot)
    ctx.dispose();
}
```

Chromium only until Firefox / WebKit ship `subgroups` and `timestamp-query` on Linux CI. The `./node`
subpath is never reachable from browser code: nothing in the core or the browser entry imports it, and
`sideEffects: false` lets bundlers drop what you do not use.

## The `degree` diagnostic

`degree(ctx, snapshot)` is the walking-skeleton kernel kept public as a diagnostic: it runs the row-walking gather with the
package's dummy-binding pattern and returns a `Uint32Array` equal to `snapshot.outDegree()`. If it disagrees, nothing else
will work; if it agrees, the device, the upload path, the pipeline cache and the readback ring all do.

```ts
import { fromEdgeArrays } from "@graphty/graph-format";
import { degree } from "@graphty/webgpu-graph-algorithms";
import { createNodeGpuContext } from "@graphty/webgpu-graph-algorithms/node";

const ctx = await createNodeGpuContext(); // Dawn through the webgpu package; { adapter: "llvmpipe" } selects the software adapter
const s = fromEdgeArrays({ directed: false, nodeCount: 3, src: new Uint32Array([0, 1]), dst: new Uint32Array([1, 2]) });
const out = await degree(ctx, s); // Uint32Array [1, 2, 1] === s.outDegree()
ctx.release(s); // destroys the snapshot's buffers (nothing is freed by GC)
ctx.dispose(); // destroys the device and lets the process exit
```

In a browser, `import { requestGpuContext } from "@graphty/webgpu-graph-algorithms/browser"` and `const ctx = await
requestGpuContext();` replace the first import and line; the rest is identical.

## Errors

Every condition the package detects itself is a `WebGpuGraphError` with a stable `code`
(`E_NO_WEBGPU`, `E_NO_ADAPTER`, `E_NO_DEVICE`, `E_SOFTWARE_ONLY`, `E_DEVICE_LOST`, `E_DISPOSED`,
`E_VALIDATION`, `E_SHADER_COMPILE`, `E_OUT_OF_MEMORY`, `E_TOO_LARGE`, `E_UNSUPPORTED`,
`E_INVALID_ARGUMENT`, `E_SNAPSHOT`, `E_RELEASED`, `E_NOT_LOADED`, `E_ABORTED`) and frozen `details`.
`isWebGpuGraphError(x)` and `hasErrorCode(x, code)` are structural brand checks, so they survive two copies
of the package. The graph-format codes `E_GPU_INELIGIBLE`, `E_UNKNOWN_NODE`, `E_UNKNOWN_COLUMN` and
`E_COLUMN_LENGTH` pass through unchanged (`PASSTHROUGH_FORMAT_CODES`). A simulation whose snapshot was
released rejects its next `step()` with `E_RELEASED`; a lost device disposes every simulation and rejects
every pending `step()` with `E_DEVICE_LOST` (create a new context from a fresh adapter and `load()` again).

## Benchmarks

```bash
pnpm run bench                                                 # every group; appends to benchmarks/out/<runner-class>.json
pnpm exec tsx benchmarks/run.ts upload roundtrip layout-exact  # selected groups; --no-save, --runs N, --allow-software
pnpm exec tsx benchmarks/layout-run.ts --nodes 100000 --edges 1000000   # the end-to-end layout driver (exit 1 on a bad result)
pnpm run gpu:report > gpu-report.json                          # the adapter report with a 10 s nvidia-smi sample
pnpm run bench:compare                                         # the last out session vs benchmarks/results/<runner-class>.json (> 3x fails)
```

The runner class is `<vendor>-<architecture>-driver<major>` (`scripts/runner-class.js`; `GRAPHTY_RUNNER_CLASS` overrides it,
which the GPU lane sets to `gpu-linux-t4`). Software adapters never time anything: `pnpm run bench` prints
`software adapter: nothing timed` on lavapipe unless `--allow-software` is given, and a software session is never a
baseline. The checked-in baselines under `benchmarks/results/` are written by the owner on the dev box (the RTX 4070 SUPER,
runner class `nvidia-lovelace-driver580` -- Dawn spells the architecture `lovelace`) and, for the GPU lane, from the
lane's own artifact; `bench:compare` skips when the card was not quiet during the report's sample. Groups: `upload` (T-1),
`roundtrip` (T-2, T-3), `layout-exact` (T-4 and the Node side of T-5: `step(1)` on the exact ladder 1k / 4k / 8k / 16k /
32k / 65k and at 10k, two rows per rung -- the wall time of one iteration with its readback, and the GPU time per iteration
the profiler reports; every rung starts with an untimed clock warm-up burst, because NVIDIA's power management leaves the
SM clock at its idle 210 MHz under sparse sub-millisecond dispatches and the kernels then measure 4-15x slower). The
Chromium number of T-5 comes from the `bench`-tagged browser test (`GRAPHTY_BROWSER_GPU=nvidia node
scripts/run-browser-project.js`), which appends its session through the Vitest commands bridge. `exactMaxNodes` is
re-fixed from the ladder by the rule of plan section 7.8 (the largest rung under 4 ms per iteration, rounded down to a
power of two; `benchmarks/layout-exact.bench.ts` `exactMaxNodesFromLadder`).

## Performance

Regenerated from `benchmarks/results/nvidia-lovelace-driver580.json` (the last session) by the procedure
recorded in `docs/decisions/G3.md` appendix A; the targets are the T-table of plan section 10.4. A missed target is re-fixed by a
recorded owner decision in `docs/decisions/G<n>.md`, never relaxed silently.

Measured on nvidia-lovelace-driver580 (NVIDIA: 580.173.02 580.173.2.0), session 2026-09-16T02:07:45.933Z, medians of 5 runs; Chromium: nvidia / lovelace (nvidia-lovelace-driver0, the description is redacted by Chromium), session 2026-09-16T02:18:11.896Z.

| Id  | What                                                                                         | Target              | Measured             |
| --- | -------------------------------------------------------------------------------------------- | ------------------- | -------------------- |
| T-1 | Upload of the 100k / 1M weighted hot prefix (16.4 MB); 1M / 10M (164 MB)                     | <= 10 ms; <= 100 ms | 6.032 ms; 127.531 ms |
| T-2 | `degree` + 400 KB readback at 100k (core resident), Node                                     | <= 2 ms             | 0.878 ms             |
| T-3 | Empty submit + 4-byte `readU32` round trip, Dawn                                             | <= 0.1 ms           | 0.170 ms             |
| T-4 | ForceAtlas2 exact tier, GPU time per iteration (profiler) at 10k; at 16k                     | <= 1 ms; <= 2 ms    | 0.586 ms; 1.052 ms   |
| T-5 | ForceAtlas2 per-frame cost, `step(1)` + the 12n readback at 10k, Chromium (Node in brackets) | <= 6 ms             | 2.400 ms (0.738 ms)  |

Two rows miss their target in this session: the 1M / 10M upload (127.5 ms against 100 ms, the open owner decision of
`docs/decisions/G1.md` section 7) and the empty-submit round trip (0.170 ms against 0.1 ms: the row is measured after the
`upload` group, whose CPU-heavy setup lets the SM clock fall to its idle state; the same row measures 0.041-0.074 ms at
the working clock -- finding G3-F2 of `docs/decisions/G3.md` section 10).

The exact curve (the `layout-exact` group: 2D, E = 10n, seeded G(n, m), one simulation per rung; ms / iteration from the profiler):

| n     | ms / iteration | step(1) wall (ms) | pairs / s |
| ----- | -------------- | ----------------- | --------- |
| 1024  | 0.096          | 0.228             | 1.09e+10  |
| 4096  | 0.255          | 0.415             | 6.58e+10  |
| 8192  | 0.478          | 0.630             | 1.40e+11  |
| 10000 | 0.586          | 0.738             | 1.71e+11  |
| 16384 | 1.052          | 1.226             | 2.55e+11  |
| 32768 | 2.560          | 2.787             | 4.19e+11  |
| 65536 | 8.405          | 8.862             | 5.11e+11  |

The end-to-end run of `benchmarks/layout-run.ts --nodes 100000 --edges 1000000` (the exact tier at 100k, 100
iterations, batches of 8) takes 18.971 ms per iteration on the same card, uploads and readbacks included (16.975 ms of
GPU time per iteration in the last batch).

## Development

```bash
cd packages && pnpm install                       # the pnpm workspace root
cd webgpu-graph-algorithms
pnpm run build:all                                # tsc + the vite bundle + the d.ts shims
pnpm run lint                                     # eslint + tsc --noEmit + the strict-consumer compile
pnpm exec vitest run --project=node               # the node suite on the default adapter
pnpm run coverage                                 # the node suite with the 80 / 80 / 75 / 80 thresholds
node scripts/run-browser-project.js               # the browser smoke suite (SwiftShader by default)
node scripts/gpu-report.js                        # the adapter report and the policy verdict (after build)
cd .. && pnpm exec knip                           # unused files / exports / dependencies
```

Environment variables of the test harness (plan section 12.2):

| Variable                                  | Default lane (GitHub, software) | GPU lane (NVIDIA T4) | Local (dev box)                                                     |
| ----------------------------------------- | ------------------------------- | -------------------- | ------------------------------------------------------------------- |
| `GRAPHTY_GPU_ADAPTER`                     | `llvmpipe`                      | unset                | unset (NVIDIA) or `llvmpipe` to mirror CI                           |
| `GRAPHTY_GPU_REQUIRE`                     | `any`                           | `nvidia`             | unset (skip with a printed reason) or `hardware`                    |
| `GRAPHTY_BROWSER_GPU`                     | `swiftshader`                   | `nvidia`             | `nvidia` (needs the libEGL tree)                                    |
| `GRAPHTY_GPU_NO_SUBGROUPS`                | `1` in a second pass            | `1` in a second pass | unset                                                               |
| `GRAPHTY_GPU_INSPECT`                     | unset                           | unset                | `1` to enable `sim.inspect(name)` / `debugRunStages` in test builds |
| `GRAPHTY_NOISE_FLOOR_WRITE`               | unset                           | unset                | `1` to (re)write this adapter's noise fixtures                      |
| `GRAPHTY_DAWN_FEATURES`                   | unset                           | unset                | optional Dawn toggles                                               |
| `GRAPHTY_EGL_LIB_DIR` / `LD_LIBRARY_PATH` | --                              | unset                | the extracted libEGL tree                                           |
| `VK_DRIVER_FILES`                         | the lavapipe ICD                | unset                | unset                                                               |
| `XDG_RUNTIME_DIR`                         | `/tmp`                          | `/tmp`               | `/tmp`                                                              |

`GRAPHTY_GPU_REQUIRE` is the one policy switch: unset skips tests that need an adapter (with the reason
printed), `any` fails when no adapter exists, `hardware` additionally rejects lavapipe / SwiftShader, a
vendor name (`nvidia`) additionally requires that vendor. A wrong result is never a skip. Every f32 tolerance
of the layout tests is derived from `benchmarks/results/noise-floor.json` (the measured summation noise across
the subgroup twins, lavapipe, SwiftShader and NVIDIA; a tolerance is at most 10x its floor), every kernel ships
with sabotage mutations that must fail its test by 10x that tolerance, and every kernel result is compared
twice for bitwise determinism before it is compared to an oracle (plan section 11.9).

## License

MIT
