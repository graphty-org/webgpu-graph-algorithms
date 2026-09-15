# @graphty/webgpu-graph-algorithms

WebGPU-accelerated graph algorithms and layouts over the `@graphty/graph-format` snapshot, for Node
(Dawn, through the `webgpu` npm package) and browsers (Chromium). One code base, three entry points:

| Entry                                      | Import        | What it gives you                                                                                                                                     |
| ------------------------------------------ | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `@graphty/webgpu-graph-algorithms`         | the core      | `WebGpuGraphError`, `isSoftwareAdapter`, the shared constants; from P1 `GpuContext` and `degree`; from P3 `createForceAtlas2` and `createAccelerator` |
| `@graphty/webgpu-graph-algorithms/node`    | Node only     | `createNodeGpu` (Dawn), `dawnFlags`; from P1 `createNodeGpuContext`, `probeNodeWebGpu`                                                                |
| `@graphty/webgpu-graph-algorithms/browser` | browsers only | from P1 `probeBrowserWebGpu`, `requestGpuContext`                                                                                                     |

**Status: phase P0 (package skeleton).** The device acquisition helpers, the error class and the
constants are published; the `degree` diagnostic (P1), the exact-tier ForceAtlas2 (P3) and the grid tier
(P4) follow the phase plan of `design/webgpu-acceleration-plan.md` section 13 and the interface contract
`docs/superpowers/plans/2026-09-14-webgpu-p0-p3-interfaces.md`. There is no CPU fallback anywhere in this
package: when no adapter or device exists it throws `WebGpuGraphError`.

## Install

```bash
npm install @graphty/webgpu-graph-algorithms @graphty/graph-format
# Node only: Dawn is an OPTIONAL peer dependency (browser consumers never install it)
npm install webgpu@0.4.0
```

`webgpu@0.4.0` is the last Linux binary linking against glibc <= 2.34 (Ubuntu 22.04 ships 2.35;
`webgpu@0.6.x` needs glibc 2.38). The peer range `>=0.4.0 <1.0.0` admits the newer builds on a newer
glibc; a Node consumer that forgets the package gets `E_NO_WEBGPU` with the message
"install the optional peer dependency webgpu@0.4.0".

## Node

```ts
import { hasErrorCode, isSoftwareAdapter } from "@graphty/webgpu-graph-algorithms";
import { createNodeGpu } from "@graphty/webgpu-graph-algorithms/node";

let handle;
try {
    // import("webgpu"), install dawn.globals, dawn.create(flags); `software: true` selects Mesa llvmpipe
    handle = await createNodeGpu({ software: false });
} catch (error) {
    if (hasErrorCode(error, "E_NO_WEBGPU")) {
        console.error("Dawn is not available:", (error as Error).message);
        process.exit(1);
    }
    throw error;
}
const adapter = await handle.gpu.requestAdapter();
if (adapter === null) {
    throw new Error("no WebGPU adapter");
}
console.log(adapter.info.vendor, adapter.info.architecture, isSoftwareAdapter(adapter.info) ? "software" : "hardware");
handle.dispose(); // drops the GPU reference so the process can exit
```

From P1, `createNodeGpuContext(options)` returns a `GpuContext` (probe / create / from / release /
dispose) and `degree(ctx, snapshot)` is the walking-skeleton diagnostic: it uploads the snapshot's CSR hot
prefix, runs the `degree` kernel and returns a `Uint32Array` equal to `snapshot.outDegree()` -- the first
thing to run when a GPU result looks wrong.

Options of `createNodeGpu`: `adapter` (Dawn `adapter=<substring>`, e.g. `"llvmpipe"` or `"4070"`),
`backend` (`"vulkan"`, `"d3d12"`, `"d3d11"`, `"metal"`, `"opengl"`, `"opengles"` or `"null"`), `dawnFeatures` (Dawn toggles), `software` (shorthand for
`adapter=llvmpipe`, Linux / Mesa specific), `installGlobals` (default `true`: `GPUBufferUsage` and friends
on `globalThis`).

On a machine where the NVIDIA Vulkan driver cannot find `libEGL.so.1` (a container without `libegl1`),
Dawn silently lists only llvmpipe; see `docs/HEADLESS_GPU_REPORT.md` appendix D for the `LD_LIBRARY_PATH`
recipe.

## Browser

From P1:

```ts
import { probeBrowserWebGpu, requestGpuContext } from "@graphty/webgpu-graph-algorithms/browser";

const probe = await probeBrowserWebGpu(); // never throws: { ok, code, reason, adapter, summary }
if (probe.ok) {
    const ctx = await requestGpuContext({ adapter: probe.adapter ?? undefined });
    // degree(ctx, snapshot) from P1; createForceAtlas2(ctx, snapshot, positions, options) from P3
    ctx.dispose();
}
```

Chromium only until Firefox / WebKit ship `subgroups` and `timestamp-query` on Linux CI. The `./node`
subpath is never reachable from browser code: nothing in the core or the browser entry imports it, and
`sideEffects: false` lets bundlers drop what you do not use.

## Errors

Every condition the package detects itself is a `WebGpuGraphError` with a stable `code`
(`E_NO_WEBGPU`, `E_NO_ADAPTER`, `E_NO_DEVICE`, `E_SOFTWARE_ONLY`, `E_DEVICE_LOST`, `E_DISPOSED`,
`E_VALIDATION`, `E_SHADER_COMPILE`, `E_OUT_OF_MEMORY`, `E_TOO_LARGE`, `E_UNSUPPORTED`,
`E_INVALID_ARGUMENT`, `E_SNAPSHOT`, `E_RELEASED`, `E_NOT_LOADED`, `E_ABORTED`) and frozen `details`.
`isWebGpuGraphError(x)` and `hasErrorCode(x, code)` are structural brand checks, so they survive two copies
of the package. The graph-format codes `E_GPU_INELIGIBLE`, `E_UNKNOWN_NODE`, `E_UNKNOWN_COLUMN` and
`E_COLUMN_LENGTH` pass through unchanged (`PASSTHROUGH_FORMAT_CODES`).

## Performance

No baseline yet: the T-table of the plan (section 10.4) is filled from `benchmarks/results/<runner-class>.json`
from P1 on (`pnpm run bench` writes `benchmarks/out/<runner-class>.json`; `pnpm run bench:compare` checks a
run against the checked-in baseline for the same runner class). This section is regenerated from those files.

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

| Variable                                  | Default lane (GitHub, software) | GPU lane (NVIDIA T4) | Local (dev box)                                  |
| ----------------------------------------- | ------------------------------- | -------------------- | ------------------------------------------------ |
| `GRAPHTY_GPU_ADAPTER`                     | `llvmpipe`                      | unset                | unset (NVIDIA) or `llvmpipe` to mirror CI        |
| `GRAPHTY_GPU_REQUIRE`                     | `any`                           | `nvidia`             | unset (skip with a printed reason) or `hardware` |
| `GRAPHTY_BROWSER_GPU`                     | `swiftshader`                   | `nvidia`             | `nvidia` (needs the libEGL tree)                 |
| `GRAPHTY_GPU_NO_SUBGROUPS`                | `1` in a second pass            | `1` in a second pass | unset                                            |
| `GRAPHTY_DAWN_FEATURES`                   | unset                           | unset                | optional Dawn toggles                            |
| `GRAPHTY_EGL_LIB_DIR` / `LD_LIBRARY_PATH` | --                              | unset                | the extracted libEGL tree                        |
| `VK_DRIVER_FILES`                         | the lavapipe ICD                | unset                | unset                                            |
| `XDG_RUNTIME_DIR`                         | `/tmp`                          | `/tmp`               | `/tmp`                                           |

`GRAPHTY_GPU_REQUIRE` is the one policy switch: unset skips tests that need an adapter (with the reason
printed), `any` fails when no adapter exists, `hardware` additionally rejects lavapipe / SwiftShader, a
vendor name (`nvidia`) additionally requires that vendor. A wrong result is never a skip.

## License

MIT
