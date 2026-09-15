# 05 - WebGPU platform facts and the runtime / test architecture for a Node + browser library

Date: 2026-09-14
Scope: what @graphty/webgpu-graph-algorithms must design around because it
runs the same WGSL under Google Dawn in Node (npm package `webgpu`) AND under
browsers, is tested primarily in Node, lightly in headless Chromium, and
needs GPU CI. Every fact below is either (a) measured on the dev box today
with the probe scripts in `tmp/webgpu-plan/probe/`, (b) read from local files
(path cited), or (c) fetched from a URL (cited). Things that could not be
verified are listed in section 12.

Dev box used for all measurements: NVIDIA GeForce RTX 4070 SUPER, driver
580.173.02, Docker container Ubuntu 22.04.5 / glibc 2.35, Node v22.22.1,
Playwright 1.54.1 with Chromium build 1181 (Chromium 139)
(HEADLESS_GPU_REPORT.md lines 25-39).

---

## 1. Summary of the decisions this note recommends

1. The library never touches `navigator`. Every entry point takes a
   `GPUDevice` (or a small `GpuContext` wrapping one). Device acquisition
   lives in two tiny adapters: `@graphty/webgpu-graph-algorithms/browser`
   (uses `navigator.gpu`) and `@graphty/webgpu-graph-algorithms/node` (uses
   `webgpu`'s `create()`), the latter behind an optional peer dependency and
   a dynamic import so browser bundles never see the native module (sec. 8).
2. Pin `webgpu` to `0.4.0` while the dev container is Ubuntu 22.04: 0.5.0 and
   0.6.x require `GLIBC_2.38` and `GLIBCXX_3.4.32` (verified by reading the
   binary's symbol versions and by a failed `require()` on this box, sec. 2.1).
   Upgrade the container (or CI image) to Ubuntu 24.04 before moving to 0.6.x.
3. Node/Dawn is the PRIMARY test target: an in-process Dawn round trip
   (`submit` + `mapAsync`) is 0.03-0.04 ms versus 0.10-0.15 ms in Chromium
   (sec. 7.2), test files are plain vitest files with `pool: "forks"`, and the
   real NVIDIA adapter is selected by putting `libEGL.so.1` on
   `LD_LIBRARY_PATH` (sec. 2.4). Without it Dawn silently uses Mesa lavapipe
   (`llvmpipe`), which is ~350x slower on an n-body kernel but is a correct,
   always-available software adapter for correctness tests on GPU-less
   runners (sec. 2.5).
4. The same test files also run in a vitest browser project (Playwright
   Chromium with the four flags from HEADLESS_GPU_REPORT.md). The browser
   project is "light": a smoke test per algorithm family and the device
   acquisition path, not the full matrix (sec. 9).
5. Design kernels to the core-spec DEFAULT limits (8 storage buffers per
   stage, 128 MiB binding, 256 MiB buffer, 256 invocations, 16 KiB workgroup
   memory, 65535 workgroups per dimension) and raise limits opportunistically
   via `requiredLimits` from `adapter.limits` (sec. 4). Treat `subgroups`,
   `timestamp-query`, `shader-f16` as optional fast paths selected at pipeline
   creation, never as requirements (sec. 5).
6. CI: two runner classes. A default `ubuntu-latest` job runs everything on
   lavapipe (Node) plus SwiftShader (Chromium); a GPU job runs on a
   self-hosted runner labelled e.g. `graphty-gpu` (the cuda-ffi pattern:
   `runs-on: <label>` + `container: { options: "--gpus all" }`) or on a
   GitHub-hosted `gpu-t4-4-core` larger runner if the org plan allows it
   (sec. 10).

---

## 2. Node runtime: Dawn via the npm package `webgpu`

### 2.1 Versions, dates, native ABI requirements

Registry metadata (fetched 2026-09-14 from https://registry.npmjs.org/webgpu):

| version | published  | deps                                   | linux-x64 binary needs (read from the `.node` file)                |
|---------|------------|----------------------------------------|---------------------------------------------------------------------|
| 0.3.8   | 2025-09-25 | @webgpu/types ^0.1.65, debug ^4.4.0    | not checked                                                         |
| 0.3.9   | 2026-03-18 | @webgpu/types ^0.1.69, debug           | not checked                                                         |
| 0.4.0   | 2026-03-27 | @webgpu/types ^0.1.69, debug           | GLIBC_2.34, GLIBCXX_3.4.30 (loads on Ubuntu 22.04)                  |
| 0.5.0   | 2026-08-28 | @webgpu/types ^0.1.72, debug           | not checked (same build window as 0.6.0; assume 0.6.x requirements) |
| 0.6.0   | 2026-08-28 | @webgpu/types ^0.1.72, debug           | not checked                                                         |
| 0.6.1   | 2026-09-12 | @webgpu/types ^0.1.72, debug           | GLIBC_2.38, GLIBCXX_3.4.32, CXXABI_1.3.9 (does NOT load here)       |

`dist-tags.latest` = 0.6.1. No `engines` field in any version. Repo:
https://github.com/dawn-gpu/node-webgpu (it "just publishes dawn.node from
the dawn project", README of the installed 0.4.0 package,
`packages/graph-format/node_modules/webgpu/README.md`).

Verification of the glibc claim: `strings linux-x64.dawn.node | grep GLIBC_`
on the 0.4.0 tarball gives GLIBC_2.29/2.32/2.34 and GLIBCXX_3.4.29/3.4.30;
on the 0.6.1 tarball gives GLIBC_2.33/2.34/2.38 and GLIBCXX_3.4.31/3.4.32.
`require()` of the 0.6.1 binary on this box fails with
"/usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.32' not
found". Ubuntu 22.04 ships glibc 2.35 and libstdc++ up to GLIBCXX_3.4.30
(`ldd --version`, `strings libstdc++.so.6`). node-webgpu issue #15 (opened
2026-04-02, open) reports the same class of problem for Ubuntu 20.04
(https://github.com/dawn-gpu/node-webgpu/issues). Conclusion: `webgpu@0.4.0`
is the last version usable on glibc < 2.38; the staged graph-format package
already pins `"webgpu": "^0.4.0"` (packages/graph-format/package.json line 73).

Packaging differences between 0.4.0 and 0.6.1 that matter for an upgrade:

- 0.4.0 layout: `dist/linux-x64.dawn.node` (index.js resolves
  `${platform}-${arch}.dawn.node`). 0.6.1 layout: `dist/linux-x64/dawn.node`
  plus a `win32-arm64` build. Nothing to do on our side (index.js hides it).
- 0.6.1's index.js monkey-patches `GPUDevice.prototype.createBuffer/destroy`
  to unmap still-mapped buffers on `device.destroy()`, because "dawn.node
  leaves them mapped, so their getMappedRange ArrayBuffers stay attached"
  (0.6.1 index.js, comment block). On 0.4.0 that shim does not exist, so our
  readback helper must always `unmap()` (or `destroy()` the staging buffer)
  itself and never rely on `device.destroy()` to detach mapped ranges.
- 0.6.1 declares `@webgpu/types ^0.1.72`; the monorepo's graph-format already
  uses `@webgpu/types ^0.1.72` (packages/graph-format/package.json line 66),
  while this repo's scaffold has `^0.1.51` (package.json line 64), which
  predates `adapter.info` and `GPUAdapterInfo.subgroupMinSize`. Bump it.

### 2.2 The API surface: `create(options)` and `globals`

Installed types (`node_modules/webgpu/types.d.ts`):

```ts
import '@webgpu/types';
export declare function create(options: string[]): GPU;
export declare const globals: Object;
```

`globals` has 42 keys on 0.4.0 (probe `dawn-probe.mjs`), including the
constant namespaces `GPUBufferUsage`, `GPUMapMode`, `GPUShaderStage` and the
interface constructors (`GPUDevice`, `GPUBuffer`, ...). A library that uses
`GPUBufferUsage.STORAGE` at module scope will throw under Node unless
`Object.assign(globalThis, globals)` ran first, so either (a) the Node entry
point installs the globals before any kernel module is imported (dynamic
import ordering), or (b) the library never reads those namespaces at module
top level and instead reads them lazily inside functions. Recommendation:
both -- the `node` entry installs globals, and library code keeps usage-flag
constants inside functions or uses numeric literals with a comment
(`0x0080 /* STORAGE */`) so the core is import-order independent.

`create(options)` string options, from the installed README and verified by
probe `dawn-select.mjs` (each option is one string in the array):

| option                                   | verified behaviour on this box                                                                                                                     |
|------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `backend=vulkan`                         | selects the Vulkan backend; `null`, `webgpu`, `d3d11`, `d3d12`, `metal`, `opengl`, `opengles` are the documented names. `backend=bogus` throws "unrecognised backend 'bogus'" at `create()`. |
| `backend=null`                           | Dawn's null backend: `requestAdapter()` returns an adapter with empty vendor/architecture and device "null-backend". Useful for validation-only tests (creates pipelines, never executes). |
| `adapter=<substring>`                    | substring match on the adapter name: `adapter=llvmpipe`, `adapter=llvm`, `adapter=4070`, `adapter=NVIDIA GeForce RTX 4070 SUPER` all work. A non-matching name makes `requestAdapter()` (not `create()`) throw "no suitable backends found" and print "Available adapters:" with `backend: 'vulkan', name: '...'` lines to stderr. |
| `verbose=1`                              | prints "using GPU adapter: NVIDIA GeForce RTX 4070 SUPER" to stderr on each `requestAdapter()`.                                                     |
| `enable-dawn-features=a,b` / `disable-dawn-features=a,b` | Dawn toggles, names from https://dawn.googlesource.com/dawn/+/refs/heads/main/src/dawn/native/Toggles.cpp (e.g. `allow_unsafe_apis`, `dump_shaders`, `disable_symbol_renaming`). `allow_unsafe_apis` accepted without error. |

Things that do NOT work the browser way in Dawn-node 0.4.0 (probe
`dawn-select.mjs`):

- `requestAdapter({ forceFallbackAdapter: true })` is ignored: it returns the
  NVIDIA adapter. Select software via `create(["adapter=llvmpipe"])` instead.
- `adapter.isFallbackAdapter` is `undefined` for every adapter (Chromium
  returns `true` for SwiftShader, `false` for NVIDIA). Detect software
  adapters with `adapter.info.architecture === "software"` (llvmpipe reports
  vendor `mesa`, architecture `software`) OR `adapter.info.vendor === "google"
  && architecture === "swiftshader"` (Chromium). Write one helper
  `isSoftwareAdapter(info)` and use it everywhere, including the "assert we
  are on real hardware" test guard.
- `powerPreference` has no observable effect (all three preferences return
  the same adapter). Adapter choice in Node is entirely `adapter=`/`backend=`
  plus the Vulkan loader's ICD order.
- `adapter.requestAdapterInfo()` does not exist (removed from the spec; the
  scaffold's `test/setup/webgpu-global.ts` line 58 still calls it and must
  change to `adapter.info`). Chrome added `GPUAdapter.info` in 127 and removed
  `requestAdapterInfo()` in 131 (https://developer.chrome.com/blog/new-in-webgpu-128).

Lifetime: the README says the process will not exit while a reference to the
object returned by `create()` is reachable from a global; drop the reference
(`delete globalThis.navigator`) to let Node exit. Probe `dawn-list2.mjs`
confirmed a script that deletes its global reference exits with status 0.
The vitest `forks` pool sidesteps this (workers are killed), but a CLI
benchmark must release its `GPU` object or call `process.exit`.

Error reporting: Dawn-node prints every uncaptured validation error to stderr
as a multi-line "validation:" block and does not throw; `pushErrorScope /
popErrorScope` returns a `GPUValidationError` exactly as in the browser, and
both `device.onuncapturederror` and `addEventListener("uncapturederror")`
fire (probe `dawn-latency.mjs`). So the same error-scope discipline works in
both runtimes, and tests should install an `uncapturederror` listener that
fails the test (otherwise a bad bind group only shows up as stderr noise).

Noise: on this container every `create()` prints four
"error: XDG_RUNTIME_DIR not set in the environment." lines and two "Warning:
maxDynamic*BuffersPerPipelineLayout artificially reduced from 1000000 to 16"
lines to stderr (Mesa and Dawn respectively). Harmless; setting
`XDG_RUNTIME_DIR` to a writable directory in the test env silences the first.

### 2.3 Environment semantics that are the same in Node and browser

Verified earlier for graph-format (research note 09-webgpu-requirements.md,
section 7.2, quoting Dawn's `src/dawn/node/binding/GPUQueue.cpp` and
`GPUBuffer.cpp`): `queue.writeBuffer` copies synchronously from the V8
backing store (4-byte size rule enforced with "size is not a multiple of 4
bytes"), `getMappedRange()` returns an external ArrayBuffer that is detached
by `unmap()`/`destroy()`. The 667-line
`packages/graph-format/test/audit/gpu-upload.test.ts` runs the full section-10
upload contract (arena hot-prefix upload, 256-byte sub-range bindings,
`unpack4xU8`, permutation gathers, windowed colIdx rebase) on Dawn-node and
compares with CPU views; it passes on both NVIDIA and llvmpipe. Its
`acquire()` (lines 38-63) is the reference pattern for device acquisition in
tests: dynamic `import("webgpu")`, `Object.assign(globalThis, dawn.globals)`,
`dawn.create([])`, `requestAdapter()`, `adapter.info`, `requestDevice()`,
each failure mapped to an `E_NO_ADAPTER: ...` skip reason; "a wrong result is
never a skip" (line 16).

### 2.4 Adapters visible to Dawn-node on the dev box (probe `dawn-probe.mjs`)

Without `libEGL.so.1` on `LD_LIBRARY_PATH` the NVIDIA ICD fails to
initialise ("loader_scanned_icd_add: Could not get 'vkCreateInstance' ...
for ICD libGLX_nvidia.so.0") and the ONLY adapter is Mesa llvmpipe. With
`LD_LIBRARY_PATH=tmp/egl/root/usr/lib/x86_64-linux-gnu`
(HEADLESS_GPU_REPORT.md appendix D) the default adapter is the RTX 4070
SUPER and llvmpipe is still listed second. This is the same root cause as the
Chromium finding (HEADLESS_GPU_REPORT.md lines 10-15): fix the container
image (`libegl1`), and until then export `LD_LIBRARY_PATH` in the vitest
config / CI step.

| item                                | NVIDIA RTX 4070 SUPER (Dawn-node 0.4.0, Vulkan)                                              | Mesa llvmpipe 23.2.1 / LLVM 15 (Dawn-node, Vulkan)                                         |
|-------------------------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `adapter.info`                      | vendor `nvidia`, architecture `lovelace`, device `nvidia-geforce-rtx-4070-super`, description `NVIDIA: 580.173.02 580.173.2.0` | vendor `mesa`, architecture `software`, device `llvmpipe-llvm-15-0-7-256-bits-`             |
| `subgroupMinSize` / `MaxSize`       | 32 / 32                                                                                       | 8 / 8                                                                                       |
| features (compute-relevant)         | `core-features-and-limits`, `subgroups`, `timestamp-query`, `indirect-first-instance`, `texture-formats-tier1/2`; NO `shader-f16` | same PLUS `shader-f16`                                                                       |
| `maxBufferSize`                     | 1,099,511,627,776 (1 TiB as reported; VRAM is 12 GB)                                          | 4,294,967,295                                                                               |
| `maxStorageBufferBindingSize`       | 2,147,483,644 (2 GiB - 4)                                                                     | 134,217,728 (128 MiB, i.e. the default; lavapipe cannot raise it)                            |
| `maxStorageBuffersPerShaderStage`   | 16                                                                                            | 16                                                                                          |
| `maxComputeInvocationsPerWorkgroup` | 1024                                                                                          | 1024                                                                                        |
| `maxComputeWorkgroupSizeX/Y/Z`      | 1024 / 1024 / 64                                                                              | 1024 / 1024 / 1024                                                                          |
| `maxComputeWorkgroupStorageSize`    | 49,152                                                                                        | 32,768                                                                                      |
| `maxComputeWorkgroupsPerDimension`  | 65,535                                                                                        | 65,535                                                                                      |
| `minStorageBufferOffsetAlignment`   | 16 (!)                                                                                        | 16 (!)                                                                                      |
| `minUniformBufferOffsetAlignment`   | 64                                                                                            | 16                                                                                          |
| `maxUniformBufferBindingSize`       | 65,536                                                                                        | 65,536                                                                                      |
| `maxBindGroups` / `maxBindingsPerBindGroup` | 4 / 1000                                                                              | 4 / 1000                                                                                    |
| `wgslLanguageFeatures`              | 9: `uniform_buffer_standard_layout`, `unrestricted_pointer_parameters`, `subgroup_id`, `texture_formats_tier1`, `subgroup_uniformity`, `pointer_composite_access`, `packed_4x8_integer_dot_product`, `readonly_and_readwrite_storage_textures`, `texture_and_sampler_let` | same 9                                                                                      |
| 1M-element compute + 4 MiB readback | submit 0.40 ms, `onSubmittedWorkDone` 0.17 ms                                                 | submit 0.09 ms, `onSubmittedWorkDone` 20.5 ms                                               |

Two Node-vs-Chromium discrepancies to design around:

1. Dawn-node reports the ADAPTER's raw Vulkan limits (1 TiB maxBufferSize,
   alignment 16). Chromium 139 on the same GPU reports adapter
   `maxBufferSize` 4 GiB, `maxStorageBufferBindingSize` 4 GiB - 4, alignment
   256 (probe `chromium-latency.mjs`, and 09-webgpu-requirements.md section
   0). A device created with `requestDevice()` and no `requiredLimits` gets
   the SPEC DEFAULTS in both runtimes (verified in Chromium:
   `defaultDeviceLimits` = 256 MiB / 128 MiB / 8 / 16 KiB / 256; in Node the
   probe requested raised limits so the default device was not separately
   captured -- treat it as the spec default per the spec's requestDevice
   rule, https://gpuweb.github.io/gpuweb/#dom-gpuadapter-requestdevice).
   The upload planner must therefore read `device.limits`, never
   `adapter.limits`, and must never hard-code 256-byte alignment as the ONLY
   valid value: graph-format's arena is 256-aligned, which satisfies any
   alignment <= 256, so this is only a "do not assert `=== 256`" rule.
2. `uniform_buffer_standard_layout` is in Dawn-node's `wgslLanguageFeatures`
   but NOT in Chromium 139's (Chromium exposes only 4: `packed_4x8_integer_dot_product`,
   `unrestricted_pointer_parameters`, `pointer_composite_access`,
   `readonly_and_readwrite_storage_textures`). Do not write uniform structs
   that need the relaxed layout; obey the strict 16-byte rules (sec. 6).

### 2.5 Software adapters in Node: lavapipe

- Selection: `create(["adapter=llvmpipe"])` (substring match) or simply not
  having a usable hardware ICD. The Dawn node README also documents
  `VK_ICD_FILENAMES=<build>/lvp_icd.json` to force a specific ICD
  (https://dawn.googlesource.com/dawn/+/refs/heads/main/src/dawn/node/README.md);
  HEADLESS_GPU_REPORT.md line 123 records that pointing `VK_ICD_FILENAMES`
  at the NVIDIA json alone made Chromium's `requestAdapter()` return null, so
  prefer the `adapter=` option in Node and leave the loader env alone.
- Availability: Ubuntu's `mesa-vulkan-drivers` package provides
  `/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`; the container already has
  the lvp/radeon/intel/virtio ICDs (HEADLESS_GPU_REPORT.md line 34). On a
  GitHub `ubuntu-latest` runner `apt-get install -y mesa-vulkan-drivers
  libvulkan1` is the whole setup. (Not yet executed on a GitHub runner --
  listed in sec. 12.)
- Correctness: llvmpipe passed the whole graph-format GPU upload audit and
  the probe kernels, advertises `subgroups` (size 8) and `shader-f16`, so it
  exercises the subgroup and f16 code paths that the NVIDIA card cannot
  (f16) or exercises differently (subgroup size 8 vs 32 -- a kernel that
  assumes 32 breaks on lavapipe, which is exactly the portability bug we
  want CI to catch).
- Performance (probe `dawn-perf.mjs`, tiled O(n^2) repulsion kernel, the
  force-directed hot loop shape, 3-component positions):

  | n      | NVIDIA 4070 SUPER | llvmpipe   | ratio |
  |--------|-------------------|------------|-------|
  | 20,000 | 1.11 ms/iter      | 388 ms/iter| ~350x |
  | 5,000  | (not run)         | 29 ms/iter |       |

  So lavapipe is fine for correctness tests on graphs of 10^3-10^4 nodes and
  useless for anything O(n^2) at 10^5. Test fixtures must scale with the
  adapter: a `gpuScale()` helper that returns 1 on hardware and 1/50 on a
  software adapter, applied to fixture sizes and iteration counts.

---

## 3. Browser runtime facts (2026)

### 3.1 Where WebGPU ships (gpuweb Implementation Status wiki, "Last Updated
August 13, 2026", https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)

| browser         | status                                                                                                                                                           |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Chrome / Edge   | 113 on Mac, Windows x64, ChromeOS; Android 121 (ARM/Qualcomm/Intel, Android 12+); Linux: Intel Gen12+ 144, NVIDIA (driver 535.183.01+, Wayland) 147, others behind a flag; Windows ARM64 behind a flag. Implementation: Dawn. |
| Firefox         | Windows 141 (Mozilla Gfx blog, 2025-07-15); macOS Apple Silicon 145 (macOS 26+), 147 (all macOS versions); other macOS: Nightly; Linux: Nightly, "expected 2026 shipping"; Android behind flag. Implementation: wgpu. |
| Safari          | 26 on macOS, iOS, iPadOS, visionOS, enabled by default (WebKit blog: "shipping in Safari 26.0 for macOS, iOS, iPadOS, and visionOS", https://webkit.org/blog/17333/webkit-features-in-safari-26-0/). Implementation: WebKit's own. |

Chrome feature timeline (Chrome for Developers blog posts):

- `shader-f16`: requestable when the adapter offers it, Chrome 120
  (https://developer.chrome.com/blog/new-in-webgpu-120). Same post: default
  `maxStorageBuffersPerShaderStage` raised from 8 to 10 in Chrome (the SPEC
  default remains 8 -- our probe shows Chromium 139 giving a default device
  8 and the adapter 10, so "10" is an adapter maximum you must request).
- `timestamp-query`: shipped Chrome 121, quantised to 100 microseconds
  (https://developer.chrome.com/blog/new-in-webgpu-121); the quantisation can
  be disabled with chrome://flags/#enable-webgpu-developer-features for local
  profiling. Consequence: browser timestamp queries cannot resolve kernels
  shorter than ~100 us; per-kernel profiling belongs in Node/Dawn where the
  quantisation toggle is not applied (unverified whether Dawn-node quantises
  -- sec. 12).
- `subgroups`: experimental in 125, origin trial 128-131, shipped Chrome 134
  (https://developer.chrome.com/blog/new-in-webgpu-134,
  https://github.com/mdn/content/issues/38416).
- `GPUAdapter.info` in 127, `requestAdapterInfo()` removed in 131
  (https://developer.chrome.com/blog/new-in-webgpu-128).

Firefox and Safari optional-feature exposure (subgroups, f16, timestamps)
could not be verified from primary sources in this session (sec. 12); the
Firefox 141 post lists gaps only in IPC overhead, GPU completion latency via
interval timers, and `importExternalTexture`
(https://mozillagfx.wordpress.com/2025/07/15/shipping-webgpu-on-windows-in-firefox-141/).
Design rule: every optional feature is probed at runtime; nothing in the
library's public behaviour depends on them.

### 3.2 Headless Chromium on the dev box (probe `chromium-latency.mjs`, Chromium 139 build 1181)

With the four flags `--enable-unsafe-webgpu --enable-features=Vulkan
--use-angle=vulkan --disable-vulkan-surface` and `libEGL.so.1` on
`LD_LIBRARY_PATH` (HEADLESS_GPU_REPORT.md lines 19-23, 183-196):

- adapter: vendor `nvidia`, architecture `lovelace`, device/description
  empty strings (Chromium redacts them; Dawn-node does not),
  `isFallbackAdapter false`, subgroup size 32.
- features: `core-features-and-limits`, `subgroups`, `timestamp-query`,
  `indirect-first-instance`, `texture-compression-bc*`, plus, because of
  `--enable-unsafe-webgpu`, `chromium-experimental-subgroup-matrix`,
  `chromium-experimental-timestamp-query-inside-passes`,
  `chromium-experimental-multi-draw-indirect`. NOT `shader-f16` (the NVIDIA
  Vulkan path in this Dawn does not offer it; matches Dawn-node).
- adapter limits: maxBufferSize 4 GiB, maxStorageBufferBindingSize 4 GiB - 4,
  10 storage buffers per stage, 48 KiB workgroup storage, 1024 invocations,
  alignment 256. Default device: 256 MiB / 128 MiB / 8 / 16 KiB / 256 / 256.
- `wgslLanguageFeatures`: 4 (listed in sec. 2.4).
- latency: 4-byte `submit`+`mapAsync` round trip 0.10 ms;
  `submit([])`+`onSubmittedWorkDone` 0.12 ms; 1 MiB copy + map + slice 2.6 ms.

Without the flags (`--enable-unsafe-webgpu` only) Chromium uses SwiftShader:
vendor `google`, architecture `swiftshader`, `isFallbackAdapter true`,
subgroup size 4, `subgroups` and `timestamp-query` present, adapter
maxBufferSize / maxStorageBufferBindingSize 1 GiB, 32 KiB workgroup storage,
256 invocations max (cannot raise to 1024), round trip 0.15 ms, 1 MiB
readback 3.8 ms. Chrome 145 (system Google Chrome) only returned the NVIDIA
adapter with an explicit `powerPreference` (HEADLESS_GPU_REPORT.md line 177,
233-235): always pass `{ powerPreference: "high-performance" }` in the
browser entry point.

SwiftShader is the browser analogue of lavapipe: a real, spec-conformant CPU
implementation, subgroup size 4, present on every GitHub runner without
setup. The subgroup-size spread we can test is therefore 4 (SwiftShader), 8
(lavapipe), 32 (NVIDIA) -- good coverage for "never assume the subgroup
size" bugs.

---

## 4. Limits: spec defaults vs what is raisable

Sources: WebGPU spec limits table via 09-webgpu-requirements.md section 0
(fetched 2026-09-13) and today's probes. "Default" is what a device gets
with no `requiredLimits`; "raisable to" is `adapter.limits` on each stack.
Per spec, `requiredLimits` may only request values between the default and
the adapter's value (worse-than-default requests are clamped; better-than-
adapter requests reject `requestDevice`).

| limit                                     | spec default (core) | compat-mode default | Chromium 139 + NVIDIA adapter | Dawn-node + NVIDIA adapter | Dawn-node + llvmpipe | Chromium SwiftShader | what the library does |
|-------------------------------------------|--------------------:|--------------------:|------------------------------:|---------------------------:|---------------------:|---------------------:|-----------------------|
| maxBufferSize                             | 268,435,456         | same                | 4,294,967,296                 | 1,099,511,627,776          | 4,294,967,295        | 1,073,741,824        | request `adapter.limits.maxBufferSize`; plan arena vs per-array vs windowed from `device.limits` (design 10.3) |
| maxStorageBufferBindingSize               | 134,217,728         | same                | 4,294,967,292                 | 2,147,483,644              | 134,217,728          | 1,073,741,824        | request adapter value; window bindings at 64-arc (256 B) boundaries when a single array exceeds it (design 10.6) |
| maxStorageBuffersPerShaderStage           | 8                   | 4                   | 10                            | 16                         | 16                   | 10                   | design every kernel for 8; pack read-only arrays into one buffer with offset bindings if a kernel needs more |
| maxUniformBufferBindingSize               | 65,536              | 16,384              | 65,536                        | 65,536                     | 65,536               | (not probed)         | params structs are tiny; irrelevant |
| minStorageBufferOffsetAlignment           | 256                 | 256                 | 256                           | 16                         | 16                   | 256                  | always align to 256 (graph-format arena already does); never assert the device value equals 256 |
| minUniformBufferOffsetAlignment           | 256                 | 256                 | 256                           | 64                         | 16                   | (not probed)         | dynamic uniform offsets stride 256 |
| maxDynamicStorageBuffersPerPipelineLayout | 4                   | 4                   | 8 (Chromium 139 probe, 09 note) | 16                       | 16                   | (not probed)         | use <= 4 dynamic storage bindings |
| maxBindGroups                             | 4                   | 4                   | 4                             | 4                          | 4                    | 4                    | group 0 = graph (immutable), 1 = algorithm state, 2 = per-dispatch params |
| maxComputeInvocationsPerWorkgroup         | 256                 | 128                 | 1024                          | 1024                       | 1024                 | 256                  | workgroup size 256 default via `override`; 128 variant for compat mode; never > 256 without checking |
| maxComputeWorkgroupSizeX                  | 256                 | 128                 | 1024                          | 1024                       | 1024                 | (not probed)         | same |
| maxComputeWorkgroupStorageSize            | 16,384              | same                | 49,152                        | 49,152                     | 32,768               | 32,768               | tiles sized for 16 KiB (e.g. 256 x vec4<f32> = 4 KiB per tile array); larger tiles only via a probed variant |
| maxComputeWorkgroupsPerDimension          | 65,535              | same                | 65,535                        | 65,535                     | 65,535               | 65,535               | 1D dispatch covers 65,535 x 256 = 16,776,960 items; above that use a 2D grid or grid-stride loop (design 10.6) |
| subgroup size                             | n/a                 | n/a                 | 32                            | 32                         | 8                    | 4                    | read `adapter.info.subgroupMinSize/MaxSize`; kernels take it as an `override` and never hard-code 32 |

Note on the NVIDIA "1 TiB" `maxBufferSize` under Dawn-node: it is what the
Vulkan driver advertises (`maxStorageBufferRange`/heap accounting), not
physical memory. Allocation failure surfaces as an `out-of-memory` error
scope result or a device loss, so the residency planner must wrap large
`createBuffer` calls in `pushErrorScope("out-of-memory")` and treat a
non-null result as "chunk smaller", not rely on the limit alone.

---

## 5. Feature matrix

"verified" = probed on this box today; "docs" = a cited document; "?" =
not verified (sec. 12).

| feature / property                        | Node Dawn 0.4.0 + NVIDIA | Node Dawn 0.4.0 + llvmpipe | Chromium 139 + NVIDIA      | Chromium 139 SwiftShader | Firefox (wgpu)              | Safari 26 (WebKit)          |
|-------------------------------------------|--------------------------|----------------------------|----------------------------|--------------------------|-----------------------------|-----------------------------|
| compute shaders / storage buffers         | yes (verified)           | yes (verified)             | yes (verified)             | yes (verified)           | yes (docs: shipped 141+)    | yes (docs: "adds compute shaders") |
| `subgroups`                               | yes, 32 (verified)       | yes, 8 (verified)          | yes, 32 (verified; shipped Chrome 134 docs) | yes, 4 (verified) | ?                           | ?                           |
| `shader-f16`                              | NO (verified)            | yes (verified)             | NO (verified)              | ? (not in printed list)  | ?                           | ?                           |
| `timestamp-query`                         | yes (verified)           | yes (verified)             | yes, 100 us quantised (docs) | yes (verified)          | ?                           | ?                           |
| `indirect-first-instance`                 | yes                      | yes                        | yes                        | yes                      | ?                           | ?                           |
| `core-features-and-limits`                | yes                      | yes                        | yes                        | yes                      | ?                           | ?                           |
| `adapter.isFallbackAdapter`               | undefined (bug)          | undefined (bug)            | false                      | true                     | ?                           | ?                           |
| `adapter.info.vendor/architecture`        | nvidia / lovelace        | mesa / software            | nvidia / lovelace          | google / swiftshader     | ?                           | ?                           |
| `adapter.info.device/description`         | filled                   | filled                     | empty strings              | empty strings            | ?                           | ?                           |
| `forceFallbackAdapter`                    | ignored                  | n/a                        | honoured (Chrome docs)     | n/a                      | ?                           | ?                           |
| `powerPreference` effect                  | none                     | none                       | needed on Chrome 145 to get NVIDIA (report) | n/a     | ?                           | ?                           |
| `wgslLanguageFeatures` count              | 9 incl. uniform_buffer_standard_layout | 9              | 4                          | 4                        | ?                           | ?                           |
| uncaptured errors                         | stderr + event           | stderr + event             | console + event            | console + event          | event (spec)                | event (spec)                |
| `mapAsync` round trip                     | 0.04 ms                  | 0.03 ms                    | 0.10 ms                    | 0.15 ms                  | "interval timers" latency (Mozilla blog, being improved) | ? |
| glibc / OS requirement                    | glibc >= 2.34 (0.4.0), >= 2.38 (0.6.x) | same           | Playwright chromium deps   | none extra               | n/a                         | macOS/iOS only              |

---

## 6. WGSL constraints that shape the kernels

From the WGSL spec (https://gpuweb.github.io/gpuweb/wgsl/, fetched
2026-09-14; section numbers as in the current draft) and
09-webgpu-requirements.md section 2:

1. Atomics: `atomic<T>` only with `T` = `u32` or `i32`, only in
   `var<workgroup>` or `var<storage, read_write>` (WGSL 6.2.8). There are no
   floating-point atomics. Consequences: (a) force accumulation into a shared
   `Float32Array` displacement buffer cannot use `atomicAdd` on f32 -- either
   gather (each node sums its own incoming forces from CSR, no atomics: the
   preferred pattern for a symmetric graph), or scatter with a fixed-point
   `i32` encoding (`atomicAdd(&acc[i], i32(f * SCALE))`, decode on read;
   overflow risk must be bounded by clamping forces), or a CAS loop on
   `bitcast<u32>(f32)` (correct, slow under contention). (b) PageRank-style
   push kernels have the same choice; pull (gather over `reverse()`) is the
   default per design section 10.
2. No recursion: "a function must not directly or indirectly call itself"
   (WGSL 11.4). Barnes-Hut tree traversal must be iterative with an explicit
   stack in registers/workgroup memory or use a stackless (Karras/Burtscher
   style) traversal; a linked "rope" octree is the standard answer.
3. `override` constants (WGSL 7.2.2) can parameterise `@workgroup_size`
   (WGSL 12.15) and are set at `createComputePipeline({ compute: {
   constants: { WG: 256, USE_PERM: 1 } } })`. Each distinct constant set is
   a separate pipeline object: the PipelineCache key must include the
   constants. Design section 10 already prescribes `USE_PERM` for identity
   permutations.
4. `bool` is not host-shareable (WGSL 6.5.2): masks are `u32` bitsets or
   packed `u8` in `u32` (design section 10; `unpack4xU8` verified in
   gpu-upload.test.ts).
5. Uniform address space layout: array element stride and struct member
   alignment must be multiples of 16 (WGSL 14.4.5) unless the
   `uniform_buffer_standard_layout` language feature is present -- it is in
   Dawn-node but NOT in Chromium 139 (sec. 2.4), so write params structs
   with explicit `vec4`/16-byte padding, e.g. `struct Params { n: u32, m:
   u32, iter: u32, _pad: u32, k: f32, gravity: f32, dt: f32, _pad2: f32 }`.
   Storage buffers use the natural (std430-like) layout; `array<vec3<f32>>`
   has stride 16, so 3-component position columns are read as `array<f32>`
   with `3u*i + k` indexing (09 note section 2, recommendation 3).
6. Runtime-sized arrays only as the last member of a storage struct or as
   the whole binding; dynamic indexing with runtime indices is allowed.
   Out-of-bounds accesses are clamped/discarded by the implementation
   (spec-defined "invalid memory reference" behaviour), which silently hides
   bugs: tests must compare against CPU results, not just "no error".
7. `enable subgroups;` and `enable f16;` are per-module directives that
   fail compilation when the device lacks the feature, so a kernel with a
   subgroup fast path is TWO modules (or one template with the directive
   spliced in by the WGSL composer) selected by `device.features.has(...)`.
   Subgroup builtins used by the primitives layer: `subgroupAdd`,
   `subgroupExclusiveAdd`, `subgroupBallot`, `subgroupShuffle`,
   `subgroupBroadcast` (WGSL 17.12); `subgroup_size` and
   `subgroup_invocation_id` builtins; `subgroup_uniformity` diagnostic
   control appears in Dawn's language feature list.
8. Integer division by zero and shifts >= 32 are defined (no traps) but
   produce implementation-specified results; `u32` wraparound is defined.
   `INVALID_INDEX = 0xFFFFFFFFu` comparisons are exact.

---

## 7. Dispatch, readback, errors, device loss

### 7.1 Dispatch

- `dispatchWorkgroups(x, y, z)` with each count <= 65,535
  (`maxComputeWorkgroupsPerDimension`). Design section 10.6's rule: 1D up
  to 16,776,960 items at workgroup size 256; beyond that a 2D grid
  (`x = 65535`, `y = ceil(groups / 65535)`) with the kernel recomputing a
  linear id `gid.x + gid.y * 65535u * WG`, or a grid-stride loop. Both are
  one helper in the DispatchPlanner and unit-testable with faked limits.
- `dispatchWorkgroupsIndirect(buffer, offset)`: the buffer needs
  `GPUBufferUsage.INDIRECT`, the offset is a multiple of 4, and the 12 bytes
  at the offset are three `u32` workgroup counts
  (https://gpuweb.github.io/gpuweb/#dom-gpucomputepassencoder-dispatchworkgroupsindirect).
  Per the spec text as fetched, counts above the limit make the dispatch do
  nothing rather than error, so a frontier-sized indirect dispatch must be
  written by a tiny "finalise" kernel that computes `ceil(frontierSize /
  WG)` and clamps/splits into 2D. This is what removes the CPU round trip
  between BFS levels: the frontier size stays on the GPU, the level loop
  records N levels' worth of commands, and the CPU only reads back a "done"
  flag every k levels.
- Multiple dispatches inside one compute pass are ordered with implicit
  storage barriers between them; one command buffer per `step()` batch is
  the unit of submission. There is no cross-dispatch synchronisation within
  a dispatch (no grid barrier), so iterative algorithms are one dispatch per
  iteration phase.

### 7.2 Readback latency and pipelining (measured today)

| operation                                       | Dawn-node NVIDIA | Dawn-node llvmpipe | Chromium NVIDIA | Chromium SwiftShader |
|-------------------------------------------------|-----------------:|-------------------:|----------------:|---------------------:|
| `submit` + `mapAsync` (4 bytes)                 | 0.040 ms         | 0.029 ms           | 0.102 ms        | 0.154 ms             |
| `submit([])` + `onSubmittedWorkDone`            | 0.013 ms         | 0.013 ms           | 0.118 ms        | 0.034 ms             |
| 1 MiB copy + `mapAsync` + `slice`               | (not run)        | (not run)          | 2.65 ms         | 3.83 ms              |
| 234 KiB readback added to a 1.1 ms kernel       | +0.07 ms         | n/a                | (not run)       | (not run)            |

Interpretation: readback cost is dominated by the copy and JS-side slice,
not by the fence wait, and it is small relative to a layout iteration on a
real GPU (the n-body iteration at 20k nodes was 1.1 ms; a 100k-node
Barnes-Hut iteration will be several ms). Still, the layout API must not
force a readback per iteration:

- `LayoutSimulation.step(iterations)` (design 14.3) records `iterations`
  full iterations into one command buffer (or a few), submits once, and
  reads positions back once at the end of the call into the caller's
  stride-3 `positions` Float32Array. The GPU buffer is authoritative between
  steps; `setPosition` / `setFixed` queue `writeBuffer` calls that precede
  the next submit.
- Use a ring of 2-3 staging buffers so `mapAsync` on iteration k's staging
  buffer overlaps the submission of iteration k+1. `mapAsync` on a buffer
  that is still in use by a queued copy is a validation error, hence the
  ring; `buffer.mapState` tells whether a staging buffer is free.
- Avoid `onSubmittedWorkDone` as a poll: `mapAsync` already waits for the
  copy, and the extra promise costs ~0.1 ms in Chromium.
- Node-specific: Dawn-node's event loop integration resolves `mapAsync` only
  when the JS thread yields (it is a promise), so never spin-wait; and a
  CLI benchmark must `await` every readback before `process.exit`.

### 7.3 Errors

- Validation errors are asynchronous and per-device; capture them with
  `pushErrorScope("validation")` around pipeline creation and bind group
  creation (cheap, and it converts a silent stderr block into a thrown
  error with a message). Keep `pushErrorScope("out-of-memory")` around large
  allocations (sec. 4). Filters: `validation`, `out-of-memory`, `internal`.
- Install `device.addEventListener("uncapturederror", ...)` in the
  `GpuContext` and re-emit as a JS event / throw in tests. Verified working
  in Dawn-node (sec. 2.2) and standard in browsers.
- Label every buffer/pipeline (`label:`), since error text says "[Buffer
  (unlabeled)]" otherwise (visible in today's probe output).

### 7.4 Device loss

- `device.lost` is a promise resolving to `{ reason, message }`; reasons
  are `"destroyed"` and `"unknown"` (spec; fetched summary also listed
  `"replaced"`, which this note does not rely on). Verified: after
  `device.destroy()` Dawn-node resolves it with reason `destroyed`, message
  "device was destroyed".
- Policy (project rule: never fall back to CPU): a lost device invalidates
  every `GraphResidency` entry and pipeline for that device; the `GpuContext`
  transitions to a `lost` state, pending `step()`/algorithm promises reject
  with a typed `GpuDeviceLostError`, and the caller (graphty-element's
  LayoutManager / DataManager) decides whether to request a new device and
  re-`load()`. The library exposes `ctx.lost: Promise<GPUDeviceLostInfo>`
  and never recreates a device on its own. Tests cover this by calling
  `device.destroy()` mid-run (deterministic, both runtimes).

---

## 8. Recommended runtime architecture (platform-facing parts only)

### 8.1 Entry points and packaging

```
package.json (sketch)
{
  "name": "@graphty/webgpu-graph-algorithms",
  "type": "module",
  "exports": {
    ".":         { "types": "./dist/index.d.ts",   "import": "./dist/index.js" },
    "./browser": { "types": "./dist/browser.d.ts", "import": "./dist/browser.js" },
    "./node":    { "types": "./dist/node.d.ts",    "import": "./dist/node.js" }
  },
  "peerDependencies": { "@graphty/graph-format": "workspace:^", "webgpu": "^0.4.0" },
  "peerDependenciesMeta": { "webgpu": { "optional": true } },
  "devDependencies": { "webgpu": "0.4.0", "@webgpu/types": "^0.1.72", ... },
  "sideEffects": false
}
```

- `.` exports the device-agnostic API: `GpuContext.from(device)`,
  `GraphResidency`, algorithms as `(ctx, snapshot, options) => Promise<TypedArray>`,
  `createForceLayout(ctx): LayoutSimulation`. It imports nothing runtime-
  specific and reads `GPUBufferUsage` etc. only inside functions.
- `./browser` exports `requestGpuContext({ powerPreference:
  "high-performance", requiredFeatures?, raiseLimits: true })` built on
  `navigator.gpu`, throwing a typed `WebGpuUnavailableError` when
  `navigator.gpu` is missing or `requestAdapter` returns null (the
  project's no-fallback rule; the CPU packages are the default path, this
  package is opt-in).
- `./node` exports `createNodeGpuContext({ adapter?: string, backend?:
  string, dawnFeatures?: string[], software?: boolean })` which does
  `const dawn = await import("webgpu")` (dynamic, so a browser bundler that
  somehow reaches this file still cannot statically pull the native module),
  installs `dawn.globals`, calls `create([...options])` translating
  `software: true` to `adapter=llvmpipe` (documented as Linux/Mesa specific)
  and holds the `GPU` object on the context so it can be released.
- Bundling: vite library build with `rollupOptions.external: ["webgpu",
  "@graphty/graph-format"]` and three entries (`index`, `browser`, `node`).
  The scaffold's `vite.config.ts` has `external: []` and one entry (lines
  27-39) -- change it. `browserslist` in the scaffold (Chrome >= 113, Safari
  >= 18, Firefox >= 128) understates Firefox (141) and Safari (26) per sec.
  3.1; set `Chrome >= 134` (subgroups shipped) or document the feature
  probing.

### 8.2 WGSL sources

- Keep every kernel in a `.wgsl` file and import it with vite's `?raw`
  suffix (`import bfsAdvance from "./kernels/bfs-advance.wgsl?raw"`;
  https://vite.dev/guide/assets "Importing Asset as String"). This works
  identically in `vite build` (library mode inlines the string), in the
  vitest node project (vite transforms test imports) and in the vitest
  browser project. TypeScript needs `/// <reference types="vite/client" />`
  in an `env.d.ts` (the monorepo's graphty app already has
  `graphty/src/vite-env.d.ts` with exactly that line).
- The WGSL composer (prelude + snippets + `enable` directives) works on
  strings, so `?raw` is sufficient; no custom plugin.
- Because `tsc` alone does not understand `?raw`, the `build` script is
  `tsc --noEmit && vite build` with `vite-plugin-dts` (or `tsc -p
  tsconfig.build.json` for declarations only), never `tsc` emitting JS.

### 8.3 Runtime capability record

`GpuContext` captures once, at creation:

```ts
interface GpuCaps {
  readonly limits: GPUSupportedLimits;      // device.limits, NOT adapter.limits
  readonly features: ReadonlySet<string>;   // device.features
  readonly subgroupMin: number; readonly subgroupMax: number; // adapter.info, 0 if absent
  readonly software: boolean;               // isSoftwareAdapter(adapter.info)
  readonly runtime: "browser" | "node";
  readonly vendor: string; readonly architecture: string;
}
```

Every planner (upload, dispatch, tile sizes, subgroup variants) is a pure
function of `GpuCaps` and the snapshot's byte lengths, so it is unit-tested
with faked caps (spec defaults, SwiftShader-like, lavapipe-like, NVIDIA-like
tables from sec. 4) without a device.

---

## 9. Recommended test architecture

### 9.1 Principles

- One set of test files, two vitest projects (`node`, `browser`), same
  `include` globs. A test gets its context from `test/helpers/gpu.ts`
  `acquireGpu()` which branches on `typeof navigator !== "undefined" &&
  "gpu" in navigator` (browser) versus dynamic `import("webgpu")` (node).
  This mirrors graph-format's `acquire()` and the scaffold's
  `test/helpers/webgpu.ts` intent, and replaces the scaffold's browser-only
  `test/setup/webgpu-global.ts` (which also calls the removed
  `requestAdapterInfo()`).
- Node project = full suite, `pool: "forks"` (native addon; graph-format
  runs 1309 tests this way, packages/graph-format/vitest.config.ts line 7).
  Browser project = a curated subset (`test/**/*.browser.test.ts` or a
  `@browser` tag via `include`), run headless on Playwright Chromium with the
  four GPU flags; passes on SwiftShader too.
- Hard-fail on no adapter when `GRAPHTY_REQUIRE_GPU=1` (CI GPU job, local
  dev); otherwise skip GPU tests with the printed `E_NO_ADAPTER` reason
  (graph-format convention). Assert real hardware
  (`!caps.software`) only when `GRAPHTY_REQUIRE_HW_GPU=1`, so lavapipe/
  SwiftShader runs are green but a silent software fallback on the GPU
  runner is red (HEADLESS_GPU_REPORT.md recommendation 3).
- Never skip on a wrong result; fixtures scale by `caps.software` (sec. 2.5).
- Every GPU test compares against the CPU reference (`@graphty/algorithms`
  or a tiny in-test reference), the property-based style graph-format uses
  (`fast-check` is already a devDependency there).

### 9.2 vitest config sketch for THIS repo on vitest 2.1.x (current scaffold)

The scaffold's `vitest.config.ts` passes a top-level `launch` that Vitest 2.1
rejects, forces SwiftShader with `--use-gl=swiftshader --use-vulkan=swiftshader`,
and runs everything in the browser (lines 8-26). Replacement (2.1 syntax:
`browser.name` + `browser.providerOptions.launch.args`,
HEADLESS_GPU_REPORT.md recommendation 2):

```ts
// vitest.config.ts (vitest 2.1.x)
import { defineConfig } from "vitest/config";
const GPU_ARGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan", "--disable-vulkan-surface"];
export default defineConfig({
  test: {
    workspace: [
      { test: { name: "node", environment: "node", pool: "forks", include: ["test/**/*.test.ts"], setupFiles: ["test/setup/node.ts"], testTimeout: 60_000 } },
      { test: { name: "browser", include: ["test/**/*.browser.test.ts"], setupFiles: ["test/setup/browser.ts"],
          browser: { enabled: true, headless: true, name: "chromium", provider: "playwright",
                     providerOptions: { launch: { args: GPU_ARGS, env: { ...process.env, LD_LIBRARY_PATH: process.env.GRAPHTY_EGL_LIB_DIR ?? "" } } } } } },
    ],
  },
});
```

Better: move the scaffold to vitest 3.2.x now, because the monorepo is on
`vitest ^3.2.4`, `@vitest/browser ^3.2.4`, `playwright ^1.54.1`, `vite ^7`
(graphty-monorepo/package.json lines 130, 148, 158, 160) and its browser
projects use the 3.x `instances` form (graphty-element/vitest.config.ts,
algorithms/vitest.config.ts). The staged graph-format package is already on
3.2.4 (packages/graph-format/package.json lines 65-73).

### 9.3 vitest config sketch on vitest 3.2.x (monorepo form)

```ts
// vitest.config.ts (vitest 3.2.x; mirrors graphty-monorepo/algorithms/vitest.config.ts)
import { defineConfig } from "vitest/config";
const GPU_ARGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan", "--disable-vulkan-surface"];
const eglDir = process.env.GRAPHTY_EGL_LIB_DIR;           // e.g. tmp/egl/root/usr/lib/x86_64-linux-gnu until the image has libegl1
export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: "node",                                    // PRIMARY: full suite on Dawn
          environment: "node",
          pool: "forks",                                   // native addon; matches graph-format
          include: ["test/**/*.test.ts"],
          setupFiles: ["test/setup/node.ts"],              // installs dawn.globals, sets XDG_RUNTIME_DIR, uncapturederror -> fail
          testTimeout: 60_000, hookTimeout: 60_000,
        },
      },
      {
        test: {
          name: "browser",                                 // LIGHT: smoke subset on Playwright Chromium
          include: ["test/**/*.browser.test.ts"],
          setupFiles: ["test/setup/browser.ts"],
          browser: {
            enabled: true, headless: true, provider: "playwright",
            fileParallelism: false,                        // one GPU-bearing browser at a time (graphty-element does the same)
            instances: [{ browser: "chromium", launch: { args: GPU_ARGS, env: eglDir ? { ...process.env, LD_LIBRARY_PATH: eglDir } : undefined } }],
          },
          testTimeout: 60_000,
        },
      },
    ],
    coverage: { provider: "v8", include: ["src/**/*.ts"], reporter: ["text", "json-summary", "json", "lcov", "html"] },
  },
});
```

`instances[].launch` is the Vitest 3 per-instance Playwright launch object
(Vitest 3 blog, "Configure multi-browser instances"; the per-instance
`launch: { devtools: true }` example). Vitest ignores `launch.headless`; use
`browser.headless` (Vitest docs, playwright provider page).

Vitest 4 changes the provider to a factory from a separate package:
`import { playwright } from "@vitest/browser-playwright"` and
`provider: playwright({ launchOptions: { args: GPU_ARGS } })`, with
per-instance `launch` removed in favour of a per-instance `provider:
playwright({...})` that does NOT merge with the parent (Vitest 4 blog and
`docs/config/browser/playwright.md`). The monorepo pins `@vitest/browser`
overrides for both `<3.2.7 -> ^3.2.7` and `>=4.0.0 <4.1.10 -> ^4.1.10`
(graphty-monorepo/package.json pnpm.overrides), i.e. a 4.x migration is on
the horizon; keep the GPU flags in one exported constant so the move is a
one-line config change.

Package scripts (sketch): `"test": "vitest"`, `"test:run": "vitest run
--project=node"`, `"test:browser": "vitest run --project=browser"`,
`"test:gpu": "GRAPHTY_REQUIRE_GPU=1 GRAPHTY_REQUIRE_HW_GPU=1 vitest run"`,
`"test:software": "GRAPHTY_GPU_ADAPTER=llvmpipe vitest run --project=node"`
(the Node setup file passes `GRAPHTY_GPU_ADAPTER` through as `adapter=`).

### 9.4 Node setup file (sketch)

```ts
// test/setup/node.ts
import { beforeAll } from "vitest";
beforeAll(async () => {
  process.env.XDG_RUNTIME_DIR ??= "/tmp";                        // silences Mesa's four stderr lines
  const dawn = await import("webgpu");                            // webgpu@0.4.0 on glibc 2.35
  Object.assign(globalThis, dawn.globals);
  const opts: string[] = [];
  if (process.env.GRAPHTY_GPU_ADAPTER) opts.push(`adapter=${process.env.GRAPHTY_GPU_ADAPTER}`);
  if (process.env.GRAPHTY_DAWN_FEATURES) opts.push(`enable-dawn-features=${process.env.GRAPHTY_DAWN_FEATURES}`);
  (globalThis as any).__graphtyGpu = dawn.create(opts);           // released in a global afterAll so the fork can exit
});
```

The browser setup file only asserts `navigator.gpu` exists and, when
`GRAPHTY_REQUIRE_HW_GPU` is set (forwarded via `test.env` or
`import.meta.env`), that `adapter.info.architecture !== "swiftshader"`.

### 9.5 What "light browser testing" contains

- device acquisition through `./browser` (typed error when `navigator.gpu`
  is absent);
- one end-to-end per family on a small fixture: upload + BFS levels, one
  PageRank, one connected components, and the force layout `load / step(10)
  / positions written back / dispose`;
- a `release()` test (buffers destroyed, no uncaptured errors);
- a subgroup-variant test that only runs if `device.features.has("subgroups")`
  (true on SwiftShader with size 4).
Everything else (property tests, large fixtures, faked-limit planners,
windowed uploads, indirect dispatch loops, device-loss) lives in the Node
project.

---

## 10. CI/CD for GPU tests

### 10.1 What cuda-ffi does (cloned to tmp/webgpu-plan/repos/cuda-ffi, HEAD 086d5d6, 2024-10-20)

`.github/workflows/build.yml`: a `lint` job on `ubuntu-latest`, then a
`build` job with

```yaml
runs-on: cudaffi-gpu-runner
container:
  image: ghcr.io/apowers313/roc-dev:1.5.2
  env: { CUDA_HOME: /usr/local/cuda, LD_LIBRARY_PATH: /usr/local/cuda/lib64:... }
  options: "--gpus all --user root"
```

i.e. a SELF-HOSTED runner registered with a custom label, a pinned dev
container image from the owner's GHCR, and Docker's `--gpus all` so the
job container sees the host NVIDIA driver. Tests, coverage upload
(coveralls) and docs deploy all run inside that job. This is the pattern to
copy: the same dev image (with `libegl1`, `mesa-vulkan-drivers`,
`libvulkan1`, Node 22, pnpm, Playwright deps) serves local development and
CI, and the GPU job is just `runs-on: <label>` + `container.options:
--gpus all`. For WebGPU (Vulkan, not CUDA) the container additionally needs
`NVIDIA_DRIVER_CAPABILITIES=all` (or at least `graphics,compute,utility`)
so the toolkit injects the Vulkan ICD and libGLX_nvidia
(HEADLESS_GPU_REPORT.md lines 33-34 and the container toolkit issue #1952
cited there), and `libEGL.so.1` (the whole root cause of the report).

### 10.2 GitHub-hosted GPU runners (verified from GitHub docs, 2026-09-14)

- GA since 2024-07-08 for Linux and Windows: "T4 GPU access", set up "in
  your organization or enterprise" through runner groups, then referenced by
  the runner's name in `runs-on`
  (https://github.blog/changelog/2024-07-08-github-actions-gpu-hosted-runners-are-now-generally-available/).
- Spec: "4 CPU, 1 GPU (Tesla T4), 28 GB Memory (RAM), 16 GB GPU memory
  (VRAM), 176 GB Storage (SSD)", Ubuntu and Windows
  (https://docs.github.com/en/actions/reference/runners/larger-runners).
- Price: Linux 4-core GPU `linux_4_core_gpu` $0.052/min, Windows
  `windows_4_core_gpu` $0.102/min, versus standard Linux 2-core $0.006/min
  (https://docs.github.com/en/billing/reference/actions-runner-pricing).
  A 10-minute GPU job is $0.52; ~50 pushes/month is ~$26.
- Larger runners are an organization/enterprise feature (the docs fetched
  did not state the plan tier explicitly -- sec. 12); `graphty-org` is an
  organization, so this is possible but requires a paid plan decision.
- A T4 (Turing) under Vulkan will expose a different limit/feature set than
  the 4070 (e.g. subgroup size 32 but different workgroup storage), which is
  good coverage; the Dawn/NVIDIA driver version on the image is whatever
  the Azure marketplace image ships -- must be probed once.

Third-party GPU runner services surfaced by search (not evaluated):
https://runs-on.com/runners/gpu/ , https://machine.dev/ .

### 10.3 Recommended layout for this repo now, and for the monorepo later

Two runner classes, one workflow:

```yaml
jobs:
  test-software:                      # every push/PR, no GPU, ubuntu-latest
    runs-on: ubuntu-latest
    steps:
      - run: sudo apt-get install -y mesa-vulkan-drivers libvulkan1 libegl1
      - run: pnpm exec playwright install chromium --with-deps
      - run: GRAPHTY_GPU_ADAPTER=llvmpipe pnpm exec vitest run --project=node      # Dawn on lavapipe
      - run: pnpm exec vitest run --project=browser                                # Chromium on SwiftShader
  test-gpu:                           # push to master + nightly + label-triggered on PRs
    runs-on: [self-hosted, linux, gpu, graphty-gpu]   # or the org's gpu-t4-4-core larger runner
    container:
      image: ghcr.io/graphty-org/dev:<tag>            # same image as the dev container, WITH libegl1
      options: "--gpus all"
      env: { NVIDIA_DRIVER_CAPABILITIES: all, GRAPHTY_REQUIRE_GPU: "1", GRAPHTY_REQUIRE_HW_GPU: "1" }
    steps:
      - run: pnpm exec vitest run --project=node
      - run: pnpm exec vitest run --project=browser
      - run: pnpm run bench -- --json > bench.json     # perf regression numbers, uploaded as an artifact
```

Notes:

- The monorepo's CI already builds once and fans out a `Test (${{
  matrix.shard }})` matrix on `ubuntu-latest` with `pnpm exec vitest run
  --project=...` per package (graphty-monorepo/.github/workflows/ci.yml
  lines 236-350) and installs Playwright with `playwright install chromium
  --with-deps` (line 423). The WebGPU package slots in as two more shards:
  `webgpu-node-software` and `webgpu-browser-software` on the default
  runner, plus a separate `test-gpu` job keyed to the GPU runner label.
  Nx's `affected` logic keeps the GPU job from running when the package is
  untouched.
- The GPU job must fail loudly on a software adapter
  (`GRAPHTY_REQUIRE_HW_GPU=1`) so a broken driver mount does not turn into a
  silently-green lavapipe run -- the exact failure mode that
  HEADLESS_GPU_REPORT.md found locally.
- `fileParallelism: false` (browser) and `poolOptions.forks.singleFork`
  (node) on the GPU job: one device at a time per GPU avoids VRAM contention
  between forked workers and makes benchmark numbers stable. The software
  job can parallelise freely.
- Self-hosted runner on the dev box: the runner agent needs Docker access
  and the NVIDIA container toolkit (already present: CUDA works in the
  container, HEADLESS_GPU_REPORT.md line 39). Register it with labels
  `[self-hosted, linux, gpu, graphty-gpu]`; restrict the GPU job to
  non-fork PRs (self-hosted runners must not execute untrusted PR code).
- Until libegl1 is in the image, the job exports `GRAPHTY_EGL_LIB_DIR` to an
  extracted `libegl1` tree exactly as the local recipe does (appendix D of
  the report); the vitest config threads it into Playwright's `env` and the
  Node setup file relies on the process env.
- Benchmarks (`vitest bench` or a tsx script) run only on the GPU job and
  publish JSON; the software job never times anything.

---

## 11. Checklist of scaffold changes implied by this note

- package.json: `@webgpu/types` to `^0.1.72`; add `webgpu@0.4.0` as
  devDependency and optional peer; vitest/vite/@vitest/browser to the
  monorepo's 3.2.x / 7.x; `exports` with `./browser` and `./node`;
  `browserslist` updated.
- vitest.config.ts: two projects as in 9.3; drop the SwiftShader flags;
  drop the rejected top-level `launch`.
- vite.config.ts: three entries, `external: ["webgpu", "@graphty/graph-format"]`.
- test/setup/webgpu-global.ts: delete (uses `requestAdapterInfo`, browser
  only); replace with `test/helpers/gpu.ts` (`acquireGpu`, `isSoftwareAdapter`,
  `gpuScale`, uncaptured-error hook) plus `test/setup/node.ts` and
  `test/setup/browser.ts`.
- src/types/index.ts: obsolete `CSRGraph` deleted per design 14.5.
- Add `tmp/webgpu-plan/probe/*.mjs` equivalents as a `scripts/gpu-probe.mjs`
  that prints adapter, features, limits, subgroup size and round-trip
  latency; run it as the first step of every CI GPU job and attach its
  output to the run.

---

## 12. Unverified / open

1. `webgpu@0.5.0`'s binary requirements were not inspected (only 0.4.0 and
   0.6.1); 0.5.0 and 0.6.0 were published the same day so they are assumed
   to share the 0.6.x toolchain.
2. Dawn-node default device limits without `requiredLimits` were not
   separately captured (the probe requested raised limits); assumed spec
   defaults per the requestDevice algorithm.
3. Whether Dawn-node quantises `timestamp-query` results (Chromium does, to
   100 us). Needs a probe with `requiredFeatures: ["timestamp-query"]`.
4. Firefox (wgpu) and Safari 26 exposure of `subgroups`, `shader-f16`,
   `timestamp-query`, and their default/adapter limits: no primary source
   found in this session; both are "probe at runtime" anyway.
5. GitHub-hosted GPU runner plan eligibility (Team vs Enterprise Cloud) was
   not stated on the pages fetched; the spec, price and setup path were.
6. lavapipe installation and Dawn-node operation on a GitHub `ubuntu-latest`
   runner has not been executed; the local container's lavapipe run is the
   evidence.
7. The spec sentence that an over-limit indirect dispatch "does nothing" came
   from a summarised fetch of the spec; re-read
   https://gpuweb.github.io/gpuweb/#dom-gpucomputepassencoder-dispatchworkgroupsindirect
   before relying on it (the safe design -- clamp in the finalise kernel --
   does not depend on it).
8. Whether Node's `worker_threads` pool (`pool: "threads"`) works with the
   Dawn addon is untested; `forks` is verified by graph-format's suite.
9. `chromium-experimental-subgroup-matrix` (present on Chromium 139 with
   `--enable-unsafe-webgpu`) was not explored; irrelevant to graph kernels.

---

## Sources

Local files (all paths absolute under the two project roots):

- /home/apowers/Projects/webgpu-graph-algorithms/HEADLESS_GPU_REPORT.md (flags, libEGL root cause, Chrome 145 powerPreference, appendix D)
- /home/apowers/Projects/webgpu-graph-algorithms/package.json, vitest.config.ts, vite.config.ts (scaffold state)
- /home/apowers/Projects/webgpu-graph-algorithms/test/setup/node.ts, browser.ts, webgpu-global.ts; test/helpers/webgpu.ts, test-utils.ts
- /home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/package.json (webgpu ^0.4.0, vitest ^3.2.4, @webgpu/types ^0.1.72)
- /home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/vitest.config.ts (pool forks, environment node)
- /home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/test/audit/gpu-upload.test.ts (acquire() lines 38-63, header lines 1-17)
- /home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/test/audit/gpu-contract.test.ts (device-less contract audit)
- /home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/node_modules/webgpu/{package.json,index.js,types.d.ts,README.md} (0.4.0 as installed)
- /home/apowers/Projects/graphty-monorepo/tmp/graph-format-design/09-webgpu-requirements.md (probed Chromium limits table, Dawn source quotes, WGSL type rules)
- /home/apowers/Projects/graphty-monorepo/package.json (vitest ^3.2.4, @vitest/browser ^3.2.4, playwright ^1.54.1, vite ^7, pnpm overrides for @vitest/browser 3.2.7 / 4.1.10)
- /home/apowers/Projects/graphty-monorepo/vitest.shared.config.ts (happy-dom default env, thresholds)
- /home/apowers/Projects/graphty-monorepo/graphty-element/vitest.config.ts and algorithms/vitest.config.ts (3.x `instances: [{ browser: "chromium" }]`, fileParallelism false)
- /home/apowers/Projects/graphty-monorepo/.github/workflows/ci.yml (shard matrix lines 236-350, playwright install line 423)
- /home/apowers/Projects/graphty-monorepo/graphty/src/vite-env.d.ts (vite/client reference)
- /home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/repos/cuda-ffi/.github/workflows/build.yml (self-hosted GPU runner pattern)
- Probe scripts written and run today: /home/apowers/Projects/webgpu-graph-algorithms/tmp/webgpu-plan/probe/dawn-probe.mjs, dawn-select.mjs, dawn-list.mjs, dawn-list2.mjs, dawn-perf.mjs, dawn-latency.mjs, chromium-latency.mjs
- npm tarballs webgpu-0.4.0.tgz and webgpu-0.6.1.tgz (symbol-version inspection, scratchpad)

URLs:

- https://registry.npmjs.org/webgpu (version/date metadata)
- https://www.npmjs.com/package/webgpu and https://github.com/dawn-gpu/node-webgpu (package home)
- https://github.com/dawn-gpu/node-webgpu/issues (issue #15, glibc / Ubuntu 20.04)
- https://dawn.googlesource.com/dawn/+/refs/heads/main/src/dawn/node/README.md (create() options, VK_ICD_FILENAMES for SwiftShader/lavapipe, "WIP" status)
- https://dawn.googlesource.com/dawn/+/refs/heads/main/src/dawn/native/Toggles.cpp (Dawn toggle names)
- https://gpuweb.github.io/gpuweb/ (WebGPU spec: limits, requestDevice, dispatchWorkgroupsIndirect, device lost, error scopes)
- https://gpuweb.github.io/gpuweb/wgsl/ (WGSL spec: atomics 6.2.8, recursion 11.4, override 7.2.2 / 12.15, uniform layout 14.4.5, bool 6.5.2, subgroups 17.12)
- https://github.com/gpuweb/gpuweb/wiki/Implementation-Status (browser matrix, last updated 2026-08-13)
- https://mozillagfx.wordpress.com/2025/07/15/shipping-webgpu-on-windows-in-firefox-141/
- https://webkit.org/blog/17333/webkit-features-in-safari-26-0/
- https://developer.chrome.com/blog/new-in-webgpu-120 (shader-f16, maxStorageBuffersPerShaderStage 10, timestamp quantisation)
- https://developer.chrome.com/blog/new-in-webgpu-121 (timestamp-query shipped, Android)
- https://developer.chrome.com/blog/new-in-webgpu-128 (adapter.info in 127, requestAdapterInfo removed in 131, subgroups origin trial)
- https://developer.chrome.com/blog/new-in-webgpu-134 and https://github.com/mdn/content/issues/38416 (subgroups shipped in 134)
- https://vite.dev/guide/assets (?raw imports, vite/client types)
- Vitest docs via context7 (/vitest-dev/vitest): docs/blog/vitest-3.md (instances with per-instance launch), docs/blog/vitest-4.md and docs/config/browser/playwright.md (@vitest/browser-playwright, launchOptions, non-merging per-instance provider, launch.headless ignored)
- https://github.com/atoms-org/cuda-ffi (owner-supplied example; workflow read from the clone)
- https://github.blog/changelog/2024-07-08-github-actions-gpu-hosted-runners-are-now-generally-available/
- https://docs.github.com/en/actions/reference/runners/larger-runners (T4 runner spec)
- https://docs.github.com/en/billing/reference/actions-runner-pricing (linux_4_core_gpu $0.052/min)
- https://runs-on.com/runners/gpu/ , https://machine.dev/ (third-party GPU runners, not evaluated)
