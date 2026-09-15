# Headless Chromium + NVIDIA GPU for WebGPU Tests

Date: 2026-09-13
Status: root cause found, working recipe verified, nothing wired into the project yet

## TL;DR

- Headless Playwright Chromium on this dev box was silently running WebGPU on
  SwiftShader (Chrome's CPU renderer). Every existing test would pass on it.
- The blocker is local, not upstream: this dev environment is a Docker
  container and the image is missing the `libegl1` package. The NVIDIA Vulkan
  driver dlopen()s `libEGL.so.1` during init and returns
  VK_ERROR_INITIALIZATION_FAILED without it, so Chrome's WebGPU backend (Dawn)
  never sees the RTX 4070 SUPER.
- With `libEGL.so.1` on the library path and four Chromium flags, headless
  Chromium 139 (Playwright build 1181) uses the real GPU. Verified with a
  compute shader and `adapter.info` reporting vendor `nvidia`, architecture
  `lovelace`.
- Required flags (all four, no more):

  ```
  --enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan --disable-vulkan-surface
  ```

## Environment

| Item | Value |
|---|---|
| Host GPU | NVIDIA GeForce RTX 4070 SUPER |
| Driver (kernel + userspace) | 580.173.02 |
| Container | Docker, Ubuntu 22.04.5, glibc 2.35, hostname dev.ato.ms |
| NVIDIA container toolkit env | NVIDIA_VISIBLE_DEVICES=0, NVIDIA_DRIVER_CAPABILITIES=all |
| Vulkan loader in container | libvulkan1 1.3.204.1 (jammy) |
| Vulkan ICDs present | /etc/vulkan/icd.d/nvidia_icd.json (mounted from host) plus mesa lvp/radeon/intel/virtio |
| Display | none (no DISPLAY, no Xvfb installed) |
| Playwright (project) | 1.54.1, wants chromium build 1181 (Chromium 139) |
| Other browsers | /usr/bin/google-chrome 145.0.7632.159 |
| Vitest | 2.1.9 with @vitest/browser 2.1.9 |
| CUDA in container | works (cuInit ok, device enumerated) |

Note: the driver libraries injected by the container toolkit show nlink=0 and
appear as "(deleted)" in /proc/self/maps. That is cosmetic (host files were
replaced after the container started) and did not affect anything.

## Research summary

### Playwright and Chromium

- Playwright does not inject any SwiftShader or `--use-angle` flags on Linux
  (checked `playwright-core/lib/server/chromium/chromiumSwitches.js`).
  Puppeteer does inject `--use-angle=swiftshader-webgl`; that gotcha does not
  apply to us.
- Playwright issues asking for GPU in headless mode were closed without a fix:
  - https://github.com/microsoft/playwright/issues/11627 (closed, not planned)
  - https://github.com/microsoft/playwright/issues/15533 (closed, no resolution)
- Chromium tracker item "New Headless Chrome on Linux via Puppeteer does not
  use GPU": https://issues.chromium.org/issues/40274484 (login required to
  read; reported workaround is `--use-angle=vulkan`, which matches what worked
  here). Related: https://issues.chromium.org/issues/40540071 "Support GPU
  hardware in headless mode".
- Chrome team guidance for headless WebGPU on NVIDIA Linux:
  - https://developer.chrome.com/blog/supercharge-web-ai-testing
  - https://github.com/jasonmayes/headless-chrome-nvidia-t4-gpu-support
  - Recommended flags: `--headless=new --no-sandbox --use-angle=vulkan
    --enable-features=Vulkan --disable-vulkan-surface --enable-unsafe-webgpu`
  - Caveat from that repo: with `--disable-vulkan-surface`, WebGPU works for
    compute but not for drawing to a canvas. Fine for this project.
- https://tigerabrodi.blog/how-to-get-webgpu-in-headless-chrome-on-cloud-gpus
  claims Dawn blocklists NVIDIA drivers 570+ and suggests
  `--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist` plus an
  explicit `powerPreference`. In our tests the Dawn flags made no difference on
  Chromium 139. The explicit powerPreference detail did matter for Chrome 145
  (see results).
- Chrome troubleshooting doc:
  https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips

### NVIDIA Vulkan in containers

The loader error we hit:

```
[Vulkan Loader] ERROR: loader_scanned_icd_add: Could not get 'vkCreateInstance'
  via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0
[Vulkan Loader] ERROR | DRIVER: vkCreateInstance: Found no drivers!
```

Reports of the same error:

- https://github.com/NVIDIA/nvidia-container-toolkit/issues/1952 (closed;
  fix was NVIDIA_DRIVER_CAPABILITIES=all and deduping ICD manifests; both
  already true for us)
- https://github.com/NVIDIA/egl-wayland/issues/97 (ICD returns NULL when the
  driver is "not fully initialised")
- https://github.com/NVIDIA/nvidia-container-toolkit/issues/1124 (missing
  libGLX/ICD files in RHEL containers)
- https://bugzilla.redhat.com/show_bug.cgi?id=2011452 (closed NOTABUG)
- https://forums.developer.nvidia.com/t/vulkan-ssh/46188 (user blamed X11)

None of them name the missing EGL library. We found it by tracing the driver's
file opens (appendix C).

## Experiment log

### Step 1: baseline

`npm run test:run` could not launch: Playwright chromium build 1181 was not
installed. `npx playwright install chromium` fixed that. The vitest config also
passes a `launch` key that Vitest 2.1 rejects (typecheck error), so the flags
in vitest.config.ts were never reaching the browser anyway.

### Step 2: flag matrix without libEGL

Probe script: appendix A. Page served via `page.route` on http://localhost so
it is a secure context. Each config in its own process with a hard timeout.

| Browser | Flags | Result |
|---|---|---|
| headless_shell 139 | none | SwiftShader |
| headless_shell 139 | `--enable-unsafe-webgpu --ignore-gpu-blocklist` (+/- Dawn flags) | SwiftShader |
| full Chromium 139 | none / unsafe / unsafe+ignore / +Dawn / `--use-angle=vulkan` alone | SwiftShader |
| full Chromium 139 | anything with `--enable-features=Vulkan` | GPU process fails to init Skia, Playwright hangs at newPage() |
| Chrome 145 | unsafe+ignore (+/- Dawn) | SwiftShader |
| any | `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` | requestAdapter() returns null, GPU process exits |

GPU process log with `--enable-features=Vulkan` and no libEGL:

```
ERROR:gpu/vulkan/vulkan_device_queue.cc:297] samplerYcbcrConversion is not supported.
ERROR:gpu/command_buffer/service/shared_context_state.cc:569] OOP raster support disabled: GrContext creation failed.
ERROR:gpu/ipc/service/gpu_channel_manager.cc:1000] ContextResult::kFatalFailure: Failed to initialize Skia for SharedContextState
```

(The NVIDIA ICD failed, so the loader picked Mesa lavapipe, which lacks that
feature.)

### Step 3: isolate the driver

Direct ctypes test of the ICD (appendix B):

```
dlopen ok
negotiate rc=-3 version=7        # VK_ERROR_INITIALIZATION_FAILED
vkCreateInstance -> NULL
```

LD_PRELOAD tracer (appendix C) showed, between "before negotiate" and the
failure:

```
[trace] dlopen(libEGL.so.1) -> FAIL
```

The container has `libGLX.so.0`, `libGL.so.1`, `libGLdispatch.so.0`, and
`libEGL_nvidia.so.0`, but not glvnd's `libEGL.so.1` (package `libegl1`).

### Step 4: supply libEGL.so.1 (no root)

```
apt-get download libegl1 libglvnd0
dpkg -x libegl1_1.4.0-1_amd64.deb root
LD_LIBRARY_PATH=$PWD/root/usr/lib/x86_64-linux-gnu python3 icd-check.py
# negotiate rc=0, vkCreateInstance -> 0x7f...
```

### Step 5: flag matrix with libEGL on LD_LIBRARY_PATH

| Browser | Flags | Adapter |
|---|---|---|
| full Chromium 139 | `--enable-unsafe-webgpu` | SwiftShader |
| full Chromium 139 | `--enable-unsafe-webgpu --enable-features=Vulkan` | SwiftShader |
| full Chromium 139 | `--enable-unsafe-webgpu --use-angle=vulkan` | SwiftShader |
| full Chromium 139 | `--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan` | SwiftShader |
| full Chromium 139 | `--enable-features=Vulkan --use-angle=vulkan --disable-vulkan-surface` (no unsafe) | null |
| full Chromium 139 | all four flags | NVIDIA lovelace, compute ok |
| full Chromium 139 | all four + `--ignore-gpu-blocklist` (+/- Dawn flags) | NVIDIA lovelace, compute ok |
| headless_shell 139 | all four + ignore | NVIDIA lovelace, compute ok |
| Chrome 145 | all four + ignore (+/- Dawn flags) | default request: null; `high-performance` / `low-power`: NVIDIA lovelace |

Adapter details on NVIDIA: `isFallbackAdapter=false`,
`maxStorageBufferBindingSize=4294967292` (SwiftShader reports 1073741824).
The 1024-element compute shader returned correct values.

Playwright launch that works:

```js
await chromium.launch({
    headless: true,
    args: [
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan",
        "--use-angle=vulkan",
        "--disable-vulkan-surface",
    ],
    env: { ...process.env, LD_LIBRARY_PATH: "<dir containing libEGL.so.1>" },
});
```

## Recommendations

1. Fix the container, not the tests. Add `libegl1` to the dev container image
   (or the `docker run` setup). That removes the LD_LIBRARY_PATH hack entirely.
   Until then, extract libegl1 into `./tmp/egl` and set LD_LIBRARY_PATH in the
   Vitest config.
2. Put the four flags into vitest.config.ts via `browser.providerOptions`
   (Vitest 2.1 rejects a top-level `launch` key):

   ```ts
   browser: {
       enabled: true,
       name: "chromium",
       provider: "playwright",
       headless: true,
       providerOptions: {
           launch: {
               args: [
                   "--enable-unsafe-webgpu",
                   "--enable-features=Vulkan",
                   "--use-angle=vulkan",
                   "--disable-vulkan-surface",
               ],
           },
       },
   },
   ```

   Remove the existing `--use-gl=swiftshader` and `--use-vulkan=swiftshader`
   flags; they explicitly force software rendering.
3. Make the WebGPU check test assert `adapter.info.vendor === "nvidia"` (or at
   least `isFallbackAdapter === false`) so a SwiftShader fallback fails loudly.
   The current test passes identically on either backend. Replace the removed
   `requestAdapterInfo()` call in test/setup/webgpu-global.ts with
   `adapter.info`.
4. Keep `requestAdapter({ powerPreference: "high-performance" })` in the
   library's device setup. It costs nothing and is the only form that returned
   the NVIDIA adapter on Chrome 145.
5. CI (.github/workflows/test.yml) runs on GitHub-hosted runners with no GPU.
   Either accept SwiftShader there (and gate the vendor assertion on an env
   var like `REQUIRE_GPU=1` for local runs) or move to a self-hosted GPU
   runner. Decide before adding the assertion in step 3.
6. Do not chase the Dawn blocklist flags or `--ignore-gpu-blocklist` for
   Chromium 139; they were not needed.

## Open questions

- Chrome 145 returning null for a default-preference `requestAdapter()` was not
  diagnosed. Playwright's bundled Chromium 139 does not have this issue, so it
  only matters if the project moves to a newer browser build.
- Whether the same recipe holds once Playwright is upgraded past build 1181.
  Re-run the probe (appendix A) after any Playwright bump.

## Appendix A: Playwright probe (gpu-probe2.mjs)

Run from the project root so it resolves the project's Playwright:
`node gpu-probe2.mjs <configName>`.

```js
import { createRequire } from "node:module";
const require = createRequire("/home/apowers/Projects/webgpu-graph-algorithms/package.json");
const { chromium } = require("playwright");

const HOME = process.env.HOME;
const FULL = `${HOME}/.cache/ms-playwright/chromium-1181/chrome-linux/chrome`;
const SHELL = `${HOME}/.cache/ms-playwright/chromium_headless_shell-1181/chrome-linux/headless_shell`;
const EGL_PATH = { LD_LIBRARY_PATH: "/path/to/egl/root/usr/lib/x86_64-linux-gnu" };
const FLAGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan", "--disable-vulkan-surface"];

const configs = {
    "full:gpu": { exe: FULL, args: FLAGS, env: EGL_PATH },
    "shell:gpu": { exe: SHELL, args: FLAGS, env: EGL_PATH },
    "full:none": { exe: FULL, args: [] },
};

const PAGE = `<!doctype html><html><body>probe</body></html>`;

async function probe(page) {
    return page.evaluate(async () => {
        const out = { hasGpu: !!navigator.gpu, secure: window.isSecureContext };
        if (!navigator.gpu) return out;
        for (const pref of ["default", "high-performance", "low-power"]) {
            const opts = pref === "default" ? {} : { powerPreference: pref };
            try {
                const a = await navigator.gpu.requestAdapter(opts);
                if (!a) { out[pref] = null; continue; }
                const i = a.info;
                out[pref] = { vendor: i.vendor, architecture: i.architecture, fallback: a.isFallbackAdapter,
                              maxBuf: a.limits.maxStorageBufferBindingSize };
                if (pref === "default") {
                    const d = await a.requestDevice();
                    const m = d.createShaderModule({ code: `
                        @group(0) @binding(0) var<storage, read_write> o: array<u32>;
                        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) { o[id.x] = id.x * 2u; }` });
                    const n = 1024;
                    const buf = d.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
                    const st = d.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
                    const p = d.createComputePipeline({ layout: "auto", compute: { module: m, entryPoint: "main" } });
                    const bg = d.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
                    const e = d.createCommandEncoder();
                    const ps = e.beginComputePass(); ps.setPipeline(p); ps.setBindGroup(0, bg); ps.dispatchWorkgroups(n / 64); ps.end();
                    e.copyBufferToBuffer(buf, 0, st, 0, n * 4);
                    d.queue.submit([e.finish()]);
                    await st.mapAsync(GPUMapMode.READ);
                    const r = new Uint32Array(st.getMappedRange());
                    out.compute = (r[1] === 2 && r[1023] === 2046) ? "ok" : "WRONG " + r[1023];
                    st.unmap(); d.destroy();
                }
            } catch (err) { out[pref] = "ERR " + String(err); }
        }
        return out;
    });
}

const cfg = configs[process.argv[2]];
if (!cfg) { console.log("known:", Object.keys(configs).join(" ")); process.exit(2); }
let browser;
try {
    browser = await chromium.launch({ executablePath: cfg.exe, headless: true, args: cfg.args, env: { ...process.env, ...(cfg.env ?? {}) } });
    const page = await browser.newPage();
    await page.route("http://localhost/**", r => r.fulfill({ contentType: "text/html", body: PAGE }));
    await page.goto("http://localhost/probe");
    const res = await Promise.race([probe(page), new Promise((_, rej) => setTimeout(() => rej(new Error("probe timed out")), 25000))]);
    console.log(JSON.stringify(res));
} catch (err) {
    console.log("ERROR:", String(err).split("\n")[0]);
} finally {
    if (browser) await browser.close().catch(() => {});
}
```

Run with `DEBUG=pw:browser` to see GPU process stderr. Wrap in
`timeout -k 5 50` because a broken Vulkan setup hangs Playwright at newPage().

## Appendix B: direct ICD check (icd-check.py)

```python
import ctypes
lib = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0")
neg = lib.vk_icdNegotiateLoaderICDInterfaceVersion
neg.restype = ctypes.c_int
ver = ctypes.c_uint32(7)
print("negotiate rc=%d version=%d" % (neg(ctypes.byref(ver)), ver.value))   # rc=0 is good, -3 is init failure
gipa = lib.vk_icdGetInstanceProcAddr
gipa.restype = ctypes.c_void_p
gipa.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
p = gipa(None, b"vkCreateInstance")
print("vkCreateInstance ->", hex(p) if p else "NULL")
```

## Appendix C: LD_PRELOAD open/dlopen tracer (tracer.c)

strace is not installed in the container; this shim was enough.
Build: `gcc -shared -fPIC -O1 -o tracer.so tracer.c -ldl`.
Use: `LD_PRELOAD=./tracer.so python3 icd-check.py 2>&1 | grep '^\[trace\]'`.

```c
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void logf_(const char *fmt, ...) {
    char buf[1024]; va_list ap; va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    if (n > 0) { ssize_t r = write(2, buf, (size_t)n); (void)r; }
}

int open(const char *path, int flags, ...) {
    static int (*real)(const char *, int, ...);
    if (!real) real = dlsym(RTLD_NEXT, "open");
    mode_t mode = 0;
    if (flags & O_CREAT) { va_list ap; va_start(ap, flags); mode = va_arg(ap, mode_t); va_end(ap); }
    int fd = real(path, flags, mode); int e = errno;
    logf_("[trace] open(%s) -> %d %s\n", path, fd, fd < 0 ? strerror(e) : "");
    errno = e; return fd;
}

void *dlopen(const char *file, int flags) {
    static void *(*real)(const char *, int);
    if (!real) real = dlsym(RTLD_NEXT, "dlopen");
    void *h = real(file, flags);
    if (file) logf_("[trace] dlopen(%s) -> %s\n", file, h ? "ok" : "FAIL");
    return h;
}
```

## Appendix D: getting libEGL.so.1 without root

```
mkdir -p tmp/egl && cd tmp/egl
apt-get download libegl1 libglvnd0
dpkg -x libegl1_1.4.0-1_amd64.deb root
# libEGL.so.1 is now at root/usr/lib/x86_64-linux-gnu/
```
