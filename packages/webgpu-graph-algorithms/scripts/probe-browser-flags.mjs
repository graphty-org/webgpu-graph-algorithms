/**
 * Informational probe for the host lane (hosts.yml): launches Chromium once per candidate flag set and browser
 * build -- Playwright's headless shell (what the browser project runs), the full Chromium build in its new headless
 * mode (`channel: "chromium"`) and, on Windows, the full build headed (the runner has a desktop session) -- and
 * reports which adapter `navigator.gpu.requestAdapter()` grants under each of three option shapes (none, low-power,
 * forceFallbackAdapter), so a host whose browser smoke gets no adapter says which flags and build would work. Prints
 * one JSON line per (flag set, build, options) triple to stdout; never exits non-zero (a probe, not a gate).
 *
 *   node scripts/probe-browser-flags.mjs            # every candidate
 *   node scripts/probe-browser-flags.mjs metal      # only the sets whose name contains "metal"
 */
import { createServer } from "node:http";

import { chromium } from "playwright";

const CANDIDATES = {
    "unsafe-only": ["--enable-unsafe-webgpu"],
    "unsafe+ignore-blocklist": ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"],
    "swiftshader-angle": ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    "swiftshader-adapter": ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader"],
    "swiftshader-adapter+unsafe-swiftshader": [
        "--enable-unsafe-webgpu",
        "--enable-unsafe-swiftshader",
        "--use-webgpu-adapter=swiftshader",
    ],
    "swiftshader-vulkan": ["--enable-unsafe-webgpu", "--use-vulkan=swiftshader", "--enable-unsafe-swiftshader"],
    "swiftshader-all": [
        "--enable-unsafe-webgpu",
        "--ignore-gpu-blocklist",
        "--use-angle=swiftshader",
        "--use-vulkan=swiftshader",
        "--enable-unsafe-swiftshader",
        "--use-webgpu-adapter=swiftshader",
    ],
    "d3d11-adapter": ["--enable-unsafe-webgpu", "--use-webgpu-adapter=d3d11"],
    "metal-angle+ignore-blocklist": ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--use-angle=metal"],
    "metal-force-gpu": ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"],
};
const OPTIONS = [
    ["default", "undefined"],
    ["low-power", '{ powerPreference: "low-power" }'],
    ["fallback", "{ forceFallbackAdapter: true }"],
];
/** The browser builds: Playwright's headless shell, the full build headless (new headless mode), and headed on Windows. */
const BUILDS = [
    ["headless-shell", { headless: true }],
    ["chromium-headless", { headless: true, channel: "chromium" }],
    ...(process.platform === "win32" ? [["chromium-headed", { headless: false, channel: "chromium" }]] : []),
];
const filter = process.argv[2] ?? "";

// navigator.gpu exists only in a secure context: about:blank and data: URLs are not one under Playwright, so the
// probe page is served from a throwaway localhost server (localhost is always secure).
const server = createServer((_request, response) => {
    response.writeHead(200, { "Content-Type": "text/html" });
    response.end("<!doctype html><title>probe</title>");
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const { port } = server.address();

for (const [build, launch] of BUILDS) {
    for (const [name, args] of Object.entries(CANDIDATES)) {
        if (filter !== "" && !name.includes(filter)) {
            continue;
        }
        let browser = null;
        try {
            browser = await chromium.launch({ ...launch, args, timeout: 60_000 });
            const page = await browser.newPage();
            await page.goto(`http://127.0.0.1:${port}/`);
            for (const [label, expr] of OPTIONS) {
                const result = await page.evaluate(`(async () => {
                    if (!navigator.gpu) { return { adapter: "no navigator.gpu" }; }
                    const a = await navigator.gpu.requestAdapter(${expr});
                    if (a === null) { return { adapter: null }; }
                    const i = a.info;
                    return { adapter: { vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description, fallback: i.isFallbackAdapter, features: [...a.features].filter((f) => f === "subgroups" || f === "timestamp-query") } };
                })()`);
                console.log(
                    JSON.stringify({ build, version: browser.version(), flags: name, options: label, ...result }),
                );
            }
        } catch (err) {
            console.log(
                JSON.stringify({
                    build,
                    flags: name,
                    error: err instanceof Error ? err.message.split("\n")[0] : String(err),
                }),
            );
        } finally {
            await browser?.close();
        }
    }
}
server.close();
