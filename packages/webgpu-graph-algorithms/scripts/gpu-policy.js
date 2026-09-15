/**
 * The ONE copy of the adapter policy (spec 2.3, 11.2, D19; contract 6.5): GRAPHTY_GPU_REQUIRE parsed by
 * parseGpuRequire(), the software test isSoftwareInfo() (a copy of src/device/acquire.ts isSoftwareAdapter,
 * kept equal by test/device/policy.test.ts) and the verdict checkAdapter(). Imported by test/setup/gpu.ts,
 * test/setup/browser.ts (through vitest.config.ts's forwarding) and scripts/gpu-report.js, so the vendor-match
 * rule has one copy. Plain ESM JavaScript (package "type": "module"); the declarations live in gpu-policy.d.ts.
 *
 * Policy levels:
 *   unset / ""       -> "skip"     (no adapter -> the test skips with the reason; a present adapter always passes)
 *   "any"            -> "any"      (an adapter must exist; lavapipe / SwiftShader count)
 *   "hardware"       -> "hardware" (additionally !isSoftwareInfo(info))
 *   any other string -> "vendor"   (additionally info.vendor === value, lower-cased; in the browser also
 *                                   isFallbackAdapter === false)
 */

/**
 * Parse GRAPHTY_GPU_REQUIRE.
 * @param {string | undefined} value - the raw environment value
 * @returns {{ level: "skip" | "any" | "hardware" | "vendor", vendor: string | null, raw: string }} the policy
 */
export function parseGpuRequire(value) {
    const raw = value ?? "";
    const normalized = raw.trim().toLowerCase();
    if (normalized === "") {
        return Object.freeze({ level: "skip", vendor: null, raw });
    }
    if (normalized === "any") {
        return Object.freeze({ level: "any", vendor: null, raw });
    }
    if (normalized === "hardware") {
        return Object.freeze({ level: "hardware", vendor: null, raw });
    }
    return Object.freeze({ level: "vendor", vendor: normalized, raw });
}

/**
 * Software adapter test: architecture "software" (Dawn on Mesa llvmpipe) or "swiftshader" (Chromium's CPU
 * Vulkan), or isFallbackAdapter === true (Chromium marks SwiftShader; Dawn marks llvmpipe).
 * @param {{ vendor: string, architecture: string, isFallbackAdapter?: boolean | undefined }} info - adapter info
 * @returns {boolean} true for a software adapter
 */
export function isSoftwareInfo(info) {
    return info.architecture === "software" || info.architecture === "swiftshader" || info.isFallbackAdapter === true;
}

/**
 * The policy verdict for an adapter (null = no adapter).
 * @param {{ vendor: string, architecture: string, isFallbackAdapter?: boolean | undefined } | null} info - adapter info
 * @param {{ level: "skip" | "any" | "hardware" | "vendor", vendor: string | null, raw: string }} policy - parsed policy
 * @param {{ browser?: boolean | undefined } | undefined} [options] - browser: apply the isFallbackAdapter clause
 * @returns {{ ok: boolean, skip: boolean, reason: string | null }} the verdict
 */
export function checkAdapter(info, policy, options) {
    const browser = options?.browser === true;
    if (info === null) {
        if (policy.level === "skip") {
            return Object.freeze({
                ok: false,
                skip: true,
                reason: "E_NO_ADAPTER: no WebGPU adapter (GRAPHTY_GPU_REQUIRE unset: skipping)",
            });
        }
        return Object.freeze({
            ok: false,
            skip: false,
            reason: `E_NO_ADAPTER: no WebGPU adapter but GRAPHTY_GPU_REQUIRE=${policy.raw} requires one`,
        });
    }
    const label = `${info.vendor}/${info.architecture}`;
    switch (policy.level) {
        case "skip":
        case "any":
            return Object.freeze({ ok: true, skip: false, reason: null });
        case "hardware":
            if (isSoftwareInfo(info)) {
                return Object.freeze({
                    ok: false,
                    skip: false,
                    reason: `E_SOFTWARE_ONLY: adapter ${label} is a software adapter but GRAPHTY_GPU_REQUIRE=hardware`,
                });
            }
            return Object.freeze({ ok: true, skip: false, reason: null });
        case "vendor":
            if (info.vendor !== policy.vendor) {
                return Object.freeze({
                    ok: false,
                    skip: false,
                    reason:
                        `E_NO_ADAPTER: adapter vendor "${info.vendor}" (${label}) ` +
                        `does not match GRAPHTY_GPU_REQUIRE=${policy.raw}`,
                });
            }
            if (browser && info.isFallbackAdapter !== false) {
                return Object.freeze({
                    ok: false,
                    skip: false,
                    reason:
                        `E_SOFTWARE_ONLY: adapter ${label} reports ` +
                        `isFallbackAdapter=${String(info.isFallbackAdapter)} ` +
                        `in the browser under GRAPHTY_GPU_REQUIRE=${policy.raw}`,
                });
            }
            return Object.freeze({ ok: true, skip: false, reason: null });
        default:
            return Object.freeze({
                ok: false,
                skip: false,
                reason: `E_INVALID_ARGUMENT: unknown policy level ${String(policy.level)}`,
            });
    }
}
