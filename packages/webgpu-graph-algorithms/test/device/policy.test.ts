/**
 * Pins the ONE copy of the adapter policy (scripts/gpu-policy.js; spec 2.3, 11.2, D19) and the ONE copy of the
 * runner-class rule (scripts/runner-class.js; spec 10.4, 11.7) as pure functions, and keeps the script-side
 * isSoftwareInfo equal to the src-side isSoftwareAdapter on a six-row table (contract 6.5, 6.9, 5.5). No device.
 */

import { checkAdapter, isSoftwareInfo, parseGpuRequire } from "../../scripts/gpu-policy.js";
import { runnerClass } from "../../scripts/runner-class.js";
import { isSoftwareAdapter } from "../../src/device/acquire.js";
import { type AdapterInfoLike } from "../../src/types/context.js";

const ANY = parseGpuRequire("any");
const HARDWARE = parseGpuRequire("hardware");
const NVIDIA = parseGpuRequire("nvidia");
const SKIP = parseGpuRequire(undefined);

const LLVMPIPE = { vendor: "mesa", architecture: "software" };
const SWIFTSHADER = { vendor: "google", architecture: "swiftshader", isFallbackAdapter: true };
/** An NVIDIA info WITHOUT the fallback field: Dawn-node 0.4.0 reports the boolean too, but the Node policy never reads it. */
const NVIDIA_NODE = { vendor: "nvidia", architecture: "lovelace" };
const NVIDIA_CHROMIUM = { vendor: "nvidia", architecture: "lovelace", isFallbackAdapter: false };
const INTEL_CHROMIUM = { vendor: "intel", architecture: "gen-12lp", isFallbackAdapter: false };

describe("parseGpuRequire (spec 11.2 table; D19)", () => {
    it("unset and empty are skip", () => {
        expect(parseGpuRequire(undefined)).toEqual({ level: "skip", vendor: null, raw: "" });
        expect(parseGpuRequire("")).toEqual({ level: "skip", vendor: null, raw: "" });
    });

    it("any and hardware are the two named levels", () => {
        expect(parseGpuRequire("any")).toEqual({ level: "any", vendor: null, raw: "any" });
        expect(parseGpuRequire("hardware")).toEqual({ level: "hardware", vendor: null, raw: "hardware" });
    });

    it("any other string is a vendor requirement, lower-cased, with the raw value kept", () => {
        expect(parseGpuRequire("nvidia")).toEqual({ level: "vendor", vendor: "nvidia", raw: "nvidia" });
        expect(parseGpuRequire("intel")).toEqual({ level: "vendor", vendor: "intel", raw: "intel" });
        expect(parseGpuRequire("NVIDIA")).toEqual({ level: "vendor", vendor: "nvidia", raw: "NVIDIA" });
    });
});

describe("checkAdapter (spec 11.2: a wrong adapter is never a skip)", () => {
    it("no adapter: skip only under the unset policy, a failure under any / hardware / vendor", () => {
        expect(checkAdapter(null, SKIP)).toMatchObject({ ok: false, skip: true });
        expect(typeof checkAdapter(null, SKIP).reason).toBe("string");
        for (const policy of [ANY, HARDWARE, NVIDIA]) {
            const verdict = checkAdapter(null, policy);
            expect(verdict.ok, policy.raw).toBe(false);
            expect(verdict.skip, policy.raw).toBe(false);
            expect(typeof verdict.reason, policy.raw).toBe("string");
        }
    });

    it("a software adapter is ok under skip and any, not ok under hardware", () => {
        expect(checkAdapter(LLVMPIPE, SKIP)).toEqual({ ok: true, skip: false, reason: null });
        expect(checkAdapter(LLVMPIPE, ANY)).toEqual({ ok: true, skip: false, reason: null });
        expect(checkAdapter(SWIFTSHADER, ANY, { browser: true })).toEqual({ ok: true, skip: false, reason: null });
        const verdict = checkAdapter(LLVMPIPE, HARDWARE);
        expect(verdict.ok).toBe(false);
        expect(verdict.skip).toBe(false);
        expect(typeof verdict.reason).toBe("string");
    });

    it("hardware accepts a hardware adapter", () => {
        expect(checkAdapter(NVIDIA_NODE, HARDWARE)).toEqual({ ok: true, skip: false, reason: null });
        expect(checkAdapter(INTEL_CHROMIUM, HARDWARE, { browser: true })).toEqual({
            ok: true,
            skip: false,
            reason: null,
        });
    });

    it("a vendor policy needs the vendor and, in the browser, isFallbackAdapter === false", () => {
        expect(checkAdapter(NVIDIA_NODE, NVIDIA)).toEqual({ ok: true, skip: false, reason: null });
        expect(checkAdapter(NVIDIA_CHROMIUM, NVIDIA, { browser: true })).toEqual({
            ok: true,
            skip: false,
            reason: null,
        });
        // vendor mismatch (a silent lavapipe / SwiftShader run under the GPU lane's policy is red)
        expect(checkAdapter(LLVMPIPE, NVIDIA).ok).toBe(false);
        expect(checkAdapter(SWIFTSHADER, NVIDIA, { browser: true }).ok).toBe(false);
        expect(checkAdapter(INTEL_CHROMIUM, NVIDIA, { browser: true }).ok).toBe(false);
        // the browser additionally needs an explicit isFallbackAdapter === false
        expect(checkAdapter({ ...NVIDIA_NODE, isFallbackAdapter: true }, NVIDIA, { browser: true }).ok).toBe(false);
        expect(checkAdapter(NVIDIA_NODE, NVIDIA, { browser: true }).ok).toBe(false);
        // outside the browser the field is not consulted (the Node lane keys on the vendor alone), so an info
        // without it passes; Dawn-node 0.4.0 does report the boolean (contract correction 5)
        expect(checkAdapter(NVIDIA_NODE, NVIDIA, { browser: false }).ok).toBe(true);
        expect(checkAdapter(NVIDIA_NODE, NVIDIA, undefined).ok).toBe(true);
        for (const verdict of [checkAdapter(LLVMPIPE, NVIDIA), checkAdapter(NVIDIA_NODE, NVIDIA, { browser: true })]) {
            expect(verdict.skip).toBe(false);
            expect(typeof verdict.reason).toBe("string");
        }
    });
});

describe("isSoftwareInfo (scripts) agrees with isSoftwareAdapter (src) -- the one-copy rule of spec 2.3", () => {
    const base = { device: "", description: "" };
    const rows: readonly [string, AdapterInfoLike, boolean][] = [
        ["Dawn-node llvmpipe", { ...base, ...LLVMPIPE }, true],
        ["Chromium SwiftShader", { ...base, ...SWIFTSHADER }, true],
        ["an NVIDIA info without the fallback field", { ...base, ...NVIDIA_NODE }, false],
        ["Dawn-node or Chromium NVIDIA (isFallbackAdapter false)", { ...base, ...NVIDIA_CHROMIUM }, false],
        ["a hardware architecture flagged as fallback", { ...base, ...NVIDIA_NODE, isFallbackAdapter: true }, true],
        ["Dawn null backend", { ...base, vendor: "", architecture: "", device: "null-backend" }, false],
    ];
    for (const [name, info, expected] of rows) {
        it(`${name} -> ${expected}`, () => {
            expect(isSoftwareAdapter(info)).toBe(expected);
            expect(isSoftwareInfo(info)).toBe(expected);
        });
    }
});

describe("runnerClass (contract 6.9: <vendor>-<architecture>-driver<major>, lower-cased, [^A-Za-z0-9_.-] -> _)", () => {
    // Dawn-node on the dev box reports exactly these strings (research note 05 section 2.4); the driver
    // version is the first run of digits in the description
    const DAWN_NVIDIA = { vendor: "nvidia", architecture: "lovelace", description: "NVIDIA: 580.173.02 580.173.2.0" };
    const rows: readonly [string, { vendor: string; architecture: string; description: string }, string][] = [
        ["Dawn-node NVIDIA on the dev box", DAWN_NVIDIA, "nvidia-lovelace-driver580"],
        [
            "the contract 6.6 illustration",
            { vendor: "nvidia", architecture: "ada-lovelace", description: "NVIDIA: 580.173.02 580.173.2.0" },
            "nvidia-ada-lovelace-driver580",
        ],
        [
            "lavapipe carries Mesa's version",
            { vendor: "mesa", architecture: "software", description: "Mesa 23.2.1-1ubuntu3.1~22.04.3 llvmpipe" },
            "mesa-software-driver23",
        ],
        [
            "a description with no digits",
            { vendor: "google", architecture: "swiftshader", description: "SwiftShader" },
            "google-swiftshader-driver0",
        ],
        [
            "a vendor with a space and an upper-case architecture",
            { vendor: "Some Vendor", architecture: "Gen 12", description: "v 7.1" },
            "some_vendor-gen_12-driver7",
        ],
    ];
    for (const [name, info, expected] of rows) {
        it(`${name} -> ${expected}`, () => {
            expect(runnerClass(info, {})).toBe(expected);
            expect(runnerClass(info, { GRAPHTY_RUNNER_CLASS: "" })).toBe(expected);
            expect(runnerClass(info, { GRAPHTY_RUNNER_CLASS: undefined })).toBe(expected);
        });
    }

    it("GRAPHTY_RUNNER_CLASS overrides the derived class verbatim (the T4 lane fixes gpu-linux-t4, contract 6.4)", () => {
        expect(runnerClass(DAWN_NVIDIA, { GRAPHTY_RUNNER_CLASS: "gpu-linux-t4" })).toBe("gpu-linux-t4");
        expect(runnerClass(DAWN_NVIDIA, { GRAPHTY_RUNNER_CLASS: "My Runner" })).toBe("My Runner");
    });
});
