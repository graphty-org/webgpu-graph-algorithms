/**
 * The browser project's setup (spec 2.3, 11.6, 5.7; contract 5.1): the policy forwarded by vitest.config.ts
 * through import.meta.env (Chromium has no process.env), requireBrowserGpu(t), acquireBrowser() over the
 * ./browser entry's requestGpuContext with the same uncaptured-error hook as the Node setup, browserScale().
 * afterEach awaits onSubmittedWorkDone() on every live context so asynchronously delivered uncapturederror
 * events land, then fails the test when the sink or a context's pending-error slot holds an error; afterAll
 * disposes every context.
 */

import { afterAll, afterEach, type TestContext } from "vitest";

import { checkAdapter, type GpuPolicy, parseGpuRequire } from "../../scripts/gpu-policy.js";
import { type BrowserGpuOptions, requestGpuContext } from "../../src/browser/index.js";
import type { GpuContext } from "../../src/context.js";
import type { WebGpuGraphError } from "../../src/errors.js";

const contexts: GpuContext[] = [];
const uncaptured: WebGpuGraphError[] = [];

/** The policy forwarded by vitest.config.ts. */
export function browserPolicy(): GpuPolicy {
    return parseGpuRequire(import.meta.env.GRAPHTY_GPU_REQUIRE);
}

/** "nvidia" | "swiftshader" as forwarded. */
export function browserGpu(): "nvidia" | "swiftshader" {
    return import.meta.env.GRAPHTY_BROWSER_GPU === "nvidia" ? "nvidia" : "swiftshader";
}

/** navigator.gpu or undefined (never throws). */
export function browserWebGpu(): GPU | undefined {
    if (typeof navigator === "undefined") {
        return undefined;
    }
    const candidate: { readonly gpu?: GPU | undefined } = navigator;
    return candidate.gpu;
}

/**
 * Skips (policy "skip") or fails (any / hardware / vendor) when navigator.gpu or an adapter is absent or the adapter
 * violates the policy; in the browser a vendor policy additionally requires isFallbackAdapter === false (spec 11.2).
 */
export async function requireBrowserGpu(t: TestContext): Promise<void> {
    const policy = browserPolicy();
    const gpu = browserWebGpu();
    let info: GPUAdapterInfo | null = null;
    let absent = "navigator.gpu is absent (the Chromium flags of spec 12.2 are missing, or this is not Chromium)";
    if (gpu !== undefined) {
        const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
        if (adapter === null) {
            absent = "requestAdapter() returned null";
        } else {
            ({ info } = adapter);
        }
    }
    const verdict = checkAdapter(info, policy, { browser: true });
    if (verdict.ok) {
        return;
    }
    const reason = info === null ? `E_NO_ADAPTER: ${absent}` : (verdict.reason ?? "adapter policy violated");
    if (verdict.skip) {
        t.skip(reason);
    }
    throw new Error(`GRAPHTY_GPU_REQUIRE=${policy.raw}: ${reason}`);
}

/**
 * A fresh context through requestGpuContext (P1) with the same uncaptured-error hook as the Node setup (installed
 * only when the caller gives no onError of its own); disposed in afterAll.
 */
export async function acquireBrowser(options: BrowserGpuOptions = {}): Promise<GpuContext> {
    const ctx = await requestGpuContext({
        ...options,
        label: options.label ?? "acquireBrowser",
        onError:
            options.onError ??
            ((error): void => {
                uncaptured.push(error);
            }),
    });
    contexts.push(ctx);
    return ctx;
}

/** The browser's gpuScale(): 1 on nvidia, 1 / 50 on swiftshader. */
export function browserScale(): number {
    return browserGpu() === "nvidia" ? 1 : 1 / 50;
}

afterEach(async () => {
    for (const ctx of contexts) {
        if (ctx.state === "ready") {
            await ctx.device.queue.onSubmittedWorkDone();
            const pending = ctx.takePendingError();
            if (pending !== null) {
                uncaptured.push(pending);
            }
        }
    }
    const errors = uncaptured.splice(0);
    if (errors.length > 0) {
        throw new Error(
            `uncaptured GPU errors during the test (spec 5.7: browsers deliver them asynchronously):\n${errors.map((e) => `${e.code}: ${e.message}`).join("\n")}`,
        );
    }
});

afterAll(() => {
    for (const ctx of contexts.splice(0)) {
        ctx.dispose();
    }
});
