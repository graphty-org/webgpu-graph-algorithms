/**
 * src/device/error-scope.ts (contract 3.4; spec 5.7, 5.1): withValidationScope, AllocationTracker and
 * formatCompilationInfo on RAW devices from acquireRaw(). The out-of-memory path is driven by a Proxy that
 * substitutes popErrorScope (a real OOM needs the node-limits project of P4); the proxy binds every other
 * member to the real device so Dawn's native `this` checks pass.
 */

import { OOM_SCOPE_THRESHOLD_BYTES } from "../../src/constants.js";
import { AllocationTracker, formatCompilationInfo, withValidationScope } from "../../src/device/error-scope.js";
import { BufferUsage, ShaderStage } from "../../src/device/webgpu-constants.js";
import { isWebGpuGraphError, type WebGpuGraphError, type WebGpuGraphErrorCode } from "../../src/errors.js";
import { acquireRaw, requireGpu } from "../setup/gpu.js";

type ScopeOverrides = Partial<{
    pushErrorScope: (filter: GPUErrorFilter) => void;
    popErrorScope: () => Promise<GPUError | null>;
}>;

/** A device whose pushErrorScope / popErrorScope are replaced; everything else is bound to the real device. */
function patched(device: GPUDevice, overrides: ScopeOverrides): GPUDevice {
    return new Proxy(device, {
        get(target, prop) {
            if (prop === "pushErrorScope" && overrides.pushErrorScope !== undefined) {
                return overrides.pushErrorScope;
            }
            if (prop === "popErrorScope" && overrides.popErrorScope !== undefined) {
                return overrides.popErrorScope;
            }
            const value = Reflect.get(target, prop) as unknown;
            return typeof value === "function" ? (value as (...args: unknown[]) => unknown).bind(target) : value;
        },
    });
}

function brokenBindGroup(device: GPUDevice, label: string): void {
    const layout = device.createBindGroupLayout({
        label: `${label}/layout`,
        entries: [{ binding: 0, visibility: ShaderStage.COMPUTE, buffer: { type: "storage", minBindingSize: 1024 } }],
    });
    const small = device.createBuffer({ label: `${label}/small`, size: 16, usage: BufferUsage.STORAGE });
    device.createBindGroup({ label, layout, entries: [{ binding: 0, resource: { buffer: small } }] });
    small.destroy();
}

async function rejection(promise: Promise<unknown>): Promise<unknown> {
    try {
        await promise;
    } catch (err) {
        return err;
    }
    throw new Error("expected the promise to reject");
}

function expectCode(err: unknown, code: WebGpuGraphErrorCode, details?: Record<string, unknown>): WebGpuGraphError {
    expect(isWebGpuGraphError(err), `expected a WebGpuGraphError, got ${String(err)}`).toBe(true);
    const error = err as WebGpuGraphError;
    expect(error.code).toBe(code);
    if (details !== undefined) {
        expect(error.details).toMatchObject(details);
    }
    return error;
}

describe("withValidationScope", () => {
    it("surfaces a bad bind group as E_VALIDATION { label, message } and keeps it out of the uncaptured path", async (t) => {
        requireGpu(t);
        const { device } = await acquireRaw();
        let uncapturedEvents = 0;
        device.addEventListener("uncapturederror", () => {
            uncapturedEvents += 1;
        });
        const err = await rejection(
            withValidationScope(device, "bad-bind-group", () => {
                brokenBindGroup(device, "bad-bind-group");
            }),
        );
        const error = expectCode(err, "E_VALIDATION");
        expect(error.details.label).toBe("bad-bind-group");
        expect(String(error.details.message)).toMatch(/1024|binding/i);
        expect(error.message).toContain("bad-bind-group");
        expect(error.message).toContain(String(error.details.message));
        await device.queue.onSubmittedWorkDone();
        expect(uncapturedEvents).toBe(0);
    });

    it("returns the function's value when nothing was captured, for a sync and for an async fn", async (t) => {
        requireGpu(t);
        const { device } = await acquireRaw();
        expect(await withValidationScope(device, "sync", () => 7)).toBe(7);
        expect(await withValidationScope(device, "async", () => Promise.resolve("ok"))).toBe("ok");
    });

    it("rethrows the function's own error unchanged and pops the scope so a later scope still works", async (t) => {
        requireGpu(t);
        const { device } = await acquireRaw();
        const own = new Error("mine");
        const err = await rejection(
            withValidationScope(device, "own", () => {
                throw own;
            }),
        );
        expect(err).toBe(own);
        expect(await withValidationScope(device, "after", () => 1)).toBe(1);
        const again = await rejection(
            withValidationScope(device, "after-own", () => {
                brokenBindGroup(device, "after-own");
            }),
        );
        expectCode(again, "E_VALIDATION", { label: "after-own" });
    });
});

describe("AllocationTracker", () => {
    it("wraps only allocations at or above the threshold in an out-of-memory scope; resident / liveBuffers follow create and destroy", async (t) => {
        requireGpu(t);
        const { device } = await acquireRaw();
        const pushes: GPUErrorFilter[] = [];
        const dev = patched(device, {
            pushErrorScope: (filter) => {
                pushes.push(filter);
                device.pushErrorScope(filter);
            },
        });
        const tracker = new AllocationTracker(dev, 1024);
        expect(tracker.resident).toBe(0);
        expect(tracker.liveBuffers).toBe(0);
        const a = tracker.createBuffer({ label: "a", size: 512, usage: BufferUsage.STORAGE });
        expect(pushes).toEqual([]);
        expect(tracker.resident).toBe(512);
        expect(tracker.liveBuffers).toBe(1);
        const b = tracker.createBuffer({ label: "b", size: 2048, usage: BufferUsage.STORAGE });
        expect(pushes).toEqual(["out-of-memory"]);
        const c = tracker.createBuffer({ label: "c", size: 1024, usage: BufferUsage.STORAGE });
        expect(pushes).toEqual(["out-of-memory", "out-of-memory"]);
        expect(tracker.resident).toBe(3584);
        expect(tracker.liveBuffers).toBe(3);
        await tracker.check();
        await tracker.check();
        tracker.destroy(a);
        expect(tracker.resident).toBe(3072);
        expect(tracker.liveBuffers).toBe(2);
        tracker.destroy(a);
        expect(tracker.resident).toBe(3072);
        expect(tracker.liveBuffers).toBe(2);
        tracker.destroy(b);
        tracker.destroy(c);
        expect(tracker.resident).toBe(0);
        expect(tracker.liveBuffers).toBe(0);
    });

    it("defaults the threshold to OOM_SCOPE_THRESHOLD_BYTES (16 MiB)", async (t) => {
        requireGpu(t);
        const { device } = await acquireRaw();
        const pushes: GPUErrorFilter[] = [];
        const dev = patched(device, {
            pushErrorScope: (filter) => {
                pushes.push(filter);
                device.pushErrorScope(filter);
            },
        });
        const tracker = new AllocationTracker(dev);
        const below = tracker.createBuffer({
            label: "below",
            size: OOM_SCOPE_THRESHOLD_BYTES - 4,
            usage: BufferUsage.STORAGE,
        });
        expect(pushes).toEqual([]);
        const at = tracker.createBuffer({ label: "at", size: OOM_SCOPE_THRESHOLD_BYTES, usage: BufferUsage.STORAGE });
        expect(pushes).toEqual(["out-of-memory"]);
        await tracker.check();
        expect(tracker.resident).toBe(2 * OOM_SCOPE_THRESHOLD_BYTES - 4);
        tracker.destroy(below);
        tracker.destroy(at);
        expect(tracker.liveBuffers).toBe(0);
    });

    it("check() surfaces a captured out-of-memory error as E_OUT_OF_MEMORY { requested, resident, label }, again until reset()", async (t) => {
        requireGpu(t);
        const { device } = await acquireRaw();
        const tracker = new AllocationTracker(
            patched(device, {
                popErrorScope: () => Promise.resolve({ message: "simulated out-of-memory" } as GPUError),
            }),
            1024,
        );
        const keep = tracker.createBuffer({ label: "keep", size: 256, usage: BufferUsage.STORAGE });
        const big = tracker.createBuffer({ label: "big", size: 4096, usage: BufferUsage.STORAGE });
        expect(tracker.resident).toBe(4352);
        const first = expectCode(await rejection(tracker.check()), "E_OUT_OF_MEMORY", {
            requested: 4096,
            resident: 256,
            label: "big",
        });
        expect(first.message).toContain("big");
        expect(await rejection(tracker.check())).toBe(first);
        expect(tracker.liveBuffers).toBe(1);
        expect(tracker.resident).toBe(256);
        tracker.reset();
        await tracker.check();
        tracker.destroy(big);
        tracker.destroy(keep);
        expect(tracker.liveBuffers).toBe(0);
        // the real "out-of-memory" scope createBuffer("big") pushed stays on this throwaway device's stack; afterAll destroys it
    });
});

describe("formatCompilationInfo", () => {
    it("subtracts the prelude line count, marks messages inside the prelude and keeps unlocated ones", () => {
        const info = {
            messages: [
                { type: "error", lineNum: 12, linePos: 5, offset: 0, length: 1, message: "unresolved value 'foo'" },
                { type: "warning", lineNum: 3, linePos: 1, offset: 0, length: 1, message: "in the prelude" },
                { type: "info", lineNum: 0, linePos: 0, offset: 0, length: 0, message: "general" },
                { type: "error", lineNum: 11, linePos: 9, offset: 0, length: 1, message: "first body line" },
            ],
        } as unknown as GPUCompilationInfo;
        expect(formatCompilationInfo(info, 10)).toEqual([
            "error 2:5 unresolved value 'foo'",
            "warning prelude:3:1 in the prelude",
            "info general",
            "error 1:9 first body line",
        ]);
        expect(formatCompilationInfo({ messages: [] } as unknown as GPUCompilationInfo, 10)).toEqual([]);
    });
});
