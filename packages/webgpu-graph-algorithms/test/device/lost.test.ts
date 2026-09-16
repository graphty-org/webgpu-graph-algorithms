/**
 * src/device/lost.ts (contract 3.4; spec 5.7, 2.2 step 5) on RAW devices, then the mid-readback device-loss
 * scenario of spec 11.5 / contract 5.5 on a GpuContext: device.destroy() while a read is pending rejects it
 * with E_DEVICE_LOST, the context is "lost", the residency is cleared without an uncaptured error, and a
 * context over a FRESH adapter uploads, runs a degree kernel and reads back afterwards. The degree kernel is
 * an ad hoc two-binding spec driven through test/helpers/kernel.ts because src/algorithms/degree.ts is
 * P1-T5's; P2-T1's extension of this file (the last describe block) uses the public degree(), destroys the
 * device while a CommandBatch readback is pending, and pins the onLost fan-out, the residency warning and
 * residentBytes. The `thrown` helper below is the plan's `caughtOf` (same shape), reused instead of a copy.
 */

import { fromEdgeArrays, type GraphSnapshot } from "@graphty/graph-format";

import { degree } from "../../src/algorithms/degree.js";
import {
    deviceLostError,
    installUncapturedErrorSink,
    PendingErrorSlot,
    watchDeviceLost,
} from "../../src/device/lost.js";
import { BufferUsage, ShaderStage } from "../../src/device/webgpu-constants.js";
import { hasErrorCode, isWebGpuGraphError, WebGpuGraphError, type WebGpuGraphErrorCode } from "../../src/errors.js";
import { CommandBatch } from "../../src/kernel/batch.js";
import { plan1d } from "../../src/kernel/dispatch.js";
import { UniformRing } from "../../src/kernel/uniform-ring.js";
import type { WgslModuleSpec } from "../../src/kernel/wgsl.js";
import { FILL_PARAMS, kernelSpec } from "../../src/kernels.js";
import { bindingOf, scratchBuffer } from "../helpers/device.js";
import { KARATE_EDGES, pathEdges, snapshotOf } from "../helpers/graphs.js";
import { runKernel } from "../helpers/kernel.js";
import { outDegreeOracle } from "../oracle/degree.js";
import { acquire, acquireRaw, requireGpu, uncapturedErrors } from "../setup/gpu.js";

/** Undirected 0-1, 1-2, 2-3, 0-2, 0-3 on five nodes; node 4 isolated. */
function smallGraph(): GraphSnapshot {
    const src = new Uint32Array([0, 1, 2, 0, 0]);
    const dst = new Uint32Array([1, 2, 3, 2, 3]);
    return fromEdgeArrays({ directed: false, nodeCount: 5, src, dst });
}

/** outDegree of smallGraph(): node 0 -> {1, 2, 3}, 1 -> {0, 2}, 2 -> {1, 3, 0}, 3 -> {2, 0}, 4 -> {}. */
const SMALL_GRAPH_DEGREES = [3, 2, 3, 2, 0];

/**
 * An ad hoc degree kernel: out[i] = rowPtr[i + 1] - rowPtr[i] for i below the bound `out` binding's length.
 * Two storage bindings in group 0, no uniform; the prelude supplies WG and linear_id.
 */
const DEGREE_ADHOC: WgslModuleSpec = {
    id: "lost-test-degree",
    body: `@compute @workgroup_size(WG)
fn degree_adhoc(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = linear_id(wid, lid.x);
    if (i >= arrayLength(&out)) { return; }
    out[i] = rowPtr[i + 1u] - rowPtr[i];
}
`,
    bindings: [
        { group: 0, binding: 0, name: "rowPtr", kind: "storage-ro", wgslType: "array<u32>" },
        { group: 0, binding: 1, name: "out", kind: "storage", wgslType: "array<u32>" },
    ],
    overrideDecls: [],
    overrides: {},
    needs: [],
    uniforms: [],
};

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

function thrown(fn: () => unknown): unknown {
    try {
        fn();
    } catch (err) {
        return err;
    }
    throw new Error("expected the function to throw");
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

describe("PendingErrorSlot", () => {
    it("starts empty, reports pending after set(), and take() returns then clears the error", () => {
        const slot = new PendingErrorSlot();
        expect(slot.pending).toBe(false);
        expect(slot.take()).toBeNull();
        const error = new WebGpuGraphError("E_VALIDATION", "one", { label: "one", message: "m1" });
        slot.set(error);
        expect(slot.pending).toBe(true);
        expect(slot.take()).toBe(error);
        expect(slot.pending).toBe(false);
        expect(slot.take()).toBeNull();
    });

    it("chains a second and a third error as details.next of the first (the first is returned)", () => {
        const slot = new PendingErrorSlot();
        slot.set(new WebGpuGraphError("E_VALIDATION", "one", { label: "one", message: "m1" }));
        slot.set(new WebGpuGraphError("E_VALIDATION", "two", { label: "two", message: "m2" }));
        slot.set(new WebGpuGraphError("E_OUT_OF_MEMORY", "three", { label: "three", requested: 8, resident: 0 }));
        const taken = slot.take();
        expect(taken).not.toBeNull();
        expect(taken?.code).toBe("E_VALIDATION");
        expect(taken?.message).toBe("one");
        expect(taken?.details.label).toBe("one");
        expect(taken?.details.message).toBe("m1");
        expect(Object.isFrozen(taken?.details)).toBe(true);
        const second = taken?.details.next as WebGpuGraphError;
        expect(second.message).toBe("two");
        expect(second.details.label).toBe("two");
        const third = second.details.next as WebGpuGraphError;
        expect(third.code).toBe("E_OUT_OF_MEMORY");
        expect(third.message).toBe("three");
        expect(third.details.next).toBeUndefined();
        expect(slot.pending).toBe(false);
        expect(slot.take()).toBeNull();
    });
});

describe("deviceLostError", () => {
    it("is E_DEVICE_LOST { reason, message } naming both in the message", () => {
        const info = { reason: "destroyed", message: "device was destroyed" } as GPUDeviceLostInfo;
        const error = deviceLostError(info);
        expect(error).toBeInstanceOf(WebGpuGraphError);
        expect(error.code).toBe("E_DEVICE_LOST");
        expect(error.details).toEqual({ reason: "destroyed", message: "device was destroyed" });
        expect(error.message).toBe("device lost (destroyed): device was destroyed");
    });
});

describe("installUncapturedErrorSink", () => {
    it("routes a broken bind group to onError as E_VALIDATION { label, message }; the uninstaller stops delivery", async (t) => {
        requireGpu(t);
        const { device } = await acquireRaw();
        const slot = new PendingErrorSlot();
        const seen: WebGpuGraphError[] = [];
        const uninstall = installUncapturedErrorSink(device, slot, (error) => {
            seen.push(error);
        });
        brokenBindGroup(device, "sink-test");
        await device.queue.onSubmittedWorkDone();
        expect(seen).toHaveLength(1);
        expect(seen[0].code).toBe("E_VALIDATION");
        expect(seen[0].details.label).toBe(device.label ?? "");
        expect(String(seen[0].details.message)).toMatch(/1024|binding/i);
        expect(seen[0].message).toContain(String(seen[0].details.message));
        expect(slot.pending).toBe(false);
        uninstall();
        brokenBindGroup(device, "after-uninstall");
        await device.queue.onSubmittedWorkDone();
        expect(seen).toHaveLength(1);
    });

    it("stores the error in the slot when onError is null; take() empties it", async (t) => {
        requireGpu(t);
        const { device } = await acquireRaw();
        const slot = new PendingErrorSlot();
        const uninstall = installUncapturedErrorSink(device, slot, null);
        brokenBindGroup(device, "slot-test");
        await device.queue.onSubmittedWorkDone();
        expect(slot.pending).toBe(true);
        const pending = slot.take();
        expect(pending?.code).toBe("E_VALIDATION");
        expect(String(pending?.details.message)).toMatch(/1024|binding/i);
        expect(slot.take()).toBeNull();
        uninstall();
    });
});

describe("watchDeviceLost", () => {
    it("resolves with the lost info after device.destroy() and calls onLost exactly once", async (t) => {
        requireGpu(t);
        const { device } = await acquireRaw();
        let calls = 0;
        const lost = watchDeviceLost(device, () => {
            calls += 1;
        });
        expect(calls).toBe(0);
        device.destroy();
        const info = await lost;
        expect(info.reason).toBe("destroyed");
        expect(typeof info.message).toBe("string");
        expect(calls).toBe(1);
        await lost;
        expect(calls).toBe(1);
    });
});

describe("device loss mid-readback and recovery (spec 5.7, 11.5)", () => {
    it("device.destroy() while a read is pending: the read rejects E_DEVICE_LOST, the context is lost, the residency is cleared, no uncaptured error", async (t) => {
        requireGpu(t);
        const ctx = await acquire({ label: "lost-mid-readback" });
        const s = smallGraph();
        const core = ctx.residency.core(s);
        expect(ctx.residency.stats().buffers).toBeGreaterThan(0);
        const pending = ctx.readback.read(core.rowPtr.buffer, core.rowPtr.size, undefined, core.rowPtr.offset);
        ctx.device.destroy();
        expectCode(await rejection(pending), "E_DEVICE_LOST");
        const info = await ctx.lost;
        expect(info.reason).toBe("destroyed");
        expect(ctx.state).toBe("lost");
        expect(ctx.residency.stats().buffers).toBe(0);
        expect(ctx.residency.residentBytes).toBe(0);
        expectCode(
            thrown(() => {
                ctx.assertReady();
            }),
            "E_DEVICE_LOST",
        );
        expect(uncapturedErrors()).toEqual([]);
    });

    it("a context over a FRESH adapter uploads, runs a degree kernel and reads back afterwards", async (t) => {
        requireGpu(t);
        const first = await acquire({ label: "lost-first" });
        first.device.destroy();
        await first.lost;
        expect(first.state).toBe("lost");
        const ctx = await acquire({ label: "lost-recovered" });
        expect(ctx.state).toBe("ready");
        const s = smallGraph();
        const core = ctx.residency.core(s);
        const n = s.nodeCount;
        const out = ctx.pool.acquire(
            n * 4,
            BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST,
            "lost-recovered/out",
        );
        await runKernel(
            ctx,
            DEGREE_ADHOC,
            { rowPtr: core.rowPtr, out: { buffer: out, offset: 0, size: n * 4, window: null } },
            plan1d(n, ctx.workgroupSize, ctx.caps),
        );
        const bytes = await ctx.readback.read(out, n * 4);
        const result = Array.from(new Uint32Array(bytes));
        expect(result).toEqual(SMALL_GRAPH_DEGREES);
        expect(result).toEqual(Array.from(s.outDegree()));
        ctx.pool.release(out);
        ctx.release(s);
        expect(ctx.residency.stats().buffers).toBe(0);
        expect(uncapturedErrors()).toEqual([]);
    });
});

describe("P2: device loss mid-batch, the loss fan-out, the residency warning and residentBytes (spec 2.2 step 5, 5.7)", () => {
    it("device.destroy() mid-batch rejects the batch readback with E_DEVICE_LOST; state lost; residency cleared; listeners ran once", async (t) => {
        requireGpu(t);
        const ctx = await acquire({ label: "lost-p2" });
        const s = snapshotOf(KARATE_EDGES, { label: "karate" });
        ctx.residency.core(s);
        expect(ctx.residency.stats().snapshots).toBe(1);
        expect(ctx.residency.residentBytes).toBe(880);

        const ran: string[] = [];
        const unregisterA = ctx.onLost((info) => {
            ran.push(`A:${info.reason}`);
        });
        ctx.onLost((info) => {
            ran.push(`B:${info.reason}`);
        });
        const unregisterC = ctx.onLost(() => {
            ran.push("C");
        });
        unregisterC();

        const kernel = await ctx.pipelines.kernel(kernelSpec("fill"));
        const ring = new UniformRing(ctx.device, ctx.allocator, 1, "test/lost/ring");
        const dst = scratchBuffer(ctx, 4_194_304 * 4, "test/lost/dst");
        ring.write(0, FILL_PARAMS, { count: 4_194_304, value: 1, mode: 1 });
        ring.flush();
        const batch = new CommandBatch(ctx, "lost-mid-batch");
        const pass = batch.pass("fill");
        const bound = kernel.bind({ dst: bindingOf(dst), P: ring.binding(FILL_PARAMS) });
        const plan = plan1d(4_194_304, ctx.workgroupSize, ctx.caps);
        for (let k = 0; k < 4; k++) {
            kernel.dispatch(pass, bound, plan, [0]);
        }
        batch.readback(dst, 0, 64);
        const submitted = batch.submit();
        expect(ctx.readback.borrowed).toBe(1);

        ctx.device.destroy();
        const outcome = await submitted.readback.then(
            () => null,
            (error: unknown) => error,
        );
        expect(isWebGpuGraphError(outcome)).toBe(true);
        expect(hasErrorCode(outcome, "E_DEVICE_LOST")).toBe(true);
        const info = await ctx.lost;
        expect(info.reason).toBe("destroyed");
        console.warn(`[lost] mid-batch readback rejected E_DEVICE_LOST (reason=${info.reason}, runtime=node)`);

        expect(ctx.state).toBe("lost");
        expect(ran).toEqual(["A:destroyed", "B:destroyed"]);
        expect(ctx.residency.stats()).toEqual({ buffers: 0, bytes: 0, snapshots: 0, perSnapshot: [] });
        expect(ctx.residency.residentBytes).toBe(0);
        expect(ctx.readback.borrowed).toBe(0);
        expect(uncapturedErrors()).toEqual([]);
        expect(
            hasErrorCode(
                thrown(() => ctx.assertReady()),
                "E_DEVICE_LOST",
            ),
        ).toBe(true);
        // a listener registered after the loss never runs; the unregistered one never ran
        ctx.onLost(() => {
            ran.push("late");
        });
        expect(ran).toEqual(["A:destroyed", "B:destroyed"]);
        // a batch submitted after the loss throws synchronously
        const late = new CommandBatch(ctx, "after-loss");
        late.pass("nothing");
        expect(
            hasErrorCode(
                thrown(() => late.submit()),
                "E_DEVICE_LOST",
            ),
        ).toBe(true);
        expect(ctx.readback.borrowed).toBe(0);
        unregisterA();
        ctx.dispose();
        expect(ctx.state).toBe("disposed");
    });

    it("recovery: a new context from a FRESH adapter uploads and runs degree afterwards", async (t) => {
        requireGpu(t);
        const lostCtx = await acquire({ label: "lost-then-recover" });
        lostCtx.device.destroy();
        await lostCtx.lost;
        expect(lostCtx.state).toBe("lost");

        const ctx = await acquire({ label: "recovered" });
        expect(ctx.state).toBe("ready");
        const s = snapshotOf(KARATE_EDGES, { label: "karate" });
        const result = await degree(ctx, s);
        expect(Array.from(result)).toEqual(Array.from(outDegreeOracle(s)));
        expect(result.length).toBe(34);
        ctx.release(s);
        expect(ctx.residency.residentBytes).toBe(0);
    });

    it("the residency warning fires once when more than warnUnreleasedSnapshots snapshots are resident", async (t) => {
        requireGpu(t);
        const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
        try {
            const ctx = await acquire({ label: "warn", warnUnreleasedSnapshots: 1 });
            const a = snapshotOf(KARATE_EDGES, { label: "karate" });
            const b = snapshotOf(pathEdges(10), { label: "path10" });
            const c = snapshotOf(pathEdges(20), { label: "path20" });
            ctx.residency.core(a);
            const before = warn.mock.calls.length;
            ctx.residency.core(b);
            expect(warn.mock.calls.length).toBe(before + 1);
            const message = String(warn.mock.calls[before][0]);
            expect(message).toContain("warnUnreleasedSnapshots");
            expect(message).toContain("2 snapshots");
            ctx.residency.core(c);
            expect(warn.mock.calls.length).toBe(before + 1);
            expect(ctx.residency.stats().snapshots).toBe(3);
            ctx.release(a);
            ctx.release(b);
            ctx.release(c);
            expect(ctx.residency.residentBytes).toBe(0);
            ctx.dispose();
        } finally {
            warn.mockRestore();
        }
    });
});
