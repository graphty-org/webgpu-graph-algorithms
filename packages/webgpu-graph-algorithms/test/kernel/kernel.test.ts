/**
 * Kernel and runKernel (spec 5.1, 5.6; contract 3.9, 5.2; 5.5 row kernel.test.ts): bind() creates one labelled group
 * per layout incl. empty ones and caches by buffer identity + offset + size; a missing / extra name and a zero-size
 * binding are E_INVALID_ARGUMENT; the host-side aliasing rule (intersecting ranges, and a writable slot next to any
 * other access mode of one buffer) is E_INVALID_ARGUMENT { argument: "aliasing" } synchronously while disjoint
 * read_write ranges are accepted; a wrong-size uniform binding reaches the pending-error slot as E_VALIDATION with
 * the `<id>/<group>` label; an empty plan records nothing; a compiled trivial kernel runs through runKernel (twice,
 * bitwise equal), through dynamic offsets, and with a bool override reaching the pipeline as a number.
 */

import { GpuContext } from "../../src/context.js";
import { isWebGpuGraphError, type WebGpuGraphError } from "../../src/errors.js";
import { type DispatchPlan, plan1d } from "../../src/kernel/dispatch.js";
import { type BoundKernel, Kernel } from "../../src/kernel/kernel.js";
import { UniformBlock } from "../../src/kernel/struct-block.js";
import { type WgslModuleSpec } from "../../src/kernel/wgsl.js";
import { type Binding } from "../../src/types/memory.js";
import { bindingOf, readU32, scratchBuffer, uploadBuffer } from "../helpers/device.js";
import { runKernel } from "../helpers/kernel.js";
import { expectBitwiseEqual } from "../helpers/matchers.js";
import { acquire, acquireRaw, requireGpu } from "../setup/gpu.js";

function catchError(fn: () => unknown): WebGpuGraphError {
    try {
        fn();
    } catch (err) {
        if (isWebGpuGraphError(err)) {
            return err;
        }
        throw err;
    }
    throw new Error("expected a WebGpuGraphError");
}

const ADD_PARAMS = UniformBlock.define("AddParams", [
    ["count", "u32"],
    ["value", "u32"],
]);
const ADD_BODY = `@compute @workgroup_size(WG)
fn add(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = linear_id(wid, lid.x);
    if (i >= P.count) { return; }
    dst[i] = select(src[i] + P.value, 2u * (src[i] + P.value), DOUBLE);
}`;
/** src (read-only) + P.value -> dst; group 0 is deliberately empty (the graph group of a graph-less kernel). */
const ADD: WgslModuleSpec = {
    id: "test-add",
    body: ADD_BODY,
    bindings: [
        { group: 1, binding: 0, name: "src", kind: "storage-ro", wgslType: "array<u32>" },
        { group: 1, binding: 1, name: "dst", kind: "storage", wgslType: "array<u32>" },
        { group: 2, binding: 0, name: "P", kind: "uniform", wgslType: "AddParams" },
    ],
    overrideDecls: [{ name: "DOUBLE", type: "bool", default: false }],
    overrides: {},
    needs: [],
    uniforms: [ADD_PARAMS],
};
const COPY_BODY = `@compute @workgroup_size(WG)
fn copy_add(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = linear_id(wid, lid.x);
    if (i >= P.count) { return; }
    b[i] = a[i] + P.value;
}`;
/** Two read_write slots, so disjoint ranges of ONE buffer are legal (the state header / trace pattern of 3.10.1). */
const COPY: WgslModuleSpec = {
    id: "test-copy",
    body: COPY_BODY,
    bindings: [
        { group: 1, binding: 0, name: "a", kind: "storage", wgslType: "array<u32>" },
        { group: 1, binding: 1, name: "b", kind: "storage", wgslType: "array<u32>" },
        { group: 2, binding: 0, name: "P", kind: "uniform", wgslType: "AddParams" },
    ],
    overrideDecls: [],
    overrides: {},
    needs: [],
    uniforms: [ADD_PARAMS],
};
const N = 1024; // 4096 bytes: the half-buffer ranges below start at a 256-aligned offset

function iota(n: number): Uint32Array<ArrayBuffer> {
    const out = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
        out[i] = i;
    }
    return out;
}

function range(buffer: GPUBuffer, offset: number, size: number): Binding {
    return { buffer, offset, size, window: null };
}

describe("Kernel.bind", () => {
    it("creates one bind group per layout incl. empty ones, labelled <id>/<group>, with the uniform group dynamic", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const kernel = await ctx.pipelines.kernel(ADD);
        const src = uploadBuffer(ctx, iota(N), "src");
        const dst = scratchBuffer(ctx, N * 4, "dst");
        const params = ctx.device.createBuffer({
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "params",
        });
        const bound: BoundKernel = kernel.bind({
            src: bindingOf(src),
            dst: bindingOf(dst),
            P: range(params, 0, ADD_PARAMS.byteLength),
        });
        expect(bound.kernel).toBe(kernel);
        expect(bound.bindGroups).toHaveLength(3);
        expect(bound.bindGroups.map((g) => g.label)).toEqual(["test-add/0", "test-add/1", "test-add/2"]);
        expect(bound.dynamicGroups).toEqual([2]);
        expect(Object.isFrozen(bound)).toBe(true);
        src.destroy();
        dst.destroy();
        params.destroy();
    });

    it("returns the same BoundKernel for the same buffers, offsets and sizes; a different range or invalidate() gives a new one", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const kernel = await ctx.pipelines.kernel(ADD);
        const src = uploadBuffer(ctx, iota(N), "src");
        const dst = scratchBuffer(ctx, N * 4, "dst");
        const params = ctx.device.createBuffer({
            size: 512,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "params",
        });
        const resources = { src: bindingOf(src), dst: bindingOf(dst), P: range(params, 0, 16) };
        const first = kernel.bind(resources);
        expect(kernel.bind({ ...resources })).toBe(first);
        expect(kernel.bind({ src: range(src, 0, N * 4), dst: range(dst, 0, N * 4), P: range(params, 0, 16) })).toBe(
            first,
        );
        const shifted = kernel.bind({ ...resources, P: range(params, 256, 16) });
        expect(shifted).not.toBe(first);
        expect(kernel.bind({ ...resources, dst: range(dst, 0, 512) })).not.toBe(first);
        kernel.invalidate();
        const again = kernel.bind(resources);
        expect(again).not.toBe(first);
        expect(again.bindGroups[1]).not.toBe(first.bindGroups[1]);
        src.destroy();
        dst.destroy();
        params.destroy();
    });

    it("a missing name, an extra name, a zero-size binding and a misaligned offset are E_INVALID_ARGUMENT", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const kernel = await ctx.pipelines.kernel(ADD);
        const src = uploadBuffer(ctx, iota(N), "src");
        const dst = scratchBuffer(ctx, N * 4, "dst");
        const params = ctx.device.createBuffer({
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "params",
        });
        const missing = catchError(() => kernel.bind({ src: bindingOf(src), P: range(params, 0, 16) }));
        expect(missing.code).toBe("E_INVALID_ARGUMENT");
        expect(missing.details).toMatchObject({ argument: "bindings", value: "dst" });
        const extra = catchError(() =>
            kernel.bind({ src: bindingOf(src), dst: bindingOf(dst), P: range(params, 0, 16), q: bindingOf(dst) }),
        );
        expect(extra.code).toBe("E_INVALID_ARGUMENT");
        expect(extra.details).toMatchObject({ argument: "bindings", value: "q" });
        const zero = catchError(() =>
            kernel.bind({ src: range(src, 0, 0), dst: bindingOf(dst), P: range(params, 0, 16) }),
        );
        expect(zero.code).toBe("E_INVALID_ARGUMENT");
        expect(zero.details.argument).toBe("src");
        const misaligned = catchError(() =>
            kernel.bind({ src: range(src, 4, N * 4 - 4), dst: bindingOf(dst), P: range(params, 0, 16) }),
        );
        expect(misaligned.code).toBe("E_INVALID_ARGUMENT");
        expect(misaligned.details).toMatchObject({ argument: "src", value: 4 });
        src.destroy();
        dst.destroy();
        params.destroy();
    });

    it("one buffer in a storage slot and any other slot of the call is E_INVALID_ARGUMENT { argument: 'aliasing' } synchronously; disjoint read_write ranges are accepted and run", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const add = await ctx.pipelines.kernel(ADD);
        const copy = await ctx.pipelines.kernel(COPY);
        const both = uploadBuffer(ctx, iota(2 * N), "both");
        const params = ctx.device.createBuffer({
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "params",
        });
        const P = range(params, 0, 16);
        // read-only + read_write on one buffer: intersecting ranges ...
        const overlap = catchError(() => add.bind({ src: range(both, 0, N * 4), dst: range(both, 256, N * 4), P }));
        expect(overlap.code).toBe("E_INVALID_ARGUMENT");
        expect(overlap.details.argument).toBe("aliasing");
        expect(overlap.details.value).toEqual(["src", "dst"]);
        // ... and disjoint ranges too: WebGPU's usage-scope rule rejects writable + another usage per BUFFER (Dawn 0.4.0 verified)
        const usage = catchError(() => add.bind({ src: range(both, 0, N * 4), dst: range(both, N * 4, N * 4), P }));
        expect(usage.details).toMatchObject({ argument: "aliasing", reason: "usage" });
        // the uniform slot on the writable buffer is the same rule
        const uniform = catchError(() =>
            add.bind({ src: range(both, 0, N * 4), dst: range(both, N * 4, N * 4), P: range(both, 0, 16) }),
        );
        expect(uniform.details.argument).toBe("aliasing");
        // two read_write slots: intersecting ranges are aliasing ...
        const rw = catchError(() => copy.bind({ a: range(both, 0, N * 4), b: range(both, 256, N * 4), P }));
        expect(rw.details).toMatchObject({ argument: "aliasing", reason: "range" });
        // ... disjoint ranges are accepted and the kernel runs: b half = a half + 5
        await runKernel(
            ctx,
            COPY,
            { a: range(both, 0, N * 4), b: range(both, N * 4, N * 4) },
            plan1d(N, copy.workgroupSize, ctx.caps),
            {
                block: ADD_PARAMS,
                values: { count: N, value: 5 },
            },
        );
        const out = await readU32(ctx, both, 2 * N);
        for (let i = 0; i < N; i++) {
            expect(out[N + i]).toBe(i + 5);
        }
        both.destroy();
        params.destroy();
    });

    it("a wrong-size uniform binding is not thrown by bind(): it reaches the pending-error slot as E_VALIDATION with the <id>/<group> label, thrown by the next assertReady()", async (t) => {
        requireGpu(t);
        // a context WITHOUT an onError sink, so the uncaptured error is stored in the pending slot (spec 5.7)
        const raw = await acquireRaw();
        const ctx = await GpuContext.create({ gpu: raw.gpu, runtime: "node", label: "kernel-pending" });
        try {
            const kernel = await ctx.pipelines.kernel(ADD);
            const src = uploadBuffer(ctx, iota(N), "src");
            const dst = scratchBuffer(ctx, N * 4, "dst");
            const params = ctx.device.createBuffer({
                size: 256,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                label: "params",
            });
            // 4 bytes where AddParams needs 16: createBindGroup fails validation (minBindingSize), bind() still returns
            const bound = kernel.bind({ src: bindingOf(src), dst: bindingOf(dst), P: range(params, 0, 4) });
            expect(bound.bindGroups).toHaveLength(3);
            const encoder = ctx.device.createCommandEncoder({ label: "wrong-size" });
            const pass = encoder.beginComputePass();
            kernel.dispatch(pass, bound, plan1d(N, kernel.workgroupSize, ctx.caps));
            pass.end();
            ctx.device.queue.submit([encoder.finish()]);
            const err = catchError(() => {
                ctx.assertReady();
            });
            expect(err.code).toBe("E_VALIDATION");
            const detail = typeof err.details.message === "string" ? err.details.message : "";
            expect(`${err.message} ${detail}`).toContain("test-add/2");
            // the invalid dispatch of the same submit adds follow-up errors (details.next / a later slot entry): drain them
            ctx.takePendingError();
            src.destroy();
            dst.destroy();
            params.destroy();
        } finally {
            ctx.dispose();
        }
    });
});

describe("Kernel.dispatch", () => {
    it("records setPipeline, every setBindGroup (dynamic offsets for the uniform groups) and one dispatchWorkgroups; an empty plan records nothing", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const kernel = await ctx.pipelines.kernel(ADD);
        const src = uploadBuffer(ctx, iota(N), "src");
        const dst = scratchBuffer(ctx, N * 4, "dst");
        const params = ctx.device.createBuffer({
            size: 512,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "params",
        });
        const bound = kernel.bind({ src: bindingOf(src), dst: bindingOf(dst), P: range(params, 0, 16) });
        const calls: unknown[][] = [];
        const fakePass = {
            setPipeline: (...args: unknown[]) => calls.push(["setPipeline", ...args]),
            setBindGroup: (...args: unknown[]) => calls.push(["setBindGroup", ...args]),
            dispatchWorkgroups: (...args: unknown[]) => calls.push(["dispatchWorkgroups", ...args]),
        } as unknown as GPUComputePassEncoder;
        kernel.dispatch(fakePass, bound, plan1d(0, kernel.workgroupSize, ctx.caps));
        expect(calls).toEqual([]);
        kernel.dispatch(fakePass, bound, plan1d(N, kernel.workgroupSize, ctx.caps));
        expect(calls).toEqual([
            ["setPipeline", kernel.pipeline],
            ["setBindGroup", 0, bound.bindGroups[0]],
            ["setBindGroup", 1, bound.bindGroups[1]],
            ["setBindGroup", 2, bound.bindGroups[2], [0]],
            ["dispatchWorkgroups", Math.ceil(N / kernel.workgroupSize), 1, 1],
        ]);
        calls.length = 0;
        const twoD: DispatchPlan = { x: 65_535, y: 2, z: 1, items: 16_776_961, stride: null };
        kernel.dispatch(fakePass, bound, twoD, [256]);
        expect(calls[3]).toEqual(["setBindGroup", 2, bound.bindGroups[2], [256]]);
        expect(calls[4]).toEqual(["dispatchWorkgroups", 65_535, 2, 1]);
        src.destroy();
        dst.destroy();
        params.destroy();
    });

    it("rejects a BoundKernel of another kernel and a dynamic-offset list of the wrong length", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const add = await ctx.pipelines.kernel(ADD);
        const copy = await ctx.pipelines.kernel(COPY);
        const src = uploadBuffer(ctx, iota(N), "src");
        const dst = scratchBuffer(ctx, N * 4, "dst");
        const params = ctx.device.createBuffer({
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "params",
        });
        const bound = copy.bind({ a: bindingOf(src), b: bindingOf(dst), P: range(params, 0, 16) });
        const fakePass = {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
        } as unknown as GPUComputePassEncoder;
        const plan = plan1d(N, add.workgroupSize, ctx.caps);
        const foreign = catchError(() => {
            add.dispatch(fakePass, bound, plan);
        });
        expect(foreign.code).toBe("E_INVALID_ARGUMENT");
        expect(foreign.details).toMatchObject({ argument: "bound", value: "test-copy", expected: "test-add" });
        const offsets = catchError(() => {
            copy.dispatch(fakePass, bound, plan, [0, 256]);
        });
        expect(offsets.code).toBe("E_INVALID_ARGUMENT");
        expect(offsets.details).toMatchObject({ argument: "dynamicOffsets", value: 2, expected: 1 });
        src.destroy();
        dst.destroy();
        params.destroy();
    });
});

describe("runKernel", () => {
    it("dispatches a compiled trivial kernel on the device: dst = src + value, twice bitwise equal; an empty plan leaves dst untouched", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const kernel = await ctx.pipelines.kernel(ADD);
        expect(kernel).toBeInstanceOf(Kernel);
        const src = uploadBuffer(ctx, iota(N), "src");
        const dst = scratchBuffer(ctx, N * 4, "dst");
        const params = ctx.device.createBuffer({
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "params",
        });
        const plan = plan1d(N, kernel.workgroupSize, ctx.caps);
        const bindings = { src: bindingOf(src), dst: bindingOf(dst) };
        await runKernel(ctx, ADD, bindings, plan, { block: ADD_PARAMS, values: { count: N, value: 7 } });
        const first = await readU32(ctx, dst, N);
        for (let i = 0; i < N; i++) {
            expect(first[i]).toBe(i + 7);
        }
        await runKernel(ctx, ADD, bindings, plan, { block: ADD_PARAMS, values: { count: N, value: 7 } });
        const second = await readU32(ctx, dst, N);
        expectBitwiseEqual(first, second, "two runs of test-add");
        // the helper's per-call params buffers never accumulate in the kernel's bind-group cache: a BoundKernel bound
        // before the two runs is forgotten (a fresh object comes back) and the held one is still usable
        const held = kernel.bind({ ...bindings, P: range(params, 0, ADD_PARAMS.byteLength) });
        await runKernel(ctx, ADD, bindings, plan, { block: ADD_PARAMS, values: { count: N, value: 7 } });
        const rebound = kernel.bind({ ...bindings, P: range(params, 0, ADD_PARAMS.byteLength) });
        expect(rebound).not.toBe(held);
        expect(held.bindGroups.map((g) => g.label)).toEqual(["test-add/0", "test-add/1", "test-add/2"]);
        const untouched = scratchBuffer(ctx, N * 4, "untouched");
        await runKernel(
            ctx,
            ADD,
            { src: bindingOf(src), dst: bindingOf(untouched) },
            plan1d(0, kernel.workgroupSize, ctx.caps),
            {
                block: ADD_PARAMS,
                values: { count: N, value: 7 },
            },
        );
        expect((await readU32(ctx, untouched, N)).every((v) => v === 0)).toBe(true);
        src.destroy();
        dst.destroy();
        params.destroy();
        untouched.destroy();
    });

    it("a bool override reaches the pipeline as a number: DOUBLE true doubles the result and is a distinct pipeline", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const doubled: WgslModuleSpec = { ...ADD, overrides: { DOUBLE: true } };
        const src = uploadBuffer(ctx, iota(N), "src");
        const dst = scratchBuffer(ctx, N * 4, "dst");
        const kernel = await ctx.pipelines.kernel(doubled);
        const plan = plan1d(N, kernel.workgroupSize, ctx.caps);
        await runKernel(ctx, doubled, { src: bindingOf(src), dst: bindingOf(dst) }, plan, {
            block: ADD_PARAMS,
            values: { count: N, value: 1 },
        });
        const out = await readU32(ctx, dst, N);
        for (let i = 0; i < N; i++) {
            expect(out[i]).toBe(2 * (i + 1));
        }
        expect(await ctx.pipelines.get(doubled)).not.toBe(await ctx.pipelines.get(ADD));
        src.destroy();
        dst.destroy();
    });

    it("a uniform bound over a 512-byte buffer reads slot k through the dynamic offset", async (t) => {
        requireGpu(t);
        const ctx = await acquire();
        const kernel = await ctx.pipelines.kernel(ADD);
        const src = uploadBuffer(ctx, iota(N), "src");
        const dst = scratchBuffer(ctx, N * 4, "dst");
        const params = ctx.device.createBuffer({
            size: 512,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "params",
        });
        const bytes = new ArrayBuffer(512);
        const view = new DataView(bytes);
        ADD_PARAMS.write(view, { count: N, value: 7 }, 0);
        ADD_PARAMS.write(view, { count: N, value: 11 }, 256);
        ctx.device.queue.writeBuffer(params, 0, bytes);
        const bound = kernel.bind({
            src: bindingOf(src),
            dst: bindingOf(dst),
            P: range(params, 0, ADD_PARAMS.byteLength),
        });
        const plan = plan1d(N, kernel.workgroupSize, ctx.caps);
        for (const [offset, value] of [
            [0, 7],
            [256, 11],
        ] as const) {
            const encoder = ctx.device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            kernel.dispatch(pass, bound, plan, [offset]);
            pass.end();
            ctx.device.queue.submit([encoder.finish()]);
            const out = await readU32(ctx, dst, N);
            expect(out[0]).toBe(value);
            expect(out[N - 1]).toBe(N - 1 + value);
        }
        src.destroy();
        dst.destroy();
        params.destroy();
    });
});
