/**
 * Adversarial audit of the GPU memcpy contract ON A REAL DEVICE (design section 10): the arrays a
 * snapshot exposes are handed to `queue.writeBuffer` exactly as they are -- no copy, no cast, no
 * conversion (the one documented exception being `gpuView()` of an f64 column) -- the arena's hot
 * prefix is uploaded with the literal expression of section 10.3 and its segments are bound at
 * `segment.byteOffset - arena.byteOffset`, and compute shaders read the uploaded bytes back through
 * the WGSL idioms the design prescribes (`unpack4xU8`, the bit test, `arcToEdge` gathers, the
 * `edgeToArc` writeback, the 256-byte colIdx window rebase of section 10.6). Every result is compared
 * with the CPU views.
 *
 * Runs on Google Dawn for Node (the `webgpu` npm package, a devDependency of this package). On
 * this machine the NVIDIA Vulkan ICD needs `libEGL.so.1` on `LD_LIBRARY_PATH` (see
 * HEADLESS_GPU_REPORT.md appendix D: `tmp/egl/root/usr/lib/x86_64-linux-gnu`); without it Dawn falls
 * back to Mesa's llvmpipe software adapter, which still exercises the same upload path. When Dawn
 * cannot load or finds no adapter at all the suite is skipped with the reason printed (E_NO_ADAPTER);
 * a wrong result is never a skip.
 */

import { afterAll, beforeAll, describe, expect, it, type TestContext } from "vitest";

import { type ArenaLayout, fromEdgeArrays, GraphBuilder, type GraphSnapshot, type U32 } from "../../src/index.js";

// ============================================================ device acquisition (top level, once)

interface DawnModule {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
}

interface GpuContext {
    readonly gpu: GPU;
    readonly adapter: GPUAdapter;
    readonly device: GPUDevice;
    readonly vendor: string;
    readonly architecture: string;
}

async function acquire(): Promise<GpuContext | string> {
    let dawn: DawnModule;
    try {
        dawn = (await import("webgpu")) as unknown as DawnModule;
    } catch (err) {
        return `E_NO_ADAPTER: the webgpu (Dawn) native module did not load: ${(err as Error).message}`;
    }
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([]);
    let adapter: GPUAdapter | null;
    try {
        adapter = await gpu.requestAdapter();
    } catch (err) {
        return `E_NO_ADAPTER: requestAdapter() threw: ${(err as Error).message}`;
    }
    if (adapter === null) {
        return "E_NO_ADAPTER: requestAdapter() returned null (no Vulkan ICD usable; is libEGL.so.1 on LD_LIBRARY_PATH?)";
    }
    const { vendor, architecture } = adapter.info;
    let device: GPUDevice;
    try {
        device = await adapter.requestDevice();
    } catch (err) {
        return `E_NO_ADAPTER: adapter ${vendor}/${architecture} found but requestDevice() failed: ${(err as Error).message}`;
    }
    return { gpu, adapter, device, vendor, architecture };
}

let context: GpuContext | null = null;
let skipReason = "E_NO_ADAPTER: device acquisition did not run";

beforeAll(async () => {
    const acquired = await acquire();
    if (typeof acquired === "string") {
        skipReason = acquired;
        console.warn(`[gpu-upload] SKIPPED: ${acquired}`);
        return;
    }
    context = acquired;
    const { adapter } = acquired;
    console.warn(
        `[gpu-upload] adapter vendor=${acquired.vendor} architecture=${acquired.architecture} device=${adapter.info.device} description=${adapter.info.description}`,
    );
    console.warn(
        `[gpu-upload] limits maxBufferSize=${adapter.limits.maxBufferSize} maxStorageBufferBindingSize=${adapter.limits.maxStorageBufferBindingSize} minStorageBufferOffsetAlignment=${adapter.limits.minStorageBufferOffsetAlignment} maxComputeWorkgroupsPerDimension=${adapter.limits.maxComputeWorkgroupsPerDimension}`,
    );
});

afterAll(() => {
    if (context !== null) {
        context.device.destroy();
        context = null;
    }
});

/** The device, or a runtime skip of the calling test with the E_NO_ADAPTER reason (never on a wrong result). */
function requireGpu(t: TestContext): GpuContext {
    if (context === null) {
        t.skip(skipReason);
    }
    return context;
}

// ============================================================ device helpers

const WORKGROUP = 256;

/** A storage buffer filled by ONE writeBuffer of the array as given: no copy, no cast, no conversion. */
function upload(device: GPUDevice, data: ArrayBufferView, usage: number = GPUBufferUsage.STORAGE): GPUBuffer {
    const buffer = device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
}

function outputBuffer(device: GPUDevice, byteLength: number): GPUBuffer {
    return device.createBuffer({ size: byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
}

async function readback(device: GPUDevice, source: GPUBuffer, byteLength: number): Promise<ArrayBuffer> {
    const staging = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, staging, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    // copy out before unmap: the mapped range is detached at unmap (design section 10.7)
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return copy;
}

interface Binding {
    readonly buffer: GPUBuffer;
    readonly offset?: number;
    readonly size?: number;
    readonly kind: "read" | "write" | "uniform";
}

function bufferBindingType(kind: Binding["kind"]): GPUBufferBindingType {
    switch (kind) {
        case "write":
            return "storage";
        case "uniform":
            return "uniform";
        case "read":
            return "read-only-storage";
        default:
            throw new Error(`unknown binding kind ${String(kind)}`);
    }
}

/** Run one compute pass over `count` invocations and surface validation errors as test failures. */
async function dispatch(device: GPUDevice, code: string, bindings: readonly Binding[], count: number): Promise<void> {
    device.pushErrorScope("validation");
    const entries: GPUBindGroupLayoutEntry[] = bindings.map((b, i) => ({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: bufferBindingType(b.kind) },
    }));
    const layout = device.createBindGroupLayout({ entries });
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === "error").map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
    expect(errors, `WGSL compile errors:\n${errors.join("\n")}`).toEqual([]);
    const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module, entryPoint: "main" },
    });
    const bindGroup = device.createBindGroup({
        layout,
        entries: bindings.map((b, i) => {
            const resource: GPUBufferBinding = { buffer: b.buffer };
            if (b.offset !== undefined) {
                resource.offset = b.offset;
            }
            if (b.size !== undefined) {
                resource.size = b.size;
            }
            return { binding: i, resource };
        }),
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    // the 1D dispatch rule of section 10.6
    const groups = Math.ceil(count / WORKGROUP);
    expect(groups).toBeLessThanOrEqual(device.limits.maxComputeWorkgroupsPerDimension);
    pass.dispatchWorkgroups(groups);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const error = await device.popErrorScope();
    expect(error === null ? null : error.message, "WebGPU validation error").toBeNull();
}

function u32Uniform(device: GPUDevice, values: readonly number[]): GPUBuffer {
    const padded = new Uint32Array(Math.max(4, Math.ceil(values.length / 4) * 4));
    padded.set(values);
    return upload(device, padded, GPUBufferUsage.UNIFORM);
}

// ============================================================ graphs

/** A deterministic random edge list with no self-loops. */
function randomEdges(n: number, m: number, seed: number): { src: U32; dst: U32 } {
    const src = new Uint32Array(m);
    const dst = new Uint32Array(m);
    let state = seed >>> 0;
    const next = (): number => {
        state = (Math.imul(state, 1103515245) + 12345) >>> 0;
        return state;
    };
    for (let e = 0; e < m; e++) {
        const u = next() % n;
        let v = next() % n;
        if (u === v) {
            v = (v + 1) % n;
        }
        src[e] = u;
        dst[e] = v;
    }
    return { src, dst };
}

const N = 4096;
const M = 50000;

function directedGraph(): GraphSnapshot {
    const { src, dst } = randomEdges(N, M, 42);
    const weights = new Float32Array(M);
    const cost = new Float32Array(M);
    for (let e = 0; e < M; e++) {
        weights[e] = 1 + (e % 7);
        cost[e] = (e % 13) * 0.5;
    }
    const s = fromEdgeArrays({ directed: true, nodeCount: N, src, dst, weights, edgeColumns: { cost } });
    expect(s.flags.arcToEdgeIsIdentity).toBe(false);
    expect(s.arena).not.toBeNull();
    return s;
}

function undirectedGraph(): GraphSnapshot {
    const { src, dst } = randomEdges(N, M, 43);
    const weights = new Float32Array(M);
    for (let e = 0; e < M; e++) {
        weights[e] = 1 + (e % 5);
    }
    const s = fromEdgeArrays({ directed: false, nodeCount: N, src, dst, weights });
    expect(s.arcCount).toBe(2 * M);
    expect(s.arena).not.toBeNull();
    return s;
}

// ============================================================ the suite

describe("audit: real GPU upload of the format's arrays (design section 10)", () => {
    it("the device accepts the format's 256-byte segment alignment and default-limit planning assumptions", (t) => {
        const { device, adapter } = requireGpu(t);
        // section 10.3: segments start at multiples of 256, the spec default of minStorageBufferOffsetAlignment
        expect(256 % device.limits.minStorageBufferOffsetAlignment).toBe(0);
        expect(device.limits.maxStorageBufferBindingSize).toBeGreaterThanOrEqual(128 * 1024 * 1024);
        expect(device.limits.maxBufferSize).toBeGreaterThanOrEqual(256 * 1024 * 1024);
        expect(device.limits.maxComputeWorkgroupsPerDimension).toBeGreaterThanOrEqual(65535);
        expect(typeof adapter.info.vendor).toBe("string");
    });

    it("per-array path: rowPtr uploaded as is, outDegree summed on the device equals the CPU view", async (t) => {
        const { device } = requireGpu(t);
        const s = directedGraph();
        const rowPtr = upload(device, s.rowPtr);
        const out = outputBuffer(device, 4 * s.nodeCount);
        await dispatch(
            device,
            `
            @group(0) @binding(0) var<storage, read> rowPtr: array<u32>;
            @group(0) @binding(1) var<storage, read_write> out: array<u32>;
            @compute @workgroup_size(${WORKGROUP})
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let i = id.x;
                if (i >= arrayLength(&rowPtr) - 1u) { return; }
                out[i] = rowPtr[i + 1u] - rowPtr[i];
            }`,
            [
                { buffer: rowPtr, kind: "read" },
                { buffer: out, kind: "write" },
            ],
            s.nodeCount,
        );
        const result = new Uint32Array(await readback(device, out, 4 * s.nodeCount));
        expect(result).toEqual(s.outDegree());
        let total = 0;
        for (let i = 0; i < result.length; i++) {
            total += result[i];
        }
        expect(total).toBe(s.arcCount);
    });

    it("whole-arena path: one writeBuffer of the hot prefix, segments bound at segment.byteOffset - arena.byteOffset", async (t) => {
        const { device } = requireGpu(t);
        const s = directedGraph();
        const arena = s.arena as ArenaLayout;
        expect(arena.hotByteLength).toBeLessThanOrEqual(device.limits.maxBufferSize);
        const gbuf = device.createBuffer({
            size: arena.hotByteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        // the literal expression of design section 10.3
        device.queue.writeBuffer(gbuf, 0, new Uint8Array(arena.buffer, arena.byteOffset, arena.hotByteLength));
        const seg = (name: "rowPtr" | "colIdx" | "weights"): Binding => {
            const segment = arena.segments[name];
            expect(segment, name).not.toBeNull();
            const { byteOffset, byteLength } = segment as { byteOffset: number; byteLength: number };
            const offset = byteOffset - arena.byteOffset;
            expect(offset % device.limits.minStorageBufferOffsetAlignment).toBe(0);
            expect(offset + byteLength).toBeLessThanOrEqual(arena.hotByteLength);
            return { buffer: gbuf, offset, size: byteLength, kind: "read" };
        };
        const weightSum = outputBuffer(device, 4 * s.nodeCount);
        const targetSum = outputBuffer(device, 4 * s.nodeCount);
        await dispatch(
            device,
            `
            @group(0) @binding(0) var<storage, read> rowPtr: array<u32>;
            @group(0) @binding(1) var<storage, read> colIdx: array<u32>;
            @group(0) @binding(2) var<storage, read> weights: array<f32>;
            @group(0) @binding(3) var<storage, read_write> weightSum: array<f32>;
            @group(0) @binding(4) var<storage, read_write> targetSum: array<u32>;
            @compute @workgroup_size(${WORKGROUP})
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let u = id.x;
                if (u >= arrayLength(&rowPtr) - 1u) { return; }
                var w = 0.0;
                var t = 0u;
                for (var a = rowPtr[u]; a < rowPtr[u + 1u]; a++) {
                    w += weights[a];
                    t += colIdx[a];
                }
                weightSum[u] = w;
                targetSum[u] = t;
            }`,
            [
                seg("rowPtr"),
                seg("colIdx"),
                seg("weights"),
                { buffer: weightSum, kind: "write" },
                { buffer: targetSum, kind: "write" },
            ],
            s.nodeCount,
        );
        const gpuWeightSum = new Float32Array(await readback(device, weightSum, 4 * s.nodeCount));
        const gpuTargetSum = new Uint32Array(await readback(device, targetSum, 4 * s.nodeCount));
        const cpuWeightSum = s.weightedOutDegree();
        const cpuTargetSum = new Uint32Array(s.nodeCount);
        for (let u = 0; u < s.nodeCount; u++) {
            let t = 0;
            for (let a = s.rowPtr[u]; a < s.rowPtr[u + 1]; a++) {
                t = (t + s.colIdx[a]) >>> 0;
            }
            cpuTargetSum[u] = t;
        }
        expect(gpuTargetSum).toEqual(cpuTargetSum);
        // integer weights 1..7 over rows of at most a few dozen arcs: f32 sums are exact
        for (let u = 0; u < s.nodeCount; u++) {
            expect(gpuWeightSum[u], `weighted out-degree of ${u}`).toBe(cpuWeightSum[u]);
        }
    });

    it("cold segments: arcToEdge gathers an edge column, edgeToArc writes per-edge results back", async (t) => {
        const { device } = requireGpu(t);
        const s = directedGraph();
        const arena = s.arena as ArenaLayout;
        const gbuf = device.createBuffer({
            size: arena.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(gbuf, 0, new Uint8Array(arena.buffer, arena.byteOffset, arena.byteLength));
        const seg = (name: "rowPtr" | "colIdx" | "arcToEdge" | "edgeToArc"): Binding => {
            const segment = arena.segments[name] as { byteOffset: number; byteLength: number };
            return {
                buffer: gbuf,
                offset: segment.byteOffset - arena.byteOffset,
                size: segment.byteLength,
                kind: "read",
            };
        };
        const cost = upload(device, s.edges.gpuView("cost"));
        const costSum = outputBuffer(device, 4 * s.nodeCount);
        const edgeTarget = outputBuffer(device, 4 * s.edgeCount);
        await dispatch(
            device,
            `
            @group(0) @binding(0) var<storage, read> rowPtr: array<u32>;
            @group(0) @binding(1) var<storage, read> colIdx: array<u32>;
            @group(0) @binding(2) var<storage, read> arcToEdge: array<u32>;
            @group(0) @binding(3) var<storage, read> edgeToArc: array<u32>;
            @group(0) @binding(4) var<storage, read> cost: array<f32>;
            @group(0) @binding(5) var<storage, read_write> costSum: array<f32>;
            @group(0) @binding(6) var<storage, read_write> edgeTarget: array<u32>;
            @compute @workgroup_size(${WORKGROUP})
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let i = id.x;
                if (i < arrayLength(&rowPtr) - 1u) {
                    var c = 0.0;
                    for (var a = rowPtr[i]; a < rowPtr[i + 1u]; a++) {
                        c += cost[arcToEdge[a]];
                    }
                    costSum[i] = c;
                }
                if (i < arrayLength(&edgeToArc)) {
                    edgeTarget[i] = colIdx[edgeToArc[i]];
                }
            }`,
            [
                seg("rowPtr"),
                seg("colIdx"),
                seg("arcToEdge"),
                seg("edgeToArc"),
                { buffer: cost, kind: "read" },
                { buffer: costSum, kind: "write" },
                { buffer: edgeTarget, kind: "write" },
            ],
            Math.max(s.nodeCount, s.edgeCount),
        );
        const gpuCostSum = new Float32Array(await readback(device, costSum, 4 * s.nodeCount));
        const gpuEdgeTarget = new Uint32Array(await readback(device, edgeTarget, 4 * s.edgeCount));
        expect(gpuEdgeTarget).toEqual(s.edgeList().dst);
        const costColumn = s.edges.requireTyped("cost", "f32").data;
        for (let u = 0; u < s.nodeCount; u++) {
            let c = 0;
            for (let a = s.rowPtr[u]; a < s.rowPtr[u + 1]; a++) {
                c = Math.fround(c + costColumn[s.arcToEdge[a]]);
            }
            expect(gpuCostSum[u], `cost sum of ${u}`).toBe(c);
        }
    });

    it("undirected: doubled storage means a per-arc write through arcToEdge lands once per edge with equal values", async (t) => {
        const { device } = requireGpu(t);
        const s = undirectedGraph();
        const arena = s.arena as ArenaLayout;
        // reverse() is the forward arrays: a cache keyed on the array object uploads once
        expect(s.reverse().rowPtr).toBe(s.rowPtr);
        expect(s.reverse().colIdx).toBe(s.colIdx);
        const colIdx = upload(device, s.colIdx);
        const weights = upload(device, s.weights as Float32Array);
        const arcToEdge = upload(device, s.arcToEdge);
        const mate = upload(device, s.mate());
        const perEdge = outputBuffer(device, 4 * s.edgeCount);
        const mateOk = outputBuffer(device, 4 * s.arcCount);
        await dispatch(
            device,
            `
            @group(0) @binding(0) var<storage, read> colIdx: array<u32>;
            @group(0) @binding(1) var<storage, read> weights: array<f32>;
            @group(0) @binding(2) var<storage, read> arcToEdge: array<u32>;
            @group(0) @binding(3) var<storage, read> mate: array<u32>;
            @group(0) @binding(4) var<storage, read_write> perEdge: array<f32>;
            @group(0) @binding(5) var<storage, read_write> mateOk: array<u32>;
            @compute @workgroup_size(${WORKGROUP})
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let a = id.x;
                if (a >= arrayLength(&colIdx)) { return; }
                perEdge[arcToEdge[a]] = weights[a];
                let m = mate[a];
                mateOk[a] = select(0u, 1u, arcToEdge[m] == arcToEdge[a] && weights[m] == weights[a] && mate[m] == a);
            }`,
            [
                { buffer: colIdx, kind: "read" },
                { buffer: weights, kind: "read" },
                { buffer: arcToEdge, kind: "read" },
                { buffer: mate, kind: "read" },
                { buffer: perEdge, kind: "write" },
                { buffer: mateOk, kind: "write" },
            ],
            s.arcCount,
        );
        const gpuPerEdge = new Float32Array(await readback(device, perEdge, 4 * s.edgeCount));
        expect(gpuPerEdge).toEqual(s.edgeList().weights as Float32Array);
        const gpuMateOk = new Uint32Array(await readback(device, mateOk, 4 * s.arcCount));
        expect(gpuMateOk.every((v) => v === 1)).toBe(true);
        expect(arena.segments.arcToEdge).not.toBeNull();
    });

    it("packed columns: u8 through unpack4xU8, bool through the bit test, f64 through the cached f32 gpuView", async (t) => {
        const { device } = requireGpu(t);
        const rows = 1000;
        const b = new GraphBuilder({ directed: true });
        b.addAnonymousNodes(rows);
        b.declareNodeColumn({ name: "bytes", dtype: "u8", nullable: false });
        b.declareNodeColumn({ name: "flag", dtype: "bool", nullable: false });
        b.declareNodeColumn({ name: "score", dtype: "f64", nullable: false });
        b.declareNodeColumn({ name: "triple", dtype: "u8", components: 3, nullable: false });
        for (let i = 0; i < rows; i++) {
            b.setNodeValue("bytes", i, (i * 37) & 0xff);
            b.setNodeValue("flag", i, i % 3 === 0);
            b.setNodeValue("score", i, i / 8 + 0.25);
            b.setNodeValue("triple", i, [i & 0xff, (i >>> 8) & 0xff, 7]);
        }
        const s = b.freeze();
        const u8 = s.nodes.requireTyped("bytes", "u8");
        const bool = s.nodes.requireTyped("flag", "bool");
        expect(u8.paddedU32View().length).toBe(Math.ceil(rows / 4));
        expect(bool.data.length).toBe(Math.ceil(rows / 32));
        const bytes = upload(device, s.nodes.gpuView("bytes"));
        const flags = upload(device, s.nodes.gpuView("flag"));
        const score = upload(device, s.nodes.gpuView("score"));
        const triple = upload(device, s.nodes.gpuView("triple"));
        const out = outputBuffer(device, 4 * rows);
        const outScore = outputBuffer(device, 4 * rows);
        const outTriple = outputBuffer(device, 4 * rows);
        await dispatch(
            device,
            `
            @group(0) @binding(0) var<storage, read> bytes: array<u32>;
            @group(0) @binding(1) var<storage, read> flags: array<u32>;
            @group(0) @binding(2) var<storage, read> score: array<f32>;
            @group(0) @binding(3) var<storage, read> triple: array<u32>;
            @group(0) @binding(4) var<storage, read_write> out: array<u32>;
            @group(0) @binding(5) var<storage, read_write> outScore: array<f32>;
            @group(0) @binding(6) var<storage, read_write> outTriple: array<u32>;
            @compute @workgroup_size(${WORKGROUP})
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let i = id.x;
                if (i >= ${rows}u) { return; }
                let byte = unpack4xU8(bytes[i >> 2u])[i & 3u];
                let bit = (flags[i >> 5u] >> (i & 31u)) & 1u;
                out[i] = byte + 1000u * bit;
                outScore[i] = score[i] * 2.0;
                let k = i * 3u;
                var t = 0u;
                for (var c = 0u; c < 3u; c++) {
                    let j = k + c;
                    t = t * 256u + unpack4xU8(triple[j >> 2u])[j & 3u];
                }
                outTriple[i] = t;
            }`,
            [
                { buffer: bytes, kind: "read" },
                { buffer: flags, kind: "read" },
                { buffer: score, kind: "read" },
                { buffer: triple, kind: "read" },
                { buffer: out, kind: "write" },
                { buffer: outScore, kind: "write" },
                { buffer: outTriple, kind: "write" },
            ],
            rows,
        );
        const gpuOut = new Uint32Array(await readback(device, out, 4 * rows));
        const gpuScore = new Float32Array(await readback(device, outScore, 4 * rows));
        const gpuTriple = new Uint32Array(await readback(device, outTriple, 4 * rows));
        for (let i = 0; i < rows; i++) {
            expect(gpuOut[i], `row ${i}`).toBe(((i * 37) & 0xff) + (i % 3 === 0 ? 1000 : 0));
            expect(gpuScore[i], `score ${i}`).toBe(Math.fround(i / 8 + 0.25) * 2);
            expect(gpuTriple[i], `triple ${i}`).toBe(((i & 0xff) * 256 + ((i >>> 8) & 0xff)) * 256 + 7);
        }
    });

    it("windowed colIdx bindings of section 10.6: a 64-arc rebase is a legal 256-byte storage offset", async (t) => {
        const { device } = requireGpu(t);
        const s = directedGraph();
        const colIdx = upload(device, s.colIdx);
        const v0 = 1234;
        const v1 = 2345;
        const start = s.rowPtr[v0] - (s.rowPtr[v0] % 64);
        const end = s.rowPtr[v1];
        expect((4 * start) % 256).toBe(0);
        const rowWindow = upload(device, s.rowPtr.subarray(v0, v1 + 1));
        const windowCopy = upload(device, s.colIdx.subarray(start, end));
        const uniforms = u32Uniform(device, [start, v1 - v0]);
        const fromOffset = outputBuffer(device, 4 * (v1 - v0));
        const fromCopy = outputBuffer(device, 4 * (v1 - v0));
        await dispatch(
            device,
            `
            struct Params { start: u32, count: u32 }
            @group(0) @binding(0) var<uniform> params: Params;
            @group(0) @binding(1) var<storage, read> rowPtr: array<u32>;
            @group(0) @binding(2) var<storage, read> windowA: array<u32>;
            @group(0) @binding(3) var<storage, read> windowB: array<u32>;
            @group(0) @binding(4) var<storage, read_write> outA: array<u32>;
            @group(0) @binding(5) var<storage, read_write> outB: array<u32>;
            @compute @workgroup_size(${WORKGROUP})
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let i = id.x;
                if (i >= params.count) { return; }
                var a = 0u;
                var b = 0u;
                for (var k = rowPtr[i]; k < rowPtr[i + 1u]; k++) {
                    a += windowA[k - params.start];
                    b += windowB[k - params.start];
                }
                outA[i] = a;
                outB[i] = b;
            }`,
            [
                { buffer: uniforms, kind: "uniform" },
                { buffer: rowWindow, kind: "read" },
                { buffer: colIdx, offset: 4 * start, size: 4 * (end - start), kind: "read" },
                { buffer: windowCopy, kind: "read" },
                { buffer: fromOffset, kind: "write" },
                { buffer: fromCopy, kind: "write" },
            ],
            v1 - v0,
        );
        const gpuA = new Uint32Array(await readback(device, fromOffset, 4 * (v1 - v0)));
        const gpuB = new Uint32Array(await readback(device, fromCopy, 4 * (v1 - v0)));
        const cpu = new Uint32Array(v1 - v0);
        for (let v = v0; v < v1; v++) {
            let t = 0;
            for (let a = s.rowPtr[v]; a < s.rowPtr[v + 1]; a++) {
                t = (t + s.colIdx[a]) >>> 0;
            }
            cpu[v - v0] = t;
        }
        expect(gpuA).toEqual(cpu);
        expect(gpuB).toEqual(cpu);
    });

    it("a container adopted by fromBytes uploads straight from the file bytes (arena.byteOffset !== 0)", async (t) => {
        const { device } = requireGpu(t);
        const source = directedGraph();
        const bytes = source.toBytes();
        const shifted = new Uint8Array(bytes.byteLength + 8);
        shifted.set(bytes, 8);
        const { fromBytes } = await import("../../src/index.js");
        const s = fromBytes(new Uint8Array(shifted.buffer, 8, bytes.byteLength), { validate: "structure" });
        const arena = s.arena as ArenaLayout;
        expect(arena.buffer).toBe(shifted.buffer);
        expect(arena.byteOffset).toBeGreaterThan(0);
        const gbuf = device.createBuffer({
            size: arena.hotByteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(gbuf, 0, new Uint8Array(arena.buffer, arena.byteOffset, arena.hotByteLength));
        const rowSeg = arena.segments.rowPtr as { byteOffset: number; byteLength: number };
        const colSeg = arena.segments.colIdx as { byteOffset: number; byteLength: number };
        const out = outputBuffer(device, 4 * s.nodeCount);
        await dispatch(
            device,
            `
            @group(0) @binding(0) var<storage, read> rowPtr: array<u32>;
            @group(0) @binding(1) var<storage, read> colIdx: array<u32>;
            @group(0) @binding(2) var<storage, read_write> out: array<u32>;
            @compute @workgroup_size(${WORKGROUP})
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let u = id.x;
                if (u >= arrayLength(&rowPtr) - 1u) { return; }
                var t = 0u;
                for (var a = rowPtr[u]; a < rowPtr[u + 1u]; a++) { t += colIdx[a]; }
                out[u] = t;
            }`,
            [
                { buffer: gbuf, offset: rowSeg.byteOffset - arena.byteOffset, size: rowSeg.byteLength, kind: "read" },
                { buffer: gbuf, offset: colSeg.byteOffset - arena.byteOffset, size: colSeg.byteLength, kind: "read" },
                { buffer: out, kind: "write" },
            ],
            s.nodeCount,
        );
        const gpu = new Uint32Array(await readback(device, out, 4 * s.nodeCount));
        const cpu = new Uint32Array(s.nodeCount);
        for (let u = 0; u < source.nodeCount; u++) {
            let t = 0;
            for (let a = source.rowPtr[u]; a < source.rowPtr[u + 1]; a++) {
                t = (t + source.colIdx[a]) >>> 0;
            }
            cpu[u] = t;
        }
        expect(gpu).toEqual(cpu);
    });
});
