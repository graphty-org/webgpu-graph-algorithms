// DESIGN-review probe: settle platform facts the plan presents as certain.
//  1. default device limits with NO requiredLimits (note 05 unverified #2; plan 2.2 step 4)
//  2. override constants: bool override in select(), override WG in @workgroup_size (plan 3.5 / 5.1)
//  3. dummy binding pattern: same buffer bound twice read-only, u32 buffer bound as array<f32> (plan 4.1 / 7.5)
//  4. explicit layout compatibility: layout type "storage" vs shader var<storage, read> (plan 5.1)
//  5. subgroups: enable subgroups; subgroup_size builtin value vs adapter.info.subgroupMinSize (plan D16)
//  6. timestamp-query granularity on Dawn-node (note 05 unverified #3; plan 2.6 / Q-16)
//  7. dispatchWorkgroupsIndirect with a count above maxComputeWorkgroupsPerDimension (note 05 unverified #7)
//  8. INDIRECT | STORAGE buffer written by a dispatch and consumed by dispatchWorkgroupsIndirect in the SAME pass (plan 5.4)
import { createRequire } from "node:module";
const require = createRequire("/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/package.json");
const { create, globals } = await import(require.resolve("webgpu"));
Object.assign(globalThis, globals);
const which = process.argv[2] ?? "";
const gpu = create(which ? [`adapter=${which}`] : []);
const adapter = await gpu.requestAdapter();
if (!adapter) { console.log("no adapter"); process.exit(1); }
console.log("adapter:", adapter.info.vendor, adapter.info.architecture, adapter.info.device, "subgroup", adapter.info.subgroupMinSize + "-" + adapter.info.subgroupMaxSize);
console.log("features:", [...adapter.features].join(","));

// ---- 1. default device limits
{
    const d = await adapter.requestDevice();
    const L = d.limits;
    console.log("[1] default device limits:", JSON.stringify({
        maxBufferSize: L.maxBufferSize, maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
        maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage, maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupsPerDimension: L.maxComputeWorkgroupsPerDimension,
        minStorageBufferOffsetAlignment: L.minStorageBufferOffsetAlignment, minUniformBufferOffsetAlignment: L.minUniformBufferOffsetAlignment,
        maxDynamicStorageBuffersPerPipelineLayout: L.maxDynamicStorageBuffersPerPipelineLayout, maxDynamicUniformBuffersPerPipelineLayout: L.maxDynamicUniformBuffersPerPipelineLayout,
    }));
    d.destroy();
}

const wantFeatures = ["subgroups", "timestamp-query"].filter((f) => adapter.features.has(f));
// the adapter is CONSUMED by the first requestDevice (spec + Dawn): request a fresh one
const adapter2 = await gpu.requestAdapter();
console.log("[1b] second requestDevice on the same adapter throws 'adapter is consumed' (seen); fresh adapter:", adapter2.info.architecture);
const device = await adapter2.requestDevice({ requiredFeatures: wantFeatures, requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
device.addEventListener("uncapturederror", (e) => console.log("UNCAPTURED:", e.error.message));
const U = GPUBufferUsage;

function mk(size, usage, label) { return device.createBuffer({ size, usage, label }); }
async function readU32(buf, n) {
    const st = mk(n * 4, U.MAP_READ | U.COPY_DST, "st");
    const e = device.createCommandEncoder(); e.copyBufferToBuffer(buf, 0, st, 0, n * 4); device.queue.submit([e.finish()]);
    await st.mapAsync(GPUMapMode.READ); const out = new Uint32Array(st.getMappedRange()).slice(); st.unmap(); st.destroy(); return out;
}
async function tryPipeline(label, code, constants, layoutEntries) {
    device.pushErrorScope("validation");
    const mod = device.createShaderModule({ code });
    const info = await mod.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === "error").map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
    let layout = "auto";
    if (layoutEntries) {
        layout = device.createPipelineLayout({ bindGroupLayouts: [device.createBindGroupLayout({ entries: layoutEntries })] });
    }
    let pipe = null; let perr = null;
    try { pipe = await device.createComputePipelineAsync({ layout, compute: { module: mod, entryPoint: "main", constants } }); } catch (e) { perr = e.message; }
    const scope = await device.popErrorScope();
    console.log(`[${label}] compile errors: ${errs.length ? errs.join(" | ") : "none"}; pipeline: ${pipe ? "ok" : "FAILED " + perr}; scope: ${scope ? scope.message.slice(0, 200) : "none"}`);
    return pipe;
}

// ---- 2 + 3 + 4. overrides, dummy bindings, explicit layout compatibility
{
    const n = 1000;
    const code = `
override WG: u32 = 64u;
override USE_PERM: bool = false;
override HAS_WEIGHTS: bool = false;
@group(0) @binding(0) var<storage, read> rowPtr: array<u32>;
@group(0) @binding(1) var<storage, read> colIdx: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> perm: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(WG) fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let row = gid.x;
    if (row >= ${n}u) { return; }
    let i = select(row, perm[row], USE_PERM);
    var acc = 0u;
    for (var a = rowPtr[i]; a < rowPtr[i + 1u]; a = a + 1u) {
        var w = 1.0;
        if (HAS_WEIGHTS) { w = weights[a]; }
        acc = acc + colIdx[a] + u32(w);
    }
    out[row] = acc + WG * 100000u;
}`;
    const rowPtr = new Uint32Array(n + 1); for (let i = 0; i <= n; i++) rowPtr[i] = i * 2;
    const colIdx = new Uint32Array(2 * n); for (let a = 0; a < 2 * n; a++) colIdx[a] = a % 7;
    const perm = new Uint32Array(n); for (let i = 0; i < n; i++) perm[i] = n - 1 - i;
    const bRow = mk(rowPtr.byteLength, U.STORAGE | U.COPY_DST, "rowPtr"); device.queue.writeBuffer(bRow, 0, rowPtr);
    const bCol = mk(colIdx.byteLength, U.STORAGE | U.COPY_DST, "colIdx"); device.queue.writeBuffer(bCol, 0, colIdx);
    const bPerm = mk(perm.byteLength, U.STORAGE | U.COPY_DST, "perm"); device.queue.writeBuffer(bPerm, 0, perm);
    const bOut = mk(n * 4, U.STORAGE | U.COPY_SRC, "out");
    const ro = (b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } });
    const rw = (b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } });
    const layoutEntries = [ro(0), ro(1), ro(2), ro(3), rw(4)];
    for (const [label, constants, useRealPerm] of [["2a override WG=256,USE_PERM=0,HAS_WEIGHTS=0", { WG: 256, USE_PERM: 0, HAS_WEIGHTS: 0 }, false], ["2b override WG=128,USE_PERM=1", { WG: 128, USE_PERM: 1, HAS_WEIGHTS: 0 }, true]]) {
        const pipe = await tryPipeline(label, code, constants, layoutEntries);
        if (!pipe) continue;
        device.pushErrorScope("validation");
        // 3: dummy pattern -- colIdx bound in the weights slot (u32 as f32), rowPtr in the perm slot when !USE_PERM
        const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: bRow } }, { binding: 1, resource: { buffer: bCol } }, { binding: 2, resource: { buffer: bCol } },
            { binding: 3, resource: { buffer: useRealPerm ? bPerm : bRow } }, { binding: 4, resource: { buffer: bOut } }] });
        const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(Math.ceil(n / constants.WG)); p.end(); device.queue.submit([e.finish()]);
        const scope = await device.popErrorScope();
        const out = await readU32(bOut, n);
        const wg = constants.WG;
        const expect = (row) => { const i = useRealPerm ? n - 1 - row : row; let acc = 0; for (let a = rowPtr[i]; a < rowPtr[i + 1]; a++) acc += colIdx[a] + 1; return acc + wg * 100000; };
        let ok = true; for (let r = 0; r < n; r++) if (out[r] !== expect(r)) { ok = false; console.log("  mismatch at", r, out[r], expect(r)); break; }
        console.log(`[${label}] dummy bindings + result: ${ok ? "ok" : "WRONG"}; scope: ${scope ? scope.message.slice(0, 200) : "none"}`);
    }
    // 4: explicit layout says "storage" (read_write) for binding 0 but the shader declares var<storage, read>
    await tryPipeline("4 layout storage(rw) vs shader var<storage,read>", code, { WG: 64 }, [rw(0), ro(1), ro(2), ro(3), rw(4)]);
    // 4b: layout says read-only-storage but shader declares read_write
    await tryPipeline("4b layout read-only vs shader read_write", code, { WG: 64 }, [ro(0), ro(1), ro(2), ro(3), ro(4)]);
}

// ---- 5. subgroups
if (device.features.has("subgroups")) {
    const code = `
enable subgroups;
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(subgroup_size) ss: u32, @builtin(subgroup_invocation_id) sid: u32) {
    let s = subgroupAdd(1u);
    if (lid.x == 0u) { out[0] = ss; out[1] = s; }
    if (lid.x == 255u) { out[2] = ss; out[3] = s; }
}`;
    const pipe = await tryPipeline("5 subgroups", code, {}, null);
    if (pipe) {
        const bOut = mk(16, U.STORAGE | U.COPY_SRC, "sg");
        const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: bOut } }] });
        const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(1); p.end(); device.queue.submit([e.finish()]);
        const out = await readU32(bOut, 4);
        console.log(`[5] subgroup_size builtin = ${out[0]} (lane 255: ${out[2]}), subgroupAdd(1) = ${out[1]} / ${out[3]}; adapter.info.subgroupMinSize = ${adapter.info.subgroupMinSize}, MaxSize = ${adapter.info.subgroupMaxSize}`);
    }
} else { console.log("[5] no subgroups feature"); }

// ---- 6. timestamp-query granularity
if (device.features.has("timestamp-query")) {
    const qs = device.createQuerySet({ type: "timestamp", count: 2 });
    const res = mk(16, U.QUERY_RESOLVE | U.COPY_SRC, "resolve");
    const code = `@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) { var a = f32(g.x); for (var i = 0u; i < 2000u; i++) { a = sin(a) * 1.0001 + 0.5; } o[g.x] = a; }`;
    const pipe = await tryPipeline("6 timestamp kernel", code, {}, null);
    const bOut = mk(1 << 20, U.STORAGE, "ts");
    const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: bOut } }] });
    const samples = [];
    for (let k = 0; k < 8; k++) {
        const e = device.createCommandEncoder();
        const p = e.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
        p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(1 + k * 64); p.end();
        e.resolveQuerySet(qs, 0, 2, res, 0);
        device.queue.submit([e.finish()]);
        const st = mk(16, U.MAP_READ | U.COPY_DST, "st2"); const e2 = device.createCommandEncoder(); e2.copyBufferToBuffer(res, 0, st, 0, 16); device.queue.submit([e2.finish()]);
        await st.mapAsync(GPUMapMode.READ); const v = new BigUint64Array(st.getMappedRange()); samples.push(Number(v[1] - v[0])); st.unmap(); st.destroy();
    }
    console.log(`[6] timestamp deltas (ns): ${samples.join(", ")}; all multiples of 100000? ${samples.every((s) => s % 100000 === 0)}; multiples of 1000? ${samples.every((s) => s % 1000 === 0)}`);
} else { console.log("[6] no timestamp-query feature"); }

// ---- 7 + 8. indirect dispatch: over-limit count, and same-pass write-then-indirect
{
    const code = `@group(0) @binding(0) var<storage, read_write> cnt: array<atomic<u32>>;
@compute @workgroup_size(1) fn main() { atomicAdd(&cnt[0], 1u); }`;
    const writer = `@group(0) @binding(0) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(1) fn main() { args[0] = 7u; args[1] = 1u; args[2] = 1u; }`;
    const pipe = await tryPipeline("7 counter kernel", code, {}, null);
    const wpipe = await tryPipeline("8 args writer", writer, {}, null);
    const cnt = mk(4, U.STORAGE | U.COPY_SRC | U.COPY_DST, "cnt");
    const args = mk(12, U.INDIRECT | U.STORAGE | U.COPY_DST, "args");
    const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: cnt } }] });
    const wbg = device.createBindGroup({ layout: wpipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: args } }] });
    // 7: x = 70000 > 65535
    device.queue.writeBuffer(cnt, 0, new Uint32Array([0]));
    device.queue.writeBuffer(args, 0, new Uint32Array([70000, 1, 1]));
    device.pushErrorScope("validation");
    { const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroupsIndirect(args, 0); p.end(); device.queue.submit([e.finish()]); }
    const s7 = await device.popErrorScope();
    console.log(`[7] indirect x=70000 (> 65535): counter = ${(await readU32(cnt, 1))[0]} (spec: 0 = 'does nothing'); scope: ${s7 ? s7.message.slice(0, 160) : "none"}`);
    // 7b: x = 65535 exactly
    device.queue.writeBuffer(cnt, 0, new Uint32Array([0]));
    device.queue.writeBuffer(args, 0, new Uint32Array([65535, 1, 1]));
    { const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroupsIndirect(args, 0); p.end(); device.queue.submit([e.finish()]); }
    console.log(`[7b] indirect x=65535: counter = ${(await readU32(cnt, 1))[0]}`);
    // 8: same pass: dispatch writes args (7,1,1) then indirect consumes them
    device.queue.writeBuffer(cnt, 0, new Uint32Array([0]));
    device.queue.writeBuffer(args, 0, new Uint32Array([0, 0, 0]));
    device.pushErrorScope("validation");
    { const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(wpipe); p.setBindGroup(0, wbg); p.dispatchWorkgroups(1); p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroupsIndirect(args, 0); p.end(); device.queue.submit([e.finish()]); }
    const s8 = await device.popErrorScope();
    console.log(`[8] same-pass write-then-indirect: counter = ${(await readU32(cnt, 1))[0]} (expect 7); scope: ${s8 ? s8.message.slice(0, 160) : "none"}`);
}
device.destroy();
