// MAINT review probe: verify three sync-hazard claims about the plan's WGSL / TS split.
//  1. What the 4070 (or whichever adapter Dawn picks) reports for the limits the plan hardcodes
//     in BOTH constants.ts and the WGSL prelude (65535 in linear_id, WG 256, 8 storage buffers).
//     Is maxComputeWorkgroupsPerDimension raisable above 65535 (plan 2.2 lists it as RaisableLimit)?
//  2. Is a TS-side `constants` key for an override the WGSL does not declare a validation error
//     (i.e. is an override-name mismatch between TS and WGSL detected at pipeline creation)?
//  3. Is an override declared in the prelude but never referenced by the entry point accepted
//     when it IS set from TS (the plan's prelude declares USE_PERM / HAS_WEIGHTS / SUBGROUP_SIZE
//     for every module whether or not the kernel uses them)?
import { createRequire } from "node:module";
const require = createRequire("/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/package.json");
const dawn = await import(require.resolve("webgpu"));
Object.assign(globalThis, dawn.globals);
const gpu = dawn.create(process.argv.slice(2));
const adapter = await gpu.requestAdapter();
if (!adapter) { console.log("no adapter"); process.exit(1); }
const L = adapter.limits;
console.log("adapter:", adapter.info.vendor, adapter.info.architecture, adapter.info.device);
console.log("adapter limits:", JSON.stringify({
    maxComputeWorkgroupsPerDimension: L.maxComputeWorkgroupsPerDimension,
    maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
    maxBindGroups: L.maxBindGroups,
    maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
    maxUniformBufferBindingSize: L.maxUniformBufferBindingSize,
    minUniformBufferOffsetAlignment: L.minUniformBufferOffsetAlignment,
}));
const dflt = await adapter.requestDevice();
console.log("default device limits:", JSON.stringify({
    maxComputeWorkgroupsPerDimension: dflt.limits.maxComputeWorkgroupsPerDimension,
    maxStorageBuffersPerShaderStage: dflt.limits.maxStorageBuffersPerShaderStage,
    maxBindGroups: dflt.limits.maxBindGroups,
}));
dflt.destroy();

let raised = null;
const adapter2 = await gpu.requestAdapter();   // Dawn-node 0.4.0: an adapter is "consumed" by one requestDevice
try {
    raised = await adapter2.requestDevice({ requiredLimits: {
        maxComputeWorkgroupsPerDimension: L.maxComputeWorkgroupsPerDimension,
        maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
    } });
    console.log("raised device limits:", JSON.stringify({
        maxComputeWorkgroupsPerDimension: raised.limits.maxComputeWorkgroupsPerDimension,
        maxStorageBuffersPerShaderStage: raised.limits.maxStorageBuffersPerShaderStage,
    }));
} catch (e) { console.log("raise failed:", String(e)); }

const device = raised ?? await (await gpu.requestAdapter()).requestDevice();
device.addEventListener("uncapturederror", (ev) => console.log("UNCAPTURED:", ev.error.message.split("\n")[0]));

const code = /* wgsl */ `
override WG: u32 = 256u;
override USE_PERM: bool = false;      // declared in the prelude, never referenced below
override HAS_WEIGHTS: bool = false;   // declared in the prelude, never referenced below
@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(WG) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < 1024u) { o[id.x] = id.x; }
}`;
const module = device.createShaderModule({ code });
const info = await module.getCompilationInfo();
console.log("compile messages:", info.messages.length);
const bgl = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });

async function tryPipeline(label, constants) {
    device.pushErrorScope("validation");
    let ok = true;
    try {
        await device.createComputePipelineAsync({ label, layout, compute: { module, entryPoint: "main", constants } });
    } catch (e) { ok = false; console.log(`${label}: createComputePipelineAsync rejected: ${String(e).split("\n")[0]}`); }
    const err = await device.popErrorScope();
    console.log(`${label}: constants=${JSON.stringify(constants)} -> ${ok ? "created" : "rejected"}; validation scope: ${err ? err.message.split("\n")[0] : "none"}`);
}
await tryPipeline("A-declared-unreferenced-set", { WG: 256, USE_PERM: 1, HAS_WEIGHTS: 0 });
await tryPipeline("B-undeclared-override-key", { WG: 256, LINLOG: 0 });
await tryPipeline("C-misspelled-key", { WG: 256, USE_PREM: 1 });
await tryPipeline("D-wg-mismatch-512", { WG: 512 });
device.destroy();
