// VERIFY-review probe: (1) does dawn.create(["adapter=llvmpipe"]) select lavapipe under webgpu@0.4.0,
// (2) what adapter.info.architecture / isFallbackAdapter report for it (plan 2.2 isSoftwareAdapter),
// (3) default device limits WITHOUT requiredLimits (note 05 unverified item 2; plan G2 "windowed at defaults").
import { createRequire } from "node:module";
const require = createRequire("/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/package.json");
const dawn = await import(require.resolve("webgpu"));
Object.assign(globalThis, dawn.globals);
process.env.XDG_RUNTIME_DIR ??= "/tmp";
const opts = process.argv.slice(2);
const gpu = dawn.create(opts);
const a = await gpu.requestAdapter();
if (!a) { console.log("adapter: null"); process.exit(2); }
console.log("create opts:", JSON.stringify(opts));
console.log("adapter.info:", JSON.stringify({ vendor: a.info.vendor, architecture: a.info.architecture, device: a.info.device, description: a.info.description, isFallbackAdapter: a.isFallbackAdapter, subgroupMinSize: a.info.subgroupMinSize, subgroupMaxSize: a.info.subgroupMaxSize }));
console.log("adapter.limits:", JSON.stringify({ maxBufferSize: a.limits.maxBufferSize, maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize, maxStorageBuffersPerShaderStage: a.limits.maxStorageBuffersPerShaderStage, minStorageBufferOffsetAlignment: a.limits.minStorageBufferOffsetAlignment, minUniformBufferOffsetAlignment: a.limits.minUniformBufferOffsetAlignment, maxComputeWorkgroupsPerDimension: a.limits.maxComputeWorkgroupsPerDimension }));
const d = await a.requestDevice();
console.log("default device.limits (no requiredLimits):", JSON.stringify({ maxBufferSize: d.limits.maxBufferSize, maxStorageBufferBindingSize: d.limits.maxStorageBufferBindingSize, maxStorageBuffersPerShaderStage: d.limits.maxStorageBuffersPerShaderStage, maxComputeWorkgroupStorageSize: d.limits.maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup: d.limits.maxComputeInvocationsPerWorkgroup, minStorageBufferOffsetAlignment: d.limits.minStorageBufferOffsetAlignment, minUniformBufferOffsetAlignment: d.limits.minUniformBufferOffsetAlignment, maxComputeWorkgroupsPerDimension: d.limits.maxComputeWorkgroupsPerDimension }));
console.log("device.features:", JSON.stringify([...d.features]));
d.destroy();
