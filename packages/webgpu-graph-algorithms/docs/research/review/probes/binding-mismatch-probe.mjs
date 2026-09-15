// MAINT review probe: is a TS BindingSpec / WGSL declaration mismatch (read-only layout vs
// read_write shader var; binding index off by one) detected at pipeline creation with an
// EXPLICIT layout, and how? (Plan 5.1 uses explicit layouts; 3.5 / 5.1 keep the two declarations
// in separate places with no generator.)
import { createRequire } from "node:module";
const require = createRequire("/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/package.json");
const dawn = await import(require.resolve("webgpu"));
Object.assign(globalThis, dawn.globals);
const gpu = dawn.create(process.argv.slice(2));
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice();
device.addEventListener("uncapturederror", (ev) => console.log("UNCAPTURED:", ev.error.message.split("\n")[0]));
const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) { if (id.x < 64u) { o[id.x] = a[id.x]; } }` });
async function tryLayout(label, entries) {
    const bgl = device.createBindGroupLayout({ entries });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    try { await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: "main" } }); console.log(label, "-> created"); }
    catch (e) { console.log(label, "-> rejected:", String(e).split("\n")[0].slice(0, 160)); }
}
const C = GPUShaderStage.COMPUTE;
await tryLayout("correct", [{ binding: 0, visibility: C, buffer: { type: "read-only-storage" } }, { binding: 1, visibility: C, buffer: { type: "storage" } }]);
await tryLayout("access-mismatch (layout read-only, shader read_write)", [{ binding: 0, visibility: C, buffer: { type: "read-only-storage" } }, { binding: 1, visibility: C, buffer: { type: "read-only-storage" } }]);
await tryLayout("index-mismatch (layout 0,2; shader 0,1)", [{ binding: 0, visibility: C, buffer: { type: "read-only-storage" } }, { binding: 2, visibility: C, buffer: { type: "storage" } }]);
await tryLayout("looser (layout storage for a read var)", [{ binding: 0, visibility: C, buffer: { type: "storage" } }, { binding: 1, visibility: C, buffer: { type: "storage" } }]);
device.destroy();
