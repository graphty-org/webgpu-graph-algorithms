// VERIFY verifier probe: does webgpu@0.4.0 (Dawn-node) deliver `uncapturederror` to
// device.addEventListener / device.onuncapturederror? Plan 11.2 / 11.5 rely on it
// ("An uncapturederror listener fails the current test").
import { createRequire } from "node:module";
const require = createRequire("/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/package.json");
const { create, globals } = await import(require.resolve("webgpu"));
Object.assign(globalThis, globals);
process.env.XDG_RUNTIME_DIR ??= "/tmp";
const which = process.argv[2] ?? "llvmpipe";
const gpu = create([`adapter=${which}`]);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice();
console.log(JSON.stringify({ adapter: adapter.info.vendor + "/" + adapter.info.architecture,
  hasAddEventListener: typeof device.addEventListener, hasOnUncaptured: "onuncapturederror" in device }));
let got = [];
if (typeof device.addEventListener === "function") device.addEventListener("uncapturederror", (e) => got.push("listener:" + e.error?.constructor?.name + ":" + String(e.error?.message).slice(0, 60)));
try { device.onuncapturederror = (e) => got.push("onprop:" + e.error?.constructor?.name); } catch (e) { got.push("onprop-set-threw:" + e.message); }
// deliberately broken: a bind group whose buffer is too small for the layout's min binding size
const mod = device.createShaderModule({ code: `@group(0) @binding(0) var<storage, read_write> b: array<u32, 1024>; @compute @workgroup_size(1) fn m() { b[0] = 1u; }` });
const pipe = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "m" } });
const small = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: small } }] });
await device.queue.onSubmittedWorkDone();
await new Promise((r) => setTimeout(r, 200));
// also test that popErrorScope still works as the alternative
device.pushErrorScope("validation");
device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: small } }] });
const scoped = await device.popErrorScope();
console.log(JSON.stringify({ uncapturedEvents: got, popErrorScope: scoped ? scoped.constructor.name : null }));
device.destroy();
