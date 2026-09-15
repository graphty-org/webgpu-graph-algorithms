// DESIGN-review probe: does Tint (Dawn-node 0.4.0) reject workgroupBarrier() / subgroupAdd() inside
// non-uniform control flow, as plan 7.6's sketch `if (valid) { epilogue(...) }` would do?
import { createRequire } from "node:module";
const require = createRequire("/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/package.json");
const { create, globals } = await import(require.resolve("webgpu"));
Object.assign(globalThis, globals);
const which = process.argv[2] ?? "";
const gpu = create(which ? [`adapter=${which}`] : []);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has("subgroups") ? ["subgroups"] : [] });
async function tryPipeline(label, code) {
    device.pushErrorScope("validation");
    const mod = device.createShaderModule({ code });
    const info = await mod.getCompilationInfo();
    const msgs = info.messages.map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message.split("\n")[0]}`);
    let pipe = null; let perr = null;
    try { pipe = await device.createComputePipelineAsync({ layout: "auto", compute: { module: mod, entryPoint: "main" } }); } catch (e) { perr = e.message.split("\n")[0]; }
    const scope = await device.popErrorScope();
    console.log(`[${label}] messages: ${msgs.length ? msgs.join(" | ") : "none"}; pipeline: ${pipe ? "ok" : "FAILED " + perr}; scope: ${scope ? scope.message.split("\n")[0].slice(0, 200) : "none"}`);
}
const head = `struct P { n: u32, a: u32, b: u32, c: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
var<workgroup> red: array<f32, 256>;`;
// A: barrier inside a per-invocation branch (plan 7.6 sketch shape)
await tryPipeline("A barrier in if(valid) [non-uniform]", `${head}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let valid = g.x < p.n;
    var f = 0.0;
    if (valid) { f = f32(g.x); red[l.x] = f; workgroupBarrier(); if (l.x == 0u) { out[g.x / 256u] = red[0] + red[1]; } }
}`);
// B: barrier outside the branch (the fix)
await tryPipeline("B barrier outside if [uniform]", `${head}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let valid = g.x < p.n;
    var f = 0.0;
    if (valid) { f = f32(g.x); }
    red[l.x] = f; workgroupBarrier(); if (l.x == 0u) { out[g.x / 256u] = red[0] + red[1]; }
}`);
// C: early return before a barrier keyed on a per-invocation id
await tryPipeline("C early return then barrier [non-uniform]", `${head}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    if (g.x >= p.n) { return; }
    red[l.x] = f32(g.x); workgroupBarrier(); if (l.x == 0u) { out[g.x / 256u] = red[0]; }
}`);
// D: early return keyed on workgroup_id only (uniform per workgroup)
await tryPipeline("D early return on workgroup_id then barrier [uniform]", `${head}
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) w: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    if (w.x >= p.n) { return; }
    red[l.x] = f32(w.x); workgroupBarrier(); if (l.x == 0u) { out[w.x] = red[0]; }
}`);
if (device.features.has("subgroups")) {
    await tryPipeline("E subgroupAdd in if(valid) [non-uniform]", `enable subgroups;
${head}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let valid = g.x < p.n;
    if (valid) { let s = subgroupAdd(f32(g.x)); out[g.x] = s; }
}`);
    await tryPipeline("F subgroupAdd outside if [uniform]", `enable subgroups;
${head}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let valid = g.x < p.n;
    var v = 0.0; if (valid) { v = f32(g.x); }
    let s = subgroupAdd(v); if (valid) { out[g.x] = s; }
}`);
}
device.destroy();
