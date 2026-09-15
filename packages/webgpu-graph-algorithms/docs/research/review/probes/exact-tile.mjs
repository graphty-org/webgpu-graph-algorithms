// PERF review probe 1: the plan's K3 body (mass, floor, kick branch, law select) versus the
// dawn-perf.mjs probe body, at the sizes the 7.8 crossover rule names. Wall time per iteration
// around submit + onSubmittedWorkDone, 10 iterations after warm-up.
import { createRequire } from "node:module";
const require = createRequire("/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/package.json");
const { create, globals } = await import(require.resolve("webgpu"));
Object.assign(globalThis, globals);
const gpu = create(["adapter=NVIDIA"]);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice();
console.log("adapter:", adapter.info.vendor, adapter.info.architecture, adapter.info.description);

const probeBody = `
struct P { n: u32, k: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> mass: array<f32>;
@group(0) @binding(3) var<storage, read_write> disp: array<f32>;
var<workgroup> tile: array<vec4<f32>, 256>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = gid.x;
  var me = vec3<f32>(0.0);
  if (i < p.n) { me = vec3<f32>(pos[3u*i], pos[3u*i+1u], pos[3u*i+2u]); }
  var acc = vec3<f32>(0.0); if (mass[0] < 0.0) { acc.x = 1.0; }
  for (var base = 0u; base < p.n; base += 256u) {
    let j = base + lid.x;
    if (j < p.n) { tile[lid.x] = vec4<f32>(pos[3u*j], pos[3u*j+1u], pos[3u*j+2u], 1.0); } else { tile[lid.x] = vec4<f32>(0.0); }
    workgroupBarrier();
    for (var t = 0u; t < 256u; t++) {
      let o = tile[t];
      if (o.w > 0.0) { let d = me - o.xyz; let d2 = dot(d, d) + 0.01; acc += d * (p.k / d2); }
    }
    workgroupBarrier();
  }
  if (i < p.n) { disp[3u*i] = acc.x; disp[3u*i+1u] = acc.y; disp[3u*i+2u] = acc.z; }
}`;

// The plan 7.6 body: mass in the tile, jj != i, coincident kick branch, d2 floor, law select.
const fa2Body = `
struct P { n: u32, k: f32, pad0: u32, pad1: u32 };
override REPULSION_LAW: u32 = 0u;
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> mass: array<f32>;
@group(0) @binding(3) var<storage, read_write> disp: array<f32>;
var<workgroup> tile: array<vec4<f32>, 256>;
fn lowbias32(x0: u32) -> u32 { var x = x0; x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u; return x; }
fn kick(i: u32, j: u32, k: f32) -> vec3<f32> { let h = lowbias32((i * 0x9E3779B9u) ^ j); let a = f32(h & 0xFFFFu) * 9.5874e-5; return vec3<f32>(cos(a), sin(a), 0.0) * (k / 1.0e-4); }
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = gid.x;
  let valid = i < p.n;
  var pi = vec3<f32>(0.0); var mi = 0.0;
  if (valid) { pi = vec3<f32>(pos[3u*i], pos[3u*i+1u], pos[3u*i+2u]); mi = mass[i]; }
  var f = vec3<f32>(0.0);
  let tiles = (p.n + 255u) / 256u;
  for (var t = 0u; t < tiles; t++) {
    let j = t * 256u + lid.x;
    if (j < p.n) { tile[lid.x] = vec4<f32>(pos[3u*j], pos[3u*j+1u], pos[3u*j+2u], mass[j]); } else { tile[lid.x] = vec4<f32>(0.0); }
    workgroupBarrier();
    for (var s = 0u; s < 256u; s++) {
      let o = tile[s];
      let jj = t * 256u + s;
      if (o.w > 0.0 && jj != i) {
        let d = pi - o.xyz;
        var d2 = dot(d, d);
        if (d2 < 1.0e-8) { f = f + kick(i, jj, mi * o.w); continue; }
        d2 = max(d2, 1.0e-4);
        let k = p.k * mi * o.w;
        if (REPULSION_LAW == 0u) { f = f + d * (k / d2); } else { f = f + d * (k / (d2 * sqrt(d2))); }
      }
    }
    workgroupBarrier();
  }
  if (valid) { disp[3u*i] = f.x; disp[3u*i+1u] = f.y; disp[3u*i+2u] = f.z; }
}`;

async function bench(label, code, n, iters = 10) {
  const mod = device.createShaderModule({ code });
  const pipe = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
  const pos = new Float32Array(3 * n); for (let i = 0; i < pos.length; i++) pos[i] = (Math.random() * 2 - 1) * 50;
  const mass = new Float32Array(n); for (let i = 0; i < n; i++) mass[i] = 1 + Math.floor(Math.random() * 20);
  const mk = (arr, usage) => { const b = device.createBuffer({ size: arr.byteLength, usage: usage | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, arr); return b; };
  const posBuf = mk(pos, GPUBufferUsage.STORAGE), massBuf = mk(mass, GPUBufferUsage.STORAGE);
  const dispBuf = device.createBuffer({ size: pos.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const ub = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(ub, 0, new Uint32Array([n, 0, 0, 0])); device.queue.writeBuffer(ub, 4, new Float32Array([2.0]));
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: ub } }, { binding: 1, resource: { buffer: posBuf } }, { binding: 2, resource: { buffer: massBuf } }, { binding: 3, resource: { buffer: dispBuf } }] });
  const step = async () => { const e = device.createCommandEncoder(); const ps = e.beginComputePass(); ps.setPipeline(pipe); ps.setBindGroup(0, bg); ps.dispatchWorkgroups(Math.ceil(n / 256)); ps.end(); device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone(); };
  await step(); await step();
  const t0 = performance.now(); for (let i = 0; i < iters; i++) await step(); const t1 = performance.now();
  const ms = (t1 - t0) / iters;
  const pairs = Math.ceil(n / 256) * 256 * Math.ceil(n / 256) * 256;
  console.log(`${label.padEnd(6)} n=${String(n).padStart(7)}  ${ms.toFixed(3)} ms/iter  ${(pairs / (ms / 1000) / 1e11).toFixed(2)}e11 pairs/s`);
  posBuf.destroy(); massBuf.destroy(); dispBuf.destroy(); ub.destroy();
  return ms;
}
const out = {};
for (const n of [4096, 8192, 16384, 20000, 32768, 65536, 100000]) {
  const a = await bench("probe", probeBody, n);
  const b = await bench("fa2", fa2Body, n);
  out[n] = { probeMs: +a.toFixed(3), fa2Ms: +b.toFixed(3), ratio: +(b / a).toFixed(2) };
}
console.log(JSON.stringify(out));
device.destroy();
