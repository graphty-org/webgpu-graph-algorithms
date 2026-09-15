// VERIFY-review probe: is an exact-tile FA2-shaped repulsion iteration (division, sqrt, max) BIT-IDENTICAL
// across lavapipe and NVIDIA under Dawn 0.4.0? Plan 11.5 / G1 requires "identical checksums on lavapipe,
// SwiftShader and NVIDIA" for the whole skeleton file, which includes one FA2 exact-tile iteration (P1).
// Usage: node verify-checksum-cross-adapter.mjs [adapter-substring] [n]
import { createRequire } from "node:module";
import { createHash } from "node:crypto";
const require = createRequire("/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/package.json");
const { create, globals } = await import(require.resolve("webgpu"));
Object.assign(globalThis, globals);
process.env.XDG_RUNTIME_DIR ??= "/tmp";
const which = process.argv[2] ?? "";
const n = Number(process.argv[3] ?? 2048);
const gpu = create(which ? [`adapter=${which}`] : []);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice();
const code = `
struct P { n: u32, k: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> mass: array<f32>;
@group(0) @binding(3) var<storage, read_write> force: array<f32>;
@group(0) @binding(4) var<storage, read_write> swing: array<f32>;
var<workgroup> tile: array<vec4<f32>, 256>;
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
        d2 = max(d2, 1.0e-4);
        let k = p.k * mi * o.w;
        f = f + d * (k / d2);                 // paper law: |F| = k m_i m_j / d
      }
    }
    workgroupBarrier();
  }
  if (valid) {
    // gravity toward origin, regular: -g m q/|q| with |q| = sqrt
    let q = pi; let ql = length(q);
    if (ql > 0.01) { f = f - 1.0 * mi * q / ql; }
    force[3u*i] = f.x; force[3u*i+1u] = f.y; force[3u*i+2u] = f.z;
    swing[i] = mi * length(f);              // per-node swing-like quantity (sqrt)
  }
}`;
const mod = device.createShaderModule({ code });
const pipe = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
// seeded LCG positions in [-1,1), masses degree-like 1..8
let s = 12345 >>> 0; const lcg = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
const pos = new Float32Array(3 * n); for (let i = 0; i < pos.length; i++) pos[i] = lcg() * 2 - 1;
const mass = new Float32Array(n); for (let i = 0; i < n; i++) mass[i] = 1 + Math.floor(lcg() * 8);
const mk = (size, usage) => device.createBuffer({ size, usage });
const posBuf = mk(pos.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST); device.queue.writeBuffer(posBuf, 0, pos);
const massBuf = mk(mass.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST); device.queue.writeBuffer(massBuf, 0, mass);
const forceBuf = mk(pos.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
const swingBuf = mk(mass.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
const ub = mk(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
device.queue.writeBuffer(ub, 0, new Uint32Array([n, 0, 0, 0])); device.queue.writeBuffer(ub, 4, new Float32Array([2.0]));
const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
  { binding: 0, resource: { buffer: ub } }, { binding: 1, resource: { buffer: posBuf } }, { binding: 2, resource: { buffer: massBuf } },
  { binding: 3, resource: { buffer: forceBuf } }, { binding: 4, resource: { buffer: swingBuf } } ] });
const stF = mk(pos.byteLength, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
const stS = mk(mass.byteLength, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
const e = device.createCommandEncoder();
const ps = e.beginComputePass(); ps.setPipeline(pipe); ps.setBindGroup(0, bg); ps.dispatchWorkgroups(Math.ceil(n / 256)); ps.end();
e.copyBufferToBuffer(forceBuf, 0, stF, 0, pos.byteLength); e.copyBufferToBuffer(swingBuf, 0, stS, 0, mass.byteLength);
device.queue.submit([e.finish()]);
await stF.mapAsync(GPUMapMode.READ); await stS.mapAsync(GPUMapMode.READ);
const fBytes = new Uint8Array(stF.getMappedRange()).slice(); const sBytes = new Uint8Array(stS.getMappedRange()).slice();
stF.unmap(); stS.unmap();
const f = new Float32Array(fBytes.buffer); const sw = new Float32Array(sBytes.buffer);
let swingSum = 0; for (let i = 0; i < n; i++) swingSum += sw[i];
console.log(JSON.stringify({ adapter: adapter.info.vendor + "/" + adapter.info.architecture, n,
  forceSha256: createHash("sha256").update(fBytes).digest("hex").slice(0, 16),
  swingSha256: createHash("sha256").update(sBytes).digest("hex").slice(0, 16),
  swingSumF64: swingSum, f0: [f[0], f[1], f[2]] }));
device.destroy();
