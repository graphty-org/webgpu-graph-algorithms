// PERF review probe 3: the plan's G6 far-field kernel (cosmos exact-once tiling, 4^2..512^2 pyramid,
// 196 cell evaluations per node) in node-index order (as 7.7 specifies) versus cell-sorted order;
// and the per-dispatch overhead of tiny dispatches inside one compute pass (plan 7.4: "~5-10 us").
import { createRequire } from "node:module";
const require = createRequire("/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/package.json");
const { create, globals } = await import(require.resolve("webgpu"));
Object.assign(globalThis, globals);
const gpu = create(["adapter=NVIDIA"]);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice();
console.log("adapter:", adapter.info.vendor, adapter.info.architecture);
let seed = 777; const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
const gauss = () => { const u = rnd() || 1e-9, v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const G = 512, LEVELS = 8; // 4,8,...,512
const levelOff = []; let cells = 0; for (let l = 0; l < LEVELS; l++) { levelOff.push(cells); cells += (4 << l) * (4 << l); }
console.log("pyramid cells", cells, "bytes", cells * 16);

function build(n, kind) {
  const p = new Float32Array(3 * n), mass = new Float32Array(n);
  const centers = []; for (let c = 0; c < 100; c++) centers.push([(rnd() * 1.6 - 0.8), (rnd() * 1.6 - 0.8)]);
  for (let i = 0; i < n; i++) { let x, y; if (kind === "uniform") { x = rnd() * 2 - 1; y = rnd() * 2 - 1; } else { const r = rnd(); if (r < 0.05) { x = rnd() * 2 - 1; y = rnd() * 2 - 1; } else { const c = centers[(rnd() * 100) | 0]; x = c[0] + gauss() * 0.03; y = c[1] + gauss() * 0.03; } }
    p[3 * i] = x; p[3 * i + 1] = y; mass[i] = 1 + ((rnd() * 20) | 0); }
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (let i = 0; i < n; i++) { const x = p[3 * i], y = p[3 * i + 1]; if (x < minx) minx = x; if (x > maxx) maxx = x; if (y < miny) miny = y; if (y > maxy) maxy = y; }
  const ext = Math.max(maxx - minx, maxy - miny) * 1.01, cell = ext / G, ox = minx - (ext - (maxx - minx)) / 2, oy = miny - (ext - (maxy - miny)) / 2;
  const key = new Uint32Array(n), count = new Uint32Array(G * G + 1);
  const pyr = new Float32Array(cells * 4);
  for (let i = 0; i < n; i++) { const cx = Math.min(G - 1, Math.max(0, Math.floor((p[3 * i] - ox) / cell))), cy = Math.min(G - 1, Math.max(0, Math.floor((p[3 * i + 1] - oy) / cell))); key[i] = cy * G + cx; count[key[i] + 1]++;
    const o = (levelOff[LEVELS - 1] + key[i]) * 4; pyr[o] += mass[i] * p[3 * i]; pyr[o + 1] += mass[i] * p[3 * i + 1]; pyr[o + 3] += mass[i]; }
  for (let l = LEVELS - 2; l >= 0; l--) { const g = 4 << l, gc = g * 2; for (let y = 0; y < g; y++) for (let x = 0; x < g; x++) { const o = (levelOff[l] + y * g + x) * 4; for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) { const oc = (levelOff[l + 1] + (2 * y + dy) * gc + (2 * x + dx)) * 4; pyr[o] += pyr[oc]; pyr[o + 1] += pyr[oc + 1]; pyr[o + 3] += pyr[oc + 3]; } } }
  const cellStart = new Uint32Array(G * G + 1); for (let c = 0; c < G * G; c++) cellStart[c + 1] = cellStart[c] + count[c + 1];
  const cursor = cellStart.slice(0, G * G), sortedIdx = new Uint32Array(n); for (let i = 0; i < n; i++) sortedIdx[cursor[key[i]]++] = i;
  return { p, mass, pyr, sortedIdx, ox, oy, cell };
}
const wgsl = (order) => `
struct P { n: u32, g: u32, pad0: u32, pad1: u32, ox: f32, oy: f32, cell: f32, k: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> mass: array<f32>;
@group(0) @binding(3) var<storage, read> pyr: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(5) var<storage, read_write> force: array<f32>;
const LEVELS: u32 = ${LEVELS}u;
fn levelOff(l: u32) -> u32 { var o = 0u; for (var i = 0u; i < l; i++) { let g = 4u << i; o += g * g; } return o; }
fn cellForce(pi: vec3<f32>, mi: f32, c: vec4<f32>, ox: f32, oy: f32) -> vec3<f32> {
  if (c.w <= 0.0) { return vec3<f32>(0.0); }
  let ctr = vec3<f32>(c.x / c.w, c.y / c.w, 0.0);
  let d = pi - ctr; let d2 = dot(d, d) + 0.0025;
  return d * (p.k * mi * c.w / d2);
}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x; if (t >= p.n) { return; }
  let i = ${order === "sorted" ? "sortedIdx[t]" : "t"};
  let pi = vec3<f32>(pos[3u*i], pos[3u*i+1u], 0.0); let mi = mass[i];
  var f = vec3<f32>(0.0);
  // coarsest level: all 16 cells minus the 3x3 around the node
  let c0x = i32(clamp(floor((pi.x - p.ox) / (p.cell * 128.0)), 0.0, 3.0)); let c0y = i32(clamp(floor((pi.y - p.oy) / (p.cell * 128.0)), 0.0, 3.0));
  for (var y = 0; y < 4; y++) { for (var x = 0; x < 4; x++) { if (abs(x - c0x) <= 1 && abs(y - c0y) <= 1) { continue; } f += cellForce(pi, mi, pyr[u32(y * 4 + x)], p.ox, p.oy); } }
  // finer levels: 6x6 block aligned to the parent's 3x3 minus own 3x3
  for (var l = 1u; l < LEVELS; l++) {
    let g = 4u << l; let cs = p.cell * f32(512u / g);
    let cx = i32(clamp(floor((pi.x - p.ox) / cs), 0.0, f32(g - 1u))); let cy = i32(clamp(floor((pi.y - p.oy) / cs), 0.0, f32(g - 1u)));
    let px = cx >> 1; let py = cy >> 1;
    let off = levelOff(l);
    for (var y = 2 * (py - 1); y < 2 * (py + 2); y++) { for (var x = 2 * (px - 1); x < 2 * (px + 2); x++) {
      if (x < 0 || y < 0 || x >= i32(g) || y >= i32(g)) { continue; }
      if (abs(x - cx) <= 1 && abs(y - cy) <= 1) { continue; }
      f += cellForce(pi, mi, pyr[off + u32(y) * g + u32(x)], p.ox, p.oy);
    } }
  }
  force[3u*i] = f.x; force[3u*i+1u] = f.y; force[3u*i+2u] = f.z;
}`;
async function run(n, kind) {
  const b = build(n, kind);
  const mk = (arr, usage = GPUBufferUsage.STORAGE) => { const x = device.createBuffer({ size: arr.byteLength, usage: usage | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(x, 0, arr); return x; };
  const bufs = { pos: mk(b.p), mass: mk(b.mass), pyr: mk(b.pyr), sortedIdx: mk(b.sortedIdx), force: device.createBuffer({ size: 12 * n, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }) };
  const ub = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const dv = new DataView(new ArrayBuffer(32)); dv.setUint32(0, n, true); dv.setUint32(4, G, true); dv.setFloat32(16, b.ox, true); dv.setFloat32(20, b.oy, true); dv.setFloat32(24, b.cell, true); dv.setFloat32(28, 2.0, true); device.queue.writeBuffer(ub, 0, dv.buffer);
  const res = { n, kind };
  for (const order of ["index", "sorted"]) {
    const pipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: wgsl(order) }), entryPoint: "main" } });
    const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ub } }, { binding: 1, resource: { buffer: bufs.pos } }, { binding: 2, resource: { buffer: bufs.mass } }, { binding: 3, resource: { buffer: bufs.pyr } }, ...(order === "sorted" ? [{ binding: 4, resource: { buffer: bufs.sortedIdx } }] : []), { binding: 5, resource: { buffer: bufs.force } }] });
    const step = async () => { const e = device.createCommandEncoder(); const ps = e.beginComputePass(); ps.setPipeline(pipe); ps.setBindGroup(0, bg); ps.dispatchWorkgroups(Math.ceil(n / 256)); ps.end(); device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone(); };
    await step(); await step(); const t0 = performance.now(); for (let k = 0; k < 10; k++) await step(); res[`${order}Ms`] = +((performance.now() - t0) / 10).toFixed(2);
  }
  for (const x of Object.values(bufs)) x.destroy(); ub.destroy();
  console.log(JSON.stringify(res));
}
for (const n of [262144, 1000000]) for (const kind of ["uniform", "clustered"]) await run(n, kind);

device.destroy(); process.exit(0);
// per-dispatch overhead: one pass with N dispatches of a trivial kernel over 1 workgroup and over 4096 workgroups
const triv = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: `@group(0) @binding(0) var<storage, read_write> a: array<u32>; @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < 1024u) { a[g.x] = a[g.x] + 1u; } }` }), entryPoint: "main" } });
const ab = device.createBuffer({ size: 4096, usage: GPUBufferUsage.STORAGE });
const abg = device.createBindGroup({ layout: triv.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ab } }] });
for (const wgs of [1, 4096]) for (const N of [1, 31, 100, 300]) {
  const step = async () => { const e = device.createCommandEncoder(); const ps = e.beginComputePass(); ps.setPipeline(triv); ps.setBindGroup(0, abg); for (let k = 0; k < N; k++) ps.dispatchWorkgroups(wgs); ps.end(); device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone(); };
  await step(); await step(); const t0 = performance.now(); for (let k = 0; k < 20; k++) await step(); const ms = (performance.now() - t0) / 20;
  console.log(`pass with ${String(N).padStart(3)} dispatches x ${String(wgs).padStart(4)} workgroups: ${ms.toFixed(3)} ms  (${(ms * 1000 / N).toFixed(1)} us/dispatch incl. submit latency)`);
}
device.destroy();
