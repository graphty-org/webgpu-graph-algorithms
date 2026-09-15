// PERF review probe 2: the plan's G7 near-field kernel dispatched in node-index order (as 7.7 specifies)
// versus dispatched in cell-sorted order, and stride-3 f32 positions versus packed vec4 (xyz + mass).
// Grid built on the CPU (G = 512 over the bbox, nearMax = 64, Horvitz-Thompson offset) so only the
// kernel's memory pattern differs. Also reports finest-cell occupancy statistics for each distribution.
import { createRequire } from "node:module";
const require = createRequire("/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/package.json");
const { create, globals } = await import(require.resolve("webgpu"));
Object.assign(globalThis, globals);
const gpu = create(["adapter=NVIDIA"]);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
console.log("adapter:", adapter.info.vendor, adapter.info.architecture);

let seed = 12345; const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
const gauss = () => { const u = rnd() || 1e-9, v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

function makePositions(n, kind) {
  const p = new Float32Array(3 * n);
  const centers = []; for (let c = 0; c < 100; c++) centers.push([(rnd() * 1.6 - 0.8), (rnd() * 1.6 - 0.8)]);
  for (let i = 0; i < n; i++) {
    let x, y;
    if (kind === "uniform") { x = rnd() * 2 - 1; y = rnd() * 2 - 1; }
    else {
      const r = rnd();
      if (kind === "clustered+outliers" && r < 0.002) { const a = rnd() * 2 * Math.PI, d = 2 + rnd() * 2; x = Math.cos(a) * d; y = Math.sin(a) * d; }
      else if (r < 0.05) { x = rnd() * 2 - 1; y = rnd() * 2 - 1; }
      else { const c = centers[(rnd() * 100) | 0]; x = c[0] + gauss() * 0.03; y = c[1] + gauss() * 0.03; }
    }
    p[3 * i] = x; p[3 * i + 1] = y; p[3 * i + 2] = 0;
  }
  return p;
}

function buildGrid(p, n, G) {
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (let i = 0; i < n; i++) { const x = p[3 * i], y = p[3 * i + 1]; if (x < minx) minx = x; if (x > maxx) maxx = x; if (y < miny) miny = y; if (y > maxy) maxy = y; }
  const ext = Math.max(maxx - minx, maxy - miny) * 1.01; const cell = ext / G;
  const ox = minx - (ext - (maxx - minx)) / 2, oy = miny - (ext - (maxy - miny)) / 2;
  const key = new Uint32Array(n); const count = new Uint32Array(G * G + 1);
  for (let i = 0; i < n; i++) {
    const cx = Math.min(G - 1, Math.max(0, Math.floor((p[3 * i] - ox) / cell))), cy = Math.min(G - 1, Math.max(0, Math.floor((p[3 * i + 1] - oy) / cell)));
    key[i] = cy * G + cx; count[key[i] + 1]++;
  }
  const cellStart = new Uint32Array(G * G + 1); for (let c = 0; c < G * G; c++) cellStart[c + 1] = cellStart[c] + count[c + 1];
  const cursor = cellStart.slice(0, G * G); const sortedIdx = new Uint32Array(n);
  for (let i = 0; i < n; i++) sortedIdx[cursor[key[i]]++] = i;
  // occupancy stats
  let maxOcc = 0, nodesOver64 = 0, pairEvals = 0, occupied = 0;
  for (let c = 0; c < G * G; c++) { const k = cellStart[c + 1] - cellStart[c]; if (k > 0) occupied++; if (k > maxOcc) maxOcc = k; if (k > 64) nodesOver64 += k; }
  // near-field pair evaluations per node = sum over 3x3 of min(count, 64)
  for (let c = 0; c < G * G; c++) { const k = cellStart[c + 1] - cellStart[c]; if (k === 0) continue; const cx = c % G, cy = (c / G) | 0; let s = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const nx = cx + dx, ny = cy + dy; if (nx < 0 || ny < 0 || nx >= G || ny >= G) continue; const cc = ny * G + nx; s += Math.min(64, cellStart[cc + 1] - cellStart[cc]); }
    pairEvals += k * s; }
  return { key, sortedIdx, cellStart, ox, oy, cell, stats: { maxOcc, fracNodesInCellsOver64: +(nodesOver64 / n).toFixed(3), occupiedCells: occupied, meanPairEvalsPerNode: +(pairEvals / n).toFixed(1) } };
}

const wgsl = (order, packed) => `
struct P { n: u32, g: u32, iter: u32, nearMax: u32, ox: f32, oy: f32, cell: f32, k: f32 };
@group(0) @binding(0) var<uniform> p: P;
${packed ? "@group(0) @binding(1) var<storage, read> pos4: array<vec4<f32>>;" : "@group(0) @binding(1) var<storage, read> pos: array<f32>;"}
@group(0) @binding(2) var<storage, read> mass: array<f32>;
@group(0) @binding(3) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(4) var<storage, read> cellStart: array<u32>;
@group(0) @binding(5) var<storage, read_write> force: array<f32>;
fn lowbias32(x0: u32) -> u32 { var x = x0; x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u; return x; }
fn load(i: u32) -> vec4<f32> { ${packed ? "return pos4[i];" : "return vec4<f32>(pos[3u*i], pos[3u*i+1u], pos[3u*i+2u], mass[i]);"} }
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x; if (t >= p.n) { return; }
  let i = ${order === "sorted" ? "sortedIdx[t]" : "t"};
  let me = load(i); let pi = me.xyz; let mi = me.w;
  let cx = i32(clamp(floor((pi.x - p.ox) / p.cell), 0.0, f32(p.g - 1u)));
  let cy = i32(clamp(floor((pi.y - p.oy) / p.cell), 0.0, f32(p.g - 1u)));
  var f = vec3<f32>(0.0);
  for (var dy = -1; dy <= 1; dy++) { for (var dx = -1; dx <= 1; dx++) {
    let nx = cx + dx; let ny = cy + dy;
    if (nx < 0 || ny < 0 || nx >= i32(p.g) || ny >= i32(p.g)) { continue; }
    let c = u32(ny) * p.g + u32(nx);
    let s0 = cellStart[c]; let cnt = cellStart[c + 1u] - s0;
    let s = min(cnt, p.nearMax);
    var off = 0u; if (cnt > p.nearMax) { off = lowbias32(c ^ (p.iter * 0x9E3779B9u)) % cnt; }
    var acc = vec3<f32>(0.0);
    for (var q = 0u; q < s; q++) {
      let j = sortedIdx[s0 + (off + q) % cnt];
      if (j == i) { continue; }
      let o = load(j);
      let d = pi - o.xyz; var d2 = dot(d, d); d2 = max(d2, 1.0e-4);
      acc = acc + d * (p.k * mi * o.w / d2);
    }
    f = f + acc * (f32(cnt) / f32(s));
  } }
  force[3u*i] = f.x; force[3u*i+1u] = f.y; force[3u*i+2u] = f.z;
}`;

async function run(n, kind) {
  const G = 512;
  const p = makePositions(n, kind);
  const grid = buildGrid(p, n, G);
  const mass = new Float32Array(n); for (let i = 0; i < n; i++) mass[i] = 1 + ((rnd() * 20) | 0);
  const p4 = new Float32Array(4 * n); for (let i = 0; i < n; i++) { p4[4 * i] = p[3 * i]; p4[4 * i + 1] = p[3 * i + 1]; p4[4 * i + 2] = 0; p4[4 * i + 3] = mass[i]; }
  const mk = (arr, usage = GPUBufferUsage.STORAGE) => { const b = device.createBuffer({ size: arr.byteLength, usage: usage | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, arr); return b; };
  const bufs = { pos: mk(p), pos4: mk(p4), mass: mk(mass), sortedIdx: mk(grid.sortedIdx), cellStart: mk(grid.cellStart), force: device.createBuffer({ size: 12 * n, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }) };
  const ub = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const setU = (iter) => { const dv = new DataView(new ArrayBuffer(32)); dv.setUint32(0, n, true); dv.setUint32(4, G, true); dv.setUint32(8, iter, true); dv.setUint32(12, 64, true); dv.setFloat32(16, grid.ox, true); dv.setFloat32(20, grid.oy, true); dv.setFloat32(24, grid.cell, true); dv.setFloat32(28, 2.0, true); device.queue.writeBuffer(ub, 0, dv.buffer); };
  const results = { n, kind, ...grid.stats };
  for (const [order, packed] of [["index", false], ["sorted", false], ["sorted", true]]) {
    const pipe = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: wgsl(order, packed) }), entryPoint: "main" } });
    const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: ub } }, { binding: 1, resource: { buffer: packed ? bufs.pos4 : bufs.pos } }, ...(packed ? [] : [{ binding: 2, resource: { buffer: bufs.mass } }]),
      { binding: 3, resource: { buffer: bufs.sortedIdx } }, { binding: 4, resource: { buffer: bufs.cellStart } }, { binding: 5, resource: { buffer: bufs.force } }] });
    const step = async (iter) => { setU(iter); const e = device.createCommandEncoder(); const ps = e.beginComputePass(); ps.setPipeline(pipe); ps.setBindGroup(0, bg); ps.dispatchWorkgroups(Math.ceil(n / 256)); ps.end(); device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone(); };
    await step(0); await step(1);
    const iters = 10; const t0 = performance.now(); for (let it = 0; it < iters; it++) await step(it + 2); const t1 = performance.now();
    results[`${order}${packed ? "+vec4" : ""}Ms`] = +((t1 - t0) / iters).toFixed(2);
  }
  for (const b of Object.values(bufs)) b.destroy(); ub.destroy();
  console.log(JSON.stringify(results));
}
for (const n of [262144, 1000000]) for (const kind of ["uniform", "clustered", "clustered+outliers"]) await run(n, kind);
device.destroy();
