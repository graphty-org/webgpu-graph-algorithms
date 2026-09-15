// PERF review probe 4 (CPU only): finest-cell occupancy of the plan's bbox-derived 512^2 grid
// when a few nodes sit far from the core, as FA2's 1/d repulsion + constant gravity produces
// for isolated nodes / small components. Reports the fraction of nodes in cells above nearMax.
let seed = 99; const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
const gauss = () => { const u = rnd() || 1e-9, v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
function stats(n, outlierFrac, outlierRadius) {
  const G = 512, nearMax = 64;
  const x = new Float32Array(n), y = new Float32Array(n);
  const centers = []; for (let c = 0; c < 100; c++) centers.push([(rnd() * 1.6 - 0.8), (rnd() * 1.6 - 0.8)]);
  let nOut = 0;
  for (let i = 0; i < n; i++) {
    if (rnd() < outlierFrac) { const a = rnd() * 2 * Math.PI; x[i] = Math.cos(a) * outlierRadius; y[i] = Math.sin(a) * outlierRadius; nOut++; continue; }
    const c = centers[(rnd() * 100) | 0]; x[i] = c[0] + gauss() * 0.03; y[i] = c[1] + gauss() * 0.03;
  }
  if (nOut === 0 && outlierFrac > 0) { x[0] = outlierRadius; y[0] = 0; nOut = 1; }
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (let i = 0; i < n; i++) { if (x[i] < minx) minx = x[i]; if (x[i] > maxx) maxx = x[i]; if (y[i] < miny) miny = y[i]; if (y[i] > maxy) maxy = y[i]; }
  const ext = Math.max(maxx - minx, maxy - miny) * 1.01, cell = ext / G;
  const count = new Uint32Array(G * G);
  for (let i = 0; i < n; i++) { const cx = Math.min(G - 1, Math.max(0, Math.floor((x[i] - minx) / cell))), cy = Math.min(G - 1, Math.max(0, Math.floor((y[i] - miny) / cell))); count[cy * G + cx]++; }
  let maxOcc = 0, over = 0, occupied = 0; for (let c = 0; c < G * G; c++) { if (count[c] > 0) occupied++; if (count[c] > maxOcc) maxOcc = count[c]; if (count[c] > nearMax) over += count[c]; }
  return { n, outliers: nOut, outlierRadiusOverCoreRadius: outlierRadius, cellOverCoreRadius: +(cell).toFixed(4), occupiedCells: occupied, maxOcc, fracNodesInCellsOverNearMax: +(over / n).toFixed(3) };
}
for (const n of [100000, 1000000]) {
  console.log(JSON.stringify(stats(n, 0, 0)));
  for (const r of [2, 10, 100, 1000]) console.log(JSON.stringify(stats(n, 1 / n, r)));   // ONE far node
}
