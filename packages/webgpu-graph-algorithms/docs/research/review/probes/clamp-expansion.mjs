// PERF verifier probe (CPU only): does the plan 7.7 / 7.11 per-iteration clamp `2 * cellSize`
// (cellSize = bbox max extent / G, G = 512) throttle the global expansion of an FA2 layout that is
// seeded in [-1, 1) (plan 7.2)? Exact O(n^2) FA2 with the plan's 7.2 laws (paper 1/d repulsion,
// linear attraction, regular gravity of constant magnitude, mass = deg + 1, Gephi speed controller
// SWING_MODE 0). Reports the bbox extent every 25 iterations, with and without the clamp.
let seed = 7; const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
const n = 1000, avgDeg = 10, G = 512, iters = 300;
const edges = []; for (let e = 0; e < n * avgDeg / 2; e++) { const a = (rnd() * n) | 0; let b = (rnd() * n) | 0; if (a === b) b = (b + 1) % n; edges.push([a, b]); }
const deg = new Uint32Array(n); for (const [a, b] of edges) { deg[a]++; deg[b]++; }
const mass = Float32Array.from(deg, (d) => d + 1);
function run(useClamp, scalingRatio = 2, gravity = 1, jitterTolerance = 1) {
  seed = 7; for (let e = 0; e < edges.length; e++) rnd(), rnd();
  const px = new Float64Array(n), py = new Float64Array(n); for (let i = 0; i < n; i++) { px[i] = rnd() * 2 - 1; py[i] = rnd() * 2 - 1; }
  const fx = new Float64Array(n), fy = new Float64Array(n), ofx = new Float64Array(n), ofy = new Float64Array(n);
  let speed = 1, eff = 1; const out = [];
  for (let it = 0; it < iters; it++) {
    let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity, cx = 0, cy = 0;
    for (let i = 0; i < n; i++) { minx = Math.min(minx, px[i]); maxx = Math.max(maxx, px[i]); miny = Math.min(miny, py[i]); maxy = Math.max(maxy, py[i]); cx += px[i]; cy += py[i]; }
    cx /= n; cy /= n; const extent = Math.max(maxx - minx, maxy - miny) * 1.01; const cellSize = extent / G;
    if (it % 25 === 0 || it === iters - 1) out.push(`${it}:${extent.toFixed(1)}`);
    fx.fill(0); fy.fill(0);
    for (const [a, b] of edges) { const dx = px[b] - px[a], dy = py[b] - py[a]; fx[a] += dx; fy[a] += dy; fx[b] -= dx; fy[b] -= dy; }
    for (let i = 0; i < n; i++) {
      let rx = 0, ry = 0;
      for (let j = 0; j < n; j++) { if (j === i) continue; const dx = px[i] - px[j], dy = py[i] - py[j]; let d2 = dx * dx + dy * dy; d2 = Math.max(d2, 1e-4); const k = scalingRatio * mass[i] * mass[j]; rx += dx * k / d2; ry += dy * k / d2; }
      const qx = px[i] - cx, qy = py[i] - cy, ql = Math.hypot(qx, qy);
      if (ql > 0.01) { rx -= gravity * mass[i] * qx / ql; ry -= gravity * mass[i] * qy / ql; }
      fx[i] += rx; fy[i] += ry;
    }
    let swing = 0, traction = 0;
    for (let i = 0; i < n; i++) { swing += mass[i] * Math.hypot(fx[i] - ofx[i], fy[i] - ofy[i]); traction += 0.5 * mass[i] * Math.hypot(fx[i] + ofx[i], fy[i] + ofy[i]); }
    const optJitter = 0.05 * Math.sqrt(n), minJitter = Math.sqrt(optJitter), maxJitter = 10;
    let jitter = jitterTolerance * Math.max(minJitter, Math.min(maxJitter, optJitter * traction / (n * n)));
    if (swing / traction > 2) { eff = Math.max(eff * 0.5, 0.05); jitter = Math.max(jitter, jitterTolerance); }
    const target = swing === 0 ? Infinity : jitter * eff * traction / swing;
    if (swing > jitter * traction) eff = Math.max(eff * 0.7, 0.05); else if (speed < 1000) eff *= 1.3;
    speed = speed + Math.min(target - speed, 0.5 * speed);
    for (let i = 0; i < n; i++) {
      const sw = mass[i] * Math.hypot(fx[i] - ofx[i], fy[i] - ofy[i]);
      const factor = speed / (1 + Math.sqrt(speed * sw));
      let dx = fx[i] * factor, dy = fy[i] * factor;
      if (useClamp) { const l = Math.hypot(dx, dy), c = 2 * cellSize; if (l > c) { dx *= c / l; dy *= c / l; } }
      px[i] += dx; py[i] += dy; ofx[i] = fx[i]; ofy[i] = fy[i];
    }
  }
  return out.join("  ");
}
console.log(`n=${n} avgDeg=${avgDeg} G=${G}: bbox extent by iteration`);
console.log("no clamp   :", run(false));
console.log("2*cellSize :", run(true));
