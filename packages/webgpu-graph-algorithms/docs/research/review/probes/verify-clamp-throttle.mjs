// DESIGN-verdicts probe (DESIGN-1 / DESIGN-2): does the plan's grid-tier clamp `|dp| <= 2 * cellSize`
// with cellSize = extent / G recomputed every iteration throttle the FA2 expansion from the [-1, 1) seed?
// CPU model of plan 7.2 (paper law 1/d, force-based swing / traction, Gephi estimateFactor), 2D,
// random graph with mean degree 10, mass = degree + 1, scalingRatio 2, gravity 1 to the centroid.
// Runs the same start (a) unclamped (exact tier) and (b) clamped to 2 * (1.01 * extent / G) (grid tier as written).
const n = Number(process.argv[2] ?? 2048);
const G = Number(process.argv[3] ?? 512);
const iters = Number(process.argv[4] ?? 100);
let s = 12345;
function rnd() { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; }
const deg = new Int32Array(n);
const edges = [];
for (let e = 0; e < n * 5; e++) { const a = Math.floor(rnd() * n); const b = Math.floor(rnd() * n); if (a !== b) { edges.push([a, b]); deg[a]++; deg[b]++; } }
const mass = Float64Array.from(deg, (d) => d + 1);
const seedX = new Float64Array(n), seedY = new Float64Array(n);
for (let i = 0; i < n; i++) { seedX[i] = rnd() * 2 - 1; seedY[i] = rnd() * 2 - 1; }

function run(clamped) {
    const x = Float64Array.from(seedX), y = Float64Array.from(seedY);
    const fx = new Float64Array(n), fy = new Float64Array(n), ox = new Float64Array(n), oy = new Float64Array(n);
    let speed = 1, eff = 1;
    const k = 2, gravity = 1, jitterTol = 1;
    const out = [];
    for (let it = 0; it < iters; it++) {
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, cx = 0, cy = 0;
        for (let i = 0; i < n; i++) { minX = Math.min(minX, x[i]); maxX = Math.max(maxX, x[i]); minY = Math.min(minY, y[i]); maxY = Math.max(maxY, y[i]); cx += x[i]; cy += y[i]; }
        cx /= n; cy /= n;
        const extent = Math.max(maxX - minX, maxY - minY);
        const cellSize = (1.01 * extent) / G;
        fx.fill(0); fy.fill(0);
        for (const [a, b] of edges) { const dx = x[b] - x[a], dy = y[b] - y[a]; fx[a] += dx; fy[a] += dy; fx[b] -= dx; fy[b] -= dy; }
        for (let i = 0; i < n; i++) {
            let rx = 0, ry = 0;
            for (let j = 0; j < n; j++) {
                if (j === i) continue;
                const dx = x[i] - x[j], dy = y[i] - y[j];
                let d2 = dx * dx + dy * dy;
                if (d2 < 1e-8) continue;
                d2 = Math.max(d2, 1e-4);
                const f = (k * mass[i] * mass[j]) / d2;
                rx += dx * f; ry += dy * f;
            }
            fx[i] += rx; fy[i] += ry;
            const qx = x[i] - cx, qy = y[i] - cy; const ql = Math.hypot(qx, qy);
            if (ql > 0.01) { fx[i] -= (gravity * mass[i] * qx) / ql; fy[i] -= (gravity * mass[i] * qy) / ql; }
        }
        let swing = 0, traction = 0;
        for (let i = 0; i < n; i++) { swing += mass[i] * Math.hypot(fx[i] - ox[i], fy[i] - oy[i]); traction += 0.5 * mass[i] * Math.hypot(fx[i] + ox[i], fy[i] + oy[i]); }
        const optJ = 0.05 * Math.sqrt(n), minJ = Math.sqrt(optJ);
        let jt = jitterTol * Math.max(minJ, Math.min(10, (optJ * traction) / (n * n)));
        if (swing / traction > 2) { if (eff > 0.05) eff *= 0.5; jt = Math.max(jt, jitterTol); }
        const target = swing === 0 ? Infinity : (jt * eff * traction) / swing;
        if (swing > jt * traction) { if (eff > 0.05) eff *= 0.7; } else if (speed < 1000) eff *= 1.3;
        speed = speed + Math.min(target - speed, 0.5 * speed);
        let maxStep = 0, meanStep = 0;
        for (let i = 0; i < n; i++) {
            const sw = mass[i] * Math.hypot(fx[i] - ox[i], fy[i] - oy[i]);
            const factor = speed / (1 + Math.sqrt(speed * sw));
            let dx = fx[i] * factor, dy = fy[i] * factor;
            const l = Math.hypot(dx, dy);
            if (clamped && l > 2 * cellSize) { dx *= (2 * cellSize) / l; dy *= (2 * cellSize) / l; }
            const ll = Math.hypot(dx, dy); maxStep = Math.max(maxStep, ll); meanStep += ll;
            x[i] += dx; y[i] += dy; ox[i] = fx[i]; oy[i] = fy[i];
        }
        if (it % 10 === 0 || it === iters - 1) out.push({ it, extent: +extent.toFixed(4), cellSize: +cellSize.toExponential(2), speed: +speed.toFixed(3), eff: +eff.toFixed(4), maxStep: +maxStep.toExponential(2), meanStep: +(meanStep / n).toExponential(2) });
    }
    return out;
}
console.log(`n=${n} G=${G} iters=${iters} edges=${edges.length}`);
console.log("(a) exact tier, no clamp:"); for (const r of run(false)) console.log(JSON.stringify(r));
console.log("(b) grid tier as written, |dp| <= 2 * cellSize with cellSize = 1.01 * extent / G:"); for (const r of run(true)) console.log(JSON.stringify(r));
