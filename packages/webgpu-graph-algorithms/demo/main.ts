/**
 * Standalone browser demo of the exact-tier GPU ForceAtlas2 (P3): a canvas 2D renderer over the simulation's own
 * stride-3 position array, stepped once per animation frame exactly the way graphty-element's frame loop will
 * (fire-and-forget step(), at most maxInFlight batches in flight), with drag / pin through setPosition / setFixed.
 * Served by `pnpm exec vite demo --host --port <9000-9099>`; not part of the package build, the tests or the lint set.
 */

import { installRemoteLog } from "./remote-log.js";
import { fromEdgeArrays, type GraphSnapshot, makeMask, maskSet, maskTest, type U32 } from "@graphty/graph-format";

import { requestGpuContext } from "../src/browser/index.js";
import type { GpuContext } from "../src/context.js";
import { createForceAtlas2 } from "../src/layouts/forceatlas2.js";
import type { ForceAtlas2Stats, GpuLayoutSimulation, GpuLayoutTuning } from "../src/types/layout.js";
import type { ForceAtlas2Options } from "../src/types/options.js";

// ------------------------------------------------------------------ graphs

interface Edges {
    readonly nodeCount: number;
    readonly src: U32;
    readonly dst: U32;
}

function lcg(seed: number): () => number {
    let state = seed >>> 0 || 1;
    return () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

function randomGraph(n: number, m: number, seed: number): Edges {
    const rnd = lcg(seed);
    const src = new Uint32Array(m);
    const dst = new Uint32Array(m);
    for (let e = 0; e < m; e++) {
        let a = Math.floor(rnd() * n);
        let b = Math.floor(rnd() * n);
        if (a === b) {
            b = (b + 1) % n;
        }
        src[e] = a;
        dst[e] = b;
    }
    return { nodeCount: n, src, dst };
}

function clusteredGraph(clusters: number, perCluster: number, inEdges: number, outEdges: number, seed: number): Edges {
    const rnd = lcg(seed);
    const n = clusters * perCluster;
    const src: number[] = [];
    const dst: number[] = [];
    for (let c = 0; c < clusters; c++) {
        for (let e = 0; e < inEdges; e++) {
            const a = c * perCluster + Math.floor(rnd() * perCluster);
            let b = c * perCluster + Math.floor(rnd() * perCluster);
            if (a === b) {
                b = c * perCluster + ((b - c * perCluster + 1) % perCluster);
            }
            src.push(a);
            dst.push(b);
        }
    }
    for (let e = 0; e < outEdges; e++) {
        src.push(Math.floor(rnd() * n));
        dst.push(Math.floor(rnd() * n));
    }
    return { nodeCount: n, src: Uint32Array.from(src), dst: Uint32Array.from(dst) };
}

function gridGraph(w: number, h: number): Edges {
    const src: number[] = [];
    const dst: number[] = [];
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const i = y * w + x;
            if (x + 1 < w) {
                src.push(i);
                dst.push(i + 1);
            }
            if (y + 1 < h) {
                src.push(i);
                dst.push(i + w);
            }
        }
    }
    return { nodeCount: w * h, src: Uint32Array.from(src), dst: Uint32Array.from(dst) };
}

function rmatGraph(scale: number, edgeFactor: number, seed: number): Edges {
    const rnd = lcg(seed);
    const n = 1 << scale;
    const m = n * edgeFactor;
    const src = new Uint32Array(m);
    const dst = new Uint32Array(m);
    const [a, b, c] = [0.57, 0.19, 0.19];
    for (let e = 0; e < m; e++) {
        let u = 0;
        let v = 0;
        for (let bit = 0; bit < scale; bit++) {
            const r = rnd();
            if (r < a) {
                // top-left quadrant
            } else if (r < a + b) {
                v |= 1 << bit;
            } else if (r < a + b + c) {
                u |= 1 << bit;
            } else {
                u |= 1 << bit;
                v |= 1 << bit;
            }
        }
        if (u === v) {
            v = (v + 1) % n;
        }
        src[e] = u;
        dst[e] = v;
    }
    return { nodeCount: n, src, dst };
}

const KARATE: readonly (readonly [number, number])[] = [
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [0, 5],
    [0, 6],
    [0, 7],
    [0, 8],
    [0, 10],
    [0, 11],
    [0, 12],
    [0, 13],
    [0, 17],
    [0, 19],
    [0, 21],
    [0, 31],
    [1, 2],
    [1, 3],
    [1, 7],
    [1, 13],
    [1, 17],
    [1, 19],
    [1, 21],
    [1, 30],
    [2, 3],
    [2, 7],
    [2, 8],
    [2, 9],
    [2, 13],
    [2, 27],
    [2, 28],
    [2, 32],
    [3, 7],
    [3, 12],
    [3, 13],
    [4, 6],
    [4, 10],
    [5, 6],
    [5, 10],
    [5, 16],
    [6, 16],
    [8, 30],
    [8, 32],
    [8, 33],
    [9, 33],
    [13, 33],
    [14, 32],
    [14, 33],
    [15, 32],
    [15, 33],
    [18, 32],
    [18, 33],
    [19, 33],
    [20, 32],
    [20, 33],
    [22, 32],
    [22, 33],
    [23, 25],
    [23, 27],
    [23, 29],
    [23, 32],
    [23, 33],
    [24, 25],
    [24, 27],
    [24, 31],
    [25, 31],
    [26, 29],
    [26, 33],
    [27, 33],
    [28, 31],
    [28, 33],
    [29, 32],
    [29, 33],
    [30, 32],
    [30, 33],
    [31, 32],
    [31, 33],
    [32, 33],
];

function karateGraph(): Edges {
    return {
        nodeCount: 34,
        src: Uint32Array.from(KARATE.map((e) => e[0])),
        dst: Uint32Array.from(KARATE.map((e) => e[1])),
    };
}

function buildEdges(kind: string, seed: number): Edges {
    switch (kind) {
        case "karate":
            return karateGraph();
        case "grid":
            return gridGraph(30, 30);
        case "clusters":
            return clusteredGraph(15, 100, 560, 600, seed);
        case "random10k":
            return randomGraph(10_000, 30_000, seed);
        case "rmat":
            return rmatGraph(14, 8, seed);
        case "random30k":
            return randomGraph(30_000, 90_000, seed);
        default:
            return randomGraph(2_000, 6_000, seed);
    }
}

// ------------------------------------------------------------------ dom

function el<T extends HTMLElement>(id: string): T {
    const node = document.getElementById(id);
    if (node === null) {
        throw new Error(`missing #${id}`);
    }
    return node as T;
}

const canvas = el<HTMLCanvasElement>("canvas");
const graphSelect = el<HTMLSelectElement>("graph");
const compatSelect = el<HTMLSelectElement>("compat");
const ipsInput = el<HTMLInputElement>("ips");
const linlogInput = el<HTMLInputElement>("linlog");
const strongInput = el<HTMLInputElement>("strong");
const edgesInput = el<HTMLInputElement>("edges");
const playButton = el<HTMLButtonElement>("play");
const restartButton = el<HTMLButtonElement>("restart");
const unpinButton = el<HTMLButtonElement>("unpin");
const statusBox = el<HTMLDivElement>("status");
const statsBox = el<HTMLDivElement>("stats");

function setStatus(text: string, error = false): void {
    statusBox.textContent = text;
    statusBox.className = error ? "error" : "";
    if (error) {
        console.error(`status: ${text}`);
    } else {
        console.log(`status: ${text}`);
    }
}

// ------------------------------------------------------------------ state

interface Demo {
    readonly snapshot: GraphSnapshot;
    readonly edges: Edges;
    readonly sim: GpuLayoutSimulation<ForceAtlas2Options, ForceAtlas2Stats>;
    readonly positions: Float32Array;
    readonly degree: Uint32Array;
    readonly fixed: U32;
    running: boolean;
    pending: Promise<void> | null;
    frames: number;
    lastFrameMs: number;
}

let ctx: GpuContext | null = null;
let demo: Demo | null = null;
let seed = 1;
const view = { scale: 1, cx: 0, cy: 0, fitted: false, zoom: 1 };
const drag = { index: -1, wasFixed: false, moved: false };

function degreesOf(edges: Edges): Uint32Array {
    const degree = new Uint32Array(edges.nodeCount);
    for (let e = 0; e < edges.src.length; e++) {
        degree[edges.src[e]]++;
        degree[edges.dst[e]]++;
    }
    return degree;
}

async function createDemo(): Promise<void> {
    if (ctx === null) {
        return;
    }
    if (demo !== null) {
        demo.running = false;
        demo.sim.dispose();
        ctx.release(demo.snapshot);
        demo = null;
    }
    const kind = graphSelect.value;
    const edges = buildEdges(kind, seed);
    const snapshot = fromEdgeArrays({ directed: false, nodeCount: edges.nodeCount, src: edges.src, dst: edges.dst });
    const n = snapshot.nodeCount;
    const positions = new Float32Array(3 * n).fill(Number.NaN); // NaN rows are seeded by the simulation's LCG
    const tuning: GpuLayoutTuning = {
        repulsion: "exact",
        exactMaxNodes: 65_536,
        compat: compatSelect.value === "networkx" ? "networkx" : "paper",
    };
    const options: ForceAtlas2Options = {
        seed,
        maxIter: 100_000,
        settleThreshold: 1e-3,
        settleWindow: 20,
        iterationsPerStep: Math.max(1, Math.min(64, Number(ipsInput.value) || 1)),
        maxInFlight: 2,
        linlog: linlogInput.checked,
        strongGravity: strongInput.checked,
    };
    const sim = createForceAtlas2(ctx, { ...options, ...tuning });
    sim.load(snapshot, positions);
    demo = {
        snapshot,
        edges,
        sim,
        positions,
        degree: degreesOf(edges),
        fixed: makeMask(n),
        running: true,
        pending: null,
        frames: 0,
        lastFrameMs: 0,
    };
    view.fitted = false;
    view.zoom = 1;
    playButton.textContent = "Pause";
    setStatus(
        `${kind}: ${n.toLocaleString()} nodes, ${edges.src.length.toLocaleString()} edges; ${ctx.caps.vendor} / ${ctx.caps.architecture}${ctx.caps.software ? " (software adapter)" : ""}`,
    );
    console.log(
        `caps: features=${[...ctx.caps.features].join(",")} wgsl=${[...ctx.caps.wgslFeatures].join(",")} subgroups=${ctx.caps.subgroupMinSize}-${ctx.caps.subgroupMaxSize} maxComputeInvocationsPerWorkgroup=${ctx.caps.limits.maxComputeInvocationsPerWorkgroup} maxStorageBuffersPerShaderStage=${ctx.caps.limits.maxStorageBuffersPerShaderStage}`,
    );
}

// ------------------------------------------------------------------ rendering

function resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }
}

function fitView(d: Demo): void {
    // Frame the CORE of the layout, not its bounding box: with the FA2 laws an isolated node sits far outside
    // the core (spec 7.7), so a bbox fit would shrink everything else to a blob. The stats carry the centroid and
    // the RMS radius of the last landed batch; 2.6 RMS radii hold the bulk of any of the demo graphs.
    const s = d.sim.stats;
    let cx = s.centroid[0];
    let cy = s.centroid[1];
    let half = 2.6 * s.rmsRadius;
    if (!Number.isFinite(cx) || !Number.isFinite(cy) || !(half > 0)) {
        // Before the first batch lands: the seeded positions in [-1, 1).
        cx = 0;
        cy = 0;
        half = 1;
    }
    const margin = 0.92;
    const target = ((Math.min(canvas.width, canvas.height) * margin) / (2 * half)) * view.zoom;
    if (!view.fitted) {
        view.scale = target;
        view.cx = cx;
        view.cy = cy;
        view.fitted = true;
    } else {
        // Smooth the framing so the picture does not jitter while the layout expands.
        view.scale += (target - view.scale) * 0.08;
        view.cx += (cx - view.cx) * 0.08;
        view.cy += (cy - view.cy) * 0.08;
    }
}

function toScreen(x: number, y: number): [number, number] {
    return [canvas.width / 2 + (x - view.cx) * view.scale, canvas.height / 2 + (y - view.cy) * view.scale];
}

function toLayout(sx: number, sy: number): [number, number] {
    return [(sx - canvas.width / 2) / view.scale + view.cx, (sy - canvas.height / 2) / view.scale + view.cy];
}

function draw(d: Demo): void {
    resize();
    fitView(d);
    const g = canvas.getContext("2d");
    if (g === null) {
        return;
    }
    const n = d.snapshot.nodeCount;
    const p = d.positions;
    g.clearRect(0, 0, canvas.width, canvas.height);
    const drawEdges = edgesInput.checked && d.edges.src.length <= 200_000;
    if (drawEdges) {
        g.strokeStyle = "rgba(150, 170, 200, 0.28)";
        g.lineWidth = 1;
        g.beginPath();
        for (let e = 0; e < d.edges.src.length; e++) {
            const a = d.edges.src[e];
            const b = d.edges.dst[e];
            const [ax, ay] = toScreen(p[3 * a], p[3 * a + 1]);
            const [bx, by] = toScreen(p[3 * b], p[3 * b + 1]);
            g.moveTo(ax, ay);
            g.lineTo(bx, by);
        }
        g.stroke();
    }
    const radius = n <= 100 ? 6 : n <= 2_000 ? 3.5 : n <= 12_000 ? 2.2 : 1.4;
    let maxDegree = 1;
    for (let i = 0; i < n; i++) {
        if (d.degree[i] > maxDegree) maxDegree = d.degree[i];
    }
    for (let i = 0; i < n; i++) {
        const [x, y] = toScreen(p[3 * i], p[3 * i + 1]);
        const t = Math.log1p(d.degree[i]) / Math.log1p(maxDegree);
        const hue = 210 - 170 * t;
        const pinned = maskTest(d.fixed, i);
        g.fillStyle = pinned ? "#ffd166" : `hsl(${hue.toFixed(0)} 85% ${(55 + 15 * t).toFixed(0)}%)`;
        g.beginPath();
        g.arc(x, y, pinned ? radius * 1.8 : radius, 0, Math.PI * 2);
        g.fill();
    }
}

function renderStats(d: Demo): void {
    const s = d.sim.stats;
    const lines = [
        `iterations   ${d.sim.iterationsDone}`,
        `settled      ${d.sim.settled}`,
        `in flight    ${d.sim.inFlight}`,
        `ms / iter    ${s.msPerIteration === null ? "-" : s.msPerIteration.toFixed(3)}`,
        `frame ms     ${d.lastFrameMs.toFixed(1)}`,
        `speed        ${s.speed.toFixed(3)}`,
        `efficiency   ${s.speedEfficiency.toFixed(3)}`,
        `swing        ${s.swing.toExponential(2)}`,
        `traction     ${s.traction.toExponential(2)}`,
        `mean disp    ${s.meanDisplacement.toExponential(2)}`,
        `rms radius   ${s.rmsRadius.toFixed(2)}`,
        `tier         ${s.repulsionTier}`,
    ];
    statsBox.textContent = lines.join("\n");
}

// ------------------------------------------------------------------ frame loop (the element bridge, D6)

let lastPromise: Promise<void> | null = null;

function frame(now: number): void {
    const d = demo;
    if (d !== null) {
        if (d.running && !d.sim.settled) {
            const promise = d.sim.step();
            if (promise !== lastPromise) {
                lastPromise = promise;
                promise.catch((error: unknown) => {
                    d.running = false;
                    setStatus(`step failed: ${error instanceof Error ? error.message : String(error)}`, true);
                });
            }
        } else if (d.running && d.sim.settled) {
            d.running = false;
            playButton.textContent = "Play";
            setStatus(`settled after ${d.sim.iterationsDone} iterations`);
        }
        draw(d);
        if ((d.frames & 7) === 0) {
            renderStats(d);
        }
        d.frames++;
        d.lastFrameMs = performance.now() - now;
    }
    requestAnimationFrame(frame);
}

// ------------------------------------------------------------------ interaction

function nearestNode(sx: number, sy: number): number {
    const d = demo;
    if (d === null) {
        return -1;
    }
    const [lx, ly] = toLayout(sx, sy);
    const reach = 12 / view.scale;
    let best = -1;
    let bestDist = reach * reach;
    const p = d.positions;
    for (let i = 0; i < d.snapshot.nodeCount; i++) {
        const dx = p[3 * i] - lx;
        const dy = p[3 * i + 1] - ly;
        const dist = dx * dx + dy * dy;
        if (dist < bestDist) {
            bestDist = dist;
            best = i;
        }
    }
    return best;
}

function canvasPoint(event: PointerEvent): [number, number] {
    const rect = canvas.getBoundingClientRect();
    const dpr = canvas.width / rect.width;
    return [(event.clientX - rect.left) * dpr, (event.clientY - rect.top) * dpr];
}

canvas.addEventListener("pointerdown", (event) => {
    const d = demo;
    if (d === null) {
        return;
    }
    const [sx, sy] = canvasPoint(event);
    const i = nearestNode(sx, sy);
    if (i < 0) {
        return;
    }
    drag.index = i;
    drag.wasFixed = maskTest(d.fixed, i);
    drag.moved = false;
    maskSet(d.fixed, i, true);
    d.sim.setFixed(d.fixed);
    canvas.classList.add("dragging");
    canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
    const d = demo;
    if (d === null || drag.index < 0) {
        return;
    }
    const [sx, sy] = canvasPoint(event);
    const [lx, ly] = toLayout(sx, sy);
    drag.moved = true;
    d.sim.setPosition(drag.index, lx, ly, 0);
    if (!d.running) {
        d.running = true;
        playButton.textContent = "Pause";
    }
});

canvas.addEventListener("pointerup", (event) => {
    const d = demo;
    if (d === null || drag.index < 0) {
        return;
    }
    if (!drag.moved && drag.wasFixed) {
        // A click on a pinned node releases it (an unpin reheats, spec 7.12).
        maskSet(d.fixed, drag.index, false);
        d.sim.setFixed(d.fixed);
        if (!d.running) {
            d.running = true;
            playButton.textContent = "Pause";
        }
    }
    drag.index = -1;
    canvas.classList.remove("dragging");
    canvas.releasePointerCapture(event.pointerId);
});

canvas.addEventListener(
    "wheel",
    (event) => {
        event.preventDefault();
        view.zoom = Math.min(20, Math.max(0.2, view.zoom * (event.deltaY < 0 ? 1.1 : 1 / 1.1)));
    },
    { passive: false },
);

playButton.addEventListener("click", () => {
    const d = demo;
    if (d === null) {
        return;
    }
    if (d.running) {
        d.running = false;
        playButton.textContent = "Play";
    } else {
        if (d.sim.settled) {
            d.sim.reheat();
        }
        d.running = true;
        playButton.textContent = "Pause";
    }
});

restartButton.addEventListener("click", () => {
    seed = (seed % 1_000_000) + 1;
    void createDemo();
});

unpinButton.addEventListener("click", () => {
    const d = demo;
    if (d === null) {
        return;
    }
    d.fixed.fill(0);
    d.sim.setFixed(d.fixed);
});

for (const input of [graphSelect, compatSelect, linlogInput, strongInput]) {
    input.addEventListener("change", () => {
        void createDemo();
    });
}
ipsInput.addEventListener("change", () => {
    const d = demo;
    if (d !== null) {
        d.sim.setParams({ iterationsPerStep: Math.max(1, Math.min(64, Number(ipsInput.value) || 1)) });
    }
});

// ------------------------------------------------------------------ start

async function main(): Promise<void> {
    const session = installRemoteLog();
    console.log(`remote log session ${session}`);
    try {
        ctx = await requestGpuContext({ powerPreference: "high-performance" });
    } catch (error) {
        setStatus(
            `no WebGPU device: ${error instanceof Error ? error.message : String(error)} (Chrome / Edge 113+ with WebGPU enabled)`,
            true,
        );
        return;
    }
    ctx.lost.then((info) => {
        setStatus(`device lost: ${info.message}`, true);
        demo = null;
    });
    await createDemo();
    requestAnimationFrame(frame);
}

void main();
