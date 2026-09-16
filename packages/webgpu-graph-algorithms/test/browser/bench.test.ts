/**
 * T-5 (spec 10.4): the FA2 per-frame cost in Chromium -- step(1) + the 12n readback -- at the 10k exact tier,
 * measured by this `bench`-tagged browser test (spec 11.6 item 8, 11.7) and appended to
 * benchmarks/out/<runner class>.json through the Vitest commands bridge (contract 2.5 appendBenchRecord, 6.4).
 * Skipped unless GRAPHTY_BROWSER_GPU === "nvidia" (software adapters never time anything, spec 11.7). The 100k
 * grid-tier half of T-5 is P4 (no grid tier exists at P3).
 *
 * The target (<= 6 ms on the 4070 at 10k) is a G3 record, not an assertion here: a missed target is re-fixed by an
 * owner decision, never relaxed silently (spec 10.4), and the T4 lane runs this test on slower hardware. The median
 * is printed on the console and stored in the session record P3-T7 copies into docs/decisions/G3.md.
 *
 * The session is built here: benchmarks/harness.ts imports node:fs / node:os and cannot enter the browser bundle
 * (its types are imported with `import type`, which esbuild erases). The runner class comes from the ONE copy of
 * the rule in scripts/runner-class.js with an explicit empty env (the .js never reads process.env in the browser);
 * Chromium redacts adapter.info.description (note 05 section 3.2), so the browser's class carries "driver0" and its
 * sessions land in their own out file, distinguishable by gpu.runtime === "browser" (spec 11.7). The empty env also
 * means the GRAPHTY_RUNNER_CLASS override of contract 6.4 (the T4 lane's fixed `gpu-linux-t4`) does not reach this
 * test: vitest.config.ts forwards no such key through `define` (contract 2.5), so on the lane this session lands in
 * nvidia-<architecture>-driver0.json beside the Node harness's gpu-linux-t4.json. Open item (section 9) for P0-T2:
 * forward import.meta.env.GRAPHTY_RUNNER_CLASS and pass it here as `env`; until then the G3 record cites this file
 * by name.
 */

import { commands } from "@vitest/browser/context";

import type { BenchResult, BenchSession, GpuSessionInfo } from "../../benchmarks/harness.js";
import { runnerClass } from "../../scripts/runner-class.js";
import { type GpuContext } from "../../src/context.js";
import { createForceAtlas2 } from "../../src/layouts/forceatlas2.js";
import { randomEdges, snapshotOf } from "../helpers/graphs.js";
import { acquireBrowser, browserGpu, requireBrowserGpu } from "../setup/browser.js";

const NODES = 10_000;
const EDGES = 100_000; // E = 10 n, the exact-ladder density of T-4 (spec 10.4)
const WARM_FRAMES = 5;
const TIMED_FRAMES = 50;

/**
 * The GpuSessionInfo of a browser context (the browser twin of benchmarks/harness.ts gpuSessionInfo, 6.1: the same
 * fields, `driver` the version string found in the description -- "" in Chromium, which redacts it).
 * @param ctx - the context
 * @returns the session's gpu field
 */
function browserGpuSessionInfo(ctx: GpuContext): GpuSessionInfo {
    const { caps } = ctx;
    const version = /\d+(?:\.\d+)+/.exec(caps.description);
    return {
        vendor: caps.vendor,
        architecture: caps.architecture,
        device: caps.device,
        description: caps.description,
        driver: version === null ? "" : version[0],
        limits: {
            maxBufferSize: caps.limits.maxBufferSize,
            maxStorageBufferBindingSize: caps.limits.maxStorageBufferBindingSize,
            maxStorageBuffersPerShaderStage: caps.limits.maxStorageBuffersPerShaderStage,
            maxComputeWorkgroupsPerDimension: caps.limits.maxComputeWorkgroupsPerDimension,
        },
        software: caps.software,
        runtime: "browser",
        subgroupMaxSize: caps.subgroupMaxSize,
    };
}

/**
 * The median of a list (the harness's arithmetic).
 * @param values - the values
 * @returns the median
 */
function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

describe.skipIf(browserGpu() !== "nvidia")(
    "[bench] T-5: FA2 exact-tier per-frame cost in Chromium (NVIDIA only)",
    () => {
        it("10k nodes: step(1) + readback timings appended through commands.appendBenchRecord", async (t) => {
            await requireBrowserGpu(t);
            const ctx = await acquireBrowser();
            expect(ctx.caps.software).toBe(false); // never time a software adapter (spec 11.7); a SwiftShader pick is red, not silent
            expect(ctx.caps.vendor).toBe("nvidia");
            const snapshot = snapshotOf(randomEdges(NODES, EDGES, 5), { nodeCount: NODES });
            expect(snapshot.nodeCount).toBe(NODES);
            const sim = createForceAtlas2(ctx, {
                seed: 1,
                maxIter: 1_000_000,
                settleThreshold: 0,
                repulsion: "exact",
                maxInFlight: 1,
            });
            const positions = new Float32Array(3 * NODES).fill(NaN);
            sim.load(snapshot, positions);
            for (let i = 0; i < WARM_FRAMES; i++) {
                await sim.step(1); // pipeline compile, the first submit, the staging ring's growth
            }
            const deltas: number[] = [];
            for (let i = 0; i < TIMED_FRAMES; i++) {
                const start = performance.now();
                await sim.step(1); // resolves once the 12n readback has landed in `positions`: the per-frame cost of T-5
                deltas.push(performance.now() - start);
            }
            expect(sim.iterationsDone).toBe(WARM_FRAMES + TIMED_FRAMES);
            expect(positions.every((v) => Number.isFinite(v))).toBe(true);
            const medianMs = median(deltas);
            expect(Number.isFinite(medianMs)).toBe(true);
            expect(medianMs).toBeGreaterThan(0);
            const result: BenchResult = {
                group: "layout-browser",
                name: "fa2-exact-10k-step1-readback",
                medianMs,
                minMs: Math.min(...deltas),
                maxMs: Math.max(...deltas),
                runs: TIMED_FRAMES,
                memoryDeltaBytes: 0, // no process.memoryUsage() in the browser
                rate: 1000 / medianMs,
                rateUnit: "frames/s",
            };
            // an explicit empty env: the browser has no process.env and no forwarded GRAPHTY_RUNNER_CLASS (see the header)
            const cls = runnerClass(
                { vendor: ctx.caps.vendor, architecture: ctx.caps.architecture, description: ctx.caps.description },
                {},
            );
            const session: BenchSession = {
                date: new Date().toISOString(),
                host: globalThis.location.hostname,
                node: navigator.userAgent,
                cpu: "unknown",
                exposeGc: false,
                gpu: browserGpuSessionInfo(ctx),
                runnerClass: cls,
                results: [result],
            };
            const file = await commands.appendBenchRecord({ runnerClass: cls, session });
            expect(file).toMatch(/benchmarks[\\/]out[\\/][A-Za-z0-9_.-]+\.json$/);
            console.warn(
                `T-5 fa2 exact 10k step(1) + readback: median ${medianMs.toFixed(3)} ms (min ${result.minMs.toFixed(3)}, max ${result.maxMs.toFixed(3)}) over ${TIMED_FRAMES} frames on ${cls} -> ${file}`,
            );
            sim.dispose();
            ctx.release(snapshot);
        });
    },
);
