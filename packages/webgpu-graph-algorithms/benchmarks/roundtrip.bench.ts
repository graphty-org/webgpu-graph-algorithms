/**
 * Round-trip benchmarks (spec 10.4 T-2, T-3; contract 6.3): `degree` plus its 400 KB readback at 100k with the core already
 * resident (T-2: <= 2 ms wall in Node), and an empty submit followed by a 4-byte readU32 through the staging ring (T-3:
 * <= 0.1 ms under Dawn). Both timings include the harness's closing onSubmittedWorkDone(), which is recorded in G1.md.
 */

import { degree } from "../src/algorithms/degree.js";
import { type GpuContext } from "../src/context.js";
import { BufferUsage } from "../src/device/webgpu-constants.js";
import { randomEdges, snapshotOf, TIERS } from "./datasets.js";
import { bench, type BenchResult } from "./harness.js";

/**
 * Run the round-trip benchmarks.
 * @param ctx - the context (a hardware adapter; run.ts refuses software ones)
 * @returns the results
 */
export async function runRoundtripBenchmarks(ctx: GpuContext): Promise<BenchResult[]> {
    const results: BenchResult[] = [];
    const tier = TIERS.find((t) => t.name === "100k/1M");
    if (tier === undefined) {
        throw new Error("TIERS lacks 100k/1M");
    }
    const s = snapshotOf(randomEdges(tier.nodes, tier.edges, 12345), { label: "roundtrip/100k" });
    ctx.residency.core(s);
    await ctx.device.queue.onSubmittedWorkDone();
    try {
        results.push(
            await bench(
                "roundtrip",
                "degree + 400 KB readback at 100k (core resident)",
                { setup: () => s, run: (input) => degree(ctx, input) },
                { device: ctx.device, items: s.nodeCount, unit: "nodes" },
            ),
        );
    } finally {
        ctx.release(s);
    }
    const counter = ctx.device.createBuffer({
        size: 4,
        usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST,
        label: "roundtrip/counter",
    });
    try {
        results.push(
            await bench(
                "roundtrip",
                "empty submit + 4-byte readU32 round trip",
                {
                    setup: () => counter,
                    run: async (buffer) => {
                        ctx.device.queue.submit([]);
                        return ctx.readback.readU32(buffer, 0);
                    },
                },
                { device: ctx.device },
            ),
        );
    } finally {
        counter.destroy();
    }
    return results;
}
