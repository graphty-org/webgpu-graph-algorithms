/**
 * Upload benchmarks (spec 10.4 T-1; contract 6.3): `residency.core` of the weighted hot prefix at 100k / 1M (16.4 MB) and
 * 1M / 10M (164 MB) -- the arena path, ONE writeBuffer. A FRESH snapshot per run (the residency uploads each array object
 * once, so re-using one snapshot would time a cache hit) and `release` in the teardown; the timer brackets the upload and
 * the queue's onSubmittedWorkDone (harness.ts). Targets: <= 10 ms and <= 100 ms ([X] 5-12 GB/s).
 */

import { type GpuContext } from "../src/context.js";
import { randomEdges, snapshotOf, TIERS } from "./datasets.js";
import { bench, type BenchResult } from "./harness.js";

/**
 * Run the upload benchmarks.
 * @param ctx - the context (a hardware adapter; run.ts refuses software ones)
 * @returns the results
 */
export async function runUploadBenchmarks(ctx: GpuContext): Promise<BenchResult[]> {
    const results: BenchResult[] = [];
    for (const tier of TIERS) {
        if (tier.nodes < 100_000) {
            continue; // T-1 names the 100k / 1M and 1M / 10M tiers
        }
        const edges = randomEdges(tier.nodes, tier.edges, 12345);
        const label = `upload/${tier.name}`;
        // one untimed probe: the hot byte length and the plan this device takes (arena unless the binding limit is below it)
        const probe = snapshotOf(edges, { label });
        const { arena } = probe;
        if (arena === null) {
            throw new Error(`fromEdgeArrays produced no arena for ${tier.name}`);
        }
        const { hotByteLength: hotBytes } = arena;
        const { plan } = ctx.residency.core(probe);
        await ctx.device.queue.onSubmittedWorkDone();
        ctx.release(probe);
        results.push(
            await bench(
                "upload",
                `residency.core ${tier.name} weighted hot prefix (${(hotBytes / 1e6).toFixed(1)} MB, ${plan})`,
                {
                    setup: () => snapshotOf(edges, { label }),
                    run: (s) => ctx.residency.core(s),
                    teardown: (s) => {
                        ctx.release(s);
                    },
                },
                { device: ctx.device, items: hotBytes, unit: "bytes" },
            ),
        );
    }
    return results;
}
