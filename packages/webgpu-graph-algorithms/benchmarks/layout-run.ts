/**
 * The end-to-end Node layout driver (contract 6.3; the P3 deliverable of spec 13: "benchmarks/layout-run.ts lays out
 * the 100k / 1M graph end to end on the exact tier: ~18 ms per iteration, correct"). Lays a seeded G(n, m) graph out
 * with the exact-tier ForceAtlas2 through `run({ batch })`, prints ms per iteration, the final stats and the wall time,
 * verifies that every position is finite and that the run ended by settling or by reaching maxIter, and exits 1
 * otherwise.
 *
 * Usage (from packages/webgpu-graph-algorithms):
 *   pnpm exec tsx benchmarks/layout-run.ts --nodes 100000 --edges 1000000          # the P3 deliverable
 *   pnpm exec tsx benchmarks/layout-run.ts --nodes 1000 --edges 10000 --iterations 50 --batch 4 --seed 7 --dim 3 --compat networkx
 *   GRAPHTY_GPU_ADAPTER=llvmpipe pnpm exec tsx benchmarks/layout-run.ts --nodes 200 --edges 2000 --iterations 20
 *
 * Options: --nodes N (required, >= 1) --edges M (required, >= 0) [--iterations 100] [--batch 8] [--seed 1] [--dim 2]
 * [--compat paper]. PLAN DECISIONS (P3-T7 item 5): the tuning is `repulsion: "exact"` because the grid tier does not
 * exist in P3 and `"auto"` refuses n above exactMaxNodes at load() (P4 adds --repulsion); a software adapter is NOT
 * refused -- this is a correctness driver, not a benchmark, and spec 11.7's "software adapters never time anything" is
 * honoured by labelling its timings "not representative"; main() runs only when this file is the entry script, so
 * test/benchmarks.test.ts imports the pure helpers without side effects. No top-level await (module ES2020).
 */

import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { createForceAtlas2 } from "../src/layouts/forceatlas2.js";
import { createNodeGpuContext } from "../src/node/index.js";
import { randomEdges, snapshotOf } from "./datasets.js";

/** The parsed command line. */
export interface LayoutRunArgs {
    readonly nodes: number;
    readonly edges: number;
    readonly iterations: number;
    readonly batch: number;
    readonly seed: number;
    readonly dim: 2 | 3;
    readonly compat: "paper" | "networkx";
}

/** The verdict of checkLayoutResult. */
export interface LayoutCheck {
    readonly ok: boolean;
    readonly problems: readonly string[];
}

/** The option names parseLayoutRunArgs knows, in the usage order. */
const KNOWN_OPTIONS = ["--nodes", "--edges", "--iterations", "--batch", "--seed", "--dim", "--compat"];

/**
 * Parses an integer option value.
 * @param flag - the option name (for the message)
 * @param text - the raw value (undefined when the flag was last)
 * @param min - the smallest accepted value
 * @returns the integer
 */
function integerOption(flag: string, text: string | undefined, min: number): number {
    const value = text === undefined ? Number.NaN : Number(text);
    if (!Number.isInteger(value) || value < min) {
        throw new Error(`${flag} expects an integer >= ${min}, got ${String(text)}`);
    }
    return value;
}

/**
 * Parses `--nodes N --edges M [--iterations 100] [--batch 8] [--seed 1] [--dim 2] [--compat paper]`.
 * @param argv - process.argv.slice(2)
 * @returns the arguments; a missing required option, an out-of-range value, an unknown option or a stray word throws
 */
export function parseLayoutRunArgs(argv: readonly string[]): LayoutRunArgs {
    let nodes: number | null = null;
    let edges: number | null = null;
    let iterations = 100;
    let batch = 8;
    let seed = 1;
    let dim: 2 | 3 = 2;
    let compat: "paper" | "networkx" = "paper";
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        const next = argv[i + 1];
        switch (a) {
            case "--nodes":
                nodes = integerOption(a, next, 1);
                i += 1;
                break;
            case "--edges":
                edges = integerOption(a, next, 0);
                i += 1;
                break;
            case "--iterations":
                iterations = integerOption(a, next, 1);
                i += 1;
                break;
            case "--batch":
                batch = integerOption(a, next, 1);
                i += 1;
                break;
            case "--seed":
                seed = integerOption(a, next, 0);
                i += 1;
                break;
            case "--dim":
                if (next !== "2" && next !== "3") {
                    throw new Error(`--dim expects 2 or 3, got ${String(next)}`);
                }
                dim = next === "2" ? 2 : 3;
                i += 1;
                break;
            case "--compat":
                if (next !== "paper" && next !== "networkx") {
                    throw new Error(`--compat expects paper or networkx, got ${String(next)}`);
                }
                compat = next;
                i += 1;
                break;
            default:
                if (a.startsWith("--")) {
                    throw new Error(`unknown option ${a}; known: ${KNOWN_OPTIONS.join(" ")}`);
                }
                throw new Error(`unexpected argument ${a}`);
        }
    }
    if (nodes === null) {
        throw new Error("--nodes N is required");
    }
    if (edges === null) {
        throw new Error("--edges M is required");
    }
    return { nodes, edges, iterations, batch, seed, dim, compat };
}

/**
 * The verification of contract 6.3: the owner's stride-3 array has 3n finite components and the run ended by settling
 * or by exhausting maxIter. Every problem is reported, the length problem first.
 * @param positions - the owner's array after run()
 * @param nodeCount - the snapshot's node count
 * @param iterationsDone - sim.iterationsDone after run()
 * @param settled - sim.settled after run()
 * @param maxIter - the iteration budget the run was given
 * @returns ok with an empty list, or the problems
 */
export function checkLayoutResult(
    positions: Float32Array,
    nodeCount: number,
    iterationsDone: number,
    settled: boolean,
    maxIter: number,
): LayoutCheck {
    const problems: string[] = [];
    if (positions.length !== 3 * nodeCount) {
        problems.push(
            `positions has ${positions.length} components, expected ${3 * nodeCount} (3 x ${nodeCount} nodes)`,
        );
    } else {
        let bad = 0;
        let firstNode = -1;
        let firstComponent = -1;
        for (let i = 0; i < positions.length; i++) {
            if (!Number.isFinite(positions[i])) {
                if (bad === 0) {
                    firstNode = Math.floor(i / 3);
                    firstComponent = i % 3;
                }
                bad += 1;
            }
        }
        if (bad > 0) {
            problems.push(
                `${bad} non-finite position components (first at node ${firstNode}, component ${firstComponent})`,
            );
        }
    }
    if (!settled && iterationsDone < maxIter) {
        problems.push(`the run ended after ${iterationsDone} of ${maxIter} iterations without settling`);
    }
    return { ok: problems.length === 0, problems };
}

/**
 * The bounding box of the finite rows of a stride-3 array, as text.
 * @param positions - the owner's array
 * @param nodeCount - the node count
 * @returns "x [lo, hi] y [lo, hi] z [lo, hi]" (or "empty" when no row is finite)
 */
function bboxText(positions: Float32Array, nodeCount: number): string {
    const lo = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
    const hi = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
    let finite = 0;
    for (let i = 0; i < nodeCount; i++) {
        const x = positions[3 * i];
        const y = positions[3 * i + 1];
        const z = positions[3 * i + 2];
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
            continue;
        }
        finite += 1;
        lo[0] = Math.min(lo[0], x);
        lo[1] = Math.min(lo[1], y);
        lo[2] = Math.min(lo[2], z);
        hi[0] = Math.max(hi[0], x);
        hi[1] = Math.max(hi[1], y);
        hi[2] = Math.max(hi[2], z);
    }
    if (finite === 0) {
        return "empty";
    }
    const axis = (k: number): string => `[${lo[k].toFixed(4)}, ${hi[k].toFixed(4)}]`;
    return `x ${axis(0)} y ${axis(1)} z ${axis(2)}`;
}

/**
 * Lays the graph out and verifies the result.
 * @returns the process exit code (0 ok, 1 a failed check or a usage error)
 */
async function main(): Promise<number> {
    const args = parseLayoutRunArgs(process.argv.slice(2));
    const ctx = await createNodeGpuContext({ adapter: process.env.GRAPHTY_GPU_ADAPTER, label: "layout-run" });
    try {
        const { caps } = ctx;
        const profiled = ctx.profiler !== null && ctx.profiler.enabled;
        console.log(
            `layout-run: n=${args.nodes} m=${args.edges} dim=${args.dim} compat=${args.compat} seed=${args.seed} ` +
                `batch=${args.batch} maxIter=${args.iterations} repulsion=exact`,
        );
        console.log(
            `adapter: ${caps.vendor} / ${caps.architecture} / ${caps.description} (profiler ${profiled ? "on" : "off"})` +
                (caps.software ? " -- software adapter: timings are not representative (spec 11.7)" : ""),
        );
        const snapshot = snapshotOf(randomEdges(args.nodes, args.edges, args.seed), {
            label: `layout-run/${args.nodes}/${args.edges}`,
        });
        const positions = new Float32Array(3 * snapshot.nodeCount).fill(Number.NaN);
        const sim = createForceAtlas2(ctx, {
            dim: args.dim,
            seed: args.seed,
            maxIter: args.iterations,
            repulsion: "exact",
            compat: args.compat,
        });
        try {
            const start = performance.now();
            sim.load(snapshot, positions);
            const stats = await sim.run({ batch: args.batch });
            const wall = performance.now() - start;
            const done = sim.iterationsDone;
            const perIteration = done > 0 ? wall / done : Number.NaN;
            const gpuPerIteration = stats.msPerIteration === null ? "n/a" : stats.msPerIteration.toFixed(3);
            console.log(
                `iterations=${done} settled=${String(sim.settled)} wall=${wall.toFixed(1)} ms ` +
                    `ms/iteration=${perIteration.toFixed(3)} (wall / iterations; upload, seeding and every readback included) ` +
                    `gpu ms/iteration=${gpuPerIteration} (last batch, stats.msPerIteration)`,
            );
            console.log(
                `stats: iteration=${stats.iteration} swing=${stats.swing.toExponential(4)} traction=${stats.traction.toExponential(4)} ` +
                    `speed=${stats.speed.toExponential(4)} speedEfficiency=${stats.speedEfficiency.toFixed(4)} ` +
                    `meanDisplacement=${stats.meanDisplacement.toExponential(4)} rmsRadius=${stats.rmsRadius.toFixed(4)} ` +
                    `layoutRadius=${stats.layoutRadius.toFixed(4)} centroid=(${stats.centroid.map((c) => c.toFixed(4)).join(", ")}) ` +
                    `tier=${stats.repulsionTier} trace=${stats.trace.length} records`,
            );
            console.log(`positions: ${snapshot.nodeCount} rows, bbox ${bboxText(positions, snapshot.nodeCount)}`);
            const check = checkLayoutResult(positions, snapshot.nodeCount, done, sim.settled, args.iterations);
            if (!check.ok) {
                for (const problem of check.problems) {
                    console.error(`FAIL: ${problem}`);
                }
                return 1;
            }
            console.log("OK");
            return 0;
        } finally {
            sim.dispose();
            ctx.release(snapshot);
        }
    } finally {
        ctx.dispose();
    }
}

// Run only as the entry script (tsx benchmarks/layout-run.ts ...), never on import.
const entry = process.argv[1];
if (entry !== undefined && resolve(entry) === fileURLToPath(import.meta.url)) {
    main().then(
        (code) => {
            process.exitCode = code;
        },
        (error: unknown) => {
            console.error(error instanceof Error ? error.message : String(error));
            process.exitCode = 1;
        },
    );
}
