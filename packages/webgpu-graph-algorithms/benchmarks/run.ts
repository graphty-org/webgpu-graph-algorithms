/**
 * Benchmark runner of @graphty/webgpu-graph-algorithms (spec 11.7; contract 6.3). Creates the Node (Dawn) context, prints
 * the adapter summary, runs the selected groups, prints one table per group and appends the session to
 * benchmarks/out/<runner-class>.json (6.4). The checked-in baseline is benchmarks/results/<runner-class>.json, written by the
 * owner's runs on the dev box (spec 12.1) and compared by scripts/bench-compare.js on the GPU lane.
 *
 * Usage (from packages/webgpu-graph-algorithms):
 *   pnpm run bench                                        # every group, 5 timed runs each after one warm-up, appended
 *   pnpm exec tsx benchmarks/run.ts upload roundtrip      # selected groups
 *   pnpm exec tsx benchmarks/run.ts layout-exact          # the T-4 ladder and the 10k frame rung (P3)
 *   pnpm exec tsx benchmarks/run.ts --no-save             # print only
 *   pnpm exec tsx benchmarks/run.ts --runs 3              # 3 timed runs per benchmark
 *   pnpm exec tsx benchmarks/run.ts --allow-software      # time on a software adapter anyway (never for a baseline)
 *   GRAPHTY_GPU_ADAPTER=llvmpipe pnpm exec tsx benchmarks/run.ts --allow-software --runs 1 roundtrip
 *
 * Software adapters never time anything (spec 11.7): without --allow-software the run prints "software adapter: nothing
 * timed" and exits 0 (on the GPU lane gpu-report.js has already failed the job on a software adapter, 12.3). No top-level
 * await: the base tsconfig compiles with module ES2020, so main() is chained.
 */

import { type GpuContext } from "../src/context.js";
import { createNodeGpuContext } from "../src/node/index.js";
import { appendSession, type BenchResult, gpuSessionInfo, printTable, runnerClass, setBenchRuns } from "./harness.js";
import { LAYOUT_EXACT_GROUP, runLayoutExactBenchmarks } from "./layout-exact.bench.js";
import { runRoundtripBenchmarks } from "./roundtrip.bench.js";
import { runUploadBenchmarks } from "./upload.bench.js";

/** The groups and the T-targets they record (6.3): upload T-1, roundtrip T-2 / T-3, layout-exact T-4 and the Node side of T-5. */
const GROUPS: Readonly<Record<string, (ctx: GpuContext) => Promise<BenchResult[]>>> = {
    upload: runUploadBenchmarks,
    roundtrip: runRoundtripBenchmarks,
    [LAYOUT_EXACT_GROUP]: runLayoutExactBenchmarks,
};

/** The parsed command line. */
interface Args {
    readonly groups: readonly string[];
    readonly save: boolean;
    readonly allowSoftware: boolean;
    readonly runs: number;
}

/**
 * Parses `[group ...] [--no-save] [--allow-software] [--runs N]`.
 * @param argv - process.argv.slice(2)
 * @returns the arguments; unknown flags are errors
 */
function parseArgs(argv: readonly string[]): Args {
    const groups: string[] = [];
    let save = true;
    let allowSoftware = false;
    let runs = 5;
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === "--no-save") {
            save = false;
        } else if (a === "--allow-software") {
            allowSoftware = true;
        } else if (a === "--runs") {
            const value = Number(argv[i + 1]);
            if (!Number.isInteger(value) || value < 1) {
                throw new Error(`--runs expects a positive integer, got ${String(argv[i + 1])}`);
            }
            runs = value;
            i += 1;
        } else if (a.startsWith("--")) {
            throw new Error(`unknown option ${a}; known: --no-save --allow-software --runs N`);
        } else {
            groups.push(a);
        }
    }
    return { groups: groups.length === 0 ? Object.keys(GROUPS) : groups, save, allowSoftware, runs };
}

/**
 * Runs the selected groups on a fresh Node context.
 * @returns the process exit code
 */
async function main(): Promise<number> {
    const args = parseArgs(process.argv.slice(2));
    for (const name of args.groups) {
        if (!(name in GROUPS)) {
            console.error(`unknown benchmark group "${name}"; known: ${Object.keys(GROUPS).join(", ")}`);
            return 1;
        }
    }
    const ctx = await createNodeGpuContext({ adapter: process.env.GRAPHTY_GPU_ADAPTER, label: "bench" });
    try {
        const gpu = gpuSessionInfo(ctx);
        console.log(
            `adapter: ${gpu.vendor} / ${gpu.architecture} / ${gpu.description} (software ${String(gpu.software)}, ` +
                `runtime ${gpu.runtime}, driver "${gpu.driver}", subgroups <= ${gpu.subgroupMaxSize})`,
        );
        console.log(
            `limits: maxBufferSize=${gpu.limits.maxBufferSize} maxStorageBufferBindingSize=${gpu.limits.maxStorageBufferBindingSize} ` +
                `maxStorageBuffersPerShaderStage=${gpu.limits.maxStorageBuffersPerShaderStage} maxComputeWorkgroupsPerDimension=${gpu.limits.maxComputeWorkgroupsPerDimension}`,
        );
        console.log(`runner class: ${runnerClass(gpu)}`);
        if (gpu.software && !args.allowSoftware) {
            console.log(
                "software adapter: nothing timed (pass --allow-software to time it anyway; never for a baseline)",
            );
            return 0;
        }
        setBenchRuns(args.runs);
        const all: BenchResult[] = [];
        for (const name of args.groups) {
            const group = GROUPS[name];
            console.log(`\n== ${name} (${process.version}, median of ${args.runs} runs)\n`);
            const results = await group(ctx);
            printTable(results);
            all.push(...results);
        }
        if (args.save && all.length > 0) {
            console.log(`\nresults appended to ${appendSession(all, gpu)}`);
        }
        return 0;
    } finally {
        ctx.dispose();
    }
}

main().then(
    (code) => {
        process.exitCode = code;
    },
    (error: unknown) => {
        console.error(error instanceof Error ? error.message : String(error));
        process.exitCode = 1;
    },
);
