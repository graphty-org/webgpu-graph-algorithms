/**
 * Benchmark runner for @graphty/graph-format (design section 15.5). Runs the freeze, id-map and
 * view benchmarks, prints one table per group and appends the session to
 * `benchmarks/results/<host>-<node>.json`.
 *
 * Usage (from the package directory):
 *
 *   npm run benchmark                       # every group
 *   npx tsx benchmarks/run.ts freeze ids    # selected groups
 *   node --expose-gc --import tsx benchmarks/run.ts   # with GC before every run for exact memory deltas
 *   npx tsx benchmarks/run.ts --no-save     # do not append to benchmarks/results
 */

import { runFreezeBenchmarks } from "./freeze.bench.js";
import { appendSession, type BenchResult, printTable } from "./harness.js";
import { runIdBenchmarks } from "./ids.bench.js";
import { runViewBenchmarks } from "./views.bench.js";

const GROUPS: Readonly<Record<string, () => BenchResult[]>> = {
    freeze: runFreezeBenchmarks,
    ids: runIdBenchmarks,
    views: runViewBenchmarks,
};

const args = process.argv.slice(2);
const save = !args.includes("--no-save");
const selected = args.filter((a) => !a.startsWith("--"));
const names = selected.length === 0 ? Object.keys(GROUPS) : selected;

const all: BenchResult[] = [];
for (const name of names) {
    const group = GROUPS[name] as (() => BenchResult[]) | undefined;
    if (group === undefined) {
        console.error(`unknown benchmark group "${name}"; known: ${Object.keys(GROUPS).join(", ")}`);
        process.exitCode = 1;
        break;
    }
    console.log(`\n== ${name} (${process.version}, median of 5 runs)\n`);
    const results = group();
    printTable(results);
    all.push(...results);
}
if (save && all.length > 0) {
    console.log(`\nresults appended to ${appendSession(all)}`);
}
