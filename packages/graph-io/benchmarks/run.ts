/**
 * Benchmark runner for @graphty/graph-io: `pnpm run benchmark -- [--quick] [--runs N]` runs the
 * import / export pipeline benchmark of design section 15.5 item 5 (benchmarks/import.bench.ts).
 */

import { runImportBenchmarks } from "./import.bench.js";

runImportBenchmarks(process.argv.slice(2)).catch((err: unknown) => {
    console.error(err);
    process.exitCode = 1;
});
