/**
 * Benchmark runner of @graphty/webgpu-graph-algorithms (spec 11.7; contract 6.3). The command line of the
 * P1 form is already the one package.json's "bench" / "benchmark" scripts and gpu.yml's `pnpm run bench` step
 * call:
 *
 *   tsx benchmarks/run.ts [group ...] [--no-save] [--allow-software] [--runs N]
 *
 * P0 form (contract 2.9 delta 4): no benchmark group exists until P1-T7 wires benchmarks/harness.ts with the
 * `upload` and `roundtrip` groups, so this prints "no benchmark groups until P1", times nothing, writes
 * nothing under benchmarks/out/ and exits 0. The group names given on the command line are echoed so a
 * mistyped invocation is visible in the log.
 */

const args: readonly string[] = process.argv.slice(2);
const requested: string[] = [];
for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--runs" && i + 1 < args.length) {
        i++;
    } else if (!a.startsWith("--")) {
        requested.push(a);
    }
}
const suffix = requested.length === 0 ? "" : ` (requested: ${requested.join(", ")})`;
console.log(`no benchmark groups until P1${suffix}: nothing timed, nothing written`);
process.exitCode = 0;
