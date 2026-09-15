/**
 * The measurement harness of the @graphty/graph-format benchmarks (design section 15.5): every
 * benchmark runs its body several times (default 5) after one warm-up and reports the MEDIAN wall
 * time plus the median `process.memoryUsage()` heap-and-buffer delta of one run; with `--expose-gc`
 * the garbage collector runs before every measured iteration so the deltas are of the run, not of
 * leftovers. Results are appended to `benchmarks/results/<host>-<node>.json` with the Node version
 * and CPU model, the pattern of algorithms/benchmarks/benchmark-sessions.json.
 */

import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { cpus, hostname } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/** One measured benchmark. */
export interface BenchResult {
    /** The benchmark group (freeze, ids, views). */
    readonly group: string;
    /** The benchmark name. */
    readonly name: string;
    /** Median wall time of one run, milliseconds. */
    readonly medianMs: number;
    /** Fastest run, milliseconds. */
    readonly minMs: number;
    /** Slowest run, milliseconds. */
    readonly maxMs: number;
    /** Measured runs (after the warm-up). */
    readonly runs: number;
    /** Median delta of heapUsed + arrayBuffers over one run, bytes (negative when a GC ran inside). */
    readonly memoryDeltaBytes: number;
    /** A throughput figure the benchmark chose to report (edges per second, lookups per second), or null. */
    readonly rate: number | null;
    /** What `rate` counts. */
    readonly rateUnit: string | null;
}

/** Options of `bench()`. */
interface BenchOptions {
    /** Measured runs (default 5). */
    readonly runs?: number;
    /** Items processed per run, for the rate column. */
    readonly items?: number;
    /** What `items` counts ("edges", "lookups"). */
    readonly unit?: string;
}

/** A prepared benchmark: `setup` builds the input once per run outside the timer, `run` is timed. */
interface BenchCase<T> {
    /** Build the input (not timed). Called once per measured run. */
    readonly setup: () => T;
    /** The timed body; its return value is kept alive until the timing ends so it cannot be optimised away. */
    readonly run: (input: T) => unknown;
}

/** The global gc hook exposed by `node --expose-gc`. */
declare const gc: (() => void) | undefined;

/**
 * Run the garbage collector when `--expose-gc` is on.
 */
function collectGarbage(): void {
    if (typeof gc === "function") {
        gc();
    }
}

/**
 * Current resident bytes: heapUsed plus arrayBuffers (typed-array storage lives outside the heap).
 * @returns the byte count
 */
function residentBytes(): number {
    const usage = process.memoryUsage();
    return usage.heapUsed + usage.arrayBuffers;
}

/**
 * The median of a list of numbers.
 * @param values - the values
 * @returns the median
 */
function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Measure one benchmark: one warm-up run, then `runs` measured runs, each with a fresh input from
 * `setup()` and a garbage collection before the timer starts.
 * @param group - the benchmark group
 * @param name - the benchmark name
 * @param benchCase - the setup and the timed body
 * @param options - run count and rate reporting
 * @returns the result
 */
export function bench<T>(
    group: string,
    name: string,
    benchCase: BenchCase<T>,
    options: BenchOptions = {},
): BenchResult {
    const runs = options.runs ?? 5;
    const times: number[] = [];
    const deltas: number[] = [];
    let keepAlive: unknown = null;
    for (let i = 0; i <= runs; i++) {
        const input = benchCase.setup();
        collectGarbage();
        const before = residentBytes();
        const start = performance.now();
        keepAlive = benchCase.run(input);
        const elapsed = performance.now() - start;
        const after = residentBytes();
        if (i > 0) {
            times.push(elapsed);
            deltas.push(after - before);
        }
    }
    if (keepAlive === undefined) {
        throw new Error("unreachable: keepAlive is only compared to keep the result live");
    }
    const medianMs = median(times);
    const items = options.items ?? null;
    return {
        group,
        name,
        medianMs,
        minMs: Math.min(...times),
        maxMs: Math.max(...times),
        runs,
        memoryDeltaBytes: median(deltas),
        rate: items === null || medianMs === 0 ? null : (items / medianMs) * 1000,
        rateUnit: items === null ? null : `${options.unit ?? "items"}/s`,
    };
}

/**
 * Format a byte count for the table.
 * @param bytes - the byte count
 * @returns "12.3 MB"-style text
 */
function formatBytes(bytes: number): string {
    const sign = bytes < 0 ? "-" : "";
    const abs = Math.abs(bytes);
    if (abs >= 1024 * 1024) {
        return `${sign}${(abs / (1024 * 1024)).toFixed(1)} MB`;
    }
    if (abs >= 1024) {
        return `${sign}${(abs / 1024).toFixed(1)} KB`;
    }
    return `${sign}${abs} B`;
}

/**
 * Format a rate for the table.
 * @param rate - items per second
 * @returns "1.23 M/s"-style text
 */
function formatRate(rate: number): string {
    if (rate >= 1e6) {
        return `${(rate / 1e6).toFixed(2)} M`;
    }
    if (rate >= 1e3) {
        return `${(rate / 1e3).toFixed(1)} k`;
    }
    return rate.toFixed(0);
}

/**
 * Print a group of results as an aligned table.
 * @param results - the results of one group
 */
export function printTable(results: readonly BenchResult[]): void {
    const nameWidth = Math.max(4, ...results.map((r) => r.name.length));
    const header = `${"name".padEnd(nameWidth)}  ${"median".padStart(10)}  ${"min".padStart(9)}  ${"max".padStart(9)}  ${"memory".padStart(10)}  rate`;
    console.log(header);
    console.log("-".repeat(header.length + 12));
    for (const r of results) {
        const rate = r.rate === null ? "" : `${formatRate(r.rate)} ${r.rateUnit ?? ""}`;
        console.log(
            `${r.name.padEnd(nameWidth)}  ${`${r.medianMs.toFixed(1)} ms`.padStart(10)}  ` +
                `${`${r.minMs.toFixed(1)} ms`.padStart(9)}  ${`${r.maxMs.toFixed(1)} ms`.padStart(9)}  ` +
                `${formatBytes(r.memoryDeltaBytes).padStart(10)}  ${rate}`,
        );
    }
}

/** One benchmark session as stored in the results file. */
interface BenchSession {
    /** ISO timestamp of the run. */
    readonly date: string;
    /** Host name. */
    readonly host: string;
    /** process.version. */
    readonly node: string;
    /** CPU model string. */
    readonly cpu: string;
    /** Whether --expose-gc was on. */
    readonly exposeGc: boolean;
    /** The results. */
    readonly results: readonly BenchResult[];
}

/**
 * Append a session to `benchmarks/results/<host>-<node>.json` (a JSON array of sessions, created
 * when absent).
 * @param results - every result of the session
 * @returns the path written
 */
export function appendSession(results: readonly BenchResult[]): string {
    const dir = join(dirname(fileURLToPath(import.meta.url)), "results");
    mkdirSync(dir, { recursive: true });
    const host = hostname().replace(/[^A-Za-z0-9_.-]/g, "_");
    const node = process.version.replace(/^v/, "");
    const file = join(dir, `${host}-node${node}.json`);
    const session: BenchSession = {
        date: new Date().toISOString(),
        host: hostname(),
        node: process.version,
        cpu: cpus()[0]?.model ?? "unknown",
        exposeGc: typeof gc === "function",
        results,
    };
    let sessions: BenchSession[] = [];
    if (existsSync(file)) {
        sessions = JSON.parse(readFileSync(file, "utf8")) as BenchSession[];
    }
    sessions.push(session);
    writeFileSync(file, `${JSON.stringify(sessions, null, 4)}\n`);
    appendFileSync(file, "");
    return file;
}

/**
 * A deterministic xorshift32 generator so every benchmark input is the same on every run.
 * @param seed - the seed
 * @returns a function returning uniform numbers in [0, 1)
 */
export function makeRandom(seed: number): () => number {
    let state = seed >>> 0 || 1;
    return () => {
        state ^= state << 13;
        state >>>= 0;
        state ^= state >>> 17;
        state ^= state << 5;
        state >>>= 0;
        return state / 0x100000000;
    };
}
