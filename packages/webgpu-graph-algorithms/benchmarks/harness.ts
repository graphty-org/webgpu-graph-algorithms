/**
 * The measurement harness of the @graphty/webgpu-graph-algorithms benchmarks (spec 11.7; contract 6.1): graph-format's
 * harness (packages/graph-format/benchmarks/harness.ts) with ONE change of substance -- `run` is async and the timer
 * brackets `await run(input); await device.queue.onSubmittedWorkDone()` so GPU work is timed, not merely queued -- plus a
 * `teardown` hook (release the snapshot a run uploaded), the `gpu` field of every session (adapter identity, driver, the
 * requested limits, software flag, runtime, subgroup size) and the runner class, so numbers from the RTX 4070 SUPER, the
 * gpu-linux-t4 lane and a browser never mix. Every benchmark runs its body `runs` times (default 5) after one warm-up and
 * reports the MEDIAN wall time plus the median process.memoryUsage() heap-and-buffer delta of one run; with `--expose-gc`
 * the garbage collector runs before every measured iteration. Sessions are appended to benchmarks/out/<runner-class>.json
 * (gitignored, 6.4); the checked-in baseline is benchmarks/results/<runner-class>.json and scripts/bench-compare.js compares
 * the last session of each. The JSON shape is graph-format's plus the two fields, so a future merge tool reads both.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { cpus, hostname } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { runnerClass } from "../scripts/runner-class.js";
import { type GpuContext } from "../src/context.js";

/** Re-exported from scripts/runner-class.js (6.9): the ONE copy of the runner-class rule, shared with scripts/gpu-report.js and scripts/bench-compare.js so the file names of 6.4 cannot drift. */
export { runnerClass } from "../scripts/runner-class.js";

/** One measured benchmark (graph-format's shape, unchanged). */
export interface BenchResult {
    /** The benchmark group (upload, roundtrip, layout-exact). */
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
    /** A throughput figure the benchmark chose to report (bytes per second, nodes per second), or null. */
    readonly rate: number | null;
    /** What `rate` counts. */
    readonly rateUnit: string | null;
}

/**
 * A prepared benchmark: setup outside the timer, an ASYNC run inside it, optional teardown.
 * @public referenced only through bench()'s signature inside this file; the group files pass object literals
 */
export interface BenchCase<T> {
    /** Build the input (not timed). Called once per run, the warm-up included. */
    readonly setup: () => T | Promise<T>;
    /** The timed body; its return value is kept alive until the timing ends so it cannot be optimised away. */
    readonly run: (input: T) => Promise<unknown> | unknown;
    /** Runs after the timing of each run (release what the run uploaded); not timed. */
    readonly teardown?: ((input: T) => void | Promise<void>) | undefined;
}

/**
 * Options of bench(): the device whose queue the timer waits on, runs (default 5), items / unit for the rate column.
 * @public referenced only through bench()'s signature inside this file
 */
export interface BenchOptions {
    /** The device whose queue.onSubmittedWorkDone() closes the timer. */
    readonly device: GPUDevice;
    /** Measured runs (default: the process default, 5 unless setBenchRuns changed it). */
    readonly runs?: number | undefined;
    /** Items processed per run, for the rate column. */
    readonly items?: number | undefined;
    /** What `items` counts ("bytes", "nodes"). */
    readonly unit?: string | undefined;
}

/** The global gc hook exposed by `node --expose-gc`. */
declare const gc: (() => void) | undefined;

/** The process-wide default run count (5); run.ts sets it from --runs N. */
let defaultRuns = 5;

/**
 * The "no run happened yet" sentinel of bench()'s keep-alive slot. A timed body may legally resolve to undefined (contract
 * 6.1: `run: (input: T) => Promise<unknown> | unknown`; the upload group's body is one), so `undefined` cannot be the
 * unreached marker the way it is in graph-format's copy.
 */
const NEVER: unique symbol = Symbol("never");

/**
 * Sets the default `runs` of bench() for this process (run.ts's --runs N; the group functions keep their (ctx) signature).
 * @param runs - the measured runs per benchmark (>= 1)
 */
export function setBenchRuns(runs: number): void {
    if (!Number.isInteger(runs) || runs < 1) {
        throw new Error(`setBenchRuns: runs must be a positive integer, got ${String(runs)}`);
    }
    defaultRuns = runs;
}

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
 * Measure one benchmark: one warm-up run, then `runs` measured runs, each with a fresh input from `setup()`, a garbage
 * collection before the timer starts, and the timer closed by the device queue's onSubmittedWorkDone() so queued GPU work
 * is inside the measurement.
 * @param group - the benchmark group
 * @param name - the benchmark name
 * @param benchCase - the setup, the timed body and the optional teardown
 * @param options - the device, run count and rate reporting
 * @returns the result
 */
export async function bench<T>(
    group: string,
    name: string,
    benchCase: BenchCase<T>,
    options: BenchOptions,
): Promise<BenchResult> {
    const runs = options.runs ?? defaultRuns;
    const times: number[] = [];
    const deltas: number[] = [];
    let keepAlive: unknown = NEVER;
    for (let i = 0; i <= runs; i++) {
        const input = await benchCase.setup();
        collectGarbage();
        const before = residentBytes();
        const start = performance.now();
        keepAlive = await benchCase.run(input);
        await options.device.queue.onSubmittedWorkDone();
        const elapsed = performance.now() - start;
        const after = residentBytes();
        if (benchCase.teardown !== undefined) {
            await benchCase.teardown(input);
        }
        if (i > 0) {
            times.push(elapsed);
            deltas.push(after - before);
        }
    }
    if (keepAlive === NEVER) {
        throw new Error("unreachable: runs >= 1 always assigns keepAlive (the read keeps the result live)");
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
    if (rate >= 1e9) {
        return `${(rate / 1e9).toFixed(2)} G`;
    }
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
    const header = `${"name".padEnd(nameWidth)}  ${"median".padStart(11)}  ${"min".padStart(10)}  ${"max".padStart(10)}  ${"memory".padStart(10)}  rate`;
    console.log(header);
    console.log("-".repeat(header.length + 12));
    for (const r of results) {
        const rate = r.rate === null ? "" : `${formatRate(r.rate)} ${r.rateUnit ?? ""}`;
        console.log(
            `${r.name.padEnd(nameWidth)}  ${`${r.medianMs.toFixed(3)} ms`.padStart(11)}  ` +
                `${`${r.minMs.toFixed(3)} ms`.padStart(10)}  ${`${r.maxMs.toFixed(3)} ms`.padStart(10)}  ` +
                `${formatBytes(r.memoryDeltaBytes).padStart(10)}  ${rate}`,
        );
    }
}

/**
 * The gpu field of a session (spec 11.7): adapter identity, driver, the requested limits, software flag, runtime.
 * @public read by scripts/bench-compare.js (plain JS) and by test/benchmarks.test.ts
 */
export interface GpuSessionInfo {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
    /** The version string found in the description ("580.173.02"; "23.2.1" for Mesa), or "" when none. */
    readonly driver: string;
    readonly limits: {
        readonly maxBufferSize: number;
        readonly maxStorageBufferBindingSize: number;
        readonly maxStorageBuffersPerShaderStage: number;
        readonly maxComputeWorkgroupsPerDimension: number;
    };
    readonly software: boolean;
    readonly runtime: "node" | "browser";
    readonly subgroupMaxSize: number;
}

/**
 * One session as stored (graph-format's fields plus gpu and runnerClass).
 * @public the file shape of benchmarks/out and benchmarks/results (6.4); read by scripts/bench-compare.js
 */
export interface BenchSession {
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
    /** The adapter the session ran on. */
    readonly gpu: GpuSessionInfo;
    /** runnerClass(gpu) at the time of the run (GRAPHTY_RUNNER_CLASS honoured). */
    readonly runnerClass: string;
    /** The results. */
    readonly results: readonly BenchResult[];
}

/**
 * The browser bridge payload (5.1 browser-commands.d.ts).
 * @public imported by test/setup/browser-commands.d.ts at P3-T6 (the appendBenchRecord payload)
 */
export interface BrowserBenchPayload {
    readonly runnerClass: string;
    readonly session: BenchSession;
}

/**
 * The GpuSessionInfo of a context.
 * @param ctx - the context the benchmarks run on
 * @returns the session's gpu record
 */
export function gpuSessionInfo(ctx: GpuContext): GpuSessionInfo {
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
        runtime: caps.runtime === "browser" ? "browser" : "node",
        subgroupMaxSize: caps.subgroupMaxSize,
    };
}

/**
 * Appends a session to benchmarks/out/<runnerClass>.json (created when absent); returns the path.
 * @param results - every result of the session
 * @param gpu - the adapter record (gpuSessionInfo)
 * @param options - `dir` overrides the output directory (tests)
 * @returns the path written
 */
export function appendSession(
    results: readonly BenchResult[],
    gpu: GpuSessionInfo,
    options?: { readonly dir?: string | undefined } | undefined,
): string {
    const dir = options?.dir ?? join(dirname(fileURLToPath(import.meta.url)), "out");
    mkdirSync(dir, { recursive: true });
    const cls = runnerClass(gpu);
    const file = join(dir, `${cls}.json`);
    const session: BenchSession = {
        date: new Date().toISOString(),
        host: hostname(),
        node: process.version,
        cpu: cpus()[0]?.model ?? "unknown",
        exposeGc: typeof gc === "function",
        gpu,
        runnerClass: cls,
        results,
    };
    let sessions: BenchSession[] = [];
    if (existsSync(file)) {
        sessions = JSON.parse(readFileSync(file, "utf8")) as BenchSession[];
    }
    sessions.push(session);
    writeFileSync(file, `${JSON.stringify(sessions, null, 4)}\n`);
    return file;
}

/**
 * A deterministic xorshift32 generator so every benchmark input is the same on every run (graph-format's). The bitwise
 * operators act on the generator's 32-bit state, never on an arc index or a byte offset.
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
