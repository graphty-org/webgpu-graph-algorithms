#!/usr/bin/env node
/**
 * bench:compare -- the benchmark regression check of the GPU lane (spec 10.4 T-13, 11.7; contract 6.8).
 *
 * Reads three files relative to the current directory (the package root under `pnpm run bench:compare`; gpu.yml runs
 * it with working-directory packages/webgpu-graph-algorithms):
 *   gpu-report.json                  written by `node scripts/gpu-report.js > gpu-report.json` (6.6): the runner class
 *                                    (computed through scripts/runner-class.js, the ONE copy of the rule, so it names the
 *                                    same file the harness wrote) and the 10-second nvidia-smi sample
 *   benchmarks/out/<class>.json      the sessions `pnpm run bench` appended on this runner (the LAST is compared)
 *   benchmarks/results/<class>.json  the checked-in baseline of the class (the LAST session is the baseline)
 *
 * Rules, applied in this order:
 *   1. no gpu-report.json, or no out file for the class          -> "nothing to compare", exit 0
 *   2. nvidiaSmi.available and (maxUtilization > 10 or memory     -> "SKIPPED: GPU not quiet", exit 0 (T-13: the card may be
 *      in use by other processes > 0) during the sample              an interactive dev GPU; busy medians mean nothing)
 *   3. no baseline file                                           -> every result printed as "new (no baseline)", exit 0
 *   4. every result present in both sessions (matched by group + name): medianMs > threshold x baseline.medianMs is a
 *      REGRESSION; any regression -> the table and exit 1, else the table and exit 0. Rows without a baseline are "new".
 *
 * "Memory in use by other processes" is maxMemoryUsedMiB minus the report's own footprint estimate, which is the sample's
 * MINIMUM memoryUsedMiB: the report process holds one device for the whole sample, so its footprint is the floor of the
 * series; a quiet card with a resident desktop compositor has a flat series (max - min = 0) and passes, while a process
 * that allocates during the sample lifts the maximum above the minimum and skips the comparison.
 *
 * Options: --threshold <factor> (default 3), --class <name> (overrides the report's runnerClass). Rows are matched by
 * `group/name`; a group name never contains a slash, so the key is unambiguous even when a benchmark name does.
 */

import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

/**
 * Parses --threshold N and --class NAME.
 * @param {readonly string[]} argv - process.argv.slice(2)
 * @returns {{ threshold: number, cls: string | null }} the options
 */
function parseArgs(argv) {
    let threshold = 3;
    let cls = null;
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === "--threshold") {
            threshold = Number(argv[i + 1]);
            if (!Number.isFinite(threshold) || threshold <= 0) {
                throw new Error(`--threshold expects a positive number, got ${String(argv[i + 1])}`);
            }
            i += 1;
        } else if (a === "--class") {
            cls = argv[i + 1];
            if (cls === undefined || cls === "") {
                throw new Error("--class expects a runner class name");
            }
            i += 1;
        } else {
            throw new Error(`unknown option ${a}; known: --threshold N --class NAME`);
        }
    }
    return { threshold, cls };
}

/**
 * The last session of a sessions file, or null when the file is absent or empty.
 * @param {string} file - the path
 * @returns {{ results: readonly { group: string, name: string, medianMs: number }[] } | null} the last session
 */
function lastSession(file) {
    if (!existsSync(file)) {
        return null;
    }
    const sessions = JSON.parse(readFileSync(file, "utf8"));
    if (!Array.isArray(sessions) || sessions.length === 0) {
        return null;
    }
    const last = sessions[sessions.length - 1];
    if (last === null || typeof last !== "object" || !Array.isArray(last.results)) {
        return null;
    }
    return last;
}

/**
 * Whether the nvidia-smi sample shows another user of the card (rule 2).
 * @param {{ available: boolean, samples: readonly { utilizationGpu: number, memoryUsedMiB: number }[], maxUtilization: number, maxMemoryUsedMiB: number } | undefined} smi - the report's nvidiaSmi field
 * @returns {string | null} the reason the GPU is not quiet, or null when it is
 */
function notQuiet(smi) {
    if (
        smi === undefined ||
        smi === null ||
        !smi.available ||
        !Array.isArray(smi.samples) ||
        smi.samples.length === 0
    ) {
        return null;
    }
    if (smi.maxUtilization > 10) {
        return `utilisation ${String(smi.maxUtilization)}% > 10% during the sample`;
    }
    const minMemory = Math.min(...smi.samples.map((s) => s.memoryUsedMiB));
    const others = smi.maxMemoryUsedMiB - minMemory;
    if (others > 0) {
        return `memory in use by other processes: ${String(others)} MiB above the report's own ${String(minMemory)} MiB floor`;
    }
    return null;
}

/**
 * Formats one table row.
 * @param {string} status - REGRESSION / ok / new (no baseline)
 * @param {{ group: string, name: string, medianMs: number }} r - the current result
 * @param {number | null} baseline - the baseline median, or null
 * @returns {string} the row
 */
function row(status, r, baseline) {
    const ratio = baseline === null || baseline === 0 ? "" : `x${(r.medianMs / baseline).toFixed(2)}`;
    const base = baseline === null ? "-" : `${baseline.toFixed(3)} ms`;
    return `${status.padEnd(18)} ${`${r.group}/${r.name}`.padEnd(70)} ${`${r.medianMs.toFixed(3)} ms`.padStart(14)} ${base.padStart(14)} ${ratio.padStart(8)}`;
}

/**
 * Applies the four rules.
 * @returns {number} the exit code
 */
function main() {
    const { threshold, cls: clsOverride } = parseArgs(process.argv.slice(2));
    const reportFile = resolve("gpu-report.json");
    if (!existsSync(reportFile)) {
        console.log("nothing to compare: no gpu-report.json in the current directory");
        return 0;
    }
    const report = JSON.parse(readFileSync(reportFile, "utf8"));
    const cls = clsOverride ?? report.runnerClass;
    if (typeof cls !== "string" || cls === "") {
        console.log("nothing to compare: gpu-report.json carries no runnerClass and --class was not given");
        return 0;
    }
    const current = lastSession(resolve("benchmarks/out", `${cls}.json`));
    if (current === null) {
        console.log(`nothing to compare: no benchmarks/out/${cls}.json (the bench step wrote nothing)`);
        return 0;
    }
    const reason = notQuiet(report.nvidiaSmi);
    if (reason !== null) {
        console.log(`SKIPPED: GPU not quiet (${reason}); the medians of this run are not compared (T-13)`);
        return 0;
    }
    const baseline = lastSession(resolve("benchmarks/results", `${cls}.json`));
    console.log(
        `${"status".padEnd(18)} ${"benchmark".padEnd(70)} ${"median".padStart(14)} ${"baseline".padStart(14)} ${"ratio".padStart(8)}`,
    );
    if (baseline === null) {
        for (const r of current.results) {
            console.log(row("new (no baseline)", r, null));
        }
        console.log(`no baseline benchmarks/results/${cls}.json: every result is new`);
        return 0;
    }
    const baselines = new Map(baseline.results.map((r) => [`${r.group}/${r.name}`, r.medianMs]));
    let regressions = 0;
    for (const r of current.results) {
        const base = baselines.get(`${r.group}/${r.name}`);
        if (base === undefined) {
            console.log(row("new (no baseline)", r, null));
        } else if (r.medianMs > threshold * base) {
            regressions += 1;
            console.log(row("REGRESSION", r, base));
        } else {
            console.log(row("ok", r, base));
        }
    }
    if (regressions > 0) {
        console.log(
            `${String(regressions)} regression(s): a median above ${String(threshold)}x the baseline of class ${cls}`,
        );
        return 1;
    }
    console.log(`no regression above ${String(threshold)}x the baseline of class ${cls}`);
    return 0;
}

try {
    process.exitCode = main();
} catch (error) {
    console.error(`[bench-compare] ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 2;
}
