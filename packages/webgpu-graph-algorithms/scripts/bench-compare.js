/**
 * The benchmark regression check of spec 10.4 (T-13) and 11.7 (contract 6.8): compares the LAST session of
 * benchmarks/out/<runner-class>.json (this run) with the LAST session of benchmarks/results/<runner-class>.json
 * (the checked-in baseline of the same runner class) and fails on any result whose medianMs exceeds
 * `threshold` (default 3) times the baseline's, unless the GPU was not quiet during scripts/gpu-report.js's
 * nvidia-smi sample (maxUtilization > 10, or memory in use by OTHER processes > 0 where that figure is the
 * sample's maxMemoryUsedMiB minus the sample's minimum -- the report's own process footprint estimate -- so a
 * quiet card with a resident desktop compositor passes), in which case the comparison is SKIPPED with exit 0.
 *
 * The runner class comes from gpu-report.json (computed there through scripts/runner-class.js, the same
 * function benchmarks/harness.ts uses to name its output), so the file this script looks for is the one the
 * harness wrote. Options: --threshold <n> (default 3), --class <name> (overrides the report's runner class).
 *
 * P0 form (contract 6.8: the rule is completed at P1-T7 together with the harness that writes the out file):
 * the option parsing, the input files and the two "nothing to compare" exits are in place; when both inputs
 * exist the script prints what it found and exits 0 -- the 3x rule and the quiet-GPU skip are P1-T7's edit.
 *
 * Exit codes: 0 nothing to compare / skipped / no regression; 1 a regression (P1-T7); 2 a malformed input file.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

/** The default regression factor (spec 10.4 T-13). */
export const DEFAULT_THRESHOLD = 3;

/**
 * Parse the command line.
 * @param {readonly string[]} argv - the arguments after the script name
 * @returns {{ threshold: number, className: string | null }} the options
 */
export function parseArgs(argv) {
    let threshold = DEFAULT_THRESHOLD;
    let className = null;
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === "--threshold" && i + 1 < argv.length) {
            threshold = Number(argv[i + 1]);
            i++;
        } else if (a.startsWith("--threshold=")) {
            threshold = Number(a.slice("--threshold=".length));
        } else if (a === "--class" && i + 1 < argv.length) {
            className = argv[i + 1];
            i++;
        } else if (a.startsWith("--class=")) {
            className = a.slice("--class=".length);
        } else {
            throw new Error(
                `unknown option ${a}; usage: node scripts/bench-compare.js [--threshold <n>] [--class <name>]`,
            );
        }
    }
    if (!Number.isFinite(threshold) || threshold <= 0) {
        throw new Error(`--threshold must be a positive number, got ${String(threshold)}`);
    }
    return { threshold, className };
}

/**
 * Read a JSON file.
 * @param {string} file - the path
 * @returns {unknown} the parsed document, or null when the file is absent
 */
export function readJson(file) {
    if (!existsSync(file)) {
        return null;
    }
    return JSON.parse(readFileSync(file, "utf8"));
}

/**
 * The last session of a sessions file (a JSON array ordered by date, contract 6.4).
 * @param {unknown} sessions - the parsed file
 * @returns {{ results: readonly { group: string, name: string, medianMs: number }[] } | null} the last session
 */
export function lastSession(sessions) {
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
 * Run the comparison and return the exit code.
 * @param {readonly string[]} argv - the arguments after the script name
 * @returns {number} the exit code
 */
export function main(argv) {
    const { threshold, className } = parseArgs(argv);
    const reportFile = resolve(packageRoot, "gpu-report.json");
    const report = readJson(reportFile);
    if (report === null && className === null) {
        console.log(
            "nothing to compare: gpu-report.json is absent " +
                "(run scripts/gpu-report.js > gpu-report.json first, or pass --class)",
        );
        return 0;
    }
    const runner =
        className ??
        (report !== null && typeof report === "object" && typeof report.runnerClass === "string"
            ? report.runnerClass
            : null);
    if (runner === null) {
        console.error("gpu-report.json carries no runnerClass string");
        return 2;
    }
    const outFile = resolve(packageRoot, "benchmarks/out", `${runner}.json`);
    const out = lastSession(readJson(outFile));
    if (out === null) {
        console.log(`nothing to compare: ${outFile} is absent or has no session`);
        return 0;
    }
    const baselineFile = resolve(packageRoot, "benchmarks/results", `${runner}.json`);
    const baseline = lastSession(readJson(baselineFile));
    const baselineCount = baseline === null ? "absent" : `${baseline.results.length} result(s)`;
    const summary = [
        `bench-compare (P0 form): ${out.results.length} result(s) in ${outFile}`,
        `baseline ${baselineCount} in ${baselineFile}`,
        `threshold ${threshold}x -- the comparison rule lands at P1-T7 (contract 6.8)`,
    ].join("; ");
    console.log(summary);
    return 0;
}

const isMain = process.argv[1] !== undefined && pathToFileURL(resolve(process.argv[1])).href === import.meta.url;
if (isMain) {
    try {
        process.exit(main(process.argv.slice(2)));
    } catch (error) {
        console.error(`[bench-compare] ${error instanceof Error ? error.message : String(error)}`);
        process.exit(2);
    }
}
