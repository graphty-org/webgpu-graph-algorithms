/**
 * The browser-project wrapper of spec 11.6 (contract 6.7): runs the vitest `browser` project under a hard
 * limit and turns the browser.close() hang of R-13 (Chromium on the NVIDIA path can hang AFTER every test
 * passed) into the verdict the JSON reporter recorded.
 *
 * Spawns
 *   timeout -k 10 600 pnpm exec vitest run --project=browser \
 *       --reporter=default --reporter=json --outputFile=browser-results.json
 * from the package root, inheriting the environment (GRAPHTY_BROWSER_GPU, GRAPHTY_GPU_REQUIRE,
 * GRAPHTY_EGL_LIB_DIR and the rest are read by vitest.config.ts), plus any extra command-line arguments given to
 * this script (forwarded to vitest verbatim, e.g. --passWithNoTests for a config smoke).
 *
 * Exit-code rule (spec 11.6, contract 6.7):
 *   vitest exit 0                       -> exit 0
 *   the limit hit (GNU timeout exit 124 -- the TERM stopped the run -- or 137, its exit when the
 *                  -k KILL was needed after the 10 s grace: coreutils timeout.c preserves 128 + 9 in that case)
 *                                       -> exit 0 iff browser-results.json parses and has
 *                                          numTotalTests > 0 && numFailedTests === 0, else exit 1 with the summary
 *   any other exit                      -> that exit code
 * On a platform without GNU timeout, vitest runs directly with a 600 s Node-side SIGTERM (spawnSync timeout),
 * which is treated as the limit case. A stale browser-results.json is deleted before the run so an old file
 * can never turn a hung run green. The summary line "numTotalTests / numPassedTests / numFailedTests" is
 * printed whenever the JSON is present.
 */

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

/** The JSON reporter output, relative to the package root. */
export const RESULTS_FILE = "browser-results.json";
/** The hard limit in seconds (spec 11.6). */
export const TIMEOUT_SECONDS = 600;
/** The KILL grace after the TERM, in seconds (spec 11.6). */
export const KILL_AFTER_SECONDS = 10;
/** GNU timeout's exit when the limit hit and the TERM stopped the command. */
export const EXIT_TIMED_OUT = 124;
/** GNU timeout's exit when the limit hit and the -k KILL was needed (128 + 9). */
export const EXIT_KILLED = 137;

const VITEST_ARGS = [
    "exec",
    "vitest",
    "run",
    "--project=browser",
    "--reporter=default",
    "--reporter=json",
    `--outputFile=${RESULTS_FILE}`,
];

/**
 * Read the three counters of a vitest JSON report.
 * @param {string} file - the JSON report path
 * @returns {{ numTotalTests: number, numPassedTests: number, numFailedTests: number } | null} the counters, or
 * null when the file is absent or does not parse to an object with numeric counters
 */
export function readSummary(file) {
    if (!existsSync(file)) {
        return null;
    }
    let parsed;
    try {
        parsed = JSON.parse(readFileSync(file, "utf8"));
    } catch {
        return null;
    }
    if (parsed === null || typeof parsed !== "object") {
        return null;
    }
    const { numTotalTests, numPassedTests, numFailedTests } = parsed;
    if (typeof numTotalTests !== "number" || typeof numPassedTests !== "number" || typeof numFailedTests !== "number") {
        return null;
    }
    return { numTotalTests, numPassedTests, numFailedTests };
}

/**
 * The exit code for a vitest / timeout exit status and the JSON summary.
 * @param {number} status - the exit status of the spawned command (124 / 137 = the limit hit)
 * @param {{ numTotalTests: number, numPassedTests: number, numFailedTests: number } | null} summary - readSummary()
 * @returns {{ exitCode: number, message: string }} the verdict and a one-line explanation
 */
export function verdict(status, summary) {
    if (status === 0) {
        return { exitCode: 0, message: "vitest exited 0" };
    }
    if (status === EXIT_TIMED_OUT || status === EXIT_KILLED) {
        const limitHit = `the ${TIMEOUT_SECONDS} s limit hit (exit ${status})`;
        if (summary === null) {
            return {
                exitCode: 1,
                message: `${limitHit} and ${RESULTS_FILE} is absent or unreadable: FAIL`,
            };
        }
        if (summary.numTotalTests > 0 && summary.numFailedTests === 0) {
            return {
                exitCode: 0,
                message: `${limitHit} after every test passed (the browser.close() hang of R-13): PASS`,
            };
        }
        const counts = `numTotalTests=${summary.numTotalTests} numFailedTests=${summary.numFailedTests}`;
        return {
            exitCode: 1,
            message: `${limitHit} with ${counts}: FAIL`,
        };
    }
    return { exitCode: status, message: `vitest exited ${status}` };
}

/**
 * Whether GNU coreutils timeout is on PATH.
 * @returns {boolean} true when `timeout --version` runs and names GNU coreutils
 */
export function hasGnuTimeout() {
    const probe = spawnSync("timeout", ["--version"], { encoding: "utf8" });
    return probe.status === 0 && typeof probe.stdout === "string" && probe.stdout.includes("GNU coreutils");
}

/**
 * Run the browser project and return the process exit code.
 * @param {readonly string[]} extraArgs - arguments appended to the vitest command line
 * @returns {number} the exit code
 */
export function main(extraArgs) {
    const resultsPath = resolve(packageRoot, RESULTS_FILE);
    rmSync(resultsPath, { force: true });
    const vitestArgs = [...VITEST_ARGS, ...extraArgs];
    let status;
    if (hasGnuTimeout()) {
        const args = ["-k", String(KILL_AFTER_SECONDS), String(TIMEOUT_SECONDS), "pnpm", ...vitestArgs];
        console.error(`[run-browser-project] timeout ${args.join(" ")}`);
        const result = spawnSync("timeout", args, { cwd: packageRoot, stdio: "inherit" });
        status = result.status ?? 1;
    } else {
        console.error(
            `[run-browser-project] no GNU timeout on PATH: pnpm ${vitestArgs.join(" ")} ` +
                `with a ${TIMEOUT_SECONDS} s Node-side limit`,
        );
        const result = spawnSync("pnpm", vitestArgs, {
            cwd: packageRoot,
            stdio: "inherit",
            timeout: TIMEOUT_SECONDS * 1000,
            killSignal: "SIGTERM",
        });
        status =
            result.error !== undefined && result.error.code === "ETIMEDOUT" ? EXIT_TIMED_OUT : (result.status ?? 1);
    }
    const summary = readSummary(resultsPath);
    if (summary !== null) {
        const counts = [
            `numTotalTests=${summary.numTotalTests}`,
            `numPassedTests=${summary.numPassedTests}`,
            `numFailedTests=${summary.numFailedTests}`,
        ].join(" ");
        console.error(`[run-browser-project] ${counts}`);
    }
    const decision = verdict(status, summary);
    console.error(`[run-browser-project] ${decision.message}`);
    return decision.exitCode;
}

const isMain = process.argv[1] !== undefined && pathToFileURL(resolve(process.argv[1])).href === import.meta.url;
if (isMain) {
    process.exit(main(process.argv.slice(2)));
}
