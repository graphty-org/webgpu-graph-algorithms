/**
 * vitest globalSetup of the node project (contract 5.1; spec 5.1, 11.3): `setup` clears the pipeline-key log
 * directory and exports its path through GRAPHTY_PIPELINE_KEY_LOG, so every worker's test/setup/gpu.ts appends the
 * keys its contexts created in its afterAll; `teardown` reads every worker's log and fails the run when a key of a
 * registry kernel is not produced by a case of test/helpers/override-matrix.ts -- the "complete" half of the compile
 * matrix rule (the "bounded" half is test/kernel/wgsl-compile.test.ts). globalSetup runs in the main process before
 * the fork pool captures process.env (vitest 3.2.7 executeTests spreads process.env when runTests is called), so the
 * variable reaches the workers; a throwing teardown rejects vitest's close() and the CLI exits 1.
 *
 * PLAN DECISION (P2-T2): the log format accepted is any of a JSON array of key strings, JSON lines (one JSON string
 * per line) or plain lines (one key per line), in any file of the directory.
 */

import { mkdirSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { checkAdapter, parseGpuRequire } from "../../scripts/gpu-policy.js";
import { summarizeAdapter } from "../../src/device/acquire.js";
import { createNodeGpu, type NodeGpuHandle } from "../../src/node/index.js";
import { CAPS_SPEC_DEFAULT } from "../helpers/caps-tables.js";
import { matrixCovers } from "../helpers/override-matrix.js";

/** The key-log directory: <package>/tmp/pipeline-keys (the package tmp/ is gitignored). */
export const KEY_LOG_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "../../tmp/pipeline-keys");

/**
 * The keys of one log file (see the format note above); blank lines are skipped.
 * @param text - the file contents
 * @returns the keys in file order
 */
function parseKeyLog(text: string): string[] {
    const trimmed = text.trim();
    if (trimmed === "") {
        return [];
    }
    if (trimmed.startsWith("[")) {
        const doc: unknown = JSON.parse(trimmed);
        if (Array.isArray(doc)) {
            return doc.filter((key: unknown): key is string => typeof key === "string");
        }
    }
    const keys: string[] = [];
    for (const line of trimmed.split("\n")) {
        const item = line.trim();
        if (item === "") {
            continue;
        }
        keys.push(item.startsWith('"') ? (JSON.parse(item) as string) : item);
    }
    return keys;
}

/**
 * Every key of every worker log in a directory (a missing directory reads as empty).
 * @param dir - the log directory
 * @returns the distinct keys
 */
export function readKeyLogs(dir: string): Set<string> {
    const keys = new Set<string>();
    let names: string[] = [];
    try {
        names = readdirSync(dir);
    } catch {
        return keys;
    }
    for (const name of names) {
        const file = resolve(dir, name);
        if (!statSync(file).isFile()) {
            continue;
        }
        for (const key of parseKeyLog(readFileSync(file, "utf8"))) {
            keys.add(key);
        }
    }
    return keys;
}

/**
 * Throws (naming every uncovered key) unless OVERRIDE_MATRIX covers every registry key in `keys`.
 * @param keys - the keys the node suite created
 */
export function assertMatrixCoverage(keys: Iterable<string>): void {
    const { ok, missing } = matrixCovers(keys, CAPS_SPEC_DEFAULT);
    if (!ok) {
        throw new Error(
            `override-matrix coverage: ${missing.length} pipeline key(s) created by the node suite are not produced by any OVERRIDE_MATRIX case (add the combination to test/helpers/override-matrix.ts):\n  ${missing.join("\n  ")}`,
        );
    }
}

/** Clears the pipeline-key log directory (tmp/pipeline-keys/) and exports its path through GRAPHTY_PIPELINE_KEY_LOG. */
export async function setup(): Promise<void> {
    rmSync(KEY_LOG_DIR, { recursive: true, force: true });
    mkdirSync(KEY_LOG_DIR, { recursive: true });
    process.env.GRAPHTY_PIPELINE_KEY_LOG = KEY_LOG_DIR;
    await failFastWithoutAdapter();
}

/**
 * The value of an environment variable, or undefined when unset / empty.
 * @param name - the variable
 * @returns the value or undefined
 */
function envOrUndefined(name: string): string | undefined {
    const value = process.env[name];
    return value === undefined || value === "" ? undefined : value;
}

/**
 * Probes the adapter ONCE in the main process under the GRAPHTY_GPU_REQUIRE policy (spec 11.2) and fails the run
 * with one readable message when the policy demands an adapter and none (or the wrong one) exists -- instead of
 * every GPU test of every worker failing with the same line (the first GitHub run printed it ~1300 times). Under
 * the unset (skip) policy nothing is probed: the workers skip each GPU test with the printed reason as before. The
 * Dawn handle is dropped again so the main process can exit.
 */
async function failFastWithoutAdapter(): Promise<void> {
    const policy = parseGpuRequire(process.env.GRAPHTY_GPU_REQUIRE);
    if (policy.level === "skip") {
        return;
    }
    const adapterFlag = envOrUndefined("GRAPHTY_GPU_ADAPTER");
    const dawnFeatures = envOrUndefined("GRAPHTY_DAWN_FEATURES");
    let handle: NodeGpuHandle | null = null;
    let reason: string | null = null;
    let info: { readonly vendor: string; readonly architecture: string } | null = null;
    try {
        handle = await createNodeGpu({
            adapter: adapterFlag,
            dawnFeatures: dawnFeatures === undefined ? undefined : dawnFeatures.split(","),
        });
        const adapter = await handle.gpu.requestAdapter();
        if (adapter === null) {
            reason = "requestAdapter() returned null: no usable WebGPU adapter";
        } else {
            const summary = summarizeAdapter(adapter);
            info = { vendor: summary.vendor, architecture: summary.architecture };
        }
    } catch (err) {
        reason = err instanceof Error ? err.message : String(err);
    } finally {
        handle?.dispose();
    }
    const verdict = checkAdapter(info, policy);
    if (verdict.ok) {
        return;
    }
    const lines = [
        `GRAPHTY_GPU_REQUIRE=${policy.raw} requires a WebGPU adapter and the run cannot satisfy it; failing up front instead of in every GPU test.`,
        `  ${verdict.reason ?? reason ?? "no WebGPU adapter"}`,
        `  Dawn flags: adapter=${adapterFlag ?? "(any)"} features=${dawnFeatures ?? "(none)"}; platform ${process.platform}/${process.arch}`,
        "  Hints: Linux needs a Vulkan ICD (lavapipe: apt install mesa-vulkan-drivers, then VK_DRIVER_FILES=<path of lvp_icd*.json>, on",
        "  noble /usr/share/vulkan/icd.d/lvp_icd.json); the NVIDIA ICD under Dawn dlopen()s libEGL.so.1 (see CLAUDE.md, LD_LIBRARY_PATH);",
        "  a missing native module means the optional peer dependency webgpu is not installed; unset GRAPHTY_GPU_REQUIRE to skip the",
        "  GPU tests on a machine without a GPU.",
    ];
    throw new Error(lines.join("\n"));
}

/** Reads every worker's key log and asserts test/helpers/override-matrix.ts covers each key (throws with the uncovered keys, which fails the run). */
export function teardown(): void {
    const keys = readKeyLogs(KEY_LOG_DIR);
    const policy = process.env.GRAPHTY_GPU_REQUIRE ?? "";
    if (keys.size === 0 && policy !== "" && process.env.CI !== undefined) {
        throw new Error(
            "override-matrix coverage: no pipeline keys were logged although an adapter was required (GRAPHTY_PIPELINE_KEY_LOG did not reach the workers, or test/setup/gpu.ts wrote no log)",
        );
    }
    assertMatrixCoverage(keys);
    console.warn(`[override-matrix] ${keys.size} distinct pipeline key(s) read from ${KEY_LOG_DIR}, all covered`);
}
