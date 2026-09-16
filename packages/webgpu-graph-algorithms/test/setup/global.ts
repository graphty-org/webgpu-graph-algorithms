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
export function setup(): void {
    rmSync(KEY_LOG_DIR, { recursive: true, force: true });
    mkdirSync(KEY_LOG_DIR, { recursive: true });
    process.env.GRAPHTY_PIPELINE_KEY_LOG = KEY_LOG_DIR;
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
