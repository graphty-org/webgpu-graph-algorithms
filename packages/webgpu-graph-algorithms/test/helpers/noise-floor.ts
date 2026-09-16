/**
 * The noise-floor fixtures and rows (spec 11.9 item 3; contract 5.2, 5.6). A kernel test writes this adapter's RAW
 * output as test/fixtures/noise/<kernel>-<fixture>-<adapterClass>.json (only when GRAPHTY_NOISE_FLOOR_WRITE=1),
 * compares against every committed adapter's output (u32 bitwise, f32 within a derived bound) and records the
 * measured spread as a row of benchmarks/results/noise-floor.json; the parity tests take their tolerances from that
 * file through noiseFloorFor(), never from a literal. Node only (node:fs); test/helpers/noise-floor-browser.ts (P1-T7)
 * is the browser twin over the commands bridge of vitest.config.ts, whose file shapes this module mirrors exactly.
 *
 * Writes are atomic (write `<file>.tmp`, then rename): the node project runs test files in parallel forks, so a reader
 * of the fixture directory must never see a truncated JSON file. GRAPHTY_NOISE_DIR (test-only; never set by CI or the
 * default lane) redirects the fixture directory so the unit test of this module writes into a temporary directory
 * instead of the committed one.
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { type GpuCaps } from "../../src/types/context.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = resolve(HERE, "..", "..");
const NOISE_DIR = resolve(PACKAGE_ROOT, "test", "fixtures", "noise");
const NOISE_FLOOR_FILE = resolve(PACKAGE_ROOT, "benchmarks", "results", "noise-floor.json");

/**
 * The fixture directory: test/fixtures/noise, or GRAPHTY_NOISE_DIR when set (test-only redirection).
 * @returns the absolute directory
 */
function noiseDir(): string {
    const override = process.env.GRAPHTY_NOISE_DIR;
    return override === undefined || override === "" ? NOISE_DIR : resolve(override);
}

/**
 * Writes a file atomically: the bytes go to `<path>.tmp`, then rename() replaces `path` in one step, so a
 * concurrent reader sees the old file or the new one, never a truncated one.
 * @param path - the destination
 * @param contents - the text to write
 */
function writeAtomic(path: string, contents: string): void {
    const tmp = `${path}.tmp`;
    writeFileSync(tmp, contents);
    renameSync(tmp, path);
}

/** The kind of comparison a row records (5.6). */
type NoiseComparison = "twin" | "cross-adapter" | "oracle-f64";

/** One row of benchmarks/results/noise-floor.json (5.6 schema). */
export interface NoiseRow {
    readonly id: string;
    readonly kernel: string;
    readonly fixture: string;
    readonly comparison: NoiseComparison;
    readonly a: string;
    readonly b: string;
    readonly maxRelError: number;
    readonly maxAbsError: number;
    readonly samples: number;
}

/** One adapter record of the noise-floor file (5.6). */
interface NoiseAdapter {
    readonly class: string;
    readonly vendor: string;
    readonly architecture: string;
    readonly description: string;
    readonly runtime: string;
}

/** The noise-floor document (5.6). */
interface NoiseFloorDocument {
    recordedAt: string;
    adapters: NoiseAdapter[];
    rows: NoiseRow[];
    tolerances: Record<string, { readonly value: number; readonly basis: string; readonly factor: number }>;
}

/** One committed fixture file (the shape the vitest.config.ts `writeNoiseFixture` command writes). */
interface NoiseFixtureDocument {
    readonly kernel: string;
    readonly fixture: string;
    readonly adapterClass: string;
    readonly dtype: "f32" | "u32";
    readonly values: readonly number[];
}

/**
 * The file-name sanitiser of vitest.config.ts (2.5): every character outside [A-Za-z0-9_.-] becomes "_".
 * @param s - a name segment
 * @returns the safe segment
 */
function safe(s: string): string {
    return s.replace(/[^A-Za-z0-9_.-]/g, "_");
}

/**
 * Whether writes are enabled (GRAPHTY_NOISE_FLOOR_WRITE=1).
 * @returns true when this run records fixtures and rows
 */
function writing(): boolean {
    return process.env.GRAPHTY_NOISE_FLOOR_WRITE === "1";
}

/**
 * Where a per-adapter raw output lives: test/fixtures/noise/<kernel>-<fixture>-<adapterClass>.json (under
 * GRAPHTY_NOISE_DIR instead when that test-only variable is set).
 * @param kernel - the kernel id
 * @param fixture - the fixture name
 * @param adapterClass - the adapter class string (adapterClass())
 * @returns the absolute path
 */
export function noiseFixturePath(kernel: string, fixture: string, adapterClass: string): string {
    return resolve(noiseDir(), `${safe(kernel)}-${safe(fixture)}-${safe(adapterClass)}.json`);
}

/**
 * Writes this adapter's raw output (only when GRAPHTY_NOISE_FLOOR_WRITE=1) in the same JSON shape as the
 * vitest.config.ts `writeNoiseFixture` command (2.5).
 * @param kernel - the kernel id
 * @param fixture - the fixture name
 * @param adapterClass - the adapter class string
 * @param values - the raw output
 * @param dtype - how the values are compared: "u32" bitwise, "f32" within a floor
 */
export function writeNoiseFixture(
    kernel: string,
    fixture: string,
    adapterClass: string,
    values: ArrayLike<number>,
    dtype: "f32" | "u32",
): void {
    if (!writing()) {
        return;
    }
    mkdirSync(noiseDir(), { recursive: true });
    const doc: NoiseFixtureDocument = { kernel, fixture, adapterClass, dtype, values: Array.from(values) };
    writeAtomic(noiseFixturePath(kernel, fixture, adapterClass), `${JSON.stringify(doc, null, 4)}\n`);
}

/**
 * Every committed adapter output for (kernel, fixture), in adapter-class order. Files are matched on their
 * `kernel` / `fixture` fields, not on their names, so "random1k" never picks up "random1k-u32". Only `*.json` names
 * are read (an in-flight `*.json.tmp` of writeAtomic is skipped).
 * @param kernel - the kernel id
 * @param fixture - the fixture name
 * @returns the committed outputs (empty when none)
 */
export function readNoiseFixtures(
    kernel: string,
    fixture: string,
): readonly { readonly adapterClass: string; readonly values: Float64Array; readonly dtype: "f32" | "u32" }[] {
    const dir = noiseDir();
    if (!existsSync(dir)) {
        return [];
    }
    const out: { readonly adapterClass: string; readonly values: Float64Array; readonly dtype: "f32" | "u32" }[] = [];
    for (const name of readdirSync(dir).sort()) {
        if (!name.endsWith(".json")) {
            continue;
        }
        const doc = JSON.parse(readFileSync(resolve(dir, name), "utf8")) as Partial<NoiseFixtureDocument>;
        if (doc.kernel !== kernel || doc.fixture !== fixture) {
            continue;
        }
        if (
            typeof doc.adapterClass !== "string" ||
            !Array.isArray(doc.values) ||
            (doc.dtype !== "f32" && doc.dtype !== "u32")
        ) {
            throw new Error(`noise fixture ${name} is malformed (kernel / fixture / adapterClass / dtype / values)`);
        }
        out.push({ adapterClass: doc.adapterClass, values: Float64Array.from(doc.values), dtype: doc.dtype });
    }
    return out.sort((a, b) => a.adapterClass.localeCompare(b.adapterClass));
}

/**
 * Reads the noise-floor document, or null when the file does not exist.
 * @returns the parsed document or null
 */
function readNoiseFloor(): NoiseFloorDocument | null {
    if (!existsSync(NOISE_FLOOR_FILE)) {
        return null;
    }
    return JSON.parse(readFileSync(NOISE_FLOOR_FILE, "utf8")) as NoiseFloorDocument;
}

/**
 * A noise-floor row (5.6 schema); appended to benchmarks/results/noise-floor.json when GRAPHTY_NOISE_FLOOR_WRITE=1 (a
 * row with the same id is replaced). Creates the skeleton document (no adapters, no tolerances) when the file is
 * absent; P1-T6 owns the committed document and its adapters / tolerances. The write is atomic but the
 * read-modify-write is not: runs that record rows use --no-file-parallelism (one test file at a time).
 * @param row - the row to record
 */
export function recordNoiseRow(row: NoiseRow): void {
    if (!writing()) {
        return;
    }
    mkdirSync(dirname(NOISE_FLOOR_FILE), { recursive: true });
    const doc: NoiseFloorDocument = readNoiseFloor() ?? {
        recordedAt: new Date().toISOString(),
        adapters: [],
        rows: [],
        tolerances: {},
    };
    doc.rows = [...doc.rows.filter((r) => r.id !== row.id), row];
    writeAtomic(NOISE_FLOOR_FILE, `${JSON.stringify(doc, null, 4)}\n`);
}

/**
 * The committed tolerance of a test id and its basis row; throws when the id is unknown (a tolerance without a floor
 * is a finding, spec 11.9 item 3) or when its basis row is not recorded.
 * @param testId - the tolerance id (e.g. "fa2-skeleton.force")
 * @returns the tolerance value and the id of its basis row
 */
export function noiseFloorFor(testId: string): { readonly value: number; readonly basis: string } {
    const doc = readNoiseFloor();
    if (doc === null) {
        throw new Error(
            `noise floor: ${NOISE_FLOOR_FILE} does not exist, so there is no committed tolerance for "${testId}"`,
        );
    }
    const entry = doc.tolerances[testId];
    if (entry === undefined) {
        throw new Error(
            `noise floor: no committed tolerance for test id "${testId}" -- a tolerance without a floor is a finding (spec 11.9 item 3)`,
        );
    }
    if (!doc.rows.some((r) => r.id === entry.basis)) {
        throw new Error(`noise floor: tolerance "${testId}" names basis row "${entry.basis}", which is not recorded`);
    }
    return { value: entry.value, basis: entry.basis };
}

/**
 * The adapter class string of the running adapter: <vendor>-<architecture>-<runtime> (e.g. nvidia-lovelace-node,
 * mesa-software-node, google-swiftshader-browser). Kept a one-liner so the browser twin cannot drift.
 * @param caps - the context's caps
 * @returns the class string
 */
export function adapterClass(caps: GpuCaps): string {
    return `${caps.vendor}-${caps.architecture}-${caps.runtime}`;
}
