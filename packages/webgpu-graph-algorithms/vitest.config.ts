/// <reference types="@vitest/browser/providers/playwright" />
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

/**
 * The two Chromium flag sets of spec 12.2, selected by GRAPHTY_BROWSER_GPU ("nvidia" | "swiftshader", default
 * swiftshader). Kept in one exported constant so the Vitest 4 provider change is a one-line move (spec 11.1).
 * @public exported for scripts/run-browser-project.js and the CLAUDE.md "Verified Platform Facts" table
 */
export const BROWSER_FLAGS = Object.freeze({
    nvidia: Object.freeze([
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan",
        "--use-angle=vulkan",
        "--disable-vulkan-surface",
    ]),
    swiftshader: Object.freeze(["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]),
    // The host lane on macOS (hosts.yml): Chromium picks Dawn's Metal backend by itself; WebKit takes no flags.
    metal: Object.freeze(["--enable-unsafe-webgpu"]),
});

const here = dirname(fileURLToPath(import.meta.url));
const browserGpuEnv = process.env.GRAPHTY_BROWSER_GPU;
const browserGpu: keyof typeof BROWSER_FLAGS =
    browserGpuEnv === "nvidia" || browserGpuEnv === "metal" ? browserGpuEnv : "swiftshader";
/** GRAPHTY_BROWSER=webkit runs the browser project in Playwright's WebKit (the host lane's Safari proxy); default chromium. */
const browserName: "chromium" | "webkit" = process.env.GRAPHTY_BROWSER === "webkit" ? "webkit" : "chromium";
const gpuRequire = process.env.GRAPHTY_GPU_REQUIRE ?? "";
const noiseFloorWrite = process.env.GRAPHTY_NOISE_FLOOR_WRITE ?? "";

/**
 * The environment of the Chromium child (spec 12.2 GRAPHTY_EGL_LIB_DIR): on the dev box headless Chromium finds
 * the NVIDIA GPU only with the extracted libEGL tree on LD_LIBRARY_PATH; the variable is prepended when set,
 * otherwise Playwright inherits process.env unchanged (undefined). Only defined values are copied (LaunchOptions.env
 * is a string map).
 * @returns the env map for `launch.env`, or undefined
 */
function browserLaunchEnv(): Record<string, string> | undefined {
    const eglDir = process.env.GRAPHTY_EGL_LIB_DIR;
    if (eglDir === undefined || eglDir === "") {
        return undefined;
    }
    const env: Record<string, string> = {};
    for (const [key, value] of Object.entries(process.env)) {
        if (value !== undefined) {
            env[key] = value;
        }
    }
    env.LD_LIBRARY_PATH = [eglDir, process.env.LD_LIBRARY_PATH].filter((v) => v !== undefined && v !== "").join(":");
    return env;
}

/**
 * The project names selected on the command line, from both `--project=name` and `--project name`.
 * @returns the names in command-line order
 */
function selectedProjects(): string[] {
    const argv = process.argv;
    const names: string[] = [];
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a.startsWith("--project=")) {
            names.push(a.slice("--project=".length));
        } else if (a === "--project" && i + 1 < argv.length) {
            names.push(argv[i + 1]);
        }
    }
    return names;
}

// Thresholds apply when the selected project set is EXACTLY `node` and no merge run is in progress (spec 11.8).
const projects = selectedProjects();
const thresholdsActive = projects.length === 1 && projects[0] === "node" && process.env.COVERAGE_DIR === undefined;

/**
 * The browser-side benchmark bridge (spec 11.6 item 8, 11.7): a browser test cannot write files, so it calls
 * `commands.appendBenchRecord(payload)` and this server-side command appends a session to
 * benchmarks/out/<runner-class>.json in the harness's file shape (section 6.4). Returns the path written.
 * @param _context - the Vitest command context (unused)
 * @param payload - the browser session: gpu summary, runner class, results
 * @returns the path of the file written
 */
async function appendBenchRecord(
    _context: unknown,
    payload: { runnerClass: string; session: unknown },
): Promise<string> {
    const dir = resolve(here, "benchmarks/out");
    mkdirSync(dir, { recursive: true });
    const file = resolve(dir, `${payload.runnerClass.replace(/[^A-Za-z0-9_.-]/g, "_")}.json`);
    const sessions: unknown[] = existsSync(file) ? (JSON.parse(readFileSync(file, "utf8")) as unknown[]) : [];
    sessions.push(payload.session);
    writeFileSync(file, `${JSON.stringify(sessions, null, 4)}\n`);
    return file;
}

/**
 * The browser side of test/helpers/noise-floor.ts writeNoiseFixture (spec 11.9 item 3, 11.5: the SwiftShader
 * leg of the cross-adapter fixtures). Writes test/fixtures/noise/<kernel>-<fixture>-<adapterClass>.json in the
 * 5.6 fixture shape ONLY when GRAPHTY_NOISE_FLOOR_WRITE=1 in the server's environment; returns the path, or ""
 * when writing is off.
 * @param _context - the Vitest command context (unused)
 * @param payload - the raw output of one kernel on one adapter
 * @returns the path written, or ""
 */
async function writeNoiseFixture(
    _context: unknown,
    payload: { kernel: string; fixture: string; adapterClass: string; values: readonly number[]; dtype: "f32" | "u32" },
): Promise<string> {
    if (noiseFloorWrite !== "1") {
        return "";
    }
    const dir = resolve(here, "test/fixtures/noise");
    mkdirSync(dir, { recursive: true });
    const safe = (s: string): string => s.replace(/[^A-Za-z0-9_.-]/g, "_");
    const file = resolve(dir, `${safe(payload.kernel)}-${safe(payload.fixture)}-${safe(payload.adapterClass)}.json`);
    const fixture = {
        kernel: payload.kernel,
        fixture: payload.fixture,
        adapterClass: payload.adapterClass,
        dtype: payload.dtype,
        values: payload.values,
    };
    writeFileSync(file, `${JSON.stringify(fixture, null, 4)}\n`);
    return file;
}

/**
 * The browser side of test/helpers/noise-floor.ts recordNoiseRow: appends one 5.6 row to
 * benchmarks/results/noise-floor.json (replacing a row with the same id) ONLY when GRAPHTY_NOISE_FLOOR_WRITE=1.
 * @param _context - the Vitest command context (unused)
 * @param row - the noise row (5.6 schema)
 * @returns the path written, or ""
 */
async function recordNoiseRow(_context: unknown, row: Record<string, unknown>): Promise<string> {
    if (noiseFloorWrite !== "1") {
        return "";
    }
    const file = resolve(here, "benchmarks/results/noise-floor.json");
    const doc = existsSync(file)
        ? (JSON.parse(readFileSync(file, "utf8")) as { rows: Record<string, unknown>[]; [k: string]: unknown })
        : { recordedAt: new Date().toISOString(), adapters: [], rows: [], tolerances: {} };
    doc.rows = [...doc.rows.filter((r) => r.id !== row.id), row];
    writeFileSync(file, `${JSON.stringify(doc, null, 4)}\n`);
    return file;
}

export default defineConfig({
    test: {
        reporters: ["verbose"],
        coverage: {
            all: true,
            provider: "v8",
            reporter: ["text", "json-summary", "json", "lcov", "html"],
            reportsDirectory: process.env.COVERAGE_DIR ?? "coverage",
            include: ["src/**/*.ts"],
            // Only the root barrel and the template strings are excluded (spec 11.8): src/node/index.ts and
            // src/browser/index.ts carry logic and count.
            exclude: ["**/*.d.ts", "**/*.test.ts", "src/index.ts", "src/wgsl/**"],
            thresholds: thresholdsActive ? { lines: 80, functions: 80, branches: 75, statements: 80 } : undefined,
        },
        projects: [
            {
                test: {
                    name: "node",
                    globals: true,
                    environment: "node",
                    pool: "forks",
                    testTimeout: 30_000,
                    hookTimeout: 60_000,
                    include: [
                        "test/*.test.ts",
                        "test/{device,node,memory,kernel,primitives,algorithms,layouts,oracle,sabotage,types}/**/*.test.ts",
                    ],
                    exclude: ["test/limits/**", "test/browser/**"],
                    setupFiles: ["test/setup/gpu.ts"],
                    globalSetup: ["test/setup/global.ts"],
                },
            },
            {
                test: {
                    name: "node-limits",
                    globals: true,
                    environment: "node",
                    pool: "forks",
                    testTimeout: 600_000,
                    hookTimeout: 120_000,
                    include: ["test/limits/**/*.test.ts"],
                    setupFiles: ["test/setup/gpu.ts"],
                },
            },
            {
                define: {
                    "import.meta.env.GRAPHTY_GPU_REQUIRE": JSON.stringify(gpuRequire),
                    "import.meta.env.GRAPHTY_BROWSER_GPU": JSON.stringify(browserGpu),
                    "import.meta.env.GRAPHTY_NOISE_FLOOR_WRITE": JSON.stringify(noiseFloorWrite),
                },
                test: {
                    name: "browser",
                    globals: true,
                    include: ["test/browser/**/*.test.ts"],
                    testTimeout: 120_000,
                    hookTimeout: 120_000,
                    env: {
                        GRAPHTY_GPU_REQUIRE: gpuRequire,
                        GRAPHTY_BROWSER_GPU: browserGpu,
                        GRAPHTY_NOISE_FLOOR_WRITE: noiseFloorWrite,
                    },
                    setupFiles: ["test/setup/browser.ts"],
                    browser: {
                        enabled: true,
                        headless: true,
                        provider: "playwright",
                        fileParallelism: false,
                        commands: { appendBenchRecord, writeNoiseFixture, recordNoiseRow },
                        instances: [
                            browserName === "webkit"
                                ? { browser: "webkit", launch: { env: browserLaunchEnv() } }
                                : {
                                      browser: "chromium",
                                      launch: { args: [...BROWSER_FLAGS[browserGpu]], env: browserLaunchEnv() },
                                  },
                        ],
                    },
                },
            },
        ],
    },
});
