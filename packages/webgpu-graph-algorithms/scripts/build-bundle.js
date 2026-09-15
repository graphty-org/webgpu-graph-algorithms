/**
 * Build Bundle Script
 *
 * One multi-entry vite library build (ES modules only) that emits, under dist/:
 * - webgpu-graph-algorithms.js, the root entry package.json "exports" ["."] points at (the core: GpuContext,
 *   the kernels, the algorithms and the layouts)
 * - browser.js, the "./browser" subpath (navigator.gpu acquisition, spec 2.3)
 * - node.js, the "./node" subpath -- the ONLY bundle that names the "webgpu" (Dawn) module, and only inside a
 *   dynamic import(), so a browser bundle never resolves it (spec 2.5; test/build-output.test.ts)
 * - chunks/*.js, the code two or more entries share, emitted once
 *
 * Runtime dependencies stay external so consumers install exactly one copy of each: every name under
 * package.json "dependencies" and "peerDependencies" (@graphty/graph-format, @webgpu/types, webgpu,
 * @graphty/algorithms, @graphty/layout), subpaths included. Run "npm run build" (tsc -p tsconfig.build.json)
 * first so that dist/src/ exists; scripts/bundle-types.js then writes one d.ts shim per entry that re-exports
 * the per-module declarations under dist/src/. At P0 the browser entry is `export {};`, so vite reports
 * "Generated an empty chunk: browser" -- a warning, not an error (contract 2.6).
 *
 * Copied from packages/graph-io/scripts/build-bundle.js; the code is unchanged, only this header differs.
 */

import { spawnSync } from "child_process";
import { readFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { build } from "vite";

import { ENTRIES } from "./entries.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const packageRoot = path.resolve(__dirname, "..");

/**
 * Every package listed in dependencies or peerDependencies is external to the
 * bundle, including its subpaths.
 */
function externalDependencies() {
    const packageJson = JSON.parse(readFileSync(path.resolve(packageRoot, "package.json"), "utf8"));
    const names = new Set([
        ...Object.keys(packageJson.dependencies ?? {}),
        ...Object.keys(packageJson.peerDependencies ?? {}),
    ]);
    return (id) => {
        for (const name of names) {
            if (id === name || id.startsWith(`${name}/`)) {
                return true;
            }
        }
        return false;
    };
}

async function buildBundle() {
    try {
        const entry = Object.fromEntries(
            Object.entries(ENTRIES).map(([name, source]) => [name, path.resolve(packageRoot, source)]),
        );
        await build({
            configFile: false,
            logLevel: "warn",
            build: {
                lib: {
                    entry,
                    formats: ["es"],
                    fileName: (_format, entryName) => `${entryName}.js`,
                },
                outDir: path.resolve(packageRoot, "dist"),
                emptyOutDir: false, // Keep the tsc output under dist/src/
                rollupOptions: {
                    external: externalDependencies(),
                    output: {
                        preserveModules: false,
                        chunkFileNames: "chunks/[name]-[hash].js",
                    },
                },
                minify: false,
                sourcemap: true,
            },
        });

        console.log(`Successfully built dist/{${Object.keys(ENTRIES).join(",")}}.js`);

        const result = spawnSync(process.execPath, [path.resolve(__dirname, "bundle-types.js")], {
            stdio: "inherit",
        });

        if (result.error) {
            throw result.error;
        }

        if (result.status !== 0) {
            throw new Error(`bundle-types.js exited with status ${String(result.status)}`);
        }
    } catch (error) {
        console.error("Error building bundle:", error);
        process.exit(1);
    }
}

buildBundle();
