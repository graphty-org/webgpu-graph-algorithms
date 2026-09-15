/**
 * Build Bundle Script
 *
 * Creates a single ES module bundle (dist/graph-format.js) that:
 * - Contains all the library code in a single file
 * - Is the entry point that package.json "exports" points at
 * - Can be distributed as a standalone file
 *
 * Run "npm run build" (tsc -p tsconfig.build.json) first so that dist/src/
 * exists; scripts/bundle-types.js then writes dist/graph-format.d.ts, which
 * re-exports the per-module declarations under dist/src/.
 */

import { spawnSync } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import { build } from "vite";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function buildBundle() {
    try {
        await build({
            configFile: false,
            logLevel: "warn",
            build: {
                lib: {
                    entry: path.resolve(__dirname, "../src/index.ts"),
                    name: "GraphFormat",
                    formats: ["es"],
                    fileName: () => "graph-format.js",
                },
                outDir: path.resolve(__dirname, "../dist"),
                emptyOutDir: false, // Keep the tsc output under dist/src/
                rollupOptions: {
                    external: [],
                    output: {
                        preserveModules: false,
                        inlineDynamicImports: true,
                    },
                },
                minify: false,
                sourcemap: true,
            },
        });

        console.log("Successfully built dist/graph-format.js");

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
