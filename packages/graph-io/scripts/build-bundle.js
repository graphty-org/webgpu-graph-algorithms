/**
 * Build Bundle Script
 *
 * One multi-entry vite library build (ES modules only) that emits, under dist/:
 * - graph-io.js, the root entry package.json "exports" ["."] points at
 * - <format>.js for every per-format subpath (gexf, graphml, gml, dot, pajek, csv, json, neo4j),
 *   so an application that imports only "@graphty/graph-io/csv" never loads the XML formats
 * - chunks/*.js, the code two or more entries share (the common helpers, the io types), emitted
 *   once so every entry sees one ImportError class and one LOSS table
 *
 * Runtime dependencies (@graphty/graph-format) stay external so
 * consumers install exactly one copy of each. Run "npm run build" (tsc -p tsconfig.build.json)
 * first so that dist/src/ exists; scripts/bundle-types.js then writes one d.ts shim per entry
 * that re-exports the per-module declarations under dist/src/.
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
