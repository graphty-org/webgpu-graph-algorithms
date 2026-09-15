/**
 * Bundle TypeScript declarations
 *
 * Writes one declaration shim per bundle entry: dist/webgpu-graph-algorithms.d.ts for the root (what
 * package.json "types" and exports["."] point at), dist/browser.d.ts and dist/node.d.ts for the two subpaths.
 * Each shim is a single re-export of the tsc output of its source entry under dist/src/ (the only barrels),
 * so a shim can never drift from the source export list. Every source entry starts with
 * `/// <reference types="@webgpu/types" />`, so the d.ts under dist/src/ resolve the GPU* names for consumers
 * (contract 2.3). tsconfig.strict-consumer.json compiles test/types/*.test-d.ts against these shims.
 *
 * Copied from packages/graph-io/scripts/bundle-types.js; the code is unchanged, only this header differs.
 */

import { existsSync, writeFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

import { declarationSpecifier, ENTRIES } from "./entries.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const distDir = path.resolve(__dirname, "../dist");

function bundleTypes() {
    for (const [name, source] of Object.entries(ENTRIES)) {
        const declaration = path.resolve(distDir, source.replace(/\.ts$/, ".d.ts"));
        if (!existsSync(declaration)) {
            console.error(
                `Error: ${path.relative(distDir, declaration)} not found under dist/. Run "npm run build" first.`,
            );
            process.exit(1);
        }
        try {
            const content = `export * from "${declarationSpecifier(source)}";\n`;
            writeFileSync(path.resolve(distDir, `${name}.d.ts`), content, "utf8");
        } catch (error) {
            console.error(`Error writing dist/${name}.d.ts:`, error);
            process.exit(1);
        }
    }
    console.log(`Successfully created dist/{${Object.keys(ENTRIES).join(",")}}.d.ts`);
}

bundleTypes();
