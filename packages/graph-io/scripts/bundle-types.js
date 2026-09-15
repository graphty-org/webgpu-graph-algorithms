/**
 * Bundle TypeScript declarations
 *
 * Writes one declaration shim per bundle entry: dist/graph-io.d.ts for the root (what package.json
 * "types" and exports["."] point at) and dist/<format>.d.ts for every per-format subpath. Each
 * shim is a single re-export of the tsc output of its source entry under dist/src/ (the only
 * barrels), so a shim can never drift from the source export list (design section 13.1, Q18).
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
