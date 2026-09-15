/**
 * Bundle TypeScript declarations
 *
 * Writes dist/graph-format.d.ts, the declaration file that package.json
 * "types" and "exports" point at. It is a single re-export of
 * dist/src/index.d.ts (the tsc output of src/index.ts, the only public
 * barrel), so it can never drift from the source export list.
 */

import { existsSync, writeFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function bundleTypes() {
    const indexPath = path.resolve(__dirname, "../dist/src/index.d.ts");

    if (!existsSync(indexPath)) {
        console.error('Error: dist/src/index.d.ts not found. Run "npm run build" first.');
        process.exit(1);
    }

    try {
        const content = 'export * from "./src/index.js";\n';
        const outputPath = path.resolve(__dirname, "../dist/graph-format.d.ts");
        writeFileSync(outputPath, content, "utf8");
        console.log("Successfully created dist/graph-format.d.ts");
    } catch (error) {
        console.error("Error bundling types:", error);
        process.exit(1);
    }
}

bundleTypes();
