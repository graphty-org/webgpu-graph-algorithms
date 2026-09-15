import { existsSync, readdirSync, readFileSync } from "fs";
import { resolve } from "path";
import { describe, expect, it } from "vitest";

/** The per-format subpaths of design section 8.2, one per directory under src/formats. */
const FORMAT_DIRS = readdirSync(resolve("./src/formats")).sort();

interface PackageJsonShape {
    name: string;
    version: string;
    type: string;
    main: string;
    types: string;
    sideEffects: boolean;
    exports: Record<string, Record<string, string>>;
    files: string[];
    dependencies: Record<string, string>;
    peerDependencies: Record<string, string>;
    devDependencies: Record<string, string>;
    publishConfig: { access: string; provenance: boolean };
    engines: { node: string };
    scripts: Record<string, string>;
}

interface TsconfigShape {
    extends?: string;
    compilerOptions: Record<string, unknown>;
    include: string[];
}

function readJson(relativePath: string): unknown {
    return JSON.parse(readFileSync(resolve(relativePath), "utf-8"));
}

describe("Build Output Tests", () => {
    const packageJson = readJson("./package.json") as PackageJsonShape;

    it("should have correct package.json configuration", () => {
        expect(packageJson.name).toBe("@graphty/graph-io");
        expect(packageJson.type).toBe("module");
        expect(packageJson.main).toBe("dist/graph-io.js");
        expect(packageJson.types).toBe("dist/graph-io.d.ts");
        expect(packageJson.sideEffects).toBe(false);

        // exports: ESM only, "types" condition first so resolvers see it before "import"
        expect(packageJson.exports).toBeDefined();
        expect(packageJson.exports["."]).toBeDefined();
        expect(Object.keys(packageJson.exports["."])[0]).toBe("types");
        expect(packageJson.exports["."].types).toBe("./dist/graph-io.d.ts");
        expect(packageJson.exports["."].import).toBe("./dist/graph-io.js");
        expect(packageJson.exports["."].default).toBe("./dist/graph-io.js");
        expect(packageJson.exports["."].require).toBeUndefined();

        expect(packageJson.files).toContain("dist/");
        expect(packageJson.files).toContain("src/");
        expect(packageJson.files).toContain("README.md");
        expect(packageJson.files).toContain("LICENSE");

        expect(packageJson.publishConfig).toEqual({ access: "public", provenance: true });
        expect(packageJson.engines.node).toBe(">=18.19.0");
    });

    it("should depend on the format package as both dependency and peer", () => {
        expect(packageJson.dependencies["@graphty/graph-format"]).toBe("workspace:*");
        expect(packageJson.peerDependencies["@graphty/graph-format"]).toMatch(/^\^\d+\.\d+\.\d+$/);
        // the design section 8.2 end state: every parser is hand-written, so neither papaparse
        // (one shared CSV record reader) nor fast-xml-parser (one shared streaming XML tokenizer)
        // is a dependency any more
        expect(packageJson.dependencies.papaparse).toBeUndefined();
        expect(packageJson.dependencies["fast-xml-parser"]).toBeUndefined();
        expect(packageJson.devDependencies["@types/papaparse"]).toBeUndefined();
        // the fuzz audit suite uses fast-check
        expect(packageJson.devDependencies["fast-check"]).toBeTypeOf("string");
    });

    it("should have the standard script set", () => {
        for (const name of [
            "build",
            "build:bundle",
            "build:all",
            "lint",
            "typecheck",
            "test",
            "test:run",
            "coverage",
            "coverage:preview",
            "benchmark",
            "ready:commit",
        ]) {
            expect(packageJson.scripts[name], `script ${name}`).toBeTypeOf("string");
        }
        expect(packageJson.scripts["coverage:preview"]).toContain("9057");
    });

    it("should have TypeScript configuration for ES modules", () => {
        const tsconfig = readJson("./tsconfig.json") as TsconfigShape;
        expect(tsconfig.extends).toContain("tsconfig.base.json");
        expect(tsconfig.compilerOptions.composite).toBe(true);
        expect(tsconfig.compilerOptions.noEmit).toBe(true);
        expect(tsconfig.compilerOptions.lib).toContain("DOM");
        expect(tsconfig.include).toContain("src/**/*.ts");
        expect(tsconfig.include).toContain("test/**/*.ts");

        const buildConfig = readJson("./tsconfig.build.json") as TsconfigShape;
        expect(buildConfig.extends).toBe("./tsconfig.json");
        expect(buildConfig.compilerOptions.noEmit).toBe(false);
        expect(buildConfig.compilerOptions.rootDir).toBe(".");
        expect(buildConfig.compilerOptions.outDir).toBe("./dist");
        expect(buildConfig.include).toEqual(["src/**/*.ts"]);
    });

    it("exports one subpath per format directory, types first, consistent with the bundle entries", () => {
        expect(FORMAT_DIRS).toEqual(["csv", "dot", "gexf", "gml", "graphml", "json", "neo4j", "pajek"]);
        const subpaths = Object.keys(packageJson.exports).filter((key) => key !== ".");
        expect(subpaths.sort()).toEqual(FORMAT_DIRS.map((dir) => `./${dir}`));
        for (const dir of FORMAT_DIRS) {
            const entry = packageJson.exports[`./${dir}`];
            expect(Object.keys(entry), dir).toEqual(["types", "import", "default"]);
            expect(entry.types, dir).toBe(`./dist/${dir}.d.ts`);
            expect(entry.import, dir).toBe(`./dist/${dir}.js`);
            expect(entry.default, dir).toBe(`./dist/${dir}.js`);
            expect(existsSync(resolve(`./src/formats/${dir}/index.ts`)), dir).toBe(true);
        }
        // scripts/entries.js drives both the vite build and the d.ts shims; it must name every subpath
        const entries = readFileSync(resolve("./scripts/entries.js"), "utf-8");
        expect(entries).toContain('"graph-io": "src/index.ts"');
        for (const dir of FORMAT_DIRS) {
            expect(entries, dir).toContain(`${dir}: "src/formats/${dir}/index.ts"`);
        }
        expect(packageJson.sideEffects).toBe(false);
    });

    const distExists = existsSync(resolve("./dist/src/index.js"));

    it.skipIf(!distExists)("should have built the main entry file", () => {
        expect(existsSync(resolve("./dist/src/index.js"))).toBe(true);
        expect(existsSync(resolve("./dist/src/index.d.ts"))).toBe(true);
    });

    const bundleExists = existsSync(resolve("./dist/graph-io.js"));

    it.skipIf(!bundleExists)("should have built the bundle and its one-line declaration", () => {
        expect(existsSync(resolve("./dist/graph-io.js"))).toBe(true);
        expect(existsSync(resolve("./dist/graph-io.js.map"))).toBe(true);
        const dts = readFileSync(resolve("./dist/graph-io.d.ts"), "utf-8");
        expect(dts.trim()).toBe('export * from "./src/index.js";');
    });

    it.skipIf(!bundleExists)("should have built every subpath entry with its one-line declaration", () => {
        for (const dir of FORMAT_DIRS) {
            expect(existsSync(resolve(`./dist/${dir}.js`)), dir).toBe(true);
            expect(existsSync(resolve(`./dist/${dir}.js.map`)), dir).toBe(true);
            expect(existsSync(resolve(`./dist/src/formats/${dir}/index.d.ts`)), dir).toBe(true);
            const dts = readFileSync(resolve(`./dist/${dir}.d.ts`), "utf-8");
            expect(dts.trim(), dir).toBe(`export * from "./src/formats/${dir}/index.js";`);
        }
    });

    it.skipIf(!bundleExists)("subpath bundles share one module instance with the root bundle", async () => {
        const root = (await import(resolve("./dist/graph-io.js"))) as Record<string, unknown>;
        expect(root.default).toBeUndefined();
        for (const dir of FORMAT_DIRS) {
            const sub = (await import(resolve(`./dist/${dir}.js`))) as Record<string, unknown>;
            expect(sub.default, dir).toBeUndefined();
            const names = Object.keys(sub).sort();
            expect(names.length, dir).toBeGreaterThan(0);
            for (const name of names) {
                // the same object, not a copy bundled twice (one ImportError, one LOSS table, one importer)
                expect(root[name], `${dir}.${name}`).toBe(sub[name]);
            }
        }
    });
});
