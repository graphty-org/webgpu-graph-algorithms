import { existsSync, readFileSync } from "fs";
import { resolve } from "path";
import { describe, expect, it } from "vitest";

import { WIRE_PRODUCER } from "../src/wire/to-wire.js";

interface PackageJsonShape {
    name: string;
    version: string;
    type: string;
    main: string;
    types: string;
    sideEffects: boolean;
    exports: Record<string, Record<string, string>>;
    files: string[];
    dependencies?: Record<string, string>;
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
        expect(packageJson.name).toBe("@graphty/graph-format");
        expect(packageJson.type).toBe("module");
        expect(packageJson.main).toBe("dist/graph-format.js");
        expect(packageJson.types).toBe("dist/graph-format.d.ts");
        expect(packageJson.sideEffects).toBe(false);

        // exports: ESM only, "types" condition first so resolvers see it before "import"
        expect(packageJson.exports).toBeDefined();
        expect(packageJson.exports["."]).toBeDefined();
        expect(Object.keys(packageJson.exports["."])[0]).toBe("types");
        expect(packageJson.exports["."].types).toBe("./dist/graph-format.d.ts");
        expect(packageJson.exports["."].import).toBe("./dist/graph-format.js");
        expect(packageJson.exports["."].default).toBe("./dist/graph-format.js");
        expect(packageJson.exports["."].require).toBeUndefined();

        expect(packageJson.files).toContain("dist/");
        expect(packageJson.files).toContain("src/");
        expect(packageJson.files).toContain("README.md");
        expect(packageJson.files).toContain("LICENSE");

        expect(packageJson.publishConfig).toEqual({ access: "public", provenance: true });
        expect(packageJson.engines.node).toBe(">=18.19.0");
    });

    it("should have zero runtime dependencies", () => {
        expect(packageJson.dependencies).toBeUndefined();
    });

    it("should declare the modules its own tests import as devDependencies (monorepo move, design 13.3)", () => {
        // test/audit/gpu-upload.test.ts does `import("webgpu")` (Google Dawn for Node) and
        // test/types/typed-arrays.test-d.ts references @webgpu/types. In the monorepo nothing hoists
        // either from a parent package.json, so `tsc --noEmit` (the lint script) fails with TS2307
        // unless this package declares them itself.
        expect(packageJson.devDependencies.webgpu).toBeTypeOf("string");
        expect(packageJson.devDependencies["@webgpu/types"]).toBeTypeOf("string");
    });

    it("should stamp the wire producer with the package name and version", () => {
        // WIRE_PRODUCER is a constant (importing package.json into src would emit it under dist/);
        // this pins it to package.json so the two cannot drift.
        expect(WIRE_PRODUCER).toBe(`${packageJson.name}@${packageJson.version}`);
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
        expect(packageJson.scripts["coverage:preview"]).toContain("9056");
    });

    it("should have TypeScript configuration for ES modules", () => {
        const tsconfig = readJson("./tsconfig.json") as TsconfigShape;
        expect(tsconfig.extends).toContain("tsconfig.base.json");
        expect(tsconfig.compilerOptions.composite).toBe(true);
        expect(tsconfig.compilerOptions.noEmit).toBe(true);
        expect(tsconfig.compilerOptions.noUncheckedIndexedAccess).toBe(false);
        expect(tsconfig.compilerOptions.exactOptionalPropertyTypes).toBe(false);
        expect(tsconfig.include).toContain("src/**/*.ts");
        expect(tsconfig.include).toContain("test/**/*.ts");

        const buildConfig = readJson("./tsconfig.build.json") as TsconfigShape;
        expect(buildConfig.extends).toBe("./tsconfig.json");
        expect(buildConfig.compilerOptions.noEmit).toBe(false);
        expect(buildConfig.compilerOptions.rootDir).toBe(".");
        expect(buildConfig.compilerOptions.outDir).toBe("./dist");
        // The root entry is compiled by tsc alone (as algorithms/algorithms.ts is), so a tsc-only build
        // (tools/prepush.sh runs `pnpm -r run build`) already yields the dist/graph-format.js and
        // dist/graph-format.d.ts that package.json "exports" point at; build:bundle then overwrites both.
        expect(buildConfig.include).toEqual(["graph-format.ts", "src/**/*.ts"]);
        expect(tsconfig.include).toContain("graph-format.ts");
        expect(readFileSync(resolve("./graph-format.ts"), "utf-8").trim()).toBe('export * from "./src/index.js";');

        const strictConfig = readJson("./tsconfig.strict-consumer.json") as TsconfigShape;
        expect(strictConfig.compilerOptions.noUncheckedIndexedAccess).toBe(true);
        expect(strictConfig.compilerOptions.exactOptionalPropertyTypes).toBe(true);
    });

    it("should ship the local resizable ArrayBuffer declaration", () => {
        expect(existsSync(resolve("./src/lib-resizable-array-buffer.d.ts"))).toBe(true);
    });

    const distExists = existsSync(resolve("./dist/src/index.js"));

    it.skipIf(!distExists)("should have built the main entry file", () => {
        expect(existsSync(resolve("./dist/src/index.js"))).toBe(true);
        expect(existsSync(resolve("./dist/src/index.d.ts"))).toBe(true);
    });

    const bundleExists = existsSync(resolve("./dist/graph-format.js"));

    it.skipIf(!bundleExists)("should have built the bundle and its one-line declaration", () => {
        expect(existsSync(resolve("./dist/graph-format.js"))).toBe(true);
        expect(existsSync(resolve("./dist/graph-format.js.map"))).toBe(true);
        const dts = readFileSync(resolve("./dist/graph-format.d.ts"), "utf-8");
        expect(dts.trim()).toBe('export * from "./src/index.js";');
    });
});
