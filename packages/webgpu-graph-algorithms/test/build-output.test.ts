/**
 * Package shape and build output (contract 5.5 P0 form; spec 2.5, 3.1; graph-io's test/build-output.test.ts
 * pattern): the manifest, the exports versus scripts/entries.js, the tsconfig trio, and the bundle files with
 * their one-line d.ts shims. The bundle assertions HARD-FAIL when the bundle is absent under CI (every lane
 * builds with `pnpm run build` = `pnpm -r run build:all`) and skip locally for tsc-only runs. P1-T7 adds the
 * import-specifier assertions (no "webgpu" specifier in the root and browser bundles; dynamic only in node).
 */

import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

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
    peerDependenciesMeta: Record<string, { optional: boolean }>;
    devDependencies: Record<string, string>;
    publishConfig: { access: string; provenance: boolean };
    engines: { node: string };
    scripts: Record<string, string>;
    repository: { type: string; url: string; directory: string };
    keywords: string[];
    browser?: unknown;
}

interface TsconfigShape {
    extends?: string;
    compilerOptions: Record<string, unknown>;
    include: string[];
}

/**
 * The reference directive every source entry starts with (contract 2.3, 3.15). `preserve="true"` is what makes
 * tsc 5.5+ copy a directive into the declaration output at all (a bare directive is dropped on emit), and the
 * emitter places it after the file's detached header comment -- still a valid pragma, because only comments
 * precede it; a consumer without @webgpu/types in its own `types` resolves the GPU* names through it.
 */
const WEBGPU_TYPES_DIRECTIVE = '/// <reference types="@webgpu/types" preserve="true" />';

function readJson(relativePath: string): unknown {
    return JSON.parse(readFileSync(resolve(relativePath), "utf-8"));
}

/** The bundle entry names of scripts/entries.js, read as text (the script has no declaration file). */
function entryNames(): string[] {
    const text = readFileSync(resolve("./scripts/entries.js"), "utf-8");
    const names: string[] = [];
    for (const match of text.matchAll(/^\s*"?([A-Za-z0-9-]+)"?:\s*"src\/[^"]+"/gm)) {
        names.push(match[1]);
    }
    return names;
}

/**
 * True when `text` carries the directive in its leading-comment region: everything before the directive line
 * is block comments, line comments or whitespace, so TypeScript treats it as a file pragma.
 */
function hasLeadingDirective(text: string): boolean {
    const at = text.indexOf(WEBGPU_TYPES_DIRECTIVE);
    if (at < 0) {
        return false;
    }
    const before = text
        .slice(0, at)
        .replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/^[ \t]*\/\/.*$/gm, "");
    return before.trim() === "";
}

const underCi = process.env.CI !== undefined && process.env.CI !== "";
const distExists = existsSync(resolve("./dist/src/index.js"));
const bundleExists = existsSync(resolve("./dist/webgpu-graph-algorithms.js"));

describe("package.json (contract 2.1)", () => {
    const packageJson = readJson("./package.json") as PackageJsonShape;

    it("is the ESM manifest with types-first exports and no require condition", () => {
        expect(packageJson.name).toBe("@graphty/webgpu-graph-algorithms");
        expect(packageJson.type).toBe("module");
        expect(packageJson.main).toBe("dist/webgpu-graph-algorithms.js");
        expect(packageJson.types).toBe("dist/webgpu-graph-algorithms.d.ts");
        expect(packageJson.sideEffects).toBe(false);
        expect(packageJson.browser).toBeUndefined();
        for (const [subpath, entry] of Object.entries(packageJson.exports)) {
            expect(Object.keys(entry), subpath).toEqual(["types", "import", "default"]);
            expect(entry.require, subpath).toBeUndefined();
            expect(entry.node, subpath).toBeUndefined();
        }
        expect(packageJson.exports["."]).toEqual({
            types: "./dist/webgpu-graph-algorithms.d.ts",
            import: "./dist/webgpu-graph-algorithms.js",
            default: "./dist/webgpu-graph-algorithms.js",
        });
        expect(packageJson.exports["./browser"]).toEqual({
            types: "./dist/browser.d.ts",
            import: "./dist/browser.js",
            default: "./dist/browser.js",
        });
        expect(packageJson.exports["./node"]).toEqual({
            types: "./dist/node.d.ts",
            import: "./dist/node.js",
            default: "./dist/node.js",
        });
        expect(packageJson.files).toEqual(["dist/", "src/", "README.md", "LICENSE"]);
        expect(packageJson.publishConfig).toEqual({ access: "public", provenance: true });
        expect(packageJson.engines.node).toBe(">=18.19.0");
        expect(packageJson.repository.directory).toBe("packages/webgpu-graph-algorithms");
        expect(packageJson.keywords).toContain("webgpu");
    });

    it("exports keys equal the bundle entries of scripts/entries.js, with '.' for the root", () => {
        const names = entryNames();
        expect(names).toEqual(["webgpu-graph-algorithms", "browser", "node"]);
        const expected = names.map((name) => (name === "webgpu-graph-algorithms" ? "." : `./${name}`));
        expect(Object.keys(packageJson.exports)).toEqual(expected);
        for (const name of ["browser", "node"]) {
            expect(existsSync(resolve(`./src/${name}/index.ts`)), name).toBe(true);
        }
    });

    it("declares the dependencies of spec 2.5: graph-format twice, @webgpu/types, webgpu as an optional peer and an exact devDependency", () => {
        expect(packageJson.dependencies["@graphty/graph-format"]).toBe("workspace:^");
        expect(packageJson.dependencies["@webgpu/types"]).toMatch(/^\^0\.1\.\d+$/);
        expect(Object.keys(packageJson.dependencies).sort()).toEqual(["@graphty/graph-format", "@webgpu/types"]);
        expect(packageJson.peerDependencies["@graphty/graph-format"]).toMatch(/^\^\d+\.\d+\.\d+$/);
        expect(packageJson.peerDependencies.webgpu).toBe(">=0.4.0 <1.0.0");
        expect(packageJson.peerDependenciesMeta.webgpu).toEqual({ optional: true });
        expect(packageJson.peerDependenciesMeta["@graphty/algorithms"]).toEqual({ optional: true });
        expect(packageJson.peerDependenciesMeta["@graphty/layout"]).toEqual({ optional: true });
        expect(packageJson.devDependencies.webgpu).toBe("0.4.0");
        expect(packageJson.devDependencies["@vitest/browser"]).toBeTypeOf("string");
        expect(packageJson.devDependencies.playwright).toBeTypeOf("string");
        expect(packageJson.devDependencies["fast-check"]).toBeTypeOf("string");
    });

    it("has the standard script set, the strict-consumer compile inside lint, and a node-only test:run", () => {
        for (const name of [
            "build",
            "build:bundle",
            "build:all",
            "lint",
            "typecheck",
            "typecheck:strict-consumer",
            "test",
            "test:run",
            "test:node",
            "test:browser",
            "test:browser:ci",
            "test:limits",
            "coverage",
            "coverage:preview",
            "bench",
            "bench:compare",
            "benchmark",
            "gpu:report",
            "ready:commit",
        ]) {
            expect(packageJson.scripts[name], `script ${name}`).toBeTypeOf("string");
        }
        expect(packageJson.scripts.lint).toContain("tsconfig.strict-consumer.json");
        expect(packageJson.scripts["test:run"]).toBe("vitest run --project=node");
        expect(packageJson.scripts["test:browser:ci"]).toBe("node scripts/run-browser-project.js");
        expect(packageJson.scripts["coverage:preview"]).toContain("9058");
    });
});

describe("the tsconfig trio (contract 2.3)", () => {
    it("tsconfig.json type-checks src, test, benchmarks and the script declarations with the four type packages", () => {
        const tsconfig = readJson("./tsconfig.json") as TsconfigShape;
        expect(tsconfig.extends).toContain("tsconfig.base.json");
        expect(tsconfig.compilerOptions.composite).toBe(true);
        expect(tsconfig.compilerOptions.noEmit).toBe(true);
        expect(tsconfig.compilerOptions.noUncheckedIndexedAccess).toBe(false);
        expect(tsconfig.compilerOptions.exactOptionalPropertyTypes).toBe(false);
        expect(tsconfig.compilerOptions.lib).toContain("DOM");
        expect(tsconfig.compilerOptions.types).toEqual(["node", "vitest/globals", "vite/client", "@webgpu/types"]);
        for (const pattern of ["src/**/*.ts", "test/**/*.ts", "benchmarks/**/*.ts", "scripts/**/*.d.ts"]) {
            expect(tsconfig.include, pattern).toContain(pattern);
        }
    });

    it("tsconfig.build.json emits src only, with declarations and stripInternal", () => {
        const build = readJson("./tsconfig.build.json") as TsconfigShape;
        expect(build.extends).toBe("./tsconfig.json");
        expect(build.compilerOptions.noEmit).toBe(false);
        expect(build.compilerOptions.rootDir).toBe(".");
        expect(build.compilerOptions.outDir).toBe("./dist");
        expect(build.compilerOptions.declaration).toBe(true);
        expect(build.compilerOptions.stripInternal).toBe(true);
        expect(build.compilerOptions.types).toEqual(["node", "@webgpu/types"]);
        expect(build.include).toEqual(["src/**/*.ts"]);
    });

    it("tsconfig.strict-consumer.json compiles the test-d samples against dist under the strict flags", () => {
        const strict = readJson("./tsconfig.strict-consumer.json") as TsconfigShape;
        expect(strict.extends).toBe("./tsconfig.json");
        expect(strict.compilerOptions.noUncheckedIndexedAccess).toBe(true);
        expect(strict.compilerOptions.exactOptionalPropertyTypes).toBe(true);
        const paths = strict.compilerOptions.paths as Record<string, string[]>;
        expect(paths["@graphty/webgpu-graph-algorithms"]).toEqual(["./dist/webgpu-graph-algorithms.d.ts"]);
        expect(paths["@graphty/webgpu-graph-algorithms/*"]).toEqual(["./dist/*.d.ts"]);
        expect(strict.include).toEqual(["test/types/**/*.test-d.ts"]);
    });

    it("vitest.config.ts declares the three projects and exports the flag sets", () => {
        const config = readFileSync(resolve("./vitest.config.ts"), "utf-8");
        expect(config).toContain("export const BROWSER_FLAGS");
        for (const name of ['name: "node"', 'name: "node-limits"', 'name: "browser"']) {
            expect(config).toContain(name);
        }
        expect(config).toContain('setupFiles: ["test/setup/gpu.ts"]');
        expect(config).toContain('setupFiles: ["test/setup/browser.ts"]');
        expect(config).toContain('globalSetup: ["test/setup/global.ts"]');
        expect(config).toContain("appendBenchRecord");
        expect(config).toContain("writeNoiseFixture");
        expect(config).toContain("recordNoiseRow");
    });
});

describe("the source entries (contract 2.3, 3.15)", () => {
    it("every entry starts with the @webgpu/types reference directive (preserve=true so tsc emits it)", () => {
        for (const entry of ["src/index.ts", "src/browser/index.ts", "src/node/index.ts"]) {
            const source = readFileSync(resolve(`./${entry}`), "utf-8");
            expect(source.startsWith(`${WEBGPU_TYPES_DIRECTIVE}\n`), entry).toBe(true);
        }
    });
});

describe("dist (spec 2.5; hard failure under CI, skip locally)", () => {
    it.runIf(underCi)("the tsc output and the bundle exist under CI", () => {
        expect(distExists).toBe(true);
        expect(bundleExists).toBe(true);
    });

    it.skipIf(!distExists)(
        "tsc emitted the three entries with their declarations, each carrying the @webgpu/types reference as a leading pragma",
        () => {
            for (const entry of ["src/index", "src/browser/index", "src/node/index"]) {
                expect(existsSync(resolve(`./dist/${entry}.js`)), entry).toBe(true);
                const dts = readFileSync(resolve(`./dist/${entry}.d.ts`), "utf-8");
                expect(
                    hasLeadingDirective(dts),
                    `${entry}.d.ts carries ${WEBGPU_TYPES_DIRECTIVE} before its first statement`,
                ).toBe(true);
            }
            // stripInternal: the @internal test seam is absent from the published declarations
            const nodeDts = readFileSync(resolve("./dist/src/node/index.d.ts"), "utf-8");
            expect(nodeDts).not.toContain("loadModule");
            expect(nodeDts).toContain("createNodeGpu");
            expect(nodeDts).toContain("dawnFlags");
        },
    );

    it.skipIf(!bundleExists)("the bundle and its one-line declaration shims exist for every entry", () => {
        const shims: Record<string, string> = {
            "webgpu-graph-algorithms": 'export * from "./src/index.js";',
            browser: 'export * from "./src/browser/index.js";',
            node: 'export * from "./src/node/index.js";',
        };
        for (const [name, shim] of Object.entries(shims)) {
            expect(existsSync(resolve(`./dist/${name}.js`)), name).toBe(true);
            expect(existsSync(resolve(`./dist/${name}.js.map`)), name).toBe(true);
            expect(readFileSync(resolve(`./dist/${name}.d.ts`), "utf-8").trim(), name).toBe(shim);
        }
    });

    it.skipIf(!bundleExists)("the node bundle shares one error class with the root bundle", async () => {
        const root = (await import(resolve("./dist/webgpu-graph-algorithms.js"))) as Record<string, unknown>;
        const node = (await import(resolve("./dist/node.js"))) as Record<string, unknown>;
        expect(root.default).toBeUndefined();
        expect(node.default).toBeUndefined();
        expect(Object.keys(node)).toEqual(expect.arrayContaining(["createNodeGpu", "dawnFlags"]));
        const createNodeGpu = node.createNodeGpu as (options: {
            loadModule: () => Promise<unknown>;
        }) => Promise<unknown>;
        const isWebGpuGraphError = root.isWebGpuGraphError as (x: unknown) => boolean;
        const WebGpuGraphErrorClass = root.WebGpuGraphError as new (...args: unknown[]) => Error;
        let thrown: unknown = null;
        try {
            await createNodeGpu({ loadModule: () => Promise.reject(new Error("no module")) });
        } catch (error) {
            thrown = error;
        }
        expect(isWebGpuGraphError(thrown)).toBe(true);
        // the same class object, not a copy bundled twice (one chunk under dist/chunks/)
        expect(thrown).toBeInstanceOf(WebGpuGraphErrorClass);
    });
});

// ============================================================ P1-T7: the bundle specifier assertions (spec 2.5 mechanism 1)

/** One `webgpu` specifier found in a bundle: where and in which import form. */
interface WebgpuSpecifier {
    readonly file: string;
    readonly form: "static" | "dynamic" | "require";
    readonly text: string;
}

/**
 * Every specifier naming the webgpu module in a bundle, in the four forms spec 2.5 lists (static `from "webgpu"`, a bare
 * side-effect `import "webgpu"`, dynamic `import("webgpu")`, `require("webgpu")`), both quote styles. The word itself also
 * appears in the package name, in kind: "webgpu" and in E_NO_WEBGPU, so this is a specifier match, not a substring match.
 * @param file - the bundle path relative to the package root
 * @param code - its text
 * @returns the specifiers found
 */
function webgpuSpecifiers(file: string, code: string): WebgpuSpecifier[] {
    const found: WebgpuSpecifier[] = [];
    const forms: readonly [WebgpuSpecifier["form"], RegExp][] = [
        ["static", /\bfrom\s*(["'])webgpu\1/g],
        ["static", /\bimport\s*(["'])webgpu\1/g],
        ["dynamic", /\bimport\s*\(\s*(["'])webgpu\1\s*\)/g],
        ["require", /\brequire\s*\(\s*(["'])webgpu\1\s*\)/g],
    ];
    for (const [form, pattern] of forms) {
        for (const match of code.matchAll(pattern)) {
            found.push({ file, form, text: match[0] });
        }
    }
    return found;
}

/**
 * The relative modules a bundle imports (vite's dist/chunks/*.js), transitively, so a specifier hidden in a shared chunk
 * is found too.
 * @param entry - a bundle path relative to the package root
 * @returns every file of the closure, the entry first
 */
function bundleClosure(entry: string): string[] {
    const seen: string[] = [];
    const pending = [entry];
    while (pending.length > 0) {
        const file = pending.pop();
        if (file === undefined || seen.includes(file)) {
            continue;
        }
        seen.push(file);
        const code = readFileSync(resolve(file), "utf-8");
        for (const match of code.matchAll(/\bfrom\s*(["'])(\.\.?\/[^"']+)\1/g)) {
            const target = match[2];
            if (target !== undefined) {
                pending.push(`${file.slice(0, file.lastIndexOf("/") + 1)}${target}`);
            }
        }
    }
    return seen;
}

describe("bundle specifiers (spec 2.5 mechanism 1; contract 5.5)", () => {
    const BUNDLES = ["./dist/webgpu-graph-algorithms.js", "./dist/browser.js", "./dist/node.js"];
    const bundlePresent = BUNDLES.every((file) => existsSync(resolve(file)));

    it("the bundle exists under CI (every lane builds with pnpm run build = build:all; hard-fail, never skip)", () => {
        if (underCi) {
            expect(bundlePresent, `dist bundle absent under CI: ${BUNDLES.join(", ")} -- run pnpm run build:all`).toBe(
                true,
            );
        } else {
            expect(typeof bundlePresent).toBe("boolean"); // locally the two assertions below skip without the bundle
        }
    });

    it.skipIf(!bundlePresent)(
        "the root and browser bundles (and their chunks) carry no webgpu specifier of any form",
        () => {
            for (const entry of ["./dist/webgpu-graph-algorithms.js", "./dist/browser.js"]) {
                const closure = bundleClosure(entry);
                expect(closure.length, `${entry} resolves to at least itself`).toBeGreaterThan(0);
                for (const file of closure) {
                    expect(webgpuSpecifiers(file, readFileSync(resolve(file), "utf-8")), file).toEqual([]);
                }
            }
        },
    );

    it.skipIf(!bundlePresent)("the node bundle names webgpu only inside a dynamic import()", () => {
        const found = bundleClosure("./dist/node.js").flatMap((file) =>
            webgpuSpecifiers(file, readFileSync(resolve(file), "utf-8")),
        );
        expect(found.length, "src/node/index.ts imports webgpu dynamically (3.7)").toBeGreaterThan(0);
        expect(found.filter((f) => f.form !== "dynamic")).toEqual([]);
        expect(found.every((f) => f.file === "./dist/node.js")).toBe(true);
    });
});
