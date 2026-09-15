/**
 * The layer rule of spec 3.2 and the entry isolation of spec 2.4, enforced by walking the import graph of
 * src/**\/*.ts with the TypeScript compiler API (lead decision (a); contract 5.5). Every file is parsed with
 * ts.createSourceFile (NOT ts.preProcessFile, which cannot tell `import type` from a value import) and yields one
 * edge per import / export-from / dynamic import() / `import("x").T` type query with its type-only flag:
 *  - the VALUE-edge graph has no cycle; the three declared type-only pairs are pinned separately;
 *  - every relative edge, type edges included, obeys the layer table below (the eslint zones of contract 2.4
 *    plus the three tightenings of PLAN DECISION 10);
 *  - src/wgsl/** is imported only by src/kernels.ts; src/browser and src/node by nothing else in src/;
 *  - `navigator` is referenced only under src/browser, `process.<x>` only under src/node, the "webgpu" module
 *    only by src/node/index.ts and only through a dynamic import();
 *  - `caps.software` is read only in src/kernel/dispatch.ts (planGridStride; spec 2.4);
 *  - ESLint's computed config for src/types/context.ts and src/types/accelerator.ts carries BOTH halves of the
 *    @typescript-eslint/no-restricted-imports options (2.4: flat config replaces rule options, never merges).
 */

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

import { ESLint } from "eslint";
import ts from "typescript";

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SRC = join(PACKAGE_ROOT, "src");

type Layer =
    | "base"
    | "types"
    | "device"
    | "context"
    | "memory"
    | "kernel"
    | "registry"
    | "wgsl"
    | "primitives"
    | "algorithms"
    | "layouts"
    | "accelerator"
    | "entries"
    | "barrel";

interface Edge {
    readonly from: string;
    readonly to: string;
    readonly specifier: string;
    readonly typeOnly: boolean;
    readonly dynamic: boolean;
}

interface ExternalRef {
    readonly specifier: string;
    readonly typeOnly: boolean;
    readonly dynamic: boolean;
}

interface FileFacts {
    readonly file: string;
    readonly edges: readonly Edge[];
    readonly externals: readonly ExternalRef[];
    readonly navigatorRefs: number;
    readonly processRefs: number;
    readonly capsSoftwareRefs: number;
}

/** The layers a file of each layer may import, VALUE edges (the eslint zones of contract 2.4, tightened per PLAN DECISION 10). */
const VALUE_TARGETS: Readonly<Record<Layer, readonly Layer[]>> = {
    base: ["base"],
    types: ["base", "types"],
    device: ["base", "types", "device"],
    memory: ["base", "types", "device", "memory"],
    kernel: ["base", "types", "device", "kernel"],
    registry: ["base", "types", "device", "memory", "kernel", "wgsl", "registry"],
    context: ["base", "types", "device", "memory", "kernel", "context"],
    wgsl: ["base"],
    primitives: ["base", "types", "device", "memory", "kernel", "registry", "primitives"],
    algorithms: ["base", "types", "device", "context", "memory", "kernel", "registry", "primitives", "algorithms"],
    layouts: ["base", "types", "device", "context", "memory", "kernel", "registry", "primitives", "layouts"],
    accelerator: [
        "base",
        "types",
        "device",
        "context",
        "memory",
        "kernel",
        "registry",
        "primitives",
        "algorithms",
        "layouts",
    ],
    entries: ["base", "types", "device", "context", "entries"],
    barrel: [
        "base",
        "types",
        "device",
        "context",
        "memory",
        "kernel",
        "registry",
        "primitives",
        "algorithms",
        "layouts",
        "accelerator",
    ],
};

/** Additional targets allowed for TYPE-ONLY edges (contract 2.4 allowTypeImports; contract 3.3 / 3.9 for kernel -> memory). */
const EXTRA_TYPE_TARGETS: Readonly<Partial<Record<Layer, readonly Layer[]>>> = {
    types: ["device", "context", "memory", "kernel", "registry", "primitives", "algorithms", "layouts", "accelerator"],
    kernel: ["memory"],
};

/** The declared type-only pairs: [from, to] where from -> to edges must all be `import type` (contract 5.5). */
const TYPE_ONLY_PAIRS: readonly [string, string][] = [
    ["memory/lease.ts", "memory/buffer-pool.ts"],
    ["kernel/batch.ts", "kernel/profiler.ts"],
    ["kernel/profiler.ts", "kernel/batch.ts"],
    ["types/layout.ts", "types/accelerator.ts"],
    ["types/accelerator.ts", "types/layout.ts"],
];

function walk(dir: string, out: string[] = []): string[] {
    for (const name of readdirSync(dir).sort()) {
        const path = join(dir, name);
        if (statSync(path).isDirectory()) {
            walk(path, out);
        } else if (path.endsWith(".ts") && !path.endsWith(".d.ts")) {
            out.push(path);
        }
    }
    return out;
}

/** src-relative posix path of a file. */
function rel(file: string): string {
    return relative(SRC, file).split(sep).join("/");
}

function layerOf(file: string): Layer {
    const path = rel(file);
    if (path === "errors.ts" || path === "constants.ts") {
        return "base";
    }
    if (path === "context.ts") {
        return "context";
    }
    if (path === "kernels.ts") {
        return "registry";
    }
    if (path === "accelerator.ts") {
        return "accelerator";
    }
    if (path === "index.ts") {
        return "barrel";
    }
    const dirs: Readonly<Record<string, Layer>> = {
        types: "types",
        device: "device",
        memory: "memory",
        kernel: "kernel",
        wgsl: "wgsl",
        primitives: "primitives",
        algorithms: "algorithms",
        layouts: "layouts",
        browser: "entries",
        node: "entries",
    };
    const [first] = path.split("/");
    const layer = dirs[first];
    if (layer === undefined) {
        throw new Error(`unclassified source file ${path}: add it to the layer table of test/layers.test.ts`);
    }
    return layer;
}

function isPropertyName(node: ts.Identifier): boolean {
    return ts.isPropertyAccessExpression(node.parent) && node.parent.name === node;
}

function factsOf(file: string): FileFacts {
    const text = readFileSync(file, "utf8");
    const sf = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true);
    const edges: Edge[] = [];
    const externals: ExternalRef[] = [];
    let navigatorRefs = 0;
    let processRefs = 0;
    let capsSoftwareRefs = 0;
    const record = (specifier: string, typeOnly: boolean, dynamic: boolean): void => {
        if (specifier.startsWith(".")) {
            const to = resolve(dirname(file), specifier.replace(/\.js$/, ".ts"));
            edges.push({ from: file, to, specifier, typeOnly, dynamic });
        } else {
            externals.push({ specifier, typeOnly, dynamic });
        }
    };
    const visit = (node: ts.Node): void => {
        if (ts.isImportDeclaration(node) && ts.isStringLiteral(node.moduleSpecifier)) {
            const clause = node.importClause;
            let typeOnly = clause?.isTypeOnly === true;
            if (
                !typeOnly &&
                clause !== undefined &&
                clause.name === undefined &&
                clause.namedBindings !== undefined &&
                ts.isNamedImports(clause.namedBindings)
            ) {
                const specs = clause.namedBindings.elements;
                typeOnly = specs.length > 0 && specs.every((spec) => spec.isTypeOnly);
            }
            record(node.moduleSpecifier.text, typeOnly, false);
        } else if (
            ts.isExportDeclaration(node) &&
            node.moduleSpecifier !== undefined &&
            ts.isStringLiteral(node.moduleSpecifier)
        ) {
            let typeOnly = node.isTypeOnly;
            if (!typeOnly && node.exportClause !== undefined && ts.isNamedExports(node.exportClause)) {
                const specs = node.exportClause.elements;
                typeOnly = specs.length > 0 && specs.every((spec) => spec.isTypeOnly);
            }
            record(node.moduleSpecifier.text, typeOnly, false);
        } else if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword) {
            const [argument] = node.arguments;
            if (argument !== undefined && ts.isStringLiteral(argument)) {
                record(argument.text, false, true);
            }
        } else if (
            ts.isImportTypeNode(node) &&
            ts.isLiteralTypeNode(node.argument) &&
            ts.isStringLiteral(node.argument.literal)
        ) {
            record(node.argument.literal.text, true, false);
        } else if (ts.isIdentifier(node) && node.text === "navigator" && !isPropertyName(node)) {
            navigatorRefs += 1;
        } else if (ts.isPropertyAccessExpression(node) && ts.isIdentifier(node.expression)) {
            if (node.expression.text === "process") {
                processRefs += 1;
            }
            if (node.expression.text === "caps" && node.name.text === "software") {
                capsSoftwareRefs += 1;
            }
        }
        ts.forEachChild(node, visit);
    };
    visit(sf);
    return { file, edges, externals, navigatorRefs, processRefs, capsSoftwareRefs };
}

/** The first cycle of the VALUE-edge graph as a path of src-relative files, or null. */
function findValueCycle(facts: readonly FileFacts[]): string[] | null {
    const successors = new Map<string, string[]>();
    for (const f of facts) {
        successors.set(
            f.file,
            f.edges.filter((edge) => !edge.typeOnly).map((edge) => edge.to),
        );
    }
    const state = new Map<string, "visiting" | "done">();
    const stack: string[] = [];
    const visit = (node: string): string[] | null => {
        const current = state.get(node);
        if (current === "done") {
            return null;
        }
        if (current === "visiting") {
            return [...stack.slice(stack.indexOf(node)), node].map(rel);
        }
        state.set(node, "visiting");
        stack.push(node);
        for (const next of successors.get(node) ?? []) {
            const cycle = visit(next);
            if (cycle !== null) {
                return cycle;
            }
        }
        stack.pop();
        state.set(node, "done");
        return null;
    };
    for (const f of facts) {
        const cycle = visit(f.file);
        if (cycle !== null) {
            return cycle;
        }
    }
    return null;
}

const FILES = walk(SRC);
const FACTS = FILES.map(factsOf);

describe("src/ import graph (spec 3.2, 2.4; contract 5.5)", () => {
    it("classifies every source file into a layer and every relative import has a .js suffix and resolves", () => {
        expect(FILES.length).toBeGreaterThan(0);
        for (const f of FACTS) {
            expect(() => layerOf(f.file), rel(f.file)).not.toThrow();
            for (const edge of f.edges) {
                expect(edge.specifier.endsWith(".js"), `${rel(f.file)} imports ${edge.specifier}`).toBe(true);
                expect(existsSync(edge.to), `${rel(f.file)} imports ${edge.specifier}`).toBe(true);
            }
        }
    });

    it("has no cycle over value edges", () => {
        expect(findValueCycle(FACTS)).toBeNull();
    });

    it("keeps the declared type-only pairs type-only", () => {
        for (const [from, to] of TYPE_ONLY_PAIRS) {
            const fromFile = join(SRC, from);
            const toFile = join(SRC, to);
            if (!existsSync(fromFile) || !existsSync(toFile)) {
                continue;
            }
            const facts = FACTS.find((f) => f.file === fromFile);
            expect(facts, from).toBeDefined();
            for (const edge of facts?.edges ?? []) {
                if (edge.to === toFile) {
                    expect(edge.typeOnly, `${from} -> ${to} must be import type`).toBe(true);
                }
            }
        }
    });

    it("respects the layer table for every relative edge, type edges included", () => {
        const violations: string[] = [];
        for (const f of FACTS) {
            const fromLayer = layerOf(f.file);
            for (const edge of f.edges) {
                const toLayer = layerOf(edge.to);
                const allowed =
                    VALUE_TARGETS[fromLayer].includes(toLayer) ||
                    (edge.typeOnly && (EXTRA_TYPE_TARGETS[fromLayer] ?? []).includes(toLayer));
                if (!allowed) {
                    violations.push(
                        `${rel(f.file)} (${fromLayer}) -> ${rel(edge.to)} (${toLayer})${edge.typeOnly ? " [type]" : ""}`,
                    );
                }
            }
        }
        expect(violations).toEqual([]);
    });

    it("src/wgsl/** is imported only by src/kernels.ts", () => {
        for (const f of FACTS) {
            for (const edge of f.edges) {
                if (layerOf(edge.to) === "wgsl") {
                    expect(rel(f.file), `${rel(f.file)} imports ${edge.specifier}`).toBe("kernels.ts");
                }
            }
        }
    });

    it("src/browser and src/node are imported by nothing else in src/", () => {
        for (const f of FACTS) {
            for (const edge of f.edges) {
                if (layerOf(edge.to) === "entries") {
                    expect(dirname(edge.to), `${rel(f.file)} imports ${edge.specifier}`).toBe(dirname(f.file));
                }
            }
        }
    });

    it("references navigator only under src/browser, process.<x> only under src/node", () => {
        for (const f of FACTS) {
            const path = rel(f.file);
            if (!path.startsWith("browser/")) {
                expect(f.navigatorRefs, `${path} references navigator`).toBe(0);
            }
            if (!path.startsWith("node/")) {
                expect(f.processRefs, `${path} references process.<x>`).toBe(0);
            }
        }
    });

    it('names the "webgpu" module only in src/node/index.ts and only through a dynamic import()', () => {
        let dynamicInNode = 0;
        for (const f of FACTS) {
            for (const ref of f.externals) {
                if (ref.specifier === "webgpu" || ref.specifier.startsWith("webgpu/")) {
                    expect(rel(f.file), `${rel(f.file)} names ${ref.specifier}`).toBe("node/index.ts");
                    expect(ref.dynamic, `${rel(f.file)} must use import("webgpu")`).toBe(true);
                    dynamicInNode += 1;
                }
            }
        }
        expect(dynamicInNode).toBe(1);
    });

    it("reads caps.software only in src/kernel/dispatch.ts (spec 2.4)", () => {
        for (const f of FACTS) {
            if (f.capsSoftwareRefs > 0) {
                expect(rel(f.file)).toBe("kernel/dispatch.ts");
            }
        }
    });
});

interface RestrictedImportsOptions {
    readonly paths?: readonly { readonly name: string; readonly allowTypeImports?: boolean }[];
    readonly patterns?: readonly { readonly group: readonly string[]; readonly allowTypeImports?: boolean }[];
}

/** The computed options of @typescript-eslint/no-restricted-imports for a src file (the file need not exist). */
async function restrictedImportsOptions(eslint: ESLint, file: string): Promise<RestrictedImportsOptions> {
    const config = (await eslint.calculateConfigForFile(join(PACKAGE_ROOT, file))) as
        { rules?: Record<string, unknown> } | undefined;
    const entry = config?.rules?.["@typescript-eslint/no-restricted-imports"];
    expect(Array.isArray(entry), `${file}: rule entry`).toBe(true);
    const [severity, options] = entry as [unknown, RestrictedImportsOptions | undefined];
    expect([2, "error"], `${file}: severity`).toContain(severity);
    expect(options, `${file}: options`).toBeDefined();
    return options ?? {};
}

describe("eslint zones (contract 2.4: flat config replaces rule options, so both halves must survive)", () => {
    it("src/types/context.ts carries the CPU-package paths AND the types-zone patterns", async () => {
        const eslint = new ESLint({ cwd: PACKAGE_ROOT });
        const options = await restrictedImportsOptions(eslint, "src/types/context.ts");
        expect((options.paths ?? []).map((p) => p.name).sort()).toEqual(["@graphty/algorithms", "@graphty/layout"]);
        for (const p of options.paths ?? []) {
            expect(p.allowTypeImports, p.name).not.toBe(true);
        }
        expect(options.patterns?.length).toBe(1);
        const [pattern] = options.patterns ?? [];
        expect(pattern?.allowTypeImports).toBe(true);
        expect(pattern?.group).toContain("../context.js");
        expect(pattern?.group).toContain("../device/*");
    });

    it("src/types/accelerator.ts carries the same patterns and paths with allowTypeImports", async () => {
        const eslint = new ESLint({ cwd: PACKAGE_ROOT });
        const options = await restrictedImportsOptions(eslint, "src/types/accelerator.ts");
        expect((options.paths ?? []).map((p) => p.name).sort()).toEqual(["@graphty/algorithms", "@graphty/layout"]);
        for (const p of options.paths ?? []) {
            expect(p.allowTypeImports, p.name).toBe(true);
        }
        expect(options.patterns?.length).toBe(1);
        expect(options.patterns?.[0]?.allowTypeImports).toBe(true);
    });
});
