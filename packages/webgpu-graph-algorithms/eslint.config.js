/**
 * Package-local ESLint config: the root flat config of packages/ spread first, then the rules only this
 * package needs (spec 2.4 / 3.2): the layer rule as `no-restricted-imports` zones (one block per layer
 * directory forbidding the relative specifiers of every HIGHER layer), the entry isolation (src/browser
 * and src/node are imported by nothing else in src/), the no-navigator / no-process rules, and the ban on
 * runtime imports of the CPU packages (type imports allowed in src/types/accelerator.ts only, D27).
 * test/layers.test.ts enforces the same rules plus cycle detection by walking the import graph.
 */

import tseslint from "typescript-eslint";

import root from "../eslint.config.js";

const LAYER_MESSAGE = "layer rule (spec 3.2): a lower layer never imports a higher one";
const ENTRY_MESSAGE = "src/browser and src/node are imported by nothing else in src/ (spec 2.4)";
const CPU_MESSAGE = "no runtime import of the CPU packages (D3, D27); type imports only in src/types/accelerator.ts";
const TYPES_MESSAGE = "src/types imports values from nothing above errors.ts / constants.ts; type imports are allowed";

// Flat config does NOT merge rule options: a later matching block that gives options REPLACES the earlier ones
// for that rule id. Every file set below therefore gets exactly ONE options object per rule id
// ("@typescript-eslint/no-restricted-imports" carries both `paths` and `patterns` where a file needs both), and
// the generic CPU-ban block excludes src/types/** so it cannot clobber the types zone. test/layers.test.ts
// asserts the computed config (ESLint.calculateConfigForFile) still carries both halves.
const CPU_PATHS = [
    { name: "@graphty/algorithms", message: CPU_MESSAGE },
    { name: "@graphty/layout", message: CPU_MESSAGE },
];
const CPU_PATHS_TYPES_ALLOWED = CPU_PATHS.map((p) => ({ ...p, allowTypeImports: true }));

// Relative specifiers of each layer as seen from one directory below src/ and from src/ itself.
const UP = {
    context: ["../context.js", "./context.js"],
    memory: ["../memory/*", "./memory/*"],
    kernel: ["../kernel/*", "./kernel/*"],
    registry: ["../kernels.js", "./kernels.js"],
    wgsl: ["../wgsl/*", "./wgsl/*"],
    primitives: ["../primitives/*", "./primitives/*"],
    algorithms: ["../algorithms/*", "../algorithms/**", "./algorithms/*", "./algorithms/**"],
    layouts: ["../layouts/*", "./layouts/*"],
    accelerator: ["../accelerator.js", "./accelerator.js"],
    entries: ["../browser/*", "../node/*", "./browser/*", "./node/*"],
    barrel: ["../index.js", "./index.js"],
};

/**
 * One layer zone: the files may not import the specifiers of the named higher layers.
 * @param files - glob(s) of the zone
 * @param higher - keys of UP that are above this zone
 * @returns a flat-config block
 */
function zone(files, higher) {
    const group = higher.flatMap((name) => UP[name]);
    return {
        files,
        rules: {
            "no-restricted-imports": ["error", { patterns: [{ group, message: LAYER_MESSAGE }] }],
        },
    };
}

// The type-only zone of src/types/**: every layer above errors.ts / constants.ts, values forbidden, `import type` allowed.
const TYPES_PATTERNS = [
    {
        group: [
            ...UP.context,
            ...UP.memory,
            ...UP.kernel,
            ...UP.registry,
            ...UP.wgsl,
            ...UP.primitives,
            ...UP.algorithms,
            ...UP.layouts,
            ...UP.accelerator,
            ...UP.entries,
            ...UP.barrel,
            "../device/*",
        ],
        message: TYPES_MESSAGE,
        allowTypeImports: true,
    },
];

export default tseslint.config(
    ...root,
    // ---- demo/: the standalone browser demo page (vite dev server over demo/, not part of the build, the tests or
    // tsconfig.json), type-checked by hand with the DOM lib; ignored here exactly as the root ignores examples/**.
    { ignores: ["demo/**"] },
    // ---- layer zones (spec 3.2: device < context < memory < kernel < kernels.ts < primitives < algorithms / layouts < accelerator)
    zone(
        ["src/errors.ts", "src/constants.ts"],
        [
            "context",
            "memory",
            "kernel",
            "registry",
            "wgsl",
            "primitives",
            "algorithms",
            "layouts",
            "accelerator",
            "entries",
            "barrel",
        ],
    ),
    zone(
        ["src/device/**/*.ts"],
        [
            "context",
            "memory",
            "kernel",
            "registry",
            "wgsl",
            "primitives",
            "algorithms",
            "layouts",
            "accelerator",
            "entries",
            "barrel",
        ],
    ),
    zone(
        ["src/memory/**/*.ts"],
        [
            "context",
            "kernel",
            "registry",
            "wgsl",
            "primitives",
            "algorithms",
            "layouts",
            "accelerator",
            "entries",
            "barrel",
        ],
    ),
    zone(
        ["src/kernel/**/*.ts"],
        ["context", "registry", "wgsl", "primitives", "algorithms", "layouts", "accelerator", "entries", "barrel"],
    ),
    zone(["src/kernels.ts"], ["context", "primitives", "algorithms", "layouts", "accelerator", "entries", "barrel"]),
    zone(
        ["src/context.ts"],
        ["registry", "wgsl", "primitives", "algorithms", "layouts", "accelerator", "entries", "barrel"],
    ),
    zone(["src/primitives/**/*.ts"], ["context", "wgsl", "algorithms", "layouts", "accelerator", "entries", "barrel"]),
    zone(["src/algorithms/**/*.ts"], ["wgsl", "layouts", "accelerator", "entries", "barrel"]),
    zone(["src/layouts/**/*.ts"], ["wgsl", "algorithms", "accelerator", "entries", "barrel"]),
    zone(["src/accelerator.ts"], ["wgsl", "entries", "barrel"]),
    zone(
        ["src/browser/**/*.ts", "src/node/**/*.ts"],
        ["memory", "kernel", "registry", "wgsl", "primitives", "algorithms", "layouts", "accelerator", "barrel"],
    ),
    // src/types/** holds types only: it may `import type` from anywhere below the accelerator (GpuContext, GraphResidency), never a value;
    // the SAME options object also carries the CPU-package ban (one options object per rule id per file set, see the note above)
    {
        files: ["src/types/**/*.ts"],
        ignores: ["src/types/accelerator.ts"],
        rules: {
            "@typescript-eslint/no-restricted-imports": ["error", { paths: CPU_PATHS, patterns: TYPES_PATTERNS }],
        },
    },
    {
        files: ["src/types/accelerator.ts"],
        rules: {
            "@typescript-eslint/no-restricted-imports": [
                "error",
                { paths: CPU_PATHS_TYPES_ALLOWED, patterns: TYPES_PATTERNS },
            ],
        },
    },
    // ---- the entries are imported by nothing else in src/ (the zones above already forbid them; the barrel is the remaining file)
    {
        files: ["src/index.ts"],
        rules: {
            "no-restricted-imports": ["error", { patterns: [{ group: UP.entries, message: ENTRY_MESSAGE }] }],
        },
    },
    // ---- no runtime import of the CPU packages anywhere else in src/ (src/types/** is handled by its own blocks above, D27)
    {
        files: ["src/**/*.ts"],
        ignores: ["src/types/**/*.ts"],
        rules: {
            "@typescript-eslint/no-restricted-imports": ["error", { paths: CPU_PATHS }],
        },
    },
    // ---- the core never references the runtime globals (spec 2.1, 2.4). Three DISJOINT file sets, because
    // `no-restricted-globals` options are replaced (not merged) by a later matching block: one block over the
    // core with every banned global, one over src/browser/** (process only), one over src/node/** (the DOM three).
    {
        files: ["src/**/*.ts"],
        ignores: ["src/browser/**/*.ts", "src/node/**/*.ts"],
        rules: {
            "no-restricted-globals": [
                "error",
                { name: "navigator", message: "only src/browser/** may read navigator.gpu (spec 2.4)" },
                { name: "window", message: "the core never references window (spec 2.1)" },
                { name: "document", message: "the core never references document (spec 2.1)" },
                {
                    name: "process",
                    message:
                        "the core never references process (spec 2.1); env vars are read by test/setup and scripts only (spec 2.3)",
                },
            ],
        },
    },
    {
        files: ["src/browser/**/*.ts"],
        rules: {
            "no-restricted-globals": [
                "error",
                {
                    name: "process",
                    message:
                        "the core never references process (spec 2.1); env vars are read by test/setup and scripts only (spec 2.3)",
                },
            ],
        },
    },
    {
        files: ["src/node/**/*.ts"],
        rules: {
            "no-restricted-globals": [
                "error",
                { name: "navigator", message: "only src/browser/** may read navigator.gpu (spec 2.4)" },
                { name: "window", message: "the core never references window (spec 2.1)" },
                { name: "document", message: "the core never references document (spec 2.1)" },
            ],
        },
    },
    // ---- test relaxations beyond the root's: tests may import any layer and any global; setup files read process.env
    {
        files: ["test/**/*.ts"],
        rules: {
            "no-restricted-imports": "off",
            "@typescript-eslint/no-restricted-imports": "off",
            "no-restricted-globals": "off",
        },
    },
);
