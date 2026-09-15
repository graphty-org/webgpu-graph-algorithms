// Root ESLint configuration for the graphty-monorepo
// This config focuses on ERROR PREVENTION, not stylistic rules
// Formatting is handled by Prettier (.prettierrc)

import eslint from "@eslint/js";
import jsdoc from "eslint-plugin-jsdoc";
import jsxA11y from "eslint-plugin-jsx-a11y";
import simpleImportSort from "eslint-plugin-simple-import-sort";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
    // ============================================
    // IGNORE PATTERNS
    // ============================================
    {
        ignores: [
            "**/dist/**",
            "**/node_modules/**",
            "**/coverage/**",
            "**/storybook-static/**",
            "**/gh-pages/**",
            "**/tmp/**",
            "**/.worktrees/**",
            // Config files are typically JS and don't need strict type checking
            "**/*.config.js",
            "**/*.config.ts",
            "**/vite.config.ts",
            "**/vitest.config.ts",
            "**/commitlint.config.js",
            // Benchmark and Storybook files not included in tsconfig
            "**/benchmarks/**",
            "**/.storybook/**",
            // Example files are for demonstration, not production
            "**/examples/**",
            "**/examples-legacy/**",
            // Build scripts are Node.js tools, not source code
            "**/scripts/**",
            // Stories are linted separately with relaxed rules but excluded from tsconfig
            "**/stories/**",
            // Vite plugins are build tools
            "**/vite-plugin-*.js",
            // VitePress cache and generated files
            "**/.vitepress/cache/**",
            "**/.vitepress/dist/**",
            // Docs directory (VitePress content)
            "docs/**",
            "**/docs/**",
            // Algorithms tests have pre-existing TypeScript issues; they work with vitest
            // but fail tsc --noEmit. Ignoring until tests can be refactored.
            "algorithms/test/**",
            // Layout tests are not in tsconfig and have parsing issues
            "layout/test/**",
            // Remote-logger tests are not in main tsconfig (see tsconfig.eslint.json)
            "remote-logger/test/**",
        ],
    },

    // ============================================
    // BASE JAVASCRIPT RULES
    // ============================================
    eslint.configs.recommended,

    // ============================================
    // JSDOC RULES FOR PUBLIC API DOCUMENTATION
    // ============================================
    jsdoc.configs["flat/recommended-typescript-error"],
    {
        plugins: {
            jsdoc,
        },
        rules: {
            // Enforce documentation quality for public APIs
            "jsdoc/require-description": "error",
            "jsdoc/require-param-description": "error",
            "jsdoc/require-returns-description": "error",
            "jsdoc/no-types": "error", // TypeScript handles types
            "jsdoc/check-tag-names": [
                "error",
                {
                    definedTags: ["since", "internal", "remarks"],
                },
            ],
            "jsdoc/require-jsdoc": [
                "error",
                {
                    publicOnly: true,
                    require: {
                        FunctionDeclaration: true,
                        MethodDefinition: true,
                        ClassDeclaration: true,
                    },
                },
            ],
        },
    },

    // ============================================
    // TYPESCRIPT STRICT RULES (for .ts/.tsx files)
    // ============================================
    {
        files: ["**/*.ts", "**/*.tsx"],
        extends: [...tseslint.configs.strictTypeChecked],
        languageOptions: {
            parserOptions: {
                projectService: true,
                tsconfigRootDir: import.meta.dirname,
            },
            globals: {
                ...globals.browser,
                ...globals.es2022,
            },
        },
        plugins: {
            "simple-import-sort": simpleImportSort,
        },
        rules: {
            // ==========================================
            // ERROR PREVENTION - Type Safety
            // ==========================================
            "@typescript-eslint/no-explicit-any": "error",
            "@typescript-eslint/explicit-function-return-type": [
                "error",
                {
                    allowExpressions: true,
                    allowIIFEs: true,
                },
            ],
            "@typescript-eslint/no-floating-promises": "error",
            "@typescript-eslint/no-misused-promises": "error",
            "@typescript-eslint/await-thenable": "error",
            "@typescript-eslint/require-await": "error",
            "@typescript-eslint/no-unnecessary-type-assertion": "error",
            "@typescript-eslint/no-unused-vars": [
                "error",
                {
                    argsIgnorePattern: "^_",
                    varsIgnorePattern: "^_",
                },
            ],
            // Keep non-null assertion as error (important for safety)
            "@typescript-eslint/no-non-null-assertion": "error",
            // TODO: Re-enable these rules and fix issues incrementally
            // These rules have many violations from strictTypeChecked
            "@typescript-eslint/no-unnecessary-condition": "off",
            "@typescript-eslint/no-unsafe-argument": "off",
            "@typescript-eslint/no-unsafe-assignment": "off",
            "@typescript-eslint/no-unsafe-call": "off",
            "@typescript-eslint/no-unsafe-member-access": "off",
            "@typescript-eslint/no-unsafe-return": "off",
            // Allow numbers and booleans in template literals (common logging pattern)
            "@typescript-eslint/restrict-template-expressions": [
                "error",
                {
                    allowNumber: true,
                    allowBoolean: true,
                    allowAny: false,
                    allowNullish: false,
                },
            ],

            // ==========================================
            // ERROR PREVENTION - Logic Safety
            // ==========================================
            eqeqeq: ["error", "always"],
            curly: "error",
            "no-var": "error",
            "prefer-const": "error",
            "consistent-return": "error",
            "no-template-curly-in-string": "error",
            "no-else-return": "error",
            "no-nested-ternary": "error",
            "no-unneeded-ternary": "error",
            "prefer-template": "error",
            "dot-notation": "error",
            "default-case": "error",
            "default-param-last": "error",
            yoda: ["error", "never"],

            // ==========================================
            // CODE QUALITY
            // ==========================================
            "no-console": ["error", { allow: ["warn", "error"] }],
            "no-duplicate-imports": "error",
            "no-useless-constructor": "error",
            "no-useless-rename": "error",
            "no-useless-computed-key": "error",
            "prefer-destructuring": ["error", { object: true, array: false }],
            "prefer-rest-params": "error",
            "prefer-spread": "error",
            camelcase: ["error", { properties: "always" }],

            // ==========================================
            // IMPORT SORTING (the only "stylistic" rules we keep)
            // ==========================================
            "simple-import-sort/imports": "error",
            "simple-import-sort/exports": "error",
        },
    },

    // ============================================
    // ACCESSIBILITY RULES FOR compact-mantine JSX
    // ============================================
    // Scoped to compact-mantine, which publishes React components that have to
    // meet WCAG 2.2 AA. The other packages are unaffected: graphty's JSX is an
    // application, and algorithms, layout and graphty-element ship no JSX.
    {
        files: ["compact-mantine/**/*.tsx"],
        plugins: {
            "jsx-a11y": jsxA11y,
        },
        rules: {
            ...jsxA11y.flatConfigs.recommended.rules,
        },
    },

    // ============================================
    // RELAXED RULES FOR TEST FILES
    // ============================================
    {
        files: [
            "**/*.test.ts",
            "**/*.test.tsx",
            "**/*.spec.ts",
            "**/*.spec.tsx",
            "**/test/**/*.ts",
            "**/test/**/*.tsx",
            "**/__tests__/**/*.ts",
            "**/__tests__/**/*.tsx",
        ],
        rules: {
            "@typescript-eslint/no-explicit-any": "off",
            "@typescript-eslint/explicit-function-return-type": "off",
            "@typescript-eslint/no-non-null-assertion": "off",
            "@typescript-eslint/unbound-method": "off",
            // Disable template expression checks in tests - tests often log various types
            "@typescript-eslint/restrict-template-expressions": "off",
            // Tests often need to call deprecated functions to verify their behavior
            "@typescript-eslint/no-deprecated": "off",
            // Tests often have void expressions in arrow functions for assertions
            "@typescript-eslint/no-confusing-void-expression": "off",
            // Allow undefined in arithmetic in tests (often testing edge cases)
            "@typescript-eslint/restrict-plus-operands": "off",
            // Allow redundant type constituents in tests (often testing type edge cases)
            "@typescript-eslint/no-redundant-type-constituents": "off",
            "no-console": "off",
            // Disable JSDoc rules for test files
            "jsdoc/require-jsdoc": "off",
            "jsdoc/require-description": "off",
            "jsdoc/require-param": "off",
            "jsdoc/require-param-description": "off",
            "jsdoc/require-returns": "off",
            "jsdoc/require-returns-description": "off",
            "jsdoc/check-param-names": "off",
            "jsdoc/check-tag-names": "off",
            "jsdoc/tag-lines": "off",
        },
    },

    // ============================================
    // RELAXED JSDOC FOR STORIES
    // ============================================
    {
        files: ["**/*.stories.ts", "**/*.stories.tsx", "**/stories/**/*.ts"],
        rules: {
            "jsdoc/require-jsdoc": "off",
            "jsdoc/require-description": "off",
            "jsdoc/require-param": "off",
            "jsdoc/require-param-description": "off",
            "jsdoc/require-returns": "off",
            "jsdoc/require-returns-description": "off",
            "jsdoc/check-param-names": "off",
            "jsdoc/check-tag-names": "off",
            "jsdoc/tag-lines": "off",
        },
    },

    // ============================================
    // RELAXED RULES FOR JAVASCRIPT FILES
    // ============================================
    {
        files: ["**/*.js", "**/*.mjs", "**/*.cjs"],
        rules: {
            "@typescript-eslint/explicit-function-return-type": "off",
        },
    },
);
