/**
 * Knip configuration for the graphty staging workspace.
 *
 * Mirrors the shape of the monorepo's knip.config.ts so the two package
 * entries below can be pasted into it on move-in (design section 13.3).
 * Run with: pnpm run lint:knip
 *
 * Policy (same as the monorepo): ignoreExportsUsedInFile is deliberately NOT
 * set. An export only referenced inside its own file is reported as dead
 * unless its JSDoc carries a "@public" clause saying why it is exported.
 *
 * @see https://knip.dev/overview/configuration
 */

import type { KnipConfig } from "knip";

const config: KnipConfig = {
    workspaces: {
        // Root workspace - shared configs copied verbatim from the monorepo
        ".": {
            entry: ["vite.shared.config.ts", "vitest.shared.config.ts"],
            project: ["*.ts", "*.js"],
            ignoreDependencies: [
                // Used by the package scripts through pnpm's hoisted node_modules/.bin
                "@vitest/coverage-v8",
                // Hoisted here for the packages' property tests
                "fast-check",
            ],
        },

        // graph-format package
        "graph-format": {
            entry: ["src/index.ts", "test/**/*.test.ts", "test/types/**/*.test-d.ts", "scripts/**/*.{ts,js}"],
            project: ["src/**/*.ts", "test/**/*.ts", "benchmarks/**/*.ts", "scripts/**/*.{ts,js}"],
        },

        // graph-io package
        "graph-io": {
            entry: ["src/index.ts", "test/**/*.test.ts", "test/types/**/*.test-d.ts", "scripts/**/*.{ts,js}"],
            project: ["src/**/*.ts", "test/**/*.ts", "benchmarks/**/*.ts", "scripts/**/*.{ts,js}"],
        },
    },
};

export default config;
