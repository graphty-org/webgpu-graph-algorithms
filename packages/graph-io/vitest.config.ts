import { defineConfig } from "vitest/config";

export default defineConfig({
    test: {
        globals: true,
        environment: "node",
        pool: "forks",
        testTimeout: 30000,
        include: ["test/**/*.test.ts"],
        reporters: ["verbose"],
        coverage: {
            all: true,
            provider: "v8",
            reporter: ["text", "json-summary", "json", "lcov", "html"],
            reportsDirectory: "coverage",
            include: ["src/**/*.ts"],
            exclude: ["**/*.d.ts", "**/*.test.ts", "**/index.ts"],
            thresholds: {
                lines: 80,
                functions: 80,
                branches: 75,
                statements: 80,
            },
        },
    },
});
