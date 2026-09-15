import { defineConfig } from "vitest/config";
import { resolve } from "path";

export default defineConfig({
    test: {
        globals: true,
        // Use browser environment for all tests since they're all WebGPU-related
        browser: {
            enabled: true,
            name: "chromium",
            provider: "playwright",
            headless: true,
            screenshotFailures: false,
            // Launch options for WebGPU support
            launch: {
                args: [
                    "--enable-unsafe-webgpu",
                    "--enable-features=Vulkan",
                    "--use-gl=swiftshader",
                    "--use-vulkan=swiftshader"
                ]
            }
        },
        include: ["test/**/*.test.ts"],
        setupFiles: ["test/setup/browser.ts"],
        testTimeout: 30000,
        hookTimeout: 30000,
        coverage: {
            provider: "v8",
            reporter: ["text", "json-summary", "json", "lcov", "html"],
            include: ["src/**/*.ts"],
            exclude: [
                "src/**/*.d.ts",
                "src/**/*.test.ts",
                "src/**/*.bench.ts"
            ]
        },
        benchmark: {
            include: ["test/**/*.bench.ts"]
        }
    },
    resolve: {
        alias: {
            "@": resolve(__dirname, "./src")
        }
    }
});