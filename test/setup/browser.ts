// Setup for browser tests with WebGPU
import { setupWebGPU, cleanupWebGPU } from "./webgpu-global";
import { beforeAll, afterAll } from "vitest";

// Initialize WebGPU before all browser tests
beforeAll(async () => {
    console.log("Initializing WebGPU for browser tests...");
    await setupWebGPU();
});

// Cleanup after all tests
afterAll(() => {
    cleanupWebGPU();
});