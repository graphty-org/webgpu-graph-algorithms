import { describe, expect, it } from "vitest";

import { version } from "../../src/index";

describe("WebGPU Graph Algorithms", () => {
    it("should export version", () => {
        expect(version).toBe("0.1.0");
    });

    it("should have WebGPU types available", () => {
        // This test ensures @webgpu/types is properly configured
        const device: GPUDevice | null = null;
        expect(device).toBeNull();
    });
});
