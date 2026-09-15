import { describe, it } from "vitest";

/**
 * Get the global WebGPU context for tests
 * Throws if WebGPU is not available (no fallbacks)
 */
export function getWebGPUContext(): { adapter: GPUAdapter; device: GPUDevice } {
    if (!globalThis.webGPUContext) {
        throw new Error("WebGPU context not initialized - WebGPU is required");
    }
    
    return globalThis.webGPUContext;
}

/**
 * Create a new GPU device for isolated tests
 */
export async function createTestDevice(): Promise<GPUDevice> {
    const { adapter } = getWebGPUContext();
    return await adapter.requestDevice();
}

/**
 * Run a test with a fresh GPU device that gets cleaned up afterward
 */
export async function withTestDevice(
    fn: (device: GPUDevice) => Promise<void>
): Promise<void> {
    const device = await createTestDevice();
    try {
        await fn(device);
    } finally {
        device.destroy();
    }
}