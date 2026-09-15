// Global WebGPU setup for all browser tests

declare global {
    var webGPUContext: {
        adapter: GPUAdapter;
        device: GPUDevice;
    };
}

/**
 * Global setup for WebGPU tests
 * This runs once before all browser tests and FAILS if WebGPU is not available
 */
export async function setupWebGPU(): Promise<void> {
    // Check environment
    if (typeof navigator === "undefined") {
        throw new Error(
            "Tests must run in a browser environment. " +
            "Make sure you're using 'npm run test:browser' or 'npm test'"
        );
    }
    
    if (!("gpu" in navigator)) {
        throw new Error(
            "WebGPU API is not available in this browser.\n" +
            "WebGPU is required - no fallbacks are supported.\n" +
            "Please use Chrome 113+, Edge 113+, Safari 18+, or Firefox with WebGPU enabled."
        );
    }
    
    // Get adapter
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error(
            "Failed to get WebGPU adapter.\n" +
            "This project requires a GPU with WebGPU support.\n" +
            "Possible issues:\n" +
            "- No compatible GPU hardware\n" +
            "- Outdated GPU drivers\n" +
            "- Running in a VM without GPU passthrough"
        );
    }
    
    // Get device
    const device = await adapter.requestDevice();
    if (!device) {
        throw new Error("Failed to create WebGPU device");
    }
    
    // Store in global context
    globalThis.webGPUContext = {
        adapter,
        device
    };
    
    // Log adapter info
    try {
        const info = await adapter.requestAdapterInfo();
        console.log("✅ WebGPU initialized successfully");
        console.log("GPU Adapter:", info.description || "Unknown");
    } catch (e) {
        console.log("✅ WebGPU initialized (adapter info not available)");
    }
}

/**
 * Global cleanup for WebGPU tests
 */
export function cleanupWebGPU(): void {
    if (globalThis.webGPUContext?.device) {
        globalThis.webGPUContext.device.destroy();
        console.log("WebGPU device cleaned up");
    }
}