// WebGPU test helpers

export interface GPUTestContext {
    adapter: GPUAdapter;
    device: GPUDevice;
}

/**
 * Initialize WebGPU for testing
 */
export async function initializeWebGPU(): Promise<GPUTestContext> {
    if (!navigator.gpu) {
        throw new Error("WebGPU is not supported");
    }
    
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("Failed to get WebGPU adapter");
    }
    
    const device = await adapter.requestDevice();
    
    return { adapter, device };
}

/**
 * Create a buffer with data
 */
export function createBuffer(
    device: GPUDevice,
    data: Float32Array | Uint32Array,
    usage: GPUBufferUsageFlags
): GPUBuffer {
    const buffer = device.createBuffer({
        size: data.byteLength,
        usage,
        mappedAtCreation: true
    });
    
    const mappedArray = buffer.getMappedRange();
    if (data instanceof Float32Array) {
        new Float32Array(mappedArray).set(data);
    } else {
        new Uint32Array(mappedArray).set(data);
    }
    buffer.unmap();
    
    return buffer;
}

/**
 * Read data from a GPU buffer
 */
export async function readBuffer(
    device: GPUDevice,
    buffer: GPUBuffer,
    size: number
): Promise<Float32Array> {
    const stagingBuffer = device.createBuffer({
        size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
    device.queue.submit([commandEncoder.finish()]);
    
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(stagingBuffer.getMappedRange());
    const result = new Float32Array(data);
    stagingBuffer.unmap();
    stagingBuffer.destroy();
    
    return result;
}

/**
 * Compare float arrays with tolerance
 */
export function expectFloatArraysEqual(
    actual: Float32Array,
    expected: Float32Array,
    tolerance = 1e-6
): void {
    expect(actual.length).toBe(expected.length);
    
    for (let i = 0; i < actual.length; i++) {
        const diff = Math.abs(actual[i] - expected[i]);
        if (diff > tolerance) {
            throw new Error(
                `Arrays differ at index ${i}: ${actual[i]} !== ${expected[i]} (diff: ${diff})`
            );
        }
    }
}