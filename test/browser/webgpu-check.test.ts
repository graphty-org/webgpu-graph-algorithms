import { describe, expect, it } from "vitest";

/**
 * WebGPU Requirements Test
 * This test suite verifies that WebGPU is available and functional.
 * Tests will FAIL if WebGPU is not supported - no fallbacks.
 */
describe("WebGPU Requirements", () => {
    it("must have WebGPU API available", () => {
        expect(navigator).toBeDefined();
        expect(navigator.gpu).toBeDefined();
    });
    
    it("must be able to get GPU adapter", async () => {
        const adapter = await navigator.gpu.requestAdapter();
        expect(adapter).not.toBeNull();
        expect(adapter).toBeDefined();
    });
    
    it("must be able to get GPU device", async () => {
        const adapter = await navigator.gpu.requestAdapter();
        expect(adapter).not.toBeNull();
        
        const device = await adapter!.requestDevice();
        expect(device).not.toBeNull();
        expect(device).toBeDefined();
        
        // Verify required methods exist
        expect(device.createBuffer).toBeDefined();
        expect(device.createShaderModule).toBeDefined();
        expect(device.createComputePipeline).toBeDefined();
        expect(device.createBindGroup).toBeDefined();
        expect(device.createCommandEncoder).toBeDefined();
        
        device.destroy();
    });
    
    describe("WebGPU Functionality", () => {
        it("should compile a basic compute shader", async () => {
            const adapter = await navigator.gpu.requestAdapter();
            const device = await adapter!.requestDevice();
            
            const shaderCode = `
                @compute @workgroup_size(1)
                fn main() {
                    // Empty compute shader
                }
            `;
            
            const shaderModule = device.createShaderModule({
                code: shaderCode
            });
            
            expect(shaderModule).toBeDefined();
            device.destroy();
        });
        
        it("should execute a simple compute operation", async () => {
            const adapter = await navigator.gpu.requestAdapter();
            const device = await adapter!.requestDevice();
            
            // Create a shader that writes a value
            const shaderModule = device.createShaderModule({
                code: `
                    @group(0) @binding(0) var<storage, read_write> output: f32;
                    
                    @compute @workgroup_size(1)
                    fn main() {
                        output = 42.0;
                    }
                `
            });
            
            // Create buffer
            const buffer = device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            });
            
            const stagingBuffer = device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
            });
            
            // Create pipeline and bind group
            const pipeline = device.createComputePipeline({
                layout: "auto",
                compute: {
                    module: shaderModule,
                    entryPoint: "main"
                }
            });
            
            const bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [{
                    binding: 0,
                    resource: { buffer }
                }]
            });
            
            // Execute
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(1);
            pass.end();
            
            encoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, 4);
            device.queue.submit([encoder.finish()]);
            
            // Read result
            await stagingBuffer.mapAsync(GPUMapMode.READ);
            const data = new Float32Array(stagingBuffer.getMappedRange());
            
            expect(data[0]).toBe(42.0);
            
            stagingBuffer.unmap();
            device.destroy();
        });
    });
});