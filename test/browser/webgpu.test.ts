import { describe, it, expect } from "vitest";
import { withTestDevice } from "../helpers/test-utils";
import { createBuffer, readBuffer, expectFloatArraysEqual } from "../helpers/webgpu";

describe("WebGPU Core Functionality", () => {
    it("should have WebGPU API available", () => {
        expect(navigator.gpu).toBeDefined();
        expect(globalThis.webGPUContext).toBeDefined();
        expect(globalThis.webGPUContext.adapter).toBeDefined();
        expect(globalThis.webGPUContext.device).toBeDefined();
    });
    
    it("should execute a compute shader that doubles values", async () => {
        await withTestDevice(async (device) => {
            // Simple shader that doubles values
            const shaderCode = `
                @group(0) @binding(0) var<storage, read> input: array<f32>;
                @group(0) @binding(1) var<storage, read_write> output: array<f32>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let index = global_id.x;
                    if (index >= arrayLength(&input)) {
                        return;
                    }
                    output[index] = input[index] * 2.0;
                }
            `;
            
            const shaderModule = device.createShaderModule({
                label: "Double values shader",
                code: shaderCode
            });
            
            // Test data
            const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
            
            // Create buffers using helper
            const inputBuffer = createBuffer(device, data, 
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            );
            
            const outputBuffer = device.createBuffer({
                label: "Output buffer",
                size: data.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            });
            
            // Create pipeline
            const computePipeline = device.createComputePipeline({
                label: "Double values pipeline",
                layout: "auto",
                compute: {
                    module: shaderModule,
                    entryPoint: "main"
                }
            });
            
            const bindGroup = device.createBindGroup({
                label: "Double values bind group",
                layout: computePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: inputBuffer } },
                    { binding: 1, resource: { buffer: outputBuffer } }
                ]
            });
            
            // Execute
            const commandEncoder = device.createCommandEncoder();
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(computePipeline);
            computePass.setBindGroup(0, bindGroup);
            computePass.dispatchWorkgroups(Math.ceil(data.length / 64));
            computePass.end();
            
            device.queue.submit([commandEncoder.finish()]);
            
            // Read results using helper
            const result = await readBuffer(device, outputBuffer, data.byteLength);
            const expected = new Float32Array([2, 4, 6, 8, 10, 12, 14, 16]);
            
            // Compare using helper
            expectFloatArraysEqual(result, expected);
        });
    });
    
    it("should handle large data arrays", async () => {
        await withTestDevice(async (device) => {
            const size = 10000;
            const data = new Float32Array(size);
            for (let i = 0; i < size; i++) {
                data[i] = i;
            }
            
            // Simple shader that adds 1 to each value
            const shaderModule = device.createShaderModule({
                code: `
                    @group(0) @binding(0) var<storage, read_write> data: array<f32>;
                    
                    @compute @workgroup_size(256)
                    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                        if (id.x >= ${size}u) { return; }
                        data[id.x] = data[id.x] + 1.0;
                    }
                `
            });
            
            const buffer = createBuffer(device, data, 
                GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            );
            
            const pipeline = device.createComputePipeline({
                layout: "auto",
                compute: { module: shaderModule, entryPoint: "main" }
            });
            
            const bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [{ binding: 0, resource: { buffer } }]
            });
            
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(Math.ceil(size / 256));
            pass.end();
            
            device.queue.submit([encoder.finish()]);
            
            const result = await readBuffer(device, buffer, data.byteLength);
            
            // Check a sample of values
            expect(result[0]).toBe(1);
            expect(result[100]).toBe(101);
            expect(result[9999]).toBe(10000);
        });
    });
});