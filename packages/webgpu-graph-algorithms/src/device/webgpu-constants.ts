/**
 * GPUBufferUsage / GPUMapMode / GPUShaderStage as numbers (spec 2.1 rule 1; contract 3.4): under Dawn-node the
 * global namespaces do not exist until `Object.assign(globalThis, dawn.globals)` has run, so the core never reads
 * them at module top level and keeps its own copies here. Each field names the WebGPU spec constant it mirrors;
 * test/device/constants.test.ts compares every field with the runtime globals so a drift is caught.
 */

/** GPUBufferUsage bits as numbers so the core never reads the global at module top level (spec 2.1 rule 1). */
export const BufferUsage: Readonly<{
    MAP_READ: 0x0001;
    MAP_WRITE: 0x0002;
    COPY_SRC: 0x0004;
    COPY_DST: 0x0008;
    INDEX: 0x0010;
    VERTEX: 0x0020;
    UNIFORM: 0x0040;
    STORAGE: 0x0080;
    INDIRECT: 0x0100;
    QUERY_RESOLVE: 0x0200;
}> = Object.freeze({
    MAP_READ: 0x0001, // GPUBufferUsage.MAP_READ
    MAP_WRITE: 0x0002, // GPUBufferUsage.MAP_WRITE
    COPY_SRC: 0x0004, // GPUBufferUsage.COPY_SRC
    COPY_DST: 0x0008, // GPUBufferUsage.COPY_DST
    INDEX: 0x0010, // GPUBufferUsage.INDEX
    VERTEX: 0x0020, // GPUBufferUsage.VERTEX
    UNIFORM: 0x0040, // GPUBufferUsage.UNIFORM
    STORAGE: 0x0080, // GPUBufferUsage.STORAGE
    INDIRECT: 0x0100, // GPUBufferUsage.INDIRECT
    QUERY_RESOLVE: 0x0200, // GPUBufferUsage.QUERY_RESOLVE
});

/** GPUMapMode bits. */
export const MapMode: Readonly<{ READ: 0x0001; WRITE: 0x0002 }> = Object.freeze({
    READ: 0x0001, // GPUMapMode.READ
    WRITE: 0x0002, // GPUMapMode.WRITE
});

/** GPUShaderStage bits. */
export const ShaderStage: Readonly<{ VERTEX: 0x1; FRAGMENT: 0x2; COMPUTE: 0x4 }> = Object.freeze({
    VERTEX: 0x1, // GPUShaderStage.VERTEX
    FRAGMENT: 0x2, // GPUShaderStage.FRAGMENT
    COMPUTE: 0x4, // GPUShaderStage.COMPUTE
});
