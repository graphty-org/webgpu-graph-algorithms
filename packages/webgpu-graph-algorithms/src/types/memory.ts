/**
 * The two types the memory AND the kernel layers share (spec 4.1, 4.2; contract 3.3 CONTRACT DECISION):
 * they live here so src/kernel/kernel.ts never imports src/memory/** for a type and the layer graph has no
 * P1 cycle (test/layers.test.ts). Types only; GPUBuffer is the ambient @webgpu/types global.
 */

/** One 64-arc-aligned window of colIdx / weights (spec 4.2). P1-P3 PLAN windows; executing them is P4. */
export interface ArcWindow {
    readonly start: number;
    readonly end: number;
    readonly rowFirst: number;
    readonly rowLast: number;
    readonly bufferIndex: number;
    readonly offset: number;
}

/** A storage-buffer range (spec 4.1); `window` is non-null only for a binding of one arc window. */
export interface Binding {
    readonly buffer: GPUBuffer;
    readonly offset: number;
    readonly size: number;
    readonly window: ArcWindow | null;
}
