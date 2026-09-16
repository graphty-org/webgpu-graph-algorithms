/**
 * The run options every algorithm function accepts beside its own (spec 3.3, design 10.7; contract 3.3).
 * Types only.
 */

/** Options every algorithm function accepts in addition to its own (spec 3.3, design 10.7). */
export interface GpuRunOptions {
    /** A preallocated destination of exact length; E_INVALID_ARGUMENT when the length does not match. */
    readonly dest?: Float32Array | Uint32Array | undefined;
    /** Checked between batches; E_ABORTED. */
    readonly signal?: AbortSignal | undefined;
    readonly onProgress?: ((done: number, total: number) => void) | undefined;
}
