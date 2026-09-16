/**
 * Error scopes (spec 5.7; contract 3.4): a validation scope that turns a captured GPUValidationError into
 * E_VALIDATION { label, message }, the compilation-info formatter with body-relative line numbers, and the
 * AllocationTracker that wraps large createBuffer calls in an out-of-memory scope and keeps the resident-byte
 * bookkeeping the residency, the pool and the staging ring share (it is owned by the context).
 */

import { OOM_SCOPE_THRESHOLD_BYTES } from "../constants.js";
import { WebGpuGraphError } from "../errors.js";

/**
 * Runs `fn` inside pushErrorScope("validation"); a captured error becomes E_VALIDATION { label, message }
 * thrown from the returned promise (spec 5.7). An error `fn` itself throws is rethrown unchanged after the
 * scope is popped, so the device's scope stack stays balanced.
 * @param device - the device
 * @param label - names the operation in the error (every object the package creates is labelled)
 * @param fn - the work; may return a promise
 * @returns the value of `fn`
 */
export async function withValidationScope<T>(device: GPUDevice, label: string, fn: () => Promise<T> | T): Promise<T> {
    device.pushErrorScope("validation");
    let result: T;
    try {
        result = await fn();
    } catch (err) {
        await device.popErrorScope().catch(() => null);
        throw err;
    }
    const error = await device.popErrorScope();
    if (error !== null) {
        throw new WebGpuGraphError("E_VALIDATION", `${label}: ${error.message}`, { label, message: error.message });
    }
    return result;
}

/**
 * Formats GPUCompilationInfo messages with line numbers relative to the kernel BODY: `preludeLines` (the
 * lines the composer emitted before the body) is subtracted, a message located inside those lines is
 * reported as `prelude:<line>`, and a message without a location (lineNum 0) carries no position (spec 5.1).
 * @param info - the compilation info of the shader module
 * @param preludeLines - the number of emitted lines before the body's first line
 * @returns one string per message: `<type> <line>:<column> <message>`
 */
export function formatCompilationInfo(info: GPUCompilationInfo, preludeLines: number): string[] {
    return info.messages.map((m) => {
        if (m.lineNum === 0) {
            return `${m.type} ${m.message}`;
        }
        const line = m.lineNum - preludeLines;
        if (line >= 1) {
            return `${m.type} ${line}:${m.linePos} ${m.message}`;
        }
        return `${m.type} prelude:${m.lineNum}:${m.linePos} ${m.message}`;
    });
}

/**
 * Creates buffers, wrapping sizes >= OOM_SCOPE_THRESHOLD_BYTES in an "out-of-memory" scope whose pop is
 * collected; `check()` surfaces the first OOM (spec 4.4, 5.7). Owned by the context, shared by the
 * residency and the pool.
 */
export class AllocationTracker {
    private readonly device: GPUDevice;
    private readonly threshold: number;
    private readonly live = new Map<GPUBuffer, number>();
    private residentBytes = 0;
    private pending: Promise<void>[] = [];
    private failure: WebGpuGraphError | null = null;
    private serial = 0;

    /**
     * Creates a tracker over one device.
     * @param device - the device buffers are created on
     * @param thresholdBytes - the byte size from which createBuffer runs inside an out-of-memory scope (default OOM_SCOPE_THRESHOLD_BYTES)
     */
    constructor(device: GPUDevice, thresholdBytes?: number) {
        this.device = device;
        this.threshold = thresholdBytes ?? OOM_SCOPE_THRESHOLD_BYTES;
    }

    /**
     * createBuffer with the scope rule; every buffer is labelled (an unlabelled descriptor becomes `buffer-<n>`).
     * @param descriptor - the buffer descriptor
     * @returns the buffer
     */
    createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer {
        this.serial += 1;
        const label =
            descriptor.label === undefined || descriptor.label === "" ? `buffer-${this.serial}` : descriptor.label;
        const { size } = descriptor;
        const scoped = size >= this.threshold;
        if (scoped) {
            this.device.pushErrorScope("out-of-memory");
        }
        const buffer = this.device.createBuffer({ ...descriptor, label });
        this.live.set(buffer, size);
        this.residentBytes += size;
        if (scoped) {
            const pop = this.device.popErrorScope().then(
                (error) => {
                    if (error === null) {
                        return;
                    }
                    if (this.live.has(buffer)) {
                        this.live.delete(buffer);
                        this.residentBytes -= size;
                    }
                    if (this.failure === null) {
                        this.failure = new WebGpuGraphError(
                            "E_OUT_OF_MEMORY",
                            `out of memory creating "${label}" (${size} bytes, ${this.residentBytes} resident): ${error.message}`,
                            { requested: size, resident: this.residentBytes, label },
                        );
                    }
                },
                () => undefined,
            );
            this.pending.push(pop);
        }
        return buffer;
    }

    /**
     * Destroys a buffer created here and decrements `resident`; a buffer the tracker does not know is destroyed
     * without touching the bookkeeping, so destroying twice is harmless.
     * @param buffer - the buffer to destroy
     */
    destroy(buffer: GPUBuffer): void {
        const size = this.live.get(buffer);
        if (size !== undefined) {
            this.live.delete(buffer);
            this.residentBytes -= size;
        }
        buffer.destroy();
    }

    /**
     * Awaits every outstanding scope pop; the first OOM -> E_OUT_OF_MEMORY { requested, resident, label }; later
     * calls after a failure throw the same error again until `reset()`.
     */
    async check(): Promise<void> {
        while (this.pending.length > 0) {
            const batch = this.pending.splice(0);
            await Promise.all(batch);
        }
        if (this.failure !== null) {
            throw this.failure;
        }
    }

    /** Forgets a recorded OOM after the caller released what it allocated. */
    reset(): void {
        this.failure = null;
    }

    /**
     * Bytes of live buffers created through this tracker.
     * @returns the resident byte count
     */
    get resident(): number {
        return this.residentBytes;
    }

    /**
     * Number of live buffers created through this tracker.
     * @returns the live buffer count
     */
    get liveBuffers(): number {
        return this.live.size;
    }
}
