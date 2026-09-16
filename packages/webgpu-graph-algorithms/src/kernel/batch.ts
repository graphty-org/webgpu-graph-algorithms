/**
 * The CommandBatch of spec 5.8: one GPUCommandEncoder, one compute pass per "phase" (a pass may hold
 * many dispatches; dispatches inside a pass are ordered), the staging copies of every readback
 * request at the end, and ONE queue.submit. The batch borrows one slot of the Readback ring for its
 * lifetime (spec 4.4) and returns it when its readback settles -- resolved, discarded or rejected --
 * so the ring never grows because of a slot left pending. mapAsync on that slot is the completion
 * signal; a batch never calls onSubmittedWorkDone.
 *
 * Errors (spec 5.7): under Dawn-node an uncapturederror is delivered SYNCHRONOUSLY inside the API
 * call that fails, so an error pending in the context's slot when submit() runs belongs to this batch
 * and rejects ITS OWN readback (an invalid command buffer is a no-op submit; the staging bytes would be
 * stale). In browsers delivery is asynchronous: the batch takes the slot again after its map completes,
 * and whatever arrives later is thrown by the context's next public call. Device loss is raced against
 * the map, so a readback rejects E_DEVICE_LOST whether the runtime rejects, resolves or never settles
 * the pending map of a destroyed buffer.
 */

import type { AllocationTracker } from "../device/error-scope.js";
import { deviceLostError } from "../device/lost.js";
import { MapMode } from "../device/webgpu-constants.js";
import { isWebGpuGraphError, WebGpuGraphError } from "../errors.js";
import type { Readback, StagingSlot } from "../memory/readback.js";
import type { Binding } from "../types/memory.js";
import type { Profiler } from "./profiler.js";

/** How long a batch waits for the context's loss fan-out after its map rejected with a foreign error (a destroyed buffer rejects before device.lost settles). */
const LOSS_GRACE_MS = 2000;

/**
 * A plain message for a thrown value (the house pattern of graph-io's report.ts).
 * @param error - whatever the runtime rejected with
 * @returns the Error message, the string itself, or a typeof note
 */
function messageOf(error: unknown): string {
    if (error instanceof Error) {
        return error.message;
    }
    if (typeof error === "string") {
        return error;
    }
    return `non-error thrown (${typeof error})`;
}

/** What a CommandBatch needs of its owner (GpuContext satisfies it structurally; kernel/ never imports context.ts). */
export interface BatchHost {
    readonly device: GPUDevice;
    readonly readback: Readback;
    readonly profiler: Profiler | null;
    readonly allocator: AllocationTracker;
    assertReady(): void;
    takePendingError(): WebGpuGraphError | null;
    nextBatchId(): number;
    /** Registers a device-loss listener and returns the unregister function (GpuContext.onLost, contract 3.5): the batch races its map against loss. PLAN DECISION 1: an amendment to contract 3.9, which declares the seven members above. */
    onLost(listener: (info: GPUDeviceLostInfo) => void): () => void;
}

/** A readback scheduled into the batch's staging slot: `offset` is the slot-relative byte offset the caller reads at. */
export interface ReadbackRequest {
    readonly src: GPUBuffer;
    readonly srcOffset: number;
    readonly byteLength: number;
    readonly offset: number;
}

/** What submit() returns (spec 5.8). `readback` resolves with the mapped bytes COPIED out (one ArrayBuffer holding every request at its offset) after allocator.check(); it rejects with E_VALIDATION { batchId } when the pending-error slot held an error right after submit and with E_DEVICE_LOST after loss; a discarded batch resolves an empty ArrayBuffer (see discard()). */
export interface SubmittedBatch {
    readonly id: number;
    readonly generation: number;
    readonly readback: Promise<ArrayBuffer>;
    /** Marks the batch stale: its readback still awaits mapAsync (so the slot is returned) but resolves with an empty ArrayBuffer and the caller ignores it (spec 7.19 item 6). */
    discard(): void;
}

/** The discard flag shared by a handle and its readback. */
interface DiscardState {
    discarded: boolean;
}

/** Records dispatches / copies into one encoder and submits once (spec 5.8). */
export class CommandBatch {
    readonly id: number;
    readonly generation: number;
    readonly label: string;
    private readonly host: BatchHost;
    private readonly encoder: GPUCommandEncoder;
    private readonly requests: ReadbackRequest[] = [];
    private openPass: GPUComputePassEncoder | null = null;
    private dispatchCount = 0;
    private stagingBytes = 0;
    private slot: StagingSlot | null = null;
    private submitted = false;

    /**
     * Creates the encoder; the id comes from host.nextBatchId().
     * @param host - the owner (the GpuContext)
     * @param label - names the encoder, its passes and the E_VALIDATION details
     * @param generation - the simulation generation the batch carries (default 0)
     */
    constructor(host: BatchHost, label: string, generation?: number) {
        this.host = host;
        this.id = host.nextBatchId();
        this.generation = generation ?? 0;
        this.label = label;
        this.encoder = host.device.createCommandEncoder({ label: `batch/${label}#${this.id}` });
    }

    /**
     * Dispatch count recorded so far (tests bound it).
     * @returns the number of dispatchWorkgroups / dispatchWorkgroupsIndirect calls on this batch's passes
     */
    get dispatches(): number {
        return this.dispatchCount;
    }

    /**
     * Begins a compute pass (ending the previous one); with a profiler present the pass carries timestampWrites and
     * `label` names it in Profiler.resolveInto / timings.
     * @param label - the pass name
     * @returns the pass encoder kernels dispatch into
     */
    pass(label: string): GPUComputePassEncoder {
        this.assertOpen("pass");
        this.endPass();
        const descriptor: GPUComputePassDescriptor = { label: `batch/${this.label}#${this.id}/${label}` };
        const timestampWrites = this.host.profiler?.beginPass(label);
        if (timestampWrites !== undefined) {
            descriptor.timestampWrites = timestampWrites;
        }
        const pass = this.encoder.beginComputePass(descriptor);
        this.countDispatches(pass);
        this.openPass = pass;
        return pass;
    }

    /** Ends the open pass, if any. */
    endPass(): void {
        if (this.openPass !== null) {
            this.openPass.end();
            this.openPass = null;
        }
    }

    /**
     * copyBufferToBuffer between two bindings (after endPass).
     * @param src - the source range
     * @param dst - the destination range
     * @param byteLength - a positive multiple of 4 no larger than either binding
     */
    copy(src: Binding, dst: Binding, byteLength: number): void {
        this.assertOpen("copy");
        this.checkByteLength(byteLength);
        if (byteLength > src.size || byteLength > dst.size) {
            throw new WebGpuGraphError(
                "E_INVALID_ARGUMENT",
                `copy of ${byteLength} bytes exceeds a binding (src ${src.size}, dst ${dst.size})`,
                {
                    argument: "byteLength",
                    value: byteLength,
                    expected: `<= ${Math.min(src.size, dst.size)}`,
                },
            );
        }
        this.endPass();
        this.encoder.copyBufferToBuffer(src.buffer, src.offset, dst.buffer, dst.offset, byteLength);
    }

    /**
     * Schedules a copy of `byteLength` bytes from `src` into the borrowed staging slot; the slot is borrowed at the
     * first request and re-borrowed larger when the sum of requests outgrows it (nothing is recorded before submit).
     * @param src - the source buffer
     * @param srcOffset - a non-negative multiple of 4
     * @param byteLength - a positive multiple of 4 with srcOffset + byteLength <= src.size
     * @returns the request; `offset` is where the caller reads inside the readback bytes
     */
    readback(src: GPUBuffer, srcOffset: number, byteLength: number): ReadbackRequest {
        this.assertOpen("readback");
        this.checkByteLength(byteLength);
        if (!Number.isInteger(srcOffset) || srcOffset < 0 || srcOffset % 4 !== 0 || srcOffset + byteLength > src.size) {
            throw new WebGpuGraphError(
                "E_INVALID_ARGUMENT",
                `readback of [${srcOffset}, ${srcOffset + byteLength}) is outside the ${src.size}-byte source or not 4-aligned`,
                {
                    argument: "srcOffset",
                    value: srcOffset,
                    expected: `a multiple of 4 with srcOffset + byteLength <= ${src.size}`,
                },
            );
        }
        const request: ReadbackRequest = { src, srcOffset, byteLength, offset: this.stagingBytes };
        this.requests.push(request);
        this.stagingBytes += byteLength;
        this.ensureSlot(this.stagingBytes);
        return request;
    }

    /**
     * Records `resolveQuerySet` after ending the open pass; used by Profiler.resolveInto only.
     * @internal
     * @param querySet - the timestamp query set
     * @param firstQuery - first query index
     * @param queryCount - number of queries
     * @param destination - a QUERY_RESOLVE | COPY_SRC buffer written at offset 0
     */
    resolveQuerySet(querySet: GPUQuerySet, firstQuery: number, queryCount: number, destination: GPUBuffer): void {
        this.assertOpen("resolveQuerySet");
        this.endPass();
        this.encoder.resolveQuerySet(querySet, firstQuery, queryCount, destination, 0);
    }

    /**
     * Ends passes, records the staging copies, submits, checks the pending-error slot (spec 5.7), returns the handle.
     * A batch can be submitted once; submit() after device loss throws E_DEVICE_LOST.
     * @returns the submitted-batch handle
     */
    submit(): SubmittedBatch {
        this.assertOpen("submit");
        this.submitted = true;
        this.endPass();
        const { slot } = this;
        const pendingBefore = this.host.takePendingError();
        try {
            this.host.assertReady();
        } catch (error) {
            this.releaseSlot();
            throw error;
        }
        if (slot !== null) {
            for (const request of this.requests) {
                this.encoder.copyBufferToBuffer(
                    request.src,
                    request.srcOffset,
                    slot.buffer,
                    request.offset,
                    request.byteLength,
                );
            }
        }
        const commandBuffer = this.encoder.finish({ label: `batch/${this.label}#${this.id}` });
        this.host.device.queue.submit([commandBuffer]);
        // the slot is drained a second time so a finish() / submit() error raised by a command buffer the first error
        // already invalidated never leaks to the next public call; the FIRST error is the one reported
        const pendingAfter = this.host.takePendingError();
        const pending = pendingBefore ?? pendingAfter;
        const state: DiscardState = { discarded: false };
        const readback = pending === null ? this.awaitReadback(slot, state) : this.rejectReadback(pending);
        return {
            id: this.id,
            generation: this.generation,
            readback,
            discard: (): void => {
                state.discarded = true;
            },
        };
    }

    private rejectReadback(pending: WebGpuGraphError): Promise<ArrayBuffer> {
        this.releaseSlot();
        return Promise.reject(this.attribute(pending));
    }

    private async awaitReadback(slot: StagingSlot | null, state: DiscardState): Promise<ArrayBuffer> {
        if (slot === null) {
            await this.host.allocator.check();
            return new ArrayBuffer(0);
        }
        const total = this.stagingBytes;
        let unregister: () => void = (): void => undefined;
        const lost = new Promise<never>((_resolve, reject) => {
            unregister = this.host.onLost((info) => {
                reject(deviceLostError(info));
            });
        });
        void lost.catch(() => undefined);
        const mapped = slot.buffer.mapAsync(MapMode.READ, 0, total);
        void mapped.catch(() => undefined);
        let bytes: ArrayBuffer;
        try {
            await Promise.race([mapped, lost]);
            this.host.assertReady();
            bytes = state.discarded ? new ArrayBuffer(0) : slot.buffer.getMappedRange(0, total).slice(0);
        } catch (error) {
            this.releaseSlot();
            throw await this.classify(error, lost);
        } finally {
            unregister();
        }
        this.releaseSlot();
        await this.host.allocator.check();
        return bytes;
    }

    private async classify(error: unknown, lost: Promise<never>): Promise<WebGpuGraphError> {
        if (isWebGpuGraphError(error)) {
            return error.code === "E_DEVICE_LOST" || error.code === "E_DISPOSED" ? error : this.attribute(error);
        }
        try {
            this.host.assertReady();
        } catch (state) {
            if (isWebGpuGraphError(state)) {
                return state.code === "E_DEVICE_LOST" || state.code === "E_DISPOSED" ? state : this.attribute(state);
            }
        }
        let timer: ReturnType<typeof setTimeout> | undefined;
        const grace = new Promise<null>((resolve) => {
            timer = setTimeout(() => {
                resolve(null);
            }, LOSS_GRACE_MS);
        });
        const late = await Promise.race([
            lost.then(
                () => null,
                (reason: unknown) => reason,
            ),
            grace,
        ]);
        clearTimeout(timer);
        if (isWebGpuGraphError(late)) {
            return late;
        }
        const message = messageOf(error);
        return new WebGpuGraphError(
            "E_VALIDATION",
            `mapAsync failed for batch ${this.id} "${this.label}": ${message}`,
            {
                label: this.label,
                message,
                batchId: this.id,
            },
        );
    }

    private attribute(error: WebGpuGraphError): WebGpuGraphError {
        return new WebGpuGraphError(error.code, `${error.message} (delivered to batch ${this.id} "${this.label}")`, {
            ...error.details,
            batchId: this.id,
            batchLabel: this.label,
        });
    }

    private ensureSlot(byteLength: number): void {
        if (this.slot !== null && this.slot.capacity >= byteLength) {
            return;
        }
        this.releaseSlot();
        this.slot = this.host.readback.borrowSlot(byteLength);
    }

    private releaseSlot(): void {
        if (this.slot === null) {
            return;
        }
        const { slot } = this;
        this.slot = null;
        try {
            this.host.readback.returnSlot(slot);
        } catch (error) {
            if (!isWebGpuGraphError(error) || error.code !== "E_DISPOSED") {
                throw error;
            }
        }
    }

    private countDispatches(pass: GPUComputePassEncoder): void {
        const direct = pass.dispatchWorkgroups.bind(pass);
        const indirect = pass.dispatchWorkgroupsIndirect.bind(pass);
        pass.dispatchWorkgroups = (x: number, y?: number, z?: number): undefined => {
            this.dispatchCount += 1;
            direct(x, y, z);
            return undefined;
        };
        pass.dispatchWorkgroupsIndirect = (buffer: GPUBuffer, offset: number): undefined => {
            this.dispatchCount += 1;
            indirect(buffer, offset);
            return undefined;
        };
    }

    private assertOpen(operation: string): void {
        if (this.submitted) {
            throw new WebGpuGraphError(
                "E_INVALID_ARGUMENT",
                `${operation}() after submit() on batch ${this.id} "${this.label}"`,
                {
                    argument: operation,
                    value: this.id,
                    expected: "a batch that has not been submitted",
                },
            );
        }
    }

    private checkByteLength(byteLength: number): void {
        if (!Number.isInteger(byteLength) || byteLength <= 0 || byteLength % 4 !== 0) {
            throw new WebGpuGraphError(
                "E_INVALID_ARGUMENT",
                `byteLength ${byteLength} is not a positive multiple of 4`,
                {
                    argument: "byteLength",
                    value: byteLength,
                    expected: "a positive multiple of 4",
                },
            );
        }
    }
}
