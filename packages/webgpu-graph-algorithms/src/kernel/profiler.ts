/**
 * Timestamp profiling of spec 5.5: when "timestamp-query" was granted the context owns a Profiler
 * whose query set (PROFILER_QUERY_SLOTS slots, two per pass) is written by the timestampWrites of
 * every CommandBatch pass; `resolveInto(batch)` records the resolve plus a copy into the batch's
 * staging slot and `timings(bytes, request)` decodes `{ label, ns }` per pass from the batch's
 * readback bytes. Chromium quantises timestamps to 100 us (`quantised: true`); Dawn-node reports
 * 1,024 ns ticks (spec 2.6). The profiler never throws: a full query set drops timings and a device
 * without the feature disables it. Slot pairs are handed out in order and the cursor restarts at 0
 * after every resolve: queue order guarantees the resolve of batch A reads its slots before batch B
 * overwrites them, and one resolve buffer suffices for the same reason.
 */

import { PROFILER_QUERY_SLOTS } from "../constants.js";
import { BufferUsage } from "../device/webgpu-constants.js";
import type { CommandBatch, ReadbackRequest } from "./batch.js";

/** One resolved pass timing. */
export interface PassTiming {
    readonly label: string;
    readonly ns: number;
}

/** A pass whose begin / end slots were handed out and not yet resolved. */
interface PendingPass {
    readonly label: string;
    readonly begin: number;
}

/** Timestamp profiling when "timestamp-query" was granted (spec 5.5). */
export class Profiler {
    /** True when the feature was granted and a query set exists. */
    readonly enabled: boolean;
    /** True in browsers (100 us quantisation), false under Dawn-node (spec 2.6). */
    readonly quantised: boolean;
    private readonly slots: number;
    private readonly querySet: GPUQuerySet | null;
    private readonly resolveBuffer: GPUBuffer | null;
    private readonly labels = new WeakMap<ReadbackRequest, readonly string[]>();
    private pending: PendingPass[] = [];
    private cursor = 0;
    private destroyed = false;

    /**
     * Creates the query set and the resolve buffer when usable; `slots` defaults to PROFILER_QUERY_SLOTS = 256
     * (spec 5.5); 2 query slots per pass. `enabled` is also false when the device lacks "timestamp-query" or
     * `slots < 2`, so no caller can make the profiler raise a validation error.
     * @param device - the device the query set is created on
     * @param enabled - whether the caller wants profiling (the context passes `features.has("timestamp-query")`)
     * @param quantised - true in browsers
     * @param slots - query slots, two per pass (default PROFILER_QUERY_SLOTS)
     */
    constructor(device: GPUDevice, enabled: boolean, quantised: boolean, slots?: number) {
        const count = slots ?? PROFILER_QUERY_SLOTS;
        this.quantised = quantised;
        this.slots = count;
        const usable = enabled && count >= 2 && device.features.has("timestamp-query");
        this.enabled = usable;
        if (usable) {
            this.querySet = device.createQuerySet({ label: "profiler/timestamps", type: "timestamp", count });
            this.resolveBuffer = device.createBuffer({
                label: "profiler/resolve",
                size: count * 8,
                usage: BufferUsage.QUERY_RESOLVE | BufferUsage.COPY_SRC,
            });
        } else {
            this.querySet = null;
            this.resolveBuffer = null;
        }
    }

    /**
     * The timestampWrites descriptor for a new pass, or undefined when disabled / out of slots.
     * @param label - the pass name reported by `timings()`
     * @returns the descriptor a CommandBatch puts on beginComputePass, or undefined
     */
    beginPass(label: string): GPUComputePassTimestampWrites | undefined {
        if (this.querySet === null || this.destroyed || this.cursor + 2 > this.slots) {
            return undefined;
        }
        const begin = this.cursor;
        this.cursor += 2;
        this.pending.push({ label, begin });
        return { querySet: this.querySet, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: begin + 1 };
    }

    /**
     * Records the resolve + copy into the batch's staging slot; returns the request's byte range, or null when
     * nothing was written (disabled, destroyed, or no pass begun since the last resolve).
     * @param batch - the batch whose passes were timed (it must still be open)
     * @returns the readback request `timings()` decodes, or null
     */
    resolveInto(batch: CommandBatch): ReadbackRequest | null {
        if (this.querySet === null || this.resolveBuffer === null || this.destroyed || this.pending.length === 0) {
            return null;
        }
        const count = this.cursor;
        batch.resolveQuerySet(this.querySet, 0, count, this.resolveBuffer);
        const request = batch.readback(this.resolveBuffer, 0, count * 8);
        this.labels.set(
            request,
            this.pending.map((pass) => pass.label),
        );
        this.pending = [];
        this.cursor = 0;
        return request;
    }

    /**
     * Decodes the timings of a batch from its readback bytes.
     * @param bytes - the resolved readback of the batch
     * @param request - the request `resolveInto()` returned for it
     * @returns one timing per pass in pass order (empty for a request this profiler did not create)
     */
    timings(bytes: ArrayBuffer, request: ReadbackRequest): readonly PassTiming[] {
        const labels = this.labels.get(request);
        if (labels === undefined) {
            return [];
        }
        const view = new DataView(bytes);
        const out: PassTiming[] = [];
        for (let i = 0; i < labels.length; i++) {
            const at = request.offset + i * 16;
            if (at + 16 > bytes.byteLength) {
                break;
            }
            const begin = view.getBigUint64(at, true);
            const end = view.getBigUint64(at + 8, true);
            out.push({ label: labels[i], ns: Number(end - begin) });
        }
        return out;
    }

    /** Destroys the query set and the resolve buffer; idempotent. */
    destroy(): void {
        if (this.destroyed) {
            return;
        }
        this.destroyed = true;
        this.pending = [];
        this.cursor = 0;
        if (this.querySet !== null) {
            this.querySet.destroy();
        }
        if (this.resolveBuffer !== null) {
            this.resolveBuffer.destroy();
        }
    }
}
