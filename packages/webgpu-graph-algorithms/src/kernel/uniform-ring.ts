/**
 * The UniformRing of spec 5.3: one UNIFORM buffer holding UNIFORM_SLOT_BYTES-stride slots for the
 * per-iteration params of a batch. A batch reserves k contiguous slots, writes their values into the
 * host shadow, flushes them with ONE queue.writeBuffer and records k dispatches that select their slot
 * through a dynamic offset -- no per-iteration writeBuffer and no host round trip. The stride is 256
 * because minUniformBufferOffsetAlignment is 256 on every runtime the package targets (the Dawn-node
 * ADAPTER advertises 64, which the package never requests; spec 2.6).
 */

import { UNIFORM_SLOT_BYTES } from "../constants.js";
import type { AllocationTracker } from "../device/error-scope.js";
import { BufferUsage } from "../device/webgpu-constants.js";
import { WebGpuGraphError } from "../errors.js";
import type { Binding } from "../types/memory.js";
import type { UniformBlock, UniformValues } from "./struct-block.js";

/** One UNIFORM buffer with UNIFORM_SLOT_BYTES-stride slots for the per-iteration params of a batch (spec 5.3). */
export class UniformRing {
    /** Number of slots. */
    readonly slots: number;
    private readonly device: GPUDevice;
    private readonly allocator: AllocationTracker;
    private readonly buffer: GPUBuffer;
    private readonly label: string;
    private readonly shadow: ArrayBuffer;
    private readonly view: DataView;
    private next = 0;
    private dirtyLo = -1;
    private dirtyHi = -1;
    private destroyed = false;

    /**
     * Creates the ring buffer (`slots x UNIFORM_SLOT_BYTES` bytes, UNIFORM | COPY_DST) through the allocator.
     * @param device - the device the buffer belongs to
     * @param allocator - the context's OOM-scoped allocator (the buffer is destroyed through it)
     * @param slots - number of slots; E_INVALID_ARGUMENT unless a positive integer
     * @param label - the buffer label
     */
    constructor(device: GPUDevice, allocator: AllocationTracker, slots: number, label: string) {
        if (!Number.isInteger(slots) || slots < 1) {
            throw new WebGpuGraphError(
                "E_INVALID_ARGUMENT",
                `UniformRing "${label}": slots must be a positive integer, got ${slots}`,
                {
                    argument: "slots",
                    value: slots,
                    expected: "an integer >= 1",
                },
            );
        }
        this.device = device;
        this.allocator = allocator;
        this.slots = slots;
        this.label = label;
        this.shadow = new ArrayBuffer(slots * UNIFORM_SLOT_BYTES);
        this.view = new DataView(this.shadow);
        this.buffer = allocator.createBuffer({
            label,
            size: slots * UNIFORM_SLOT_BYTES,
            usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
        });
    }

    /**
     * The whole-buffer binding a kernel binds once (size = the block's byteLength; the slot is the dynamic offset).
     * @param block - the params block the kernel declares for this binding
     * @returns the binding at offset 0 of the ring buffer
     */
    binding(block: UniformBlock): Binding {
        this.assertLive();
        this.assertFits(block);
        return { buffer: this.buffer, offset: 0, size: block.byteLength, window: null };
    }

    /**
     * Byte offset of a slot (the dynamic offset a dispatch passes for it).
     * @param slot - a slot index in [0, slots)
     * @returns slot x UNIFORM_SLOT_BYTES
     */
    offsetOf(slot: number): number {
        this.checkSlot(slot);
        return slot * UNIFORM_SLOT_BYTES;
    }

    /**
     * Reserves `count` contiguous slots for a batch, wrapping to 0 when the tail is too short; E_INVALID_ARGUMENT when count > slots.
     * @param count - slots needed, in [1, slots]
     * @returns the first slot of the reservation
     */
    reserve(count: number): number {
        this.assertLive();
        if (!Number.isInteger(count) || count < 1 || count > this.slots) {
            throw new WebGpuGraphError(
                "E_INVALID_ARGUMENT",
                `UniformRing "${this.label}": cannot reserve ${count} of ${this.slots} slots`,
                {
                    argument: "count",
                    value: count,
                    expected: `an integer in [1, ${this.slots}]`,
                },
            );
        }
        if (this.next + count > this.slots) {
            this.next = 0;
        }
        const first = this.next;
        this.next += count;
        return first;
    }

    /**
     * Writes one block's values into a slot of the host shadow; `flush()` sends the dirty range with one writeBuffer.
     * @param slot - the slot to fill
     * @param block - the block whose layout the bytes follow
     * @param values - the field values (a missing field is written as 0 by the block)
     */
    write(slot: number, block: UniformBlock, values: UniformValues): void {
        this.checkSlot(slot);
        this.assertFits(block);
        block.write(this.view, values, slot * UNIFORM_SLOT_BYTES);
        if (this.dirtyLo < 0 || slot < this.dirtyLo) {
            this.dirtyLo = slot;
        }
        if (slot > this.dirtyHi) {
            this.dirtyHi = slot;
        }
    }

    /**
     * queue.writeBuffer of the dirty slots (called by the batch driver before submit): one call covering
     * [lowest dirty slot, highest dirty slot]; nothing when no slot is dirty.
     */
    flush(): void {
        this.assertLive();
        if (this.dirtyLo < 0) {
            return;
        }
        const begin = this.dirtyLo * UNIFORM_SLOT_BYTES;
        const end = (this.dirtyHi + 1) * UNIFORM_SLOT_BYTES;
        this.device.queue.writeBuffer(this.buffer, begin, this.shadow, begin, end - begin);
        this.dirtyLo = -1;
        this.dirtyHi = -1;
    }

    /** Destroys the buffer (through the allocator); idempotent. */
    destroy(): void {
        if (this.destroyed) {
            return;
        }
        this.destroyed = true;
        this.allocator.destroy(this.buffer);
    }

    private assertLive(): void {
        if (this.destroyed) {
            throw new WebGpuGraphError("E_DISPOSED", `UniformRing "${this.label}" is destroyed`, { label: this.label });
        }
    }

    private checkSlot(slot: number): void {
        this.assertLive();
        if (!Number.isInteger(slot) || slot < 0 || slot >= this.slots) {
            throw new WebGpuGraphError(
                "E_INVALID_ARGUMENT",
                `UniformRing "${this.label}": slot ${slot} is outside [0, ${this.slots})`,
                {
                    argument: "slot",
                    value: slot,
                    expected: `an integer in [0, ${this.slots})`,
                },
            );
        }
    }

    private assertFits(block: UniformBlock): void {
        if (block.byteLength > UNIFORM_SLOT_BYTES) {
            throw new WebGpuGraphError(
                "E_INVALID_ARGUMENT",
                `UniformRing "${this.label}": block ${block.name} is ${block.byteLength} bytes, a slot holds ${UNIFORM_SLOT_BYTES}`,
                { argument: "block", value: block.name, expected: `byteLength <= ${UNIFORM_SLOT_BYTES}` },
            );
        }
    }
}
