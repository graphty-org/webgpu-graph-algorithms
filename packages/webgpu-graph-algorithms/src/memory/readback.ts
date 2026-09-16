/**
 * The staging ring of spec 4.4: a ring of MAP_READ | COPY_DST buffers (default DEFAULT_STAGING_SLOTS slots of 4 MiB,
 * PLAN DECISION 15) through which every readback of the package flows. `read()` copies with its own encoder and
 * submit, awaits mapAsync, copies out of the mapped range BEFORE unmap (the range is detached at unmap, design 10.7)
 * and never polls onSubmittedWorkDone; requests above the slot size are chunked. Ownership is one-directional: the
 * ring owns its buffers; a CommandBatch (P2) BORROWS one slot through borrowSlot and always returns it through
 * returnSlot, which unmaps it. Slots at or above OOM_SCOPE_THRESHOLD_BYTES are created through the AllocationTracker
 * (the out-of-memory scope of spec 5.7), the default-size ones directly on the device, so the allocator counts
 * residency and pool buffers only. A mapAsync rejection is E_DEVICE_LOST (or E_DISPOSED after destroyAll).
 */

import { DEFAULT_STAGING_SLOTS, OOM_SCOPE_THRESHOLD_BYTES } from "../constants.js";
import { type AllocationTracker } from "../device/error-scope.js";
import { BufferUsage, MapMode } from "../device/webgpu-constants.js";
import { WebGpuGraphError } from "../errors.js";

/** Default capacity of one staging slot (PLAN DECISION 15). */
const DEFAULT_SLOT_BYTES = 4 * 1024 * 1024;

/** Every staging buffer is MAP_READ | COPY_DST. */
const STAGING_USAGE = BufferUsage.MAP_READ | BufferUsage.COPY_DST;

/** Grown slots are sized in 256-byte steps. */
const SLOT_GRANULE = 256;

/**
 * One MAP_READ | COPY_DST staging buffer of the ring. Exported for the borrowers of borrowSlot / returnSlot
 * (contract 3.8): P2-T1's CommandBatch imports it as a type; nothing at P1 imports it by name.
 * @public
 */
export interface StagingSlot {
    readonly index: number;
    readonly buffer: GPUBuffer;
    readonly capacity: number;
}

/** The ring's bookkeeping of one slot. */
interface SlotEntry {
    readonly slot: StagingSlot;
    borrowed: boolean;
    /** True between the ring's own successful mapAsync and the unmap of returnSlot (a fallback when mapState is absent). */
    mapped: boolean;
    /** True when the buffer was created through the allocator (and is destroyed through it). */
    readonly tracked: boolean;
}

/**
 * Whether an ArrayBuffer-like is a SharedArrayBuffer (checked by tag so the global need not exist).
 * @param buffer - the buffer to test
 * @returns true for a SharedArrayBuffer
 */
function isSharedBuffer(buffer: ArrayBufferLike): boolean {
    return Object.prototype.toString.call(buffer) === "[object SharedArrayBuffer]";
}

/**
 * The E_INVALID_ARGUMENT error of a bad readback argument.
 * @param argument - the argument name
 * @param value - the value given
 * @param expected - what was expected
 * @returns the error to throw
 */
function invalid(argument: string, value: unknown, expected: string): WebGpuGraphError {
    return new WebGpuGraphError("E_INVALID_ARGUMENT", `${argument}: expected ${expected}`, {
        argument,
        value,
        expected,
    });
}

/** The staging ring (spec 4.4): default 3 slots; grows when every slot is busy; always unmaps and destroys its own buffers. */
export class Readback {
    private readonly device: GPUDevice;
    private readonly allocator: AllocationTracker;
    private readonly slotBytes: number;
    private readonly ring: SlotEntry[] = [];
    private disposed = false;

    /**
     * Creates the ring with `slots` slots of `slotBytes` each.
     * @param device - the device the copies run on
     * @param allocator - the context's OOM-scoped allocator (used for slots at or above OOM_SCOPE_THRESHOLD_BYTES)
     * @param options - ring options
     * @param options.slots - slots created up front; default DEFAULT_STAGING_SLOTS
     * @param options.slotBytes - capacity of each slot and the chunk size of read(); default 4 MiB; a positive multiple of 4
     */
    constructor(
        device: GPUDevice,
        allocator: AllocationTracker,
        options?: { readonly slots?: number | undefined; readonly slotBytes?: number | undefined },
    ) {
        const slotBytes = options?.slotBytes ?? DEFAULT_SLOT_BYTES;
        if (!Number.isInteger(slotBytes) || slotBytes <= 0 || slotBytes % 4 !== 0) {
            throw invalid("slotBytes", slotBytes, "a positive multiple of 4");
        }
        const slots = options?.slots ?? DEFAULT_STAGING_SLOTS;
        if (!Number.isInteger(slots) || slots < 0) {
            throw invalid("slots", slots, "a non-negative integer");
        }
        this.device = device;
        this.allocator = allocator;
        this.slotBytes = slotBytes;
        for (let i = 0; i < slots; i++) {
            this.addSlot(slotBytes);
        }
    }

    /**
     * Copies `byteLength` bytes from `src` at `srcOffset` (own encoder, own submit), maps, copies out BEFORE unmap;
     * resolves the bytes (a fresh ArrayBuffer, or `dest.buffer` after `dest.set` when given). Requests above the slot
     * size are chunked.
     * @param src - a COPY_SRC buffer
     * @param byteLength - bytes to read (a positive multiple of 4, within src)
     * @param dest - an optional destination view of at least byteLength bytes over a plain ArrayBuffer
     * @param srcOffset - the byte offset in src (default 0; a multiple of 4)
     * @returns the bytes read
     */
    async read(src: GPUBuffer, byteLength: number, dest?: ArrayBufferView, srcOffset?: number): Promise<ArrayBuffer> {
        this.assertLive();
        const offset = srcOffset ?? 0;
        if (!Number.isInteger(byteLength) || byteLength <= 0 || byteLength % 4 !== 0) {
            throw invalid("byteLength", byteLength, "a positive multiple of 4");
        }
        if (!Number.isInteger(offset) || offset < 0 || offset % 4 !== 0) {
            throw invalid("srcOffset", offset, "a non-negative multiple of 4");
        }
        if (offset + byteLength > src.size) {
            throw invalid("byteLength", byteLength, `srcOffset + byteLength <= src.size (${src.size})`);
        }
        if ((src.usage & BufferUsage.COPY_SRC) === 0) {
            throw invalid("src", src.label, "a buffer with COPY_SRC usage");
        }
        let out: ArrayBuffer;
        let outBytes: Uint8Array;
        if (dest === undefined) {
            out = new ArrayBuffer(byteLength);
            outBytes = new Uint8Array(out);
        } else {
            if (isSharedBuffer(dest.buffer)) {
                throw invalid("dest", "SharedArrayBuffer", "a view over a plain ArrayBuffer");
            }
            if (dest.byteLength < byteLength) {
                throw invalid("dest", dest.byteLength, `at least ${byteLength} bytes`);
            }
            out = dest.buffer as ArrayBuffer;
            outBytes = new Uint8Array(out, dest.byteOffset, byteLength);
        }
        let done = 0;
        while (done < byteLength) {
            const chunk = Math.min(this.slotBytes, byteLength - done);
            const slot = this.borrowSlot(chunk);
            const entry = this.ring[slot.index];
            try {
                const encoder = this.device.createCommandEncoder({ label: "readback" });
                encoder.copyBufferToBuffer(src, offset + done, slot.buffer, 0, chunk);
                this.device.queue.submit([encoder.finish()]);
                await this.map(entry, chunk);
                outBytes.set(new Uint8Array(slot.buffer.getMappedRange(0, chunk)), done);
            } finally {
                this.returnSlot(slot);
            }
            done += chunk;
        }
        return out;
    }

    /**
     * Reads one u32 counter through the same ring.
     * @param src - a COPY_SRC buffer
     * @param byteOffset - the counter's byte offset (a multiple of 4)
     * @returns the counter value
     */
    async readU32(src: GPUBuffer, byteOffset: number): Promise<number> {
        const bytes = await this.read(src, 4, undefined, byteOffset);
        return new Uint32Array(bytes)[0];
    }

    /**
     * Borrows an unmapped slot of at least byteLength (grows the ring when none is free); the borrower ALWAYS
     * returns it (spec 4.4).
     * @param byteLength - the bytes the borrower will copy into the slot
     * @returns the slot
     */
    borrowSlot(byteLength: number): StagingSlot {
        this.assertLive();
        if (!Number.isInteger(byteLength) || byteLength <= 0) {
            throw invalid("byteLength", byteLength, "a positive integer");
        }
        // a free slot is unmapped by construction: returnSlot unmaps, and only a borrower maps
        for (const entry of this.ring) {
            if (!entry.borrowed && entry.slot.capacity >= byteLength) {
                entry.borrowed = true;
                return entry.slot;
            }
        }
        const capacity = Math.max(this.slotBytes, Math.ceil(byteLength / SLOT_GRANULE) * SLOT_GRANULE);
        const entry = this.addSlot(capacity);
        entry.borrowed = true;
        return entry.slot;
    }

    /**
     * Returns a borrowed slot (unmapping it if mapped or pending). `mapState` decides when the runtime exposes it;
     * otherwise the ring's own record of its mapAsync calls does (a borrower that maps a slot itself must unmap it
     * before returning it, spec 4.4). A no-op after destroyAll() (the ring is gone; a borrower's finally must never
     * mask the E_DISPOSED / E_DEVICE_LOST of the read it was serving).
     * @param slot - a slot borrowed from this ring and not yet returned
     */
    returnSlot(slot: StagingSlot): void {
        if (this.disposed) {
            return;
        }
        const entry = this.ring[slot.index];
        if (entry === undefined || entry.slot !== slot || !entry.borrowed) {
            throw invalid("slot", slot.index, "a slot borrowed from this ring and not yet returned");
        }
        const state: string | undefined = slot.buffer.mapState;
        const mapped = state === undefined ? entry.mapped : state !== "unmapped";
        if (mapped) {
            slot.buffer.unmap();
        }
        entry.mapped = false;
        entry.borrowed = false;
    }

    /**
     * Number of slots (grows).
     * @returns the slot count
     */
    get slots(): number {
        return this.ring.length;
    }

    /**
     * Slots currently borrowed.
     * @returns the borrowed count
     */
    get borrowed(): number {
        let count = 0;
        for (const entry of this.ring) {
            if (entry.borrowed) {
                count++;
            }
        }
        return count;
    }

    /** Destroys every staging buffer (ctx.dispose()); idempotent. @internal */
    destroyAll(): void {
        for (const entry of this.ring) {
            if (entry.tracked) {
                this.allocator.destroy(entry.slot.buffer);
            } else {
                entry.slot.buffer.destroy();
            }
        }
        this.ring.length = 0;
        this.disposed = true;
    }

    /**
     * Creates one slot at the end of the ring.
     * @param capacity - the slot's byte capacity
     * @returns the new entry
     */
    private addSlot(capacity: number): SlotEntry {
        const index = this.ring.length;
        const descriptor: GPUBufferDescriptor = {
            label: `readback:slot:${index}`,
            size: capacity,
            usage: STAGING_USAGE,
        };
        const tracked = capacity >= OOM_SCOPE_THRESHOLD_BYTES;
        const buffer = tracked ? this.allocator.createBuffer(descriptor) : this.device.createBuffer(descriptor);
        const entry: SlotEntry = {
            slot: Object.freeze({ index, buffer, capacity }),
            borrowed: false,
            mapped: false,
            tracked,
        };
        this.ring.push(entry);
        return entry;
    }

    /**
     * mapAsync(READ) on a slot; a rejection becomes E_DISPOSED (after destroyAll) or E_DEVICE_LOST.
     * @param entry - the ring entry of the borrowed slot
     * @param byteLength - the bytes to map
     */
    private async map(entry: SlotEntry, byteLength: number): Promise<void> {
        try {
            await entry.slot.buffer.mapAsync(MapMode.READ, 0, byteLength);
            entry.mapped = true;
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : String(err);
            if (this.disposed) {
                throw new WebGpuGraphError("E_DISPOSED", "the readback ring was disposed while a read was pending", {
                    label: "readback",
                });
            }
            throw new WebGpuGraphError("E_DEVICE_LOST", `mapAsync rejected: ${message}`, {
                reason: "mapAsync",
                message,
            });
        }
    }

    /** Throws E_DISPOSED after destroyAll(). */
    private assertLive(): void {
        if (this.disposed) {
            throw new WebGpuGraphError("E_DISPOSED", "the readback ring was disposed", { label: "readback" });
        }
    }
}
