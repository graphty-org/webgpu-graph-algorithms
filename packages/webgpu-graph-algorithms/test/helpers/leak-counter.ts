/**
 * A counting proxy around ONE device's createBuffer and the destroy / mapAsync of every buffer it creates (contract 5.2;
 * spec 11.3 "Device loss / errors / leaks", 11.5 "the leak counter at 0", spec 8 / P7 "mapAsync count through the leak
 * counter"). Installed BEFORE GpuContext.from / create sees the device, so every buffer the package creates through the
 * device is counted; `destroy()` counts once per buffer (WebGPU's destroy is idempotent) and every `mapAsync()` on a counted
 * buffer counts, so a test can bound the readbacks of one call or one batch.
 *
 * The proxies are OWN properties that shadow the prototype methods: Dawn-node (webgpu@0.4.0) and Chromium both expose
 * createBuffer / destroy / mapAsync on the prototypes of extensible wrapper objects (verified on Dawn / llvmpipe on
 * 2026-09-15: `Object.getOwnPropertyNames(device)` is `features, limits, adapterInfo, queue, lost`; a buffer has no own
 * properties; own-property assignment and `delete` both work). restore() removes the own createBuffer so the prototype
 * method shows through again; buffers instrumented before restore() keep their counting proxies (they count into this
 * counter, which is what a test that restores mid-way expects).
 */

/** Counting proxy around createBuffer / destroy / mapAsync of a device (installed BEFORE GpuContext.from / create sees the device). */
export class LeakCounter {
    private createdCount = 0;
    private destroyedCount = 0;
    private mapAsyncCount = 0;
    private installed = true;
    private readonly destroyedBuffers = new WeakSet<GPUBuffer>();
    private readonly device: GPUDevice;
    private readonly ownCreateBuffer: PropertyDescriptor | undefined;

    private constructor(device: GPUDevice) {
        this.device = device;
        // an own createBuffer that existed before wrap() is restored verbatim; otherwise restore() deletes ours
        this.ownCreateBuffer = Object.getOwnPropertyDescriptor(device, "createBuffer");
        const original: GPUDevice["createBuffer"] = device.createBuffer;
        const counting = (descriptor: GPUBufferDescriptor): GPUBuffer => {
            const buffer = original.call(device, descriptor);
            this.createdCount += 1;
            this.instrument(buffer);
            return buffer;
        };
        Object.defineProperty(device, "createBuffer", { value: counting, configurable: true, writable: true });
    }

    /**
     * Installs the proxies on a device.
     * @param device - the device to count (raw, before any context adopts it)
     * @returns the counter
     */
    static wrap(device: GPUDevice): LeakCounter {
        return new LeakCounter(device);
    }

    /** Buffers created through the device since wrap() (until restore()). */
    get created(): number {
        return this.createdCount;
    }

    /** Distinct counted buffers destroyed so far. */
    get destroyed(): number {
        return this.destroyedCount;
    }

    /** created - destroyed: the buffers still alive. */
    get live(): number {
        return this.createdCount - this.destroyedCount;
    }

    /** mapAsync calls on counted buffers since wrap() or the last resetMapAsync(). */
    get mapAsyncCalls(): number {
        return this.mapAsyncCount;
    }

    /** Resets the mapAsync counter (per batch bounds). */
    resetMapAsync(): void {
        this.mapAsyncCount = 0;
    }

    /** Removes the proxies. */
    restore(): void {
        if (!this.installed) {
            return;
        }
        this.installed = false;
        if (this.ownCreateBuffer === undefined) {
            Reflect.deleteProperty(this.device, "createBuffer");
        } else {
            Object.defineProperty(this.device, "createBuffer", this.ownCreateBuffer);
        }
    }

    /**
     * Shadows destroy() and mapAsync() of one buffer with counting versions.
     * @param buffer - a buffer createBuffer just returned
     */
    private instrument(buffer: GPUBuffer): void {
        // destructured (the root config's prefer-destructuring rule); both are re-bound with .call below, and the
        // unbound-method rule is off under test/**
        const { destroy, mapAsync } = buffer;
        const countingDestroy = (): undefined => {
            if (!this.destroyedBuffers.has(buffer)) {
                this.destroyedBuffers.add(buffer);
                this.destroyedCount += 1;
            }
            return destroy.call(buffer);
        };
        const countingMapAsync = (mode: GPUMapModeFlags, offset?: GPUSize64, size?: GPUSize64): Promise<undefined> => {
            this.mapAsyncCount += 1;
            // forward exactly the arguments given: an explicit undefined is not the same as an absent optional on every runtime
            if (offset === undefined) {
                return mapAsync.call(buffer, mode);
            }
            if (size === undefined) {
                return mapAsync.call(buffer, mode, offset);
            }
            return mapAsync.call(buffer, mode, offset, size);
        };
        Object.defineProperty(buffer, "destroy", { value: countingDestroy, configurable: true, writable: true });
        Object.defineProperty(buffer, "mapAsync", { value: countingMapAsync, configurable: true, writable: true });
    }
}
