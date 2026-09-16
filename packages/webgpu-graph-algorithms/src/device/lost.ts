/**
 * Device-loss and uncaptured-error plumbing (spec 5.7, 2.2 step 5; contract 3.4): the pending-error slot a
 * context throws from its next public call, the uncapturederror listener that routes each event to the
 * caller's sink or to the slot, the device.lost watcher and the E_DEVICE_LOST constructor. Nothing here
 * throws; errors are constructed and handed on.
 */

import { WebGpuGraphError } from "../errors.js";

/** The context's pending-error slot (spec 5.7): the last uncaptured error not yet delivered; `take()` clears it. */
export class PendingErrorSlot {
    private readonly queue: WebGpuGraphError[] = [];

    /**
     * Stores an error; a second error before `take()` is kept as `details.next` of the first.
     * @param error - the error to hold until the next public call
     */
    set(error: WebGpuGraphError): void {
        this.queue.push(error);
    }

    /**
     * Returns and clears the pending error, or null. With several errors queued the FIRST is returned and
     * each later one is chained as `details.next` of its predecessor (a copy: `details` is frozen).
     * @returns the first pending error with the chain attached, or null
     */
    take(): WebGpuGraphError | null {
        if (this.queue.length === 0) {
            return null;
        }
        const errors = this.queue.splice(0);
        let chained: WebGpuGraphError | null = null;
        for (let i = errors.length - 1; i >= 0; i -= 1) {
            const error = errors[i];
            chained =
                chained === null
                    ? error
                    : new WebGpuGraphError(error.code, error.message, { ...error.details, next: chained });
        }
        return chained;
    }

    /**
     * True when an error is pending.
     * @returns whether `take()` would return an error
     */
    get pending(): boolean {
        return this.queue.length > 0;
    }
}

/**
 * The WebGpuGraphError for one uncaptured GPUError: E_OUT_OF_MEMORY for a GPUOutOfMemoryError (recognised
 * by its class name so no global is read, spec 2.1 rule 1), E_VALIDATION otherwise.
 * @param error - the event's error
 * @param label - the device label the error is attributed to
 * @returns the wrapped error
 */
function uncapturedError(error: GPUError, label: string): WebGpuGraphError {
    const { message } = error;
    if (error.constructor.name === "GPUOutOfMemoryError") {
        // the documented shape is { requested, resident, label } (contract 3.1); the event names no size and the
        // sink tracks no residency, so both are 0 and the runtime's text lives in the message only
        return new WebGpuGraphError("E_OUT_OF_MEMORY", `uncaptured out-of-memory error on "${label}": ${message}`, {
            requested: 0,
            resident: 0,
            label,
        });
    }
    return new WebGpuGraphError("E_VALIDATION", `uncaptured validation error on "${label}": ${message}`, {
        label,
        message,
    });
}

/**
 * Installs the uncapturederror listener: each event becomes E_VALIDATION { label, message } (or
 * E_OUT_OF_MEMORY for GPUOutOfMemoryError) routed to `onError`, else stored in `slot`. Returns the
 * uninstaller. `label` is the device's label (the descriptor label create() gave it).
 * @param device - the device to listen on
 * @param slot - where errors go when no sink is given
 * @param onError - the caller's sink, or null
 * @returns a function that removes the listener
 */
export function installUncapturedErrorSink(
    device: GPUDevice,
    slot: PendingErrorSlot,
    onError: ((error: WebGpuGraphError) => void) | null,
): () => void {
    let installed = true;
    const label = device.label ?? "";
    const listener = (event: GPUUncapturedErrorEvent): void => {
        if (!installed) {
            return;
        }
        const wrapped = uncapturedError(event.error, label);
        if (onError !== null) {
            onError(wrapped);
        } else {
            slot.set(wrapped);
        }
    };
    device.addEventListener("uncapturederror", listener);
    return () => {
        installed = false;
        if (typeof device.removeEventListener === "function") {
            device.removeEventListener("uncapturederror", listener);
        }
    };
}

/**
 * Chains device.lost into `onLost` exactly once; returns the promise the context exposes as `ctx.lost`.
 * @param device - the device whose loss is watched
 * @param onLost - called once with the lost info
 * @returns the lost info, after `onLost` ran
 */
export function watchDeviceLost(
    device: GPUDevice,
    onLost: (info: GPUDeviceLostInfo) => void,
): Promise<GPUDeviceLostInfo> {
    let delivered = false;
    return device.lost.then((info) => {
        if (!delivered) {
            delivered = true;
            onLost(info);
        }
        return info;
    });
}

/**
 * The E_DEVICE_LOST error for a lost-info record.
 * @param info - the device.lost result
 * @returns E_DEVICE_LOST { reason, message }
 */
export function deviceLostError(info: GPUDeviceLostInfo): WebGpuGraphError {
    return new WebGpuGraphError("E_DEVICE_LOST", `device lost (${info.reason}): ${info.message}`, {
        reason: info.reason,
        message: info.message,
    });
}
