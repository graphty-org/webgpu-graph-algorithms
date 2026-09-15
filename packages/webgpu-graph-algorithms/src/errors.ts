/**
 * The error class and the closed code set of @graphty/webgpu-graph-algorithms (spec 3.3, 5.7; contract 3.1).
 *
 * Every condition the package detects itself is thrown as a WebGpuGraphError carrying a stable `code` and a
 * frozen shallow copy of the `details` the thrower supplied. Errors raised by @graphty/graph-format accessors
 * the package calls on the caller's behalf pass through unchanged (D12); PASSTHROUGH_FORMAT_CODES lists the
 * codes a public call may let escape. Constructors never throw. Messages are plain ASCII.
 *
 * `details` keys per code (contract 3.1, so tests can assert them): E_NO_WEBGPU { reason, hint };
 * E_NO_ADAPTER { reason }; E_NO_DEVICE { reason, adapter, limit?, requested?, available? } (reason "consumed" |
 * "requestDevice" | "limit" | "feature" | "maxComputeWorkgroupsPerDimension"); E_SOFTWARE_ONLY { adapter };
 * E_DEVICE_LOST { reason, message }; E_DISPOSED { label }; E_VALIDATION { label, message, batchId? };
 * E_SHADER_COMPILE { id, stage: "compose" | "compile", slot?, messages? }; E_OUT_OF_MEMORY { requested, resident,
 * label }; E_TOO_LARGE { needed, limit, path, algorithm }; E_UNSUPPORTED { feature? | option?, hint? } (exactly
 * one of feature / option); E_INVALID_ARGUMENT { argument, value, expected? }; E_SNAPSHOT { reason, serial };
 * E_RELEASED { serial }; E_NOT_LOADED { state }; E_ABORTED { batchId? }.
 */

/** Every condition the package detects itself carries one of these codes (spec 3.3). */
export type WebGpuGraphErrorCode =
    | "E_NO_WEBGPU"
    | "E_NO_ADAPTER"
    | "E_NO_DEVICE"
    | "E_SOFTWARE_ONLY"
    | "E_DEVICE_LOST"
    | "E_DISPOSED"
    | "E_VALIDATION"
    | "E_SHADER_COMPILE"
    | "E_OUT_OF_MEMORY"
    | "E_TOO_LARGE"
    | "E_UNSUPPORTED"
    | "E_INVALID_ARGUMENT"
    | "E_SNAPSHOT"
    | "E_RELEASED"
    | "E_NOT_LOADED"
    | "E_ABORTED";

/** The graph-format error codes a public call lets propagate unchanged (D12, spec 5.7): raised by accessors the package calls on the caller's behalf. */
export const PASSTHROUGH_FORMAT_CODES: readonly [
    "E_GPU_INELIGIBLE",
    "E_UNKNOWN_NODE",
    "E_UNKNOWN_COLUMN",
    "E_COLUMN_LENGTH",
] = Object.freeze(["E_GPU_INELIGIBLE", "E_UNKNOWN_NODE", "E_UNKNOWN_COLUMN", "E_COLUMN_LENGTH"] as const);

const EMPTY_DETAILS: Readonly<Record<string, unknown>> = Object.freeze({});

/** The one error class the package throws for conditions it detects itself: a stable `code` and frozen `details` (spec 3.3). */
export class WebGpuGraphError extends Error {
    /** The stable code. */
    readonly code: WebGpuGraphErrorCode;

    /** Frozen shallow copy of the details given to the constructor (`{}` when none). */
    readonly details: Readonly<Record<string, unknown>>;

    /** `name` is always "WebGpuGraphError". */
    override readonly name: "WebGpuGraphError";

    /**
     * Create a WebGpuGraphError.
     * @param code - the stable code the caller can branch on
     * @param message - a plain-ASCII human-readable message naming the offending label, limit or value
     * @param details - optional machine-readable context; copied shallowly and frozen
     */
    constructor(code: WebGpuGraphErrorCode, message: string, details?: Record<string, unknown>) {
        super(message);
        this.name = "WebGpuGraphError";
        this.code = code;
        this.details = details === undefined ? EMPTY_DETAILS : Object.freeze({ ...details });
    }
}

/**
 * Brand check that survives two package copies: `name === "WebGpuGraphError"` and a string `code`.
 * @param x - anything, typically a caught value
 * @returns true when `x` is structurally a WebGpuGraphError (instanceof is NOT required)
 */
export function isWebGpuGraphError(x: unknown): x is WebGpuGraphError {
    if (typeof x !== "object" || x === null) {
        return false;
    }
    const candidate = x as { readonly name?: unknown; readonly code?: unknown };
    return candidate.name === "WebGpuGraphError" && typeof candidate.code === "string";
}

/**
 * Brand check that survives two package copies: `code === x.code` for an `isWebGpuGraphError(x)`.
 * @param x - anything, typically a caught value
 * @param code - the code to test for
 * @returns true when `x` is a WebGpuGraphError carrying `code`
 */
export function hasErrorCode(x: unknown, code: WebGpuGraphErrorCode): boolean {
    return isWebGpuGraphError(x) && x.code === code;
}
