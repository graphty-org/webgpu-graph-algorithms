/**
 * Local declarations for resizable ArrayBuffer (ES2024).
 *
 * The package compiles against lib ES2020 (tsconfig.base.json plus this package's
 * lib override), which predates the resizable ArrayBuffer proposal. The growable
 * builder staging (design section 6.2) uses exactly these members, feature-detected
 * at runtime, so they are declared here and nowhere else. This file is a global
 * script declaration (no import/export): its interfaces merge with the lib's own
 * ArrayBuffer and ArrayBufferConstructor. It is not emitted and is not part of the
 * public declaration surface.
 */

interface ArrayBufferOptions {
    /** Upper bound, in bytes, that `resize()` may grow the buffer to. */
    maxByteLength?: number;
}

interface ArrayBuffer {
    /** True when the buffer was created with a `maxByteLength` and can be resized in place. */
    readonly resizable: boolean;

    /** The maximum byte length the buffer can be resized to; equals `byteLength` when not resizable. */
    readonly maxByteLength: number;

    /**
     * Resize the buffer in place to `newByteLength` bytes.
     * @param newByteLength - the new length in bytes; must not exceed `maxByteLength`
     */
    resize(newByteLength: number): void;
}

interface ArrayBufferConstructor {
    new (byteLength: number, options?: ArrayBufferOptions): ArrayBuffer;
}
