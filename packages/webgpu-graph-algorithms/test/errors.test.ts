/**
 * WebGpuGraphError (spec 3.3, 5.7; contract 3.1): the stable code, the message, the frozen shallow copy of
 * `details`, the fixed `name`, the two brand checks that survive a second package copy, and the four
 * graph-format codes a public call lets pass through (D12).
 */

import { type GraphFormatErrorCode } from "@graphty/graph-format";
import fc from "fast-check";

import {
    hasErrorCode,
    isWebGpuGraphError,
    PASSTHROUGH_FORMAT_CODES,
    WebGpuGraphError,
    type WebGpuGraphErrorCode,
} from "../src/errors.js";

/** The closed union of contract 3.1, in document order. */
const CODES: readonly WebGpuGraphErrorCode[] = [
    "E_NO_WEBGPU",
    "E_NO_ADAPTER",
    "E_NO_DEVICE",
    "E_SOFTWARE_ONLY",
    "E_DEVICE_LOST",
    "E_DISPOSED",
    "E_VALIDATION",
    "E_SHADER_COMPILE",
    "E_OUT_OF_MEMORY",
    "E_TOO_LARGE",
    "E_UNSUPPORTED",
    "E_INVALID_ARGUMENT",
    "E_SNAPSHOT",
    "E_RELEASED",
    "E_NOT_LOADED",
    "E_ABORTED",
];

describe("WebGpuGraphError (contract 3.1)", () => {
    it("carries the code, the message and the fixed name, and is an Error", () => {
        const error = new WebGpuGraphError("E_NO_ADAPTER", "requestAdapter() returned null", { reason: "null" });
        expect(error.code).toBe("E_NO_ADAPTER");
        expect(error.message).toBe("requestAdapter() returned null");
        expect(error.name).toBe("WebGpuGraphError");
        expect(error).toBeInstanceOf(Error);
        expect(error).toBeInstanceOf(WebGpuGraphError);
        expect(typeof error.stack).toBe("string");
        expect(String(error)).toBe("WebGpuGraphError: requestAdapter() returned null");
    });

    it("accepts every code of the closed union exactly once", () => {
        expect(new Set(CODES).size).toBe(16);
        for (const code of CODES) {
            expect(new WebGpuGraphError(code, code).code).toBe(code);
        }
    });

    it("freezes a SHALLOW copy of details and uses a frozen empty object when none are given", () => {
        const nested = { a: 1 };
        const details: Record<string, unknown> = { reason: "x", nested };
        const error = new WebGpuGraphError("E_NO_DEVICE", "m", details);
        expect(error.details).toEqual({ reason: "x", nested });
        expect(error.details).not.toBe(details);
        expect(Object.isFrozen(error.details)).toBe(true);
        details.reason = "changed";
        expect(error.details.reason).toBe("x");
        // shallow: the nested object is the caller's own
        expect(error.details.nested).toBe(nested);
        expect(Object.isFrozen(nested)).toBe(false);

        const bare = new WebGpuGraphError("E_DISPOSED", "m");
        expect(bare.details).toEqual({});
        expect(Object.isFrozen(bare.details)).toBe(true);
        expect(new WebGpuGraphError("E_DISPOSED", "m", undefined).details).toEqual({});
        // the empty object is shared, so it must not be writable
        expect(() => {
            (bare.details as Record<string, unknown>).x = 1;
        }).toThrow(TypeError);
    });

    it("property: for any details record the copy is frozen, equal, and independent of the original", () => {
        fc.assert(
            fc.property(
                fc.dictionary(
                    fc.string().filter((key) => key !== "__proto__"),
                    fc.jsonValue(),
                ),
                (record) => {
                    const error = new WebGpuGraphError("E_INVALID_ARGUMENT", "m", record);
                    expect(Object.isFrozen(error.details)).toBe(true);
                    expect(error.details).toEqual(record);
                    expect(error.details).not.toBe(record);
                    expect(Object.keys(error.details).sort()).toEqual(Object.keys(record).sort());
                },
            ),
            { numRuns: 200 },
        );
    });

    it("isWebGpuGraphError and hasErrorCode accept a structural clone from another package copy", () => {
        const error = new WebGpuGraphError("E_TOO_LARGE", "big", { needed: 2, limit: 1 });
        const clone = { name: error.name, message: error.message, code: error.code, details: { ...error.details } };
        expect(clone).not.toBeInstanceOf(WebGpuGraphError);
        expect(isWebGpuGraphError(clone)).toBe(true);
        expect(hasErrorCode(clone, "E_TOO_LARGE")).toBe(true);
        expect(hasErrorCode(clone, "E_DISPOSED")).toBe(false);
        expect(isWebGpuGraphError(error)).toBe(true);
        expect(hasErrorCode(error, "E_TOO_LARGE")).toBe(true);
    });

    it("rejects non-errors, plain Errors, errors of another name and a non-string code", () => {
        expect(isWebGpuGraphError(null)).toBe(false);
        expect(isWebGpuGraphError(undefined)).toBe(false);
        expect(isWebGpuGraphError(42)).toBe(false);
        expect(isWebGpuGraphError("E_NO_ADAPTER")).toBe(false);
        expect(isWebGpuGraphError(new Error("x"))).toBe(false);
        expect(isWebGpuGraphError({ name: "GraphFormatError", code: "E_UNKNOWN_NODE" })).toBe(false);
        expect(isWebGpuGraphError({ name: "WebGpuGraphError", code: 7 })).toBe(false);
        expect(isWebGpuGraphError({ name: "WebGpuGraphError" })).toBe(false);
        expect(hasErrorCode(new Error("x"), "E_NO_ADAPTER")).toBe(false);
        expect(hasErrorCode(null, "E_NO_ADAPTER")).toBe(false);
    });

    it("narrows unknown to WebGpuGraphError", () => {
        const thrown: unknown = new WebGpuGraphError("E_ABORTED", "aborted", { batchId: 3 });
        if (isWebGpuGraphError(thrown)) {
            expect(thrown.code).toBe("E_ABORTED");
            expect(thrown.details.batchId).toBe(3);
        } else {
            expect.unreachable("isWebGpuGraphError must accept its own instances");
        }
    });
});

describe("PASSTHROUGH_FORMAT_CODES (D12, spec 5.7)", () => {
    it("lists the four graph-format codes in order and is frozen", () => {
        expect([...PASSTHROUGH_FORMAT_CODES]).toEqual([
            "E_GPU_INELIGIBLE",
            "E_UNKNOWN_NODE",
            "E_UNKNOWN_COLUMN",
            "E_COLUMN_LENGTH",
        ]);
        expect(Object.isFrozen(PASSTHROUGH_FORMAT_CODES)).toBe(true);
        expect(PASSTHROUGH_FORMAT_CODES.length).toBe(4);
    });

    it("names codes of graph-format's GraphFormatErrorCode union (a compile-time check)", () => {
        // the assignment compiles only if every literal is a member of graph-format's union (type-only import,
        // so nothing of graph-format is loaded at run time)
        const codes: readonly GraphFormatErrorCode[] = PASSTHROUGH_FORMAT_CODES;
        expect(codes.length).toBe(4);
    });
});
