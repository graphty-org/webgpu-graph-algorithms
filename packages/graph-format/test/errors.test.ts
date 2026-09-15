import { describe, expect, it } from "vitest";

import { GraphFormatError, type GraphFormatErrorCode } from "../src/errors.js";

// The closed code set of design section 11.2, in document order. The type-level test in
// test/types/errors.test-d.ts pins the union itself; this list pins the runtime count and lets every
// code be exercised through the constructor.
const ALL_CODES: readonly GraphFormatErrorCode[] = [
    "E_INVALID_ID",
    "E_UNKNOWN_NODE",
    "E_INDEX_RANGE",
    "E_TOO_LARGE",
    "E_INVALID_WEIGHT",
    "E_DIRECTED",
    "E_SELF_LOOP",
    "E_DUPLICATE_EDGE",
    "E_DUPLICATE_EDGE_ID",
    "E_DUPLICATE_ID",
    "E_DUPLICATE_ROLE",
    "E_UNKNOWN_COLUMN",
    "E_COLUMN_TYPE",
    "E_COLUMN_LENGTH",
    "E_COLUMN_ALIGNMENT",
    "E_COLUMN_EXISTS",
    "E_COLUMN_IMMUTABLE",
    "E_NO_DEFAULT",
    "E_PARTITION",
    "E_INVALID_PERMUTATION",
    "E_MASK_LENGTH",
    "E_GPU_INELIGIBLE",
    "E_INVALID_SNAPSHOT",
    "E_BAD_SERIALIZATION",
    "E_UNSUPPORTED_VERSION",
    "E_DETACHED",
    "E_BUILDER_DISPOSED",
    "E_UNSUPPORTED",
    "E_IMPORT",
];

describe("GraphFormatError", () => {
    it("carries the code, the message and the details", () => {
        const err = new GraphFormatError("E_INDEX_RANGE", "index 7 out of range (size 3)", { index: 7, size: 3 });
        expect(err.code).toBe("E_INDEX_RANGE");
        expect(err.message).toBe("index 7 out of range (size 3)");
        expect(err.details).toEqual({ index: 7, size: 3 });
    });

    it("is an Error with the name GraphFormatError", () => {
        const err = new GraphFormatError("E_DETACHED", "core was transferred away");
        expect(err).toBeInstanceOf(Error);
        expect(err).toBeInstanceOf(GraphFormatError);
        expect(err.name).toBe("GraphFormatError");
        expect(String(err)).toBe("GraphFormatError: core was transferred away");
        expect(err.stack).toContain("core was transferred away");
    });

    it("defaults details to a frozen empty object", () => {
        const err = new GraphFormatError("E_BUILDER_DISPOSED", "builder is disposed");
        expect(err.details).toEqual({});
        expect(Object.isFrozen(err.details)).toBe(true);
        expect(Object.keys(err.details)).toHaveLength(0);
    });

    it("freezes a shallow copy of the supplied details", () => {
        const details: Record<string, unknown> = { invariant: "I4", row: 17 };
        const err = new GraphFormatError("E_INVALID_SNAPSHOT", "row 17 is not sorted", details);
        expect(Object.isFrozen(err.details)).toBe(true);
        expect(err.details).not.toBe(details);
        details.row = 99;
        expect(err.details.row).toBe(17);
        expect(() => {
            (err.details as Record<string, unknown>).row = 1;
        }).toThrow(TypeError);
    });

    it("keeps typed-array and array values in details by reference", () => {
        const edges = [3, 9];
        const remap = new Uint32Array([1, 2, 3]);
        const err = new GraphFormatError("E_DUPLICATE_EDGE", "duplicate edge 0 -> 1", {
            source: 0,
            target: 1,
            edges,
            remap,
        });
        expect(err.details.edges).toBe(edges);
        expect(err.details.remap).toBe(remap);
    });

    it("accepts every one of the 29 codes of design section 11.2", () => {
        expect(ALL_CODES).toHaveLength(29);
        expect(new Set(ALL_CODES).size).toBe(29);
        for (const code of ALL_CODES) {
            const err = new GraphFormatError(code, `message for ${code}`);
            expect(err.code).toBe(code);
            expect(err.message).toBe(`message for ${code}`);
        }
    });

    it("works with the vitest toThrow matchers", () => {
        const throwing = (): never => {
            throw new GraphFormatError("E_UNKNOWN_NODE", "unknown node id 'x'", { id: "x" });
        };
        expect(throwing).toThrow(GraphFormatError);
        expect(throwing).toThrow("unknown node id 'x'");
        expect(throwing).toThrowError(expect.objectContaining({ code: "E_UNKNOWN_NODE", details: { id: "x" } }));
    });

    it("can be subclassed, which is how @graphty/graph-io declares ImportError", () => {
        class ImportErrorLike extends GraphFormatError {
            readonly report: { readonly errorCount: number };

            constructor(message: string, report: { readonly errorCount: number }) {
                super("E_IMPORT", message, { errorCount: report.errorCount });
                this.name = "ImportErrorLike";
                this.report = report;
            }
        }
        const err = new ImportErrorLike("importer aborted", { errorCount: 2 });
        expect(err).toBeInstanceOf(GraphFormatError);
        expect(err).toBeInstanceOf(ImportErrorLike);
        expect(err.code).toBe("E_IMPORT");
        expect(err.report.errorCount).toBe(2);
        expect(err.details).toEqual({ errorCount: 2 });
    });

    it("has read-only code and details properties at the type level and stable values at runtime", () => {
        const err = new GraphFormatError("E_TOO_LARGE", "count would exceed MAX_COUNT");
        const { code, details } = err;
        expect(code).toBe("E_TOO_LARGE");
        expect(details).toBe(err.details);
    });
});
