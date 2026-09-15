import {
    FORMAT_VERSION,
    GraphFormatError,
    type GraphFormatErrorCode,
    INVALID_INDEX,
    MAX_COUNT,
    SNAPSHOT_BRAND,
} from "@graphty/graph-format";
import { expectTypeOf } from "vitest";

// ---- constants (design section 12.2): literal types so they can discriminate and brand
expectTypeOf<typeof INVALID_INDEX>().toEqualTypeOf<0xffffffff>();
expectTypeOf<typeof MAX_COUNT>().toEqualTypeOf<0xfffffffe>();
expectTypeOf<typeof FORMAT_VERSION>().toEqualTypeOf<1>();
expectTypeOf(SNAPSHOT_BRAND).toBeSymbol();
expectTypeOf<typeof SNAPSHOT_BRAND>().not.toEqualTypeOf<symbol>();

// ---- the closed code union of design section 11.2, in document order
expectTypeOf<GraphFormatErrorCode>().toEqualTypeOf<
    | "E_INVALID_ID"
    | "E_UNKNOWN_NODE"
    | "E_INDEX_RANGE"
    | "E_TOO_LARGE"
    | "E_INVALID_WEIGHT"
    | "E_DIRECTED"
    | "E_SELF_LOOP"
    | "E_DUPLICATE_EDGE"
    | "E_DUPLICATE_EDGE_ID"
    | "E_DUPLICATE_ID"
    | "E_DUPLICATE_ROLE"
    | "E_UNKNOWN_COLUMN"
    | "E_COLUMN_TYPE"
    | "E_COLUMN_LENGTH"
    | "E_COLUMN_ALIGNMENT"
    | "E_COLUMN_EXISTS"
    | "E_COLUMN_IMMUTABLE"
    | "E_NO_DEFAULT"
    | "E_PARTITION"
    | "E_INVALID_PERMUTATION"
    | "E_MASK_LENGTH"
    | "E_GPU_INELIGIBLE"
    | "E_INVALID_SNAPSHOT"
    | "E_BAD_SERIALIZATION"
    | "E_UNSUPPORTED_VERSION"
    | "E_DETACHED"
    | "E_BUILDER_DISPOSED"
    | "E_UNSUPPORTED"
    | "E_IMPORT"
>();
expectTypeOf<"E_SOMETHING_ELSE">().not.toMatchTypeOf<GraphFormatErrorCode>();
expectTypeOf<string>().not.toMatchTypeOf<GraphFormatErrorCode>();

// ---- the error class
expectTypeOf(GraphFormatError).toBeConstructibleWith("E_INDEX_RANGE", "index 3 out of range");
expectTypeOf(GraphFormatError).toBeConstructibleWith("E_INVALID_SNAPSHOT", "row 17 unsorted", {
    invariant: "I4",
    row: 17,
});
type CtorParams = ConstructorParameters<typeof GraphFormatError>;
expectTypeOf<CtorParams[0]>().toEqualTypeOf<GraphFormatErrorCode>();
expectTypeOf<CtorParams[1]>().toBeString();
expectTypeOf<CtorParams[2]>().toEqualTypeOf<Readonly<Record<string, unknown>> | undefined>();
expectTypeOf<CtorParams["length"]>().toEqualTypeOf<2 | 3>();
expectTypeOf<["E_NOT_A_CODE", string]>().not.toMatchTypeOf<CtorParams>();

declare const err: GraphFormatError;
expectTypeOf(err).toMatchTypeOf<Error>();
expectTypeOf(err.code).toEqualTypeOf<GraphFormatErrorCode>();
expectTypeOf(err.details).toEqualTypeOf<Readonly<Record<string, unknown>>>();
expectTypeOf(err.details.invariant).toBeUnknown();
expectTypeOf(err.message).toBeString();
expectTypeOf(err.name).toBeString();

// code and details are read-only
expectTypeOf<GraphFormatError>().toHaveProperty("code");
expectTypeOf<Readonly<Pick<GraphFormatError, "code" | "details">>>().toEqualTypeOf<
    Pick<GraphFormatError, "code" | "details">
>();

// narrowing on the code is exhaustive
declare function handle(e: GraphFormatError): number;
if (err.code === "E_DETACHED") {
    expectTypeOf(err.code).toEqualTypeOf<"E_DETACHED">();
}
expectTypeOf(handle).returns.toBeNumber();
