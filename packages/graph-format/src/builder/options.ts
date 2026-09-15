/**
 * The enum vocabularies of the builder options (design section 12.2) shared by the constructor and
 * the per-freeze overrides, so a value outside its set is refused the same way on every path
 * (E_UNSUPPORTED, never a silent substitute).
 */

import { type GraphFormatError } from "../errors.js";
import { type DuplicatePolicy } from "../types/index.js";

/** The DuplicatePolicy members. */
const DUPLICATE_POLICIES: ReadonlySet<string> = new Set(["keep", "error", "first", "last", "sum", "min", "max"]);

/**
 * Check a duplicateEdges value.
 * @param field - the option name for the error
 * @param value - the value
 * @param error - builds the error for an unsupported value
 * @returns the value as a DuplicatePolicy
 */
export function assertDuplicatePolicy(
    field: string,
    value: unknown,
    error: (field: string, found: unknown) => GraphFormatError,
): DuplicatePolicy {
    if (typeof value !== "string" || !DUPLICATE_POLICIES.has(value)) {
        throw error(field, value);
    }
    return value as DuplicatePolicy;
}
