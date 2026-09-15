/**
 * Enum-valued option checking (design section 11.1): a value outside its documented set is refused
 * with E_UNSUPPORTED (`details.field`, `details.found`, `details.reason` "unsupported option") at
 * the call, never taken as a default. Every public entry point that reads an enum option -- the
 * builder constructor and freeze overrides, the derived graphs, `degreeOrder`, `validate`,
 * `set`, `indicesOf`, `fromRecords`, `foldArcs` -- goes through here, so typed callers see no change
 * and an untyped caller's typo is an error rather than a silently different result.
 */

import { GraphFormatError } from "../errors.js";

/**
 * Check that an option value is one of the allowed literals (undefined is accepted and returned so
 * the caller applies its default).
 * @param field - the option name for the error
 * @param value - the value
 * @param allowed - the documented literals
 * @returns the value, narrowed
 */
export function assertOneOf<T extends string>(field: string, value: unknown, allowed: readonly T[]): T | undefined {
    if (value === undefined) {
        return undefined;
    }
    if (typeof value !== "string" || !(allowed as readonly string[]).includes(value)) {
        throw new GraphFormatError(
            "E_UNSUPPORTED",
            `option ${field} has an unsupported value ${describeOption(value)}`,
            {
                field,
                found: value,
                reason: "unsupported option",
            },
        );
    }
    return value as T;
}

/**
 * A plain-ASCII rendering of an option value for an error message.
 * @param value - the value
 * @returns the rendering
 */
function describeOption(value: unknown): string {
    if (typeof value === "string") {
        return JSON.stringify(value.length > 40 ? `${value.slice(0, 37)}...` : value);
    }
    if (typeof value === "number" || typeof value === "boolean" || value === null) {
        return String(value);
    }
    return `a ${typeof value}`;
}

/** The WeightReducer literals (design section 7.3). */
export const WEIGHT_REDUCERS = ["first", "last", "sum", "min", "max"] as const;

/** The ColumnReducer literals (design section 7.3). */
export const COLUMN_REDUCERS = [...WEIGHT_REDUCERS, "mean", "count", "drop"] as const;
