/**
 * Weight resolution shared by every importer (design sections 3.7 and 8.4): which source field is
 * THE weight (`weightFrom`, per-format default, null = unweighted), how its text or JSON value
 * becomes the number passed to `addEdge`, and the E_INVALID_WEIGHT boundary.
 *
 * An edge whose weight field is absent or blank is pushed WITHOUT a weight argument, never with an
 * explicit 1: the builder's weightSet bitmap then records that the weight was defaulted, and the
 * exporter re-emits no weight for it (decision D-WSET). NaN and non-numeric text are
 * E_INVALID_WEIGHT before the sink is touched, so the importer records the issue and skips the edge
 * with the sink unchanged; Infinity and negative values are legal weights (design section 3.7).
 */

import { type Column, GraphFormatError, type GraphSnapshot } from "@graphty/graph-format";

import { parseDecimalText } from "./declared-types.js";
import { formatF32, formatF64, formatInteger } from "./format.js";

/** The weight of an edge added without one (design section 3.7). */
export const DEFAULT_WEIGHT = 1;

/**
 * Whether a source field is the weight field under the resolved `weightFrom` option.
 * @param name - the field name (attribute title, CSV header, JSON key)
 * @param weightFrom - the resolved option; null means unweighted
 * @returns true when the field is THE weight
 */
export function isWeightField(name: string, weightFrom: string | null): boolean {
    return weightFrom !== null && name === weightFrom;
}

/**
 * The weight argument of addEdge from a text cell: undefined for a blank cell (the weight is
 * omitted), the number otherwise.
 * @param text - the cell text
 * @returns the weight, or undefined when blank; E_INVALID_WEIGHT for NaN or non-numeric text
 */
export function parseWeightText(text: string): number | undefined {
    const trimmed = text.trim();
    if (trimmed.length === 0) {
        return undefined;
    }
    let value: number;
    try {
        value = parseDecimalText(trimmed);
    } catch (err) {
        throw invalidWeight(text, err);
    }
    if (Number.isNaN(value)) {
        throw invalidWeight(text, null);
    }
    return value;
}

/**
 * The weight argument of addEdge from a typed value (JSON, a record field): undefined for null /
 * undefined, the number for a finite or infinite number, the parsed number for numeric text.
 * @param value - the field value
 * @returns the weight, or undefined when absent; E_INVALID_WEIGHT for NaN, a boolean, an object or non-numeric text
 */
export function weightFromValue(value: unknown): number | undefined {
    if (value === undefined || value === null) {
        return undefined;
    }
    if (typeof value === "number") {
        if (Number.isNaN(value)) {
            throw invalidWeight(value, null);
        }
        return value;
    }
    if (typeof value === "string") {
        return parseWeightText(value);
    }
    throw invalidWeight(value, null);
}

/**
 * The E_INVALID_WEIGHT error of a rejected value.
 * @param value - the rejected value
 * @param cause - the parse error, or null
 * @returns the error
 */
function invalidWeight(value: unknown, cause: unknown): GraphFormatError {
    let shown: string;
    switch (typeof value) {
        case "string":
            shown = JSON.stringify(value);
            break;
        case "number":
            shown = String(value);
            break;
        default:
            shown = typeof value;
    }
    return new GraphFormatError("E_INVALID_WEIGHT", `invalid edge weight ${shown}`, {
        value: typeof value === "object" ? typeof value : value,
        cause: cause instanceof Error ? cause.message : null,
    });
}

// ============================================================ the exporter side

/**
 * The explicit weights of a snapshot for exporters (design sections 3.7 and 8.5): the weight
 * role column when present (its validity says which edges had an explicit weight, its dtype how
 * the value is written), else `edgeList().weights` as f32 for every edge of a weighted snapshot;
 * nothing for an unweighted one. One implementation for every exporter.
 * Consumed by the per-format exporters under src/formats.
 * @public
 */
export interface ExplicitWeights {
    /** Whether any edge can have an explicit weight (the snapshot is weighted). */
    readonly weighted: boolean;
    /** The dtype the values are read from: the shadow column's, or f32 for the arc array. */
    readonly dtype: "f32" | "f64";
    /**
     * Whether an edge's weight was explicit in the source.
     * @param e - the logical edge index
     * @returns true when the exporter must write it
     */
    isExplicit(e: number): boolean;
    /**
     * The weight value of an edge (meaningful when explicit).
     * @param e - the logical edge index
     * @returns the value
     */
    value(e: number): number;
    /**
     * The weight text of an edge: the shortest round-tripping decimal for its dtype, or the
     * integer digits when `integral` is requested and the value is an integer; null when the
     * weight was defaulted.
     * @param e - the logical edge index
     * @param integral - write an integral value without a decimal part (formats declaring an int weight)
     * @returns the text, or null
     */
    text(e: number, integral?: boolean): string | null;
}

/** The dtypes a weight role column may have (an integer column declared by a caller included). */
const NUMERIC_DTYPES: ReadonlySet<string> = new Set(["f32", "f64", "i32", "u32", "u8"]);

/**
 * The shortest round-tripping text of a weight by the shadow column's dtype.
 * @param dtype - the shadow column's dtype
 * @returns the formatter
 */
function weightFormatter(dtype: string): (value: number) => string {
    switch (dtype) {
        case "f32":
            return formatF32;
        case "f64":
            return formatF64;
        default:
            return formatInteger;
    }
}

/**
 * Build the explicit-weight view of a snapshot.
 * @param snapshot - the snapshot
 * @returns the view
 */
export function explicitWeights(snapshot: GraphSnapshot): ExplicitWeights {
    const shadowColumn = snapshot.edges.byRole("weight");
    const shadow: Column | null = shadowColumn !== null && NUMERIC_DTYPES.has(shadowColumn.dtype) ? shadowColumn : null;
    if (shadow !== null) {
        const dtype = shadow.dtype === "f32" ? "f32" : "f64";
        const format = weightFormatter(shadow.dtype);
        return {
            weighted: true,
            dtype,
            isExplicit: (e: number): boolean => shadow.isSet(e),
            value: (e: number): number => shadow.value(e) as number,
            text: (e: number, integral = false): string | null => {
                if (!shadow.isSet(e)) {
                    return null;
                }
                const value = shadow.value(e) as number;
                return integral && Number.isInteger(value) ? formatInteger(value) : format(value);
            },
        };
    }
    const { weights } = snapshot.edgeList();
    if (weights === null || !snapshot.flags.weighted) {
        return {
            weighted: false,
            dtype: "f32",
            isExplicit: (): boolean => false,
            value: (): number => DEFAULT_WEIGHT,
            text: (): string | null => null,
        };
    }
    return {
        weighted: true,
        dtype: "f32",
        isExplicit: (): boolean => true,
        value: (e: number): number => weights[e],
        text: (e: number, integral = false): string => {
            const value = weights[e];
            return integral && Number.isInteger(value) ? formatInteger(value) : formatF32(value);
        },
    };
}
