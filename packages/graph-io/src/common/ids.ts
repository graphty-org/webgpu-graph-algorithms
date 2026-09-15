/**
 * Id coercion (design section 4.1): the importer-side rule that turns a text cell or a JSON value
 * into a NodeId before it reaches the sink. The core never coerces.
 *
 * - "keep": typed values pass through; anything that is not a string or a number is E_INVALID_ID.
 * - "canonical": text becomes a number iff it is canonical integer text (`/^-?(0|[1-9][0-9]*)$/`
 *   and a safe integer, excluding "-0" which would collide with "0"); every other text stays a
 *   string. Injective on text, so `String(id)` on export reproduces the cell exactly: "01", "1.0"
 *   and "+1" stay strings, "1" becomes 1. Typed values pass through as under "keep".
 * - "string": `String(v)` for strings, numbers, booleans, bigints and null; anything else is
 *   E_INVALID_ID.
 * - "number": `Number(text)` for text (empty or whitespace-only text, NaN and non-finite results
 *   are E_INVALID_ID); numbers pass through; anything else is E_INVALID_ID. This rule can merge
 *   distinct cells ("01" and "1"); IdCoercer counts such merges so the importer can report them as
 *   coercion issues.
 */

import { GraphFormatError, type IdCoercion, type NodeId } from "@graphty/graph-format";

import { ID_MERGED_CODE } from "./codes.js";

const CANONICAL_INTEGER = /^-?(0|[1-9][0-9]*)$/;

export { ID_MERGED_CODE };

/**
 * The "canonical" rule on one text cell.
 * @param text - the cell text, exactly as read
 * @returns the number for canonical safe-integer text (never for "-0"), the text itself otherwise
 */
export function canonicalId(text: string): NodeId {
    if (text !== "-0" && CANONICAL_INTEGER.test(text)) {
        const n = Number(text);
        if (Number.isSafeInteger(n)) {
            return n;
        }
    }
    return text;
}

/**
 * Whether a text cell is canonical integer text under the "canonical" rule (a number after import).
 * @param text - the cell text
 * @returns true when canonicalId(text) returns a number
 */
export function isCanonicalIntegerText(text: string): boolean {
    return typeof canonicalId(text) === "number";
}

/**
 * Coerce a text cell under a rule.
 * @param text - the cell text, exactly as read
 * @param mode - the coercion rule
 * @returns the id; E_INVALID_ID when the rule rejects the text
 */
export function coerceIdText(text: string, mode: IdCoercion): NodeId {
    switch (mode) {
        case "keep":
        case "string":
            return text;
        case "canonical":
            return canonicalId(text);
        case "number": {
            if (text.trim().length === 0) {
                throw invalidId(text, "empty text is not a number");
            }
            const n = Number(text);
            if (!Number.isFinite(n)) {
                throw invalidId(text, "not a finite number");
            }
            return n;
        }
        default: {
            const name: string = mode;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown id coercion ${name}`, { option: "ids", found: name });
        }
    }
}

/**
 * Coerce a typed value (a JSON scalar, a record field) under a rule.
 * @param value - the value
 * @param mode - the coercion rule
 * @returns the id; E_INVALID_ID when the rule rejects the value
 */
export function coerceId(value: unknown, mode: IdCoercion): NodeId {
    if (typeof value === "string") {
        return coerceIdText(value, mode);
    }
    if (typeof value === "number") {
        if (!Number.isFinite(value)) {
            throw invalidId(value, "not a finite number");
        }
        return mode === "string" ? String(value) : value;
    }
    if (mode === "string" && (typeof value === "boolean" || typeof value === "bigint" || value === null)) {
        return String(value);
    }
    throw invalidId(value, `a ${value === null ? "null" : typeof value} is not an id under ids: "${mode}"`);
}

/**
 * The E_INVALID_ID error of a rejected value.
 * @param value - the rejected value
 * @param reason - why
 * @returns the error
 */
function invalidId(value: unknown, reason: string): GraphFormatError {
    const shown = typeof value === "string" || typeof value === "number" ? value : typeof value;
    return new GraphFormatError("E_INVALID_ID", `invalid node id ${JSON.stringify(shown)}: ${reason}`, {
        reason,
        value: typeof value === "bigint" ? value.toString() : value,
    });
}

/**
 * A stateful coercer for one import call: applies the rule and, under "number", detects merges
 * (two distinct texts mapping to one number) so the importer can report them as coercion issues
 * (design section 4.1).
 */
export class IdCoercer {
    /** The rule. */
    readonly mode: IdCoercion;

    /** Texts merged into an id another text already produced. */
    mergeCount = 0;

    /** The text that first produced each numeric id, kept only under "number". */
    private readonly firstText: Map<number, string> | null;

    /** The merge detected by the most recent text() call, or null. */
    lastMerge: { readonly id: number; readonly text: string; readonly previousText: string } | null = null;

    /**
     * Create a coercer.
     * @param mode - the rule
     */
    constructor(mode: IdCoercion) {
        this.mode = mode;
        this.firstText = mode === "number" ? new Map() : null;
    }

    /**
     * Coerce a text cell, recording a merge under "number" when a different text already produced
     * the same number.
     * @param text - the cell text
     * @returns the id
     */
    text(text: string): NodeId {
        const id = coerceIdText(text, this.mode);
        this.lastMerge = null;
        if (this.firstText !== null && typeof id === "number") {
            const previous = this.firstText.get(id);
            if (previous === undefined) {
                this.firstText.set(id, text);
            } else if (previous !== text) {
                this.mergeCount++;
                this.lastMerge = { id, text, previousText: previous };
            }
        }
        return id;
    }

    /**
     * Coerce a typed value.
     * @param value - the value
     * @returns the id
     */
    value(value: unknown): NodeId {
        if (typeof value === "string") {
            return this.text(value);
        }
        this.lastMerge = null;
        return coerceId(value, this.mode);
    }
}
