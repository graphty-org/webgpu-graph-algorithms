/**
 * Dictionary encoding for `dict` columns (design section 5.4): a string dictionary in first-seen (or
 * declared) order, dense codes `0..size - 1`, a reverse map built eagerly by the builder and lazily
 * by a snapshot, and the re-interning pass that merges one dictionary into another (`GraphBuilder.from`
 * / `addGraph`, O(rows)). The lone-surrogate check every string value must pass (design section 11.3)
 * is `assertWellFormedString`, over the one `hasLoneSurrogate` of the string store.
 */

import { INVALID_INDEX } from "../constants.js";
import { GraphFormatError } from "../errors.js";
import { detachString, hasLoneSurrogate } from "../ids/string-store.js";

/**
 * Reject a string value that contains a lone surrogate with E_COLUMN_TYPE (details.reason "lone
 * surrogate"); every string that enters a column or a dictionary passes through here.
 * @param value - the string to check
 * @param details - extra details (column, row, field) to attach to the error
 */
export function assertWellFormedString(value: string, details?: Readonly<Record<string, unknown>>): void {
    if (hasLoneSurrogate(value)) {
        throw new GraphFormatError("E_COLUMN_TYPE", "string value contains a lone surrogate", {
            ...details,
            reason: "lone surrogate",
        });
    }
}

/**
 * Build the reverse map of a dictionary (value -> code). Used by a snapshot's DictColumn on the
 * first codeOf() call; the builder keeps one live in its DictionaryBuilder instead.
 * @param dictionary - the dictionary in code order
 * @returns a map from value to code
 */
export function buildCodeMap(dictionary: readonly string[]): Map<string, number> {
    const map = new Map<string, number>();
    for (let i = 0; i < dictionary.length; i++) {
        const value = dictionary[i];
        if (!map.has(value)) {
            map.set(value, i);
        }
    }
    return map;
}

/**
 * An interning dictionary: values are appended in first-seen order and every value has exactly one
 * dense code. The builder keeps one per dict column while staging and keeps interning into it across
 * freezes; a frozen column takes a COPY of the values array (invariant I17, design section 5.4).
 */
export class DictionaryBuilder {
    /** The dictionary in code order; copied by the frozen column. */
    readonly values: string[];

    private readonly codes: Map<string, number>;

    /**
     * Create a dictionary, optionally seeded with declared options (GEXF `<options>`) in order.
     * @param initial - the initial members in code order; duplicates keep their first code
     */
    constructor(initial?: readonly string[]) {
        this.values = [];
        this.codes = new Map();
        if (initial !== undefined) {
            for (const value of initial) {
                this.intern(value);
            }
        }
    }

    /**
     * Number of distinct values interned so far.
     * @returns the dictionary size
     */
    get size(): number {
        return this.values.length;
    }

    /**
     * The code of a value, appending it when unseen.
     * @param value - the string to intern; E_COLUMN_TYPE when it holds a lone surrogate
     * @returns the dense code of the value
     */
    intern(value: string): number {
        const existing = this.codes.get(value);
        if (existing !== undefined) {
            return existing;
        }
        assertWellFormedString(value);
        const code = this.values.length;
        const owned = detachString(value);
        this.values.push(owned);
        this.codes.set(owned, code);
        return code;
    }

    /**
     * The code of a value without interning it.
     * @param value - the string to look up
     * @returns its code, or INVALID_INDEX when absent
     */
    codeOf(value: string): number {
        const code = this.codes.get(value);
        return code === undefined ? INVALID_INDEX : code;
    }

    /**
     * Whether a value has been interned.
     * @param value - the string to test
     * @returns true when present
     */
    has(value: string): boolean {
        return this.codes.has(value);
    }
}
