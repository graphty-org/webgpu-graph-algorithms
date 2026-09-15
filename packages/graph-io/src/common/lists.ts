/**
 * List value syntaxes of the text formats (design section 5.1, research note 07 section 2.1):
 * GEXF 1.3 bracket lists `[1, 2, 3]` / `[foo, 'bar baz']`, GEXF 1.2 `liststring` values separated
 * by `|`, `,` or `;` ("an unsafe type"), Gephi and Neo4j `;`-separated arrays. The importer splits
 * the text into item texts with splitListText() and parses each item by the declared item type.
 */

import { GraphFormatError } from "@graphty/graph-format";

/** The list syntaxes an importer or exporter names. */
export type ListSyntax = "gexf" | "brackets" | "pipe" | "comma" | "semicolon";

/**
 * Split a list value into its item texts.
 *
 * - "gexf": a bracketed text `[a, b]` is parsed with single or double quotes around items that
 *   contain commas; otherwise the 1.2 rule applies: the text is split on `|` when it contains one,
 *   else on `,` when it contains one, else on `;` when it contains one, else it is one item.
 * - "brackets": bracketed only; an unbracketed text is one item.
 * - "pipe" / "comma" / "semicolon": split on that separator.
 *
 * Items are trimmed; an empty text (or `[]`) is an empty list; an empty item between separators
 * is kept as an empty string.
 * @param text - the value text
 * @param syntax - the syntax
 * @returns the item texts
 */
export function splitListText(text: string, syntax: ListSyntax): string[] {
    const trimmed = text.trim();
    if (trimmed.length === 0) {
        return [];
    }
    switch (syntax) {
        case "gexf": {
            if (isBracketed(trimmed)) {
                return splitBracketed(trimmed);
            }
            const separator = firstSeparator(trimmed, ["|", ",", ";"]);
            return separator === null ? [trimmed] : splitOn(trimmed, separator);
        }
        case "brackets":
            return isBracketed(trimmed) ? splitBracketed(trimmed) : [trimmed];
        case "pipe":
            return splitOn(trimmed, "|");
        case "comma":
            return splitOn(trimmed, ",");
        case "semicolon":
            return splitOn(trimmed, ";");
        default: {
            const name: string = syntax;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown list syntax ${name}`, {
                option: "listSyntax",
                found: name,
            });
        }
    }
}

/**
 * Join item texts into a list value of one syntax, the inverse of splitListText() for the exporters.
 * Bracket syntaxes quote an item that contains a comma, a bracket or a quote (with double quotes,
 * a double quote inside doubled); separator syntaxes cannot escape and leave items as they are.
 * @param items - the item texts
 * @param syntax - the syntax; "gexf" writes the 1.3 bracket form
 * @returns the list text
 */
export function joinListText(items: readonly string[], syntax: ListSyntax): string {
    switch (syntax) {
        case "gexf":
        case "brackets":
            return `[${items.map(quoteBracketItem).join(", ")}]`;
        case "pipe":
            return items.join("|");
        case "comma":
            return items.join(",");
        case "semicolon":
            return items.join(";");
        default: {
            const name: string = syntax;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown list syntax ${name}`, {
                option: "listSyntax",
                found: name,
            });
        }
    }
}

/**
 * Whether a trimmed text is `[...]`.
 * @param text - the trimmed text
 * @returns true when bracketed
 */
function isBracketed(text: string): boolean {
    return text.length >= 2 && text.startsWith("[") && text.endsWith("]");
}

/**
 * The first of several separators that occurs in a text.
 * @param text - the text
 * @param candidates - separators in priority order
 * @returns the first candidate found, or null
 */
function firstSeparator(text: string, candidates: readonly string[]): string | null {
    for (const candidate of candidates) {
        if (text.includes(candidate)) {
            return candidate;
        }
    }
    return null;
}

/**
 * Split on a separator and trim every item.
 * @param text - the text
 * @param separator - the separator
 * @returns the trimmed items
 */
function splitOn(text: string, separator: string): string[] {
    return text.split(separator).map((item) => item.trim());
}

/**
 * Parse the inside of a bracketed list: comma-separated items, each optionally wrapped in single
 * or double quotes (a doubled quote inside a quoted item is one quote).
 * @param text - the bracketed text
 * @returns the items
 */
function splitBracketed(text: string): string[] {
    const inner = text.slice(1, -1);
    const items: string[] = [];
    const n = inner.length;
    if (inner.trim().length === 0) {
        return items;
    }
    let i = 0;
    for (;;) {
        while (i < n && isSpace(inner.charCodeAt(i))) {
            i++;
        }
        let item = "";
        if (i < n && (inner[i] === '"' || inner[i] === "'")) {
            const quote = inner[i];
            i++;
            while (i < n) {
                if (inner[i] === quote) {
                    if (inner[i + 1] === quote) {
                        item += quote;
                        i += 2;
                        continue;
                    }
                    i++;
                    break;
                }
                item += inner[i];
                i++;
            }
        }
        // unquoted text, or text after a closing quote that is not a separator, runs to the next comma
        let end = inner.indexOf(",", i);
        if (end < 0) {
            end = n;
        }
        item += inner.slice(i, end).trim();
        items.push(item);
        i = end;
        if (i >= n) {
            return items;
        }
        i++;
        if (inner.slice(i).trim().length === 0) {
            items.push("");
            return items;
        }
    }
}

/**
 * Whether a char code is ASCII whitespace.
 * @param c - the char code
 * @returns true for space, tab, CR, LF
 */
function isSpace(c: number): boolean {
    return c === 32 || c === 9 || c === 13 || c === 10;
}

/**
 * Quote one item for a bracket list when it needs it.
 * @param item - the item text
 * @returns the item, quoted when it contains a comma, a bracket, a quote or surrounding whitespace
 */
function quoteBracketItem(item: string): string {
    if (item.length === 0 || /[,[\]"']/.test(item) || item !== item.trim()) {
        return `"${item.replace(/"/g, '""')}"`;
    }
    return item;
}
