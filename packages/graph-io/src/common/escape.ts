/**
 * Id, label and value escaping for the text formats (design section 8.5), one function per
 * syntax, with the inverse where the importer needs it: XML attribute and element text (GEXF,
 * GraphML), GML quoted strings (`"` and `&` and non-ASCII as `&#NN;` character entities, the
 * NetworkX convention), DOT ids (bare when the grammar allows, quoted with `\"` otherwise), RFC
 * 4180 CSV cells and Pajek labels.
 */

import { GraphFormatError } from "@graphty/graph-format";

import { hasIllegalXmlChar } from "./xml.js";

const XML_TEXT_SPECIAL = /[&<>\r]/g;
const XML_ATTR_SPECIAL = /[&<>"\t\n\r]/g;
const XML_REPLACEMENTS: Readonly<Record<string, string>> = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "\t": "&#9;",
    "\n": "&#10;",
    "\r": "&#13;",
};

/**
 * Escape text for an XML element body: `&`, `<`, `>` and a carriage return (which XML
 * end-of-line handling would turn into a line feed on every read) as character references. A
 * character XML 1.0 forbids (a C0 control other than tab / LF / CR, U+FFFE, U+FFFF, a lone
 * surrogate) has no XML spelling at all and is E_COLUMN_TYPE; the exporters' check() counts them.
 * @param text - the text
 * @returns the escaped text
 */
export function escapeXmlText(text: string): string {
    if (hasIllegalXmlChar(text)) {
        throw illegalXml(text);
    }
    return text.replace(XML_TEXT_SPECIAL, (c) => XML_REPLACEMENTS[c]);
}

/**
 * Escape text for a double-quoted XML attribute value: `&`, `<`, `>`, `"` and the whitespace
 * characters attribute normalisation would fold; a character XML 1.0 forbids is E_COLUMN_TYPE.
 * @param text - the text
 * @returns the escaped text
 */
export function escapeXmlAttribute(text: string): string {
    if (hasIllegalXmlChar(text)) {
        throw illegalXml(text);
    }
    return text.replace(XML_ATTR_SPECIAL, (c) => XML_REPLACEMENTS[c]);
}

/**
 * The error of a text no XML 1.0 document can carry.
 * @param text - the text
 * @returns the error
 */
function illegalXml(text: string): GraphFormatError {
    return new GraphFormatError("E_COLUMN_TYPE", "the text holds a character XML 1.0 cannot carry", {
        reason: "xml illegal char",
        length: text.length,
    });
}

const GML_SPECIAL = /["&]|[^ -~]/gu;

/**
 * Quote a GML string value: double quotes around the text with `"`, `&` and every character
 * outside printable ASCII written as `&#NN;` (NetworkX's writer convention, which its reader and
 * Gephi's decode).
 * A lone surrogate cannot be written and is E_COLUMN_TYPE.
 * @param text - the text
 * @returns the quoted GML string
 */
export function quoteGmlString(text: string): string {
    const escaped = text.replace(GML_SPECIAL, (c) => {
        const code = c.codePointAt(0);
        if (code === undefined) {
            return c;
        }
        if (code >= 0xd800 && code <= 0xdfff) {
            throw new GraphFormatError("E_COLUMN_TYPE", "a lone surrogate cannot be written as a GML string", {
                reason: "lone surrogate",
            });
        }
        return `&#${code};`;
    });
    return `"${escaped}"`;
}

const GML_ENTITY = /&(#[0-9]+|#x[0-9a-fA-F]+|amp|quot|lt|gt|apos);/g;
const GML_NAMED: Readonly<Record<string, string>> = { amp: "&", quot: '"', lt: "<", gt: ">", apos: "'" };

/**
 * Decode the character entities of a GML string body (the inverse of quoteGmlString on the text
 * between the quotes). An unknown entity is left as written.
 * @param body - the text between the quotes
 * @returns the decoded text
 */
export function decodeGmlString(body: string): string {
    return body.replace(GML_ENTITY, (whole, entity: string) => {
        if (entity.startsWith("#x")) {
            return String.fromCodePoint(Number.parseInt(entity.slice(2), 16));
        }
        if (entity.startsWith("#")) {
            return String.fromCodePoint(Number.parseInt(entity.slice(1), 10));
        }
        return GML_NAMED[entity] ?? whole;
    });
}

const DOT_NUMERAL = /^-?(\.[0-9]+|[0-9]+(\.[0-9]*)?)$/;
const DOT_KEYWORDS: ReadonlySet<string> = new Set(["node", "edge", "graph", "digraph", "subgraph", "strict"]);

/**
 * Whether a text is a DOT ID that needs no quotes: an alphanumeric identifier not starting with
 * a digit, or a numeral, and not a keyword (case-insensitive).
 * @param text - the text
 * @returns true when the text can be written bare
 */
export function isBareDotId(text: string): boolean {
    if (DOT_KEYWORDS.has(text.toLowerCase())) {
        return false;
    }
    return isDotIdentifier(text) || DOT_NUMERAL.test(text);
}

/**
 * Whether a text is a DOT identifier: a letter, underscore or any character of code 128 and
 * above, followed by letters, digits, underscores or such characters.
 * @param text - the text
 * @returns true for a bare identifier
 */
function isDotIdentifier(text: string): boolean {
    if (text.length === 0) {
        return false;
    }
    for (let i = 0; i < text.length; i++) {
        const c = text.charCodeAt(i);
        const letter = (c >= 65 && c <= 90) || (c >= 97 && c <= 122) || c === 95 || c >= 128;
        const digit = c >= 48 && c <= 57;
        if (!letter && (i === 0 || !digit)) {
            return false;
        }
    }
    return true;
}

/**
 * Whether a text can be written as a DOT ID at all. Graphviz's scanner consumes a backslash pair
 * `\\` as one unit and `\"` as an escaped quote, left to right, so a backslash that precedes a
 * double quote or ends the text cannot be written: the written `\\"` reads as a pair and a closing
 * quote, and a trailing backslash escapes the closing quote. Every other text is writable;
 * quoteDotId() writes it.
 * @param text - the id, name or value text
 * @returns true when quoteDotId(text) reads back as `text`
 */
export function isWritableDotText(text: string): boolean {
    return !text.endsWith("\\") && !text.includes('\\"');
}

/**
 * Write a DOT ID: bare when the grammar allows, otherwise double-quoted with `"` escaped as `\"`.
 * In DOT quoted strings the dyad `\"` is the only escape; every other character, a backslash and
 * a raw line break included, is kept as written, so nothing else is rewritten. A text ending in a
 * backslash has no DOT spelling (isWritableDotText); the caller refuses it before writing.
 * @param text - the id or label text
 * @returns the DOT ID
 */
export function quoteDotId(text: string): string {
    if (isBareDotId(text)) {
        return text;
    }
    return `"${text.replace(/"/g, '\\"')}"`;
}

/**
 * Write a set CSV cell per RFC 4180: quoted with `"` doubled when the text contains the
 * delimiter, a quote, a CR or LF, or leading / trailing whitespace; bare otherwise. The empty
 * string is written as the quoted empty cell `""`, so the importers can tell a set empty string
 * (and an empty-string node id, legal under design section 4.1) from an unset cell, which every
 * exporter writes as nothing between the delimiters.
 * @param text - the cell text of a set cell
 * @param delimiter - the field delimiter (default ",")
 * @returns the cell as written
 */
export function quoteCsvCell(text: string, delimiter = ","): string {
    if (text.length === 0) {
        return '""';
    }
    if (
        !text.includes(delimiter) &&
        !text.includes('"') &&
        !text.includes("\n") &&
        !text.includes("\r") &&
        text === text.trim()
    ) {
        return text;
    }
    return `"${text.replace(/"/g, '""')}"`;
}

/**
 * Whether a text can be a Pajek label: Pajek has no escape mechanism inside its double-quoted
 * labels, so a text containing a double quote, a CR or an LF cannot be written.
 * @param text - the label text
 * @returns true when quotePajekLabel() can write it
 */
export function isPajekLabel(text: string): boolean {
    return !text.includes('"') && !text.includes("\n") && !text.includes("\r");
}

/**
 * Write a Pajek label: bare when it is a single run of non-space, non-quote characters, otherwise
 * double-quoted. E_UNSUPPORTED (reason "pajek label") for a text isPajekLabel() rejects; the
 * exporter's check() counts those.
 * @param text - the label text
 * @returns the label as written
 */
export function quotePajekLabel(text: string): string {
    if (!isPajekLabel(text)) {
        throw new GraphFormatError("E_UNSUPPORTED", "a Pajek label cannot contain a double quote or a line break", {
            reason: "pajek label",
            value: text,
        });
    }
    if (text.length > 0 && !/[\s"]/.test(text)) {
        return text;
    }
    return `"${text}"`;
}
