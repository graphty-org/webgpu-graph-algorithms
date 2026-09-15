/**
 * The GML lexical layer shared by the importer and the exporter (research note 07 section 2.2,
 * NetworkX readwrite/gml.py as the reference reader): the token list a whole GML text lexes into,
 * the structural validation that guarantees every later walk sees balanced `key value` pairs, the
 * value grammar (int, real, quoted string with `&#NN;` character references, `[ ... ]` record),
 * the NetworkX list conventions (`_networkx_list_start`, `"[]"`) and the key charset.
 *
 * Tokens are kept in typed arrays (kind, start, end, line, matching bracket) rather than objects:
 * the token list is the one intermediate structure of the GML importer, and it costs 17 bytes per
 * token instead of an object per node.
 */

import { SYNTAX_CODE } from "../../common/codes.js";
import { decodeGmlString } from "../../common/escape.js";

/** A bare word: a key, or `INF` / `NAN` at a value position. */
export const TOKEN_WORD = 0;
/** An integer literal. */
export const TOKEN_INT = 1;
/** A real literal, including the non-finite spellings. */
export const TOKEN_REAL = 2;
/** A double-quoted string; start / end delimit the body between the quotes. */
export const TOKEN_STRING = 3;
/** `[`. */
export const TOKEN_OPEN = 4;
/** `]`. */
export const TOKEN_CLOSE = 5;

/** The kind of one token. */
type TokenKind = 0 | 1 | 2 | 3 | 4 | 5;

/** The NetworkX marker that starts a one-element list written as repeated keys. */
export const LIST_START_MARKER = "_networkx_list_start";

/** The string NetworkX writes for an empty list. */
export const EMPTY_LIST_TEXT = "[]";

/** The string NetworkX writes for an empty tuple, read as an empty list too. */
export const EMPTY_TUPLE_TEXT = "()";

/** The node key the exporter writes a mangled node's original id into (design section 8.5). */
export const ORIGINAL_ID_KEY = "graphty_originalId";

/**
 * Issue code of every grammar violation (shared with the other text formats; the message carries
 * the detail): a bare token that is neither a key nor a number, an unclosed string, an unclosed
 * `[`, a `]` with no open `[`, a value without a key or a key without a value.
 */
export const SYNTAX_STRUCTURE_CODE = SYNTAX_CODE;
/** The code of an untokenisable bare token (E_SYNTAX). */
export const SYNTAX_TOKEN_CODE = SYNTAX_CODE;
/** The code of an unclosed string (E_SYNTAX). */
export const SYNTAX_STRING_CODE = SYNTAX_CODE;
/** The code of an unclosed `[` (E_SYNTAX). */
export const SYNTAX_BRACKET_CODE = SYNTAX_CODE;

const KEY_TEXT = /^[A-Za-z_][0-9A-Za-z_]*$/;
const STRICT_KEY_TEXT = /^[A-Za-z][0-9A-Za-z_]*$/;
const INT_TEXT = /^[+-]?[0-9]+$/;
const REAL_TEXT = /^[+-]?(?:(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)$/;
const NON_FINITE_TEXT = /^[+-]?(?:inf|infinity|nan)$/i;
const NON_FINITE_WORD = /^(?:inf|infinity|nan)$/i;

/**
 * A lexical or structural error of a GML text, with the 1-based line it was found on. The importer
 * turns it into a fatal parse-error issue.
 */
export class GmlSyntaxError extends Error {
    /** The issue code. */
    readonly code: string;

    /** The 1-based line. */
    readonly line: number;

    /**
     * Create a syntax error.
     * @param code - the issue code
     * @param message - a plain-ASCII message
     * @param line - the 1-based line
     */
    constructor(code: string, message: string, line: number) {
        super(message);
        this.name = "GmlSyntaxError";
        this.code = code;
        this.line = line;
    }
}

/**
 * The tokens of one GML text, structurally validated: every `[` has its matching `]` recorded and
 * every key is followed by a value, so walkers need no error paths.
 */
export class GmlTokens {
    /** The source text the offsets index into. */
    readonly text: string;

    /** The kind of every token. */
    kind: Uint8Array;

    /** The start offset of every token (a string's body start). */
    start: Uint32Array;

    /** The end offset (exclusive) of every token (a string's body end). */
    end: Uint32Array;

    /** The 1-based line of every token. */
    line: Uint32Array;

    /** For an open bracket, the index of its matching close bracket; 0 elsewhere. */
    match: Uint32Array;

    /** The number of tokens. */
    count = 0;

    /**
     * Create an empty token list over a text.
     * @param text - the source text
     * @param capacity - the initial capacity
     */
    constructor(text: string, capacity = 1024) {
        this.text = text;
        this.kind = new Uint8Array(capacity);
        this.start = new Uint32Array(capacity);
        this.end = new Uint32Array(capacity);
        this.line = new Uint32Array(capacity);
        this.match = new Uint32Array(0);
    }

    /**
     * Append a token.
     * @param kind - the kind
     * @param start - the start offset
     * @param end - the end offset
     * @param line - the 1-based line
     */
    push(kind: TokenKind, start: number, end: number, line: number): void {
        if (this.count === this.kind.length) {
            this.grow();
        }
        const i = this.count++;
        this.kind[i] = kind;
        this.start[i] = start;
        this.end[i] = end;
        this.line[i] = line;
    }

    /**
     * The raw text of a token (a string's body without the quotes, entities undecoded).
     * @param i - the token index
     * @returns the text
     */
    textOf(i: number): string {
        return this.text.slice(this.start[i], this.end[i]);
    }

    /**
     * The decoded value of a string token.
     * @param i - the token index
     * @returns the body with character references decoded
     */
    stringOf(i: number): string {
        return decodeGmlString(this.textOf(i));
    }

    /**
     * Whether a token is a `[` or `]`.
     * @param i - the token index
     * @returns true for a bracket
     */
    isBracket(i: number): boolean {
        return this.kind[i] >= TOKEN_OPEN;
    }

    /**
     * The index of the pair following the pair whose key is at `key`.
     * @param key - the index of a key token
     * @returns the index after the pair's value (after the matching `]` for a record)
     */
    nextPair(key: number): number {
        const value = key + 1;
        return this.kind[value] === TOKEN_OPEN ? this.match[value] + 1 : value + 1;
    }

    /** Double every array. */
    private grow(): void {
        const capacity = this.kind.length * 2;
        const kind = new Uint8Array(capacity);
        kind.set(this.kind);
        this.kind = kind;
        const start = new Uint32Array(capacity);
        start.set(this.start);
        this.start = start;
        const end = new Uint32Array(capacity);
        end.set(this.end);
        this.end = end;
        const line = new Uint32Array(capacity);
        line.set(this.line);
        this.line = line;
    }
}

/**
 * Lex and structurally validate a whole GML text.
 *
 * Lexical rules (NetworkX): whitespace separates tokens; `#` outside a string starts a comment
 * that runs to the end of the line; a string is delimited by double quotes and must close on its
 * line (there is no escape mechanism; `&#NN;` references are decoded later); `[` and `]` are
 * single-character tokens; every other run of characters is a key (`[A-Za-z_][0-9A-Za-z_]*`),
 * an integer (`[+-]?[0-9]+`), a real (with a decimal point or an exponent, or `INF` / `NAN` in
 * any case with an optional sign) or an error.
 *
 * Structural rules: the text is a sequence of `key value` pairs where a value is a number, a
 * string, `INF` / `NAN`, or a bracketed sequence of pairs; every `[` has a `]`.
 * @param text - the GML text
 * @returns the validated tokens; GmlSyntaxError with the line of the first problem
 */
export function tokenizeGml(text: string): GmlTokens {
    const tokens = new GmlTokens(text, Math.max(1024, text.length >>> 3));
    const n = text.length;
    let i = 0;
    let line = 1;
    while (i < n) {
        const c = text.charCodeAt(i);
        if (c === 10) {
            line++;
            i++;
        } else if (c === 13) {
            line++;
            i++;
            if (i < n && text.charCodeAt(i) === 10) {
                i++;
            }
        } else if (c === 32 || c === 9 || c === 11 || c === 12) {
            i++;
        } else if (c === 35) {
            i = endOfLine(text, i);
        } else if (c === 91) {
            tokens.push(TOKEN_OPEN, i, i + 1, line);
            i++;
        } else if (c === 93) {
            tokens.push(TOKEN_CLOSE, i, i + 1, line);
            i++;
        } else if (c === 34) {
            const close = text.indexOf('"', i + 1);
            const eol = endOfLine(text, i + 1);
            if (close < 0 || close > eol) {
                throw new GmlSyntaxError(SYNTAX_STRING_CODE, `unclosed string at line ${line}`, line);
            }
            tokens.push(TOKEN_STRING, i + 1, close, line);
            i = close + 1;
        } else {
            let j = i + 1;
            while (j < n && !isDelimiter(text.charCodeAt(j))) {
                j++;
            }
            tokens.push(classifyBare(text, i, j, line), i, j, line);
            i = j;
        }
    }
    validateStructure(tokens);
    return tokens;
}

/**
 * The offset of the line terminator at or after `from`, or the text length.
 * @param text - the text
 * @param from - the offset to search from
 * @returns the offset of the next `\n` or `\r`
 */
function endOfLine(text: string, from: number): number {
    const n = text.length;
    for (let i = from; i < n; i++) {
        const c = text.charCodeAt(i);
        if (c === 10 || c === 13) {
            return i;
        }
    }
    return n;
}

/**
 * Whether a character ends a bare token.
 * @param c - the char code
 * @returns true for whitespace, brackets, a quote or `#`
 */
function isDelimiter(c: number): boolean {
    return (
        c === 32 ||
        c === 9 ||
        c === 10 ||
        c === 13 ||
        c === 11 ||
        c === 12 ||
        c === 91 ||
        c === 93 ||
        c === 34 ||
        c === 35
    );
}

/**
 * Classify a bare token.
 * @param text - the text
 * @param start - the token start
 * @param end - the token end
 * @param line - the token line, for the error
 * @returns WORD, INT or REAL; GmlSyntaxError for anything else
 */
function classifyBare(text: string, start: number, end: number, line: number): TokenKind {
    const token = text.slice(start, end);
    if (KEY_TEXT.test(token)) {
        return TOKEN_WORD;
    }
    if (INT_TEXT.test(token)) {
        return TOKEN_INT;
    }
    if (REAL_TEXT.test(token) || NON_FINITE_TEXT.test(token)) {
        return TOKEN_REAL;
    }
    const shown = token.length > 20 ? `${token.slice(0, 20)}...` : token;
    throw new GmlSyntaxError(SYNTAX_TOKEN_CODE, `cannot tokenize "${shown}" at line ${line}`, line);
}

/**
 * Check the `key value` structure and record the matching bracket of every `[`.
 * @param tokens - the lexed tokens
 */
function validateStructure(tokens: GmlTokens): void {
    const { count, kind } = tokens;
    const match = new Uint32Array(count);
    const stack: number[] = [];
    let i = 0;
    while (i < count) {
        const k = kind[i];
        if (k === TOKEN_CLOSE) {
            const open = stack.pop();
            if (open === undefined) {
                throw new GmlSyntaxError(
                    SYNTAX_STRUCTURE_CODE,
                    `unexpected "]" at line ${tokens.line[i]}: no open "["`,
                    tokens.line[i],
                );
            }
            match[open] = i;
            i++;
            continue;
        }
        if (k !== TOKEN_WORD) {
            throw new GmlSyntaxError(
                SYNTAX_STRUCTURE_CODE,
                `expected a key at line ${tokens.line[i]}, found ${describeToken(tokens, i)}`,
                tokens.line[i],
            );
        }
        const value = i + 1;
        if (value >= count) {
            throw new GmlSyntaxError(
                SYNTAX_STRUCTURE_CODE,
                `key "${tokens.textOf(i)}" at line ${tokens.line[i]} has no value`,
                tokens.line[i],
            );
        }
        const vk = kind[value];
        if (vk === TOKEN_OPEN) {
            stack.push(value);
        } else if (vk === TOKEN_CLOSE || (vk === TOKEN_WORD && !NON_FINITE_WORD.test(tokens.textOf(value)))) {
            throw new GmlSyntaxError(
                SYNTAX_STRUCTURE_CODE,
                `expected a value for key "${tokens.textOf(i)}" at line ${tokens.line[i]}, found ${describeToken(tokens, value)}`,
                tokens.line[value],
            );
        }
        i = value + 1;
    }
    if (stack.length > 0) {
        const open = stack[stack.length - 1];
        throw new GmlSyntaxError(
            SYNTAX_BRACKET_CODE,
            `unexpected end of input: ${stack.length} bracket(s) still open, the last opened at line ${tokens.line[open]}`,
            tokens.line[open],
        );
    }
    tokens.match = match;
}

/**
 * A short description of a token for messages.
 * @param tokens - the tokens
 * @param i - the token index
 * @returns the quoted text or the bracket
 */
function describeToken(tokens: GmlTokens, i: number): string {
    switch (tokens.kind[i]) {
        case TOKEN_OPEN:
            return '"["';
        case TOKEN_CLOSE:
            return '"]"';
        case TOKEN_STRING:
            return `the string "${tokens.textOf(i)}"`;
        default:
            return `"${tokens.textOf(i)}"`;
    }
}

/**
 * The numeric value of an INT or REAL token (or a WORD that is `INF` / `NAN` at a value position).
 * @param text - the token text
 * @returns the number; Infinity / -Infinity / NaN for the non-finite spellings
 */
export function numberOfText(text: string): number {
    if (NON_FINITE_TEXT.test(text)) {
        const lower = text.toLowerCase();
        if (lower.endsWith("nan")) {
            return NaN;
        }
        return lower.startsWith("-") ? -Infinity : Infinity;
    }
    return Number(text);
}

/**
 * Whether a value token is one of the non-finite bare words (`INF`, `NAN`), which lex as words.
 * @param tokens - the tokens
 * @param i - the value token index
 * @returns true when the word is a real
 */
export function isNonFiniteWord(tokens: GmlTokens, i: number): boolean {
    return tokens.kind[i] === TOKEN_WORD && NON_FINITE_WORD.test(tokens.textOf(i));
}

/**
 * Whether a text is a GML key the strict grammar accepts (`[A-Za-z][0-9A-Za-z_]*`, what NetworkX
 * writes and reads).
 * @param text - the candidate key
 * @returns true when it can be written as a key
 */
export function isGmlKey(text: string): boolean {
    return STRICT_KEY_TEXT.test(text);
}

/**
 * Rewrite a text as a GML key: every character outside `[0-9A-Za-z_]` becomes `_` and a leading
 * non-letter is prefixed with `x` (the exporter's `sanitizeKeys: "mangle"`).
 * @param text - the text
 * @returns a valid key
 */
export function mangleGmlKey(text: string): string {
    const body = text.replace(/[^0-9A-Za-z_]/g, "_");
    return /^[A-Za-z]/.test(body) ? body : `x${body}`;
}

/** One open record while parseRecord() walks the tokens without recursion. */
interface RecordFrame {
    readonly record: Record<string, unknown>;
    listKeys: Set<string> | null;
    /** The index of the record's `]`. */
    readonly close: number;
    /** The next key token to read. */
    next: number;
    /** The key the record is stored under in its parent, or null for the root. */
    readonly key: string | null;
}

/**
 * Parse a bracketed record into a JSON object (design section 8.5: nested GML records map to
 * `json`). Values follow the value grammar; repeated keys become arrays with the NetworkX
 * conventions: the `_networkx_list_start` marker starts a list so a one-element list survives,
 * `"[]"` and `"()"` are empty lists, and a key seen twice without the marker becomes a
 * two-element array. Nested records are walked with an explicit stack, so the nesting depth is
 * bounded by memory, not by the call stack.
 * @param tokens - the validated tokens
 * @param open - the index of the record's `[`
 * @returns a null-prototype object
 */
export function parseRecord(tokens: GmlTokens, open: number): Record<string, unknown> {
    const root = Object.create(null) as Record<string, unknown>;
    const stack: RecordFrame[] = [
        { record: root, listKeys: null, close: tokens.match[open], next: open + 1, key: null },
    ];
    while (stack.length > 0) {
        const frame = stack[stack.length - 1];
        if (frame.next >= frame.close) {
            stack.pop();
            if (frame.key !== null) {
                storeValue(stack[stack.length - 1], frame.key, frame.record);
            }
            continue;
        }
        const p = frame.next;
        frame.next = tokens.nextPair(p);
        const key = tokens.textOf(p);
        const v = p + 1;
        if (tokens.kind[v] === TOKEN_STRING) {
            const text = tokens.stringOf(v);
            if (text === LIST_START_MARKER) {
                frame.record[key] = [];
                frame.listKeys ??= new Set();
                frame.listKeys.add(key);
                continue;
            }
            storeValue(frame, key, text === EMPTY_LIST_TEXT || text === EMPTY_TUPLE_TEXT ? [] : text);
        } else if (tokens.kind[v] === TOKEN_OPEN) {
            stack.push({
                record: Object.create(null) as Record<string, unknown>,
                listKeys: null,
                close: tokens.match[v],
                next: v + 1,
                key,
            });
        } else {
            storeValue(frame, key, numberOfText(tokens.textOf(v)));
        }
    }
    return root;
}

/**
 * Store one value under a key: appended to the key's list when the key is a list, paired with the
 * earlier value when the key repeats, set otherwise.
 * @param frame - the open record
 * @param key - the key
 * @param value - the value
 */
function storeValue(frame: RecordFrame, key: string, value: unknown): void {
    const { record } = frame;
    if (frame.listKeys !== null && frame.listKeys.has(key)) {
        (record[key] as unknown[]).push(value);
    } else if (key in record) {
        record[key] = [record[key], value];
        frame.listKeys ??= new Set();
        frame.listKeys.add(key);
    } else {
        record[key] = value;
    }
}

/**
 * The JSON value of a non-string value token: a number, or a record.
 * @param tokens - the tokens
 * @param v - the value token index (INT, REAL, non-finite WORD or OPEN)
 * @returns the value
 */
export function scalarOrRecord(tokens: GmlTokens, v: number): unknown {
    return tokens.kind[v] === TOKEN_OPEN ? parseRecord(tokens, v) : numberOfText(tokens.textOf(v));
}
