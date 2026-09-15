/**
 * The DOT lexer (graphviz.org/doc/info/lang.html): IDs in their four spellings (an alphanumeric
 * identifier, a numeral, a double-quoted string with `\"` as its only escape, backslash pairs
 * consumed as units and backslash-newline as a line continuation, an HTML string in balanced
 * angle brackets), the punctuation of the
 * grammar, the two edge operators, the `+` of quoted-string concatenation, and the three comment
 * forms (`//`, `/* ... *\/`, and a `#` line, which is C preprocessor output). Keywords are not
 * distinguished here: the parser matches a bare id case-insensitively, since a quoted `"node"` is
 * an ordinary id. Line numbers are 1-based and attached to every token for the report.
 */

/** The kinds of token. */
type DotTokenKind = "id" | "punct" | "eof";

/** One token of a DOT document. */
export interface DotToken {
    /** The kind. */
    readonly kind: DotTokenKind;
    /**
     * The decoded text of an id (quotes removed, `\"` unescaped, an HTML string kept with its outer
     * angle brackets), the punctuation itself for "punct" (`{ } [ ] ; , = : + -> --`), "" at the end.
     */
    readonly text: string;
    /** Whether an id was double-quoted (a quoted keyword is an ordinary id). */
    readonly quoted: boolean;
    /** Whether an id was an HTML string. */
    readonly html: boolean;
    /** The 1-based line the token starts on. */
    readonly line: number;
}

/** The error the lexer and parser throw for a grammar violation; the importer turns it into a fatal parse-error. */
export class DotSyntaxError extends Error {
    /** The 1-based line of the violation. */
    readonly line: number;

    /**
     * Create a syntax error.
     * @param message - a plain-ASCII message
     * @param line - the 1-based line
     */
    constructor(message: string, line: number) {
        super(message);
        this.name = "DotSyntaxError";
        this.line = line;
    }
}

const EOF_TOKEN_TEXT = "";

/**
 * Whether a UTF-16 code unit may start a bare DOT identifier: a letter, an underscore or any code
 * of 128 and above (the grammar's `\200-\377` range, read here as every non-ASCII unit).
 * @param c - the code unit
 * @returns true for an identifier start
 */
function isIdentifierStart(c: number): boolean {
    return (c >= 65 && c <= 90) || (c >= 97 && c <= 122) || c === 95 || c >= 128;
}

/**
 * Whether a UTF-16 code unit may continue a bare DOT identifier.
 * @param c - the code unit
 * @returns true for an identifier character
 */
function isIdentifierPart(c: number): boolean {
    return isIdentifierStart(c) || isDigit(c);
}

/**
 * Whether a code unit is an ASCII digit.
 * @param c - the code unit
 * @returns true for 0-9
 */
function isDigit(c: number): boolean {
    return c >= 48 && c <= 57;
}

/**
 * A pull lexer over one DOT document with arbitrary lookahead.
 */
export class DotTokenizer {
    private readonly text: string;

    private pos = 0;

    private line = 1;

    private readonly lookahead: DotToken[] = [];

    private readonly onAmbiguity: ((numeral: string, line: number) => void) | null;

    /**
     * Create a lexer over a whole document.
     * @param text - the DOT text (BOM already removed)
     * @param onAmbiguity - called for a badly delimited numeral (`1e3`, which Graphviz splits into
     * `1` and `e3` with a warning), with the numeral text and its line
     */
    constructor(text: string, onAmbiguity: ((numeral: string, line: number) => void) | null = null) {
        this.text = text;
        this.onAmbiguity = onAmbiguity;
    }

    /**
     * Look at a token without consuming it.
     * @param n - how far ahead (0 = the next token)
     * @returns the token; an "eof" token past the end
     */
    peek(n = 0): DotToken {
        while (this.lookahead.length <= n) {
            this.lookahead.push(this.lex());
        }
        return this.lookahead[n];
    }

    /**
     * Consume the next token.
     * @returns the token; an "eof" token past the end
     */
    next(): DotToken {
        const ahead = this.lookahead.shift();
        return ahead ?? this.lex();
    }

    /**
     * Produce the next token from the text.
     * @returns the token
     */
    private lex(): DotToken {
        this.skipTrivia();
        const { text } = this;
        if (this.pos >= text.length) {
            return { kind: "eof", text: EOF_TOKEN_TEXT, quoted: false, html: false, line: this.line };
        }
        const start = this.pos;
        const { line } = this;
        const c = text.charCodeAt(start);
        switch (c) {
            case 0x7b: // {
            case 0x7d: // }
            case 0x5b: // [
            case 0x5d: // ]
            case 0x3b: // ;
            case 0x2c: // ,
            case 0x3d: // =
            case 0x3a: // :
            case 0x2b: // +
                this.pos++;
                return { kind: "punct", text: text[start], quoted: false, html: false, line };
            case 0x2d: {
                // - : an edge operator, or the sign of a numeral
                const d = text.charCodeAt(start + 1);
                if (d === 0x3e || d === 0x2d) {
                    this.pos += 2;
                    return { kind: "punct", text: text.slice(start, start + 2), quoted: false, html: false, line };
                }
                if (isDigit(d) || (d === 0x2e && isDigit(text.charCodeAt(start + 2)))) {
                    return this.numeral(start, line);
                }
                throw new DotSyntaxError("unexpected '-': not an edge operator or a numeral", line);
            }
            case 0x22:
                return this.quoted(start, line);
            case 0x3c:
                return this.html(start, line);
            default:
                break;
        }
        if (isIdentifierStart(c)) {
            let end = start + 1;
            while (end < text.length && isIdentifierPart(text.charCodeAt(end))) {
                end++;
            }
            this.pos = end;
            return { kind: "id", text: text.slice(start, end), quoted: false, html: false, line };
        }
        if (isDigit(c) || (c === 0x2e && isDigit(text.charCodeAt(start + 1)))) {
            return this.numeral(start, line);
        }
        throw new DotSyntaxError(`unexpected character ${JSON.stringify(text[start])}`, line);
    }

    /**
     * Lex a numeral `-?(.[0-9]+ | [0-9]+(.[0-9]*)?)` starting at `start`.
     * @param start - the offset of the first character (a sign, a digit or a dot)
     * @param line - the line
     * @returns the id token holding the numeral text
     */
    private numeral(start: number, line: number): DotToken {
        const { text } = this;
        let end = start;
        if (text.charCodeAt(end) === 0x2d) {
            end++;
        }
        while (end < text.length && isDigit(text.charCodeAt(end))) {
            end++;
        }
        if (text.charCodeAt(end) === 0x2e) {
            end++;
            while (end < text.length && isDigit(text.charCodeAt(end))) {
                end++;
            }
        }
        this.pos = end;
        const numeral = text.slice(start, end);
        if (end < text.length && isIdentifierStart(text.charCodeAt(end)) && this.onAmbiguity !== null) {
            // Graphviz: "syntax ambiguity - badly delimited number '1e' ... splits into two tokens"
            this.onAmbiguity(numeral, line);
        }
        return { kind: "id", text: numeral, quoted: false, html: false, line };
    }

    /**
     * Lex a double-quoted string starting at the opening quote: `\"` is the only escape, a
     * backslash followed by a line break is a continuation (both removed), every other backslash is
     * kept as written, and raw line breaks are allowed inside.
     * @param start - the offset of the opening quote
     * @param line - the line of the opening quote
     * @returns the id token holding the decoded text
     */
    private quoted(start: number, line: number): DotToken {
        const { text } = this;
        let out = "";
        let segment = start + 1;
        let i = start + 1;
        for (;;) {
            if (i >= text.length) {
                throw new DotSyntaxError("unterminated quoted string (missing closing quote)", line);
            }
            const c = text.charCodeAt(i);
            if (c === 0x22) {
                out += text.slice(segment, i);
                this.pos = i + 1;
                return { kind: "id", text: out, quoted: true, html: false, line };
            }
            if (c === 0x5c) {
                const d = text.charCodeAt(i + 1);
                if (d === 0x22) {
                    out += `${text.slice(segment, i)}"`;
                    i += 2;
                    segment = i;
                    continue;
                }
                if (d === 0x5c) {
                    // Graphviz's scanner consumes a backslash pair as one unit (both kept), so the
                    // second backslash never escapes a closing quote: "x\\" is the text x\\
                    i += 2;
                    continue;
                }
                if (d === 0x0a || d === 0x0d) {
                    out += text.slice(segment, i);
                    i += d === 0x0d && text.charCodeAt(i + 2) === 0x0a ? 3 : 2;
                    this.line++;
                    segment = i;
                    continue;
                }
                i++;
                continue;
            }
            if (c === 0x0a || c === 0x0d) {
                if (c === 0x0d && text.charCodeAt(i + 1) === 0x0a) {
                    i++;
                }
                this.line++;
            }
            i++;
        }
    }

    /**
     * Lex an HTML string starting at the opening angle bracket: the text up to the matching
     * closing bracket, brackets balanced, kept verbatim including the outer pair.
     * @param start - the offset of the opening bracket
     * @param line - the line of the opening bracket
     * @returns the id token holding the whole `<...>` text
     */
    private html(start: number, line: number): DotToken {
        const { text } = this;
        let depth = 0;
        for (let i = start; i < text.length; i++) {
            const c = text.charCodeAt(i);
            if (c === 0x3c) {
                depth++;
            } else if (c === 0x3e) {
                depth--;
                if (depth === 0) {
                    this.pos = i + 1;
                    return { kind: "id", text: text.slice(start, i + 1), quoted: false, html: true, line };
                }
            } else if (c === 0x0a || (c === 0x0d && text.charCodeAt(i + 1) !== 0x0a)) {
                this.line++;
            }
        }
        throw new DotSyntaxError("unterminated HTML string (missing closing '>')", line);
    }

    /**
     * Skip whitespace and comments, counting lines.
     */
    private skipTrivia(): void {
        const { text } = this;
        for (;;) {
            if (this.pos >= text.length) {
                return;
            }
            const c = text.charCodeAt(this.pos);
            if (c === 0x0a) {
                this.line++;
                this.pos++;
                this.skipPreprocessorLine();
                continue;
            }
            if (c === 0x0d) {
                this.line++;
                this.pos += text.charCodeAt(this.pos + 1) === 0x0a ? 2 : 1;
                this.skipPreprocessorLine();
                continue;
            }
            if (c === 0x20 || c === 0x09 || c === 0x0b || c === 0x0c) {
                this.pos++;
                continue;
            }
            if (c === 0x2f) {
                const d = text.charCodeAt(this.pos + 1);
                if (d === 0x2f) {
                    this.skipToLineEnd();
                    continue;
                }
                if (d === 0x2a) {
                    this.skipBlockComment();
                    continue;
                }
            }
            if (c === 0x23 && this.pos === 0) {
                this.skipToLineEnd();
                continue;
            }
            return;
        }
    }

    /**
     * Discard a `#` line when one starts at the current position (the start of a line).
     */
    private skipPreprocessorLine(): void {
        if (this.text.charCodeAt(this.pos) === 0x23) {
            this.skipToLineEnd();
        }
    }

    /**
     * Advance to the next line break without consuming it.
     */
    private skipToLineEnd(): void {
        const { text } = this;
        while (this.pos < text.length) {
            const c = text.charCodeAt(this.pos);
            if (c === 0x0a || c === 0x0d) {
                return;
            }
            this.pos++;
        }
    }

    /**
     * Skip a block comment starting at the current `/*`.
     */
    private skipBlockComment(): void {
        const { text } = this;
        const startLine = this.line;
        let i = this.pos + 2;
        for (; i < text.length; i++) {
            const c = text.charCodeAt(i);
            if (c === 0x2a && text.charCodeAt(i + 1) === 0x2f) {
                this.pos = i + 2;
                return;
            }
            if (c === 0x0a || (c === 0x0d && text.charCodeAt(i + 1) !== 0x0a)) {
                this.line++;
            }
        }
        throw new DotSyntaxError("unterminated block comment", startLine);
    }
}
