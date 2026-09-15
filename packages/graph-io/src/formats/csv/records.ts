/**
 * The one streaming CSV record reader of the package (design sections 8.2 and 8.4: hand-written
 * tokenisers per format; CSV and Neo4j CSV are line-oriented over a byte stream): RFC 4180 records
 * with a configurable single-character delimiter and quote, a doubled quote inside a quoted field,
 * quoted fields spanning lines, LF / CRLF / lone-CR terminators, read from the common text reader
 * one chunk at a time as a character state machine. Only the open field is ever held, so a field
 * spanning many chunks costs its length once and a multi-gigabyte file is never buffered.
 *
 * `RecordReader` (used by the Neo4j importer) keeps one reusable cell array and one reusable
 * "was quoted" array: a quoted empty field (`""`) is an empty string while an unquoted empty field
 * means "not set", and only the tokeniser can tell the two apart. `CsvRecordReader` (the CSV
 * importer) wraps it with delimiter sniffing over a bounded preview and yields a fresh cell array
 * per row. A malformed quoted field is fatal for both: an unterminated quote swallows the rest of
 * the file, and text after a closing quote makes every later cell boundary unreliable.
 */

import { GraphFormatError } from "@graphty/graph-format";

import { type ReadOptions, textChunks } from "../../common/input.js";
import { type ImportReportBuilder } from "../../common/report.js";
import { type ImportInput } from "../../types.js";

/** Issue code: a quoted field is never closed; the import aborts (everything after it would be one cell). */
export const UNCLOSED_QUOTE_CODE = "E_CSV_UNCLOSED_QUOTE";

/** Issue code: a closing quote is followed by text other than a delimiter or a line break; the import aborts. */
export const BAD_QUOTE_CODE = "E_CSV_QUOTE";

/** The delimiters tried, in priority order, when none is given. */
const DELIMITER_CANDIDATES: readonly string[] = Object.freeze([",", "\t", ";", "|", " "]);

/** Rows the delimiter sniff looks at. */
const PREVIEW_ROWS = 10;

/** Characters buffered for the sniff when the input has fewer than PREVIEW_ROWS line breaks. */
const PREVIEW_CHARS = 64 * 1024;

const LF = 10;
const CR = 13;

/** Tokeniser state: at the start of a field. */
const START = 0;
/** Tokeniser state: inside an unquoted field. */
const UNQUOTED = 1;
/** Tokeniser state: inside a quoted field. */
const QUOTED = 2;
/** Tokeniser state: just after a quote inside a quoted field (a second quote is a literal, anything else ends the field). */
const CLOSING = 3;
/** Tokeniser state: after a closed quoted field, before the delimiter (text here is malformed). */
const AFTER_QUOTED = 4;
/** Inside a leading comment line (skipped up to its line break). */
const COMMENT = 5;

type State = typeof START | typeof UNQUOTED | typeof QUOTED | typeof CLOSING | typeof AFTER_QUOTED | typeof COMMENT;

/** The delimiter and quote of a reader. */
export interface RecordSyntax {
    /** The field delimiter, one character; null sniffs it from the first rows. */
    readonly delimiter: string | null;
    /** The quote character, one character. */
    readonly quote: string;
    /** The delimiters tried when `delimiter` is null; DELIMITER_CANDIDATES by default. */
    readonly candidates?: readonly string[] | undefined;
    /**
     * Characters that open a comment line while no record has been read yet (SNAP `#`, KONECT
     * `%`): such leading lines are skipped, kept in `leadingComments` and left out of the delimiter
     * sniff. A later line starting with one of them is an ordinary record. Default: none.
     */
    readonly comments?: readonly string[] | undefined;
}

/**
 * Check a delimiter or quote option: exactly one character that is not a line break, and the two
 * must differ.
 * @param syntax - the delimiter (or null for sniffing) and quote
 * @returns the syntax unchanged; E_UNSUPPORTED when invalid
 */
export function checkRecordSyntax(syntax: RecordSyntax): RecordSyntax {
    for (const [option, value] of [
        ["delimiter", syntax.delimiter],
        ["quote", syntax.quote],
    ] as const) {
        if (value === null) {
            continue;
        }
        if (value.length !== 1 || value === "\n" || value === "\r") {
            throw new GraphFormatError(
                "E_UNSUPPORTED",
                `option ${option}: ${JSON.stringify(value)} is not a single non-line-break character`,
                { option, found: value },
            );
        }
    }
    if (syntax.delimiter !== null && syntax.delimiter === syntax.quote) {
        throw new GraphFormatError("E_UNSUPPORTED", "options delimiter and quote must differ", {
            option: "delimiter",
            found: syntax.delimiter,
        });
    }
    return syntax;
}

/**
 * Sniff the line terminator of a text: a lone `\r` only when the text has `\r` and no `\n` at all
 * (classic Mac files); `\n` otherwise.
 * @param text - the preview text
 * @returns "\n" or "\r"
 */
export function sniffNewline(text: string): "\n" | "\r" {
    return !text.includes("\n") && text.includes("\r") ? "\r" : "\n";
}

/**
 * Split a text into records synchronously (the first `maxRows` of them), honouring quotes; for
 * the delimiter sniff and the registry's head sniff, where the input is a bounded preview.
 * @param text - the text
 * @param delimiter - the delimiter
 * @param quote - the quote character
 * @param maxRows - the most rows to return
 * @returns the rows as cell arrays (blank lines skipped)
 */
function splitRecords(text: string, delimiter: string, quote: string, maxRows: number): string[][] {
    const rows: string[][] = [];
    const delimiterCode = delimiter.charCodeAt(0);
    const quoteCode = quote.charCodeAt(0);
    let cells: string[] = [];
    let state: State = START;
    let segment = 0;
    let field = "";
    const n = text.length;
    for (let i = 0; i < n && rows.length < maxRows; i++) {
        const c = text.charCodeAt(i);
        if (state === QUOTED) {
            if (c === quoteCode) {
                field += text.slice(segment, i);
                state = CLOSING;
            }
            continue;
        }
        if (state === CLOSING) {
            if (c === quoteCode) {
                field += quote;
                segment = i + 1;
                state = QUOTED;
                continue;
            }
            state = AFTER_QUOTED;
            segment = i;
        }
        if (c === delimiterCode || c === LF || c === CR) {
            if (state === UNQUOTED || state === AFTER_QUOTED) {
                field += text.slice(segment, i);
            }
            if (c === delimiterCode) {
                cells.push(field);
                field = "";
                state = START;
            } else {
                if (state !== START || cells.length > 0) {
                    cells.push(field);
                    rows.push(cells);
                    cells = [];
                    field = "";
                }
                state = START;
                if (c === CR && text.charCodeAt(i + 1) === LF) {
                    i++;
                }
            }
            segment = i + 1;
            continue;
        }
        if (state === START) {
            if (c === quoteCode) {
                state = QUOTED;
                segment = i + 1;
            } else {
                state = UNQUOTED;
                segment = i;
            }
        }
    }
    if (rows.length < maxRows && (state !== START || cells.length > 0)) {
        if (state === UNQUOTED || state === AFTER_QUOTED || state === QUOTED) {
            field += text.slice(segment, n);
        }
        cells.push(field);
        rows.push(cells);
    }
    return rows;
}

/**
 * The preview text without its leading comment lines (those starting with one of the comment
 * characters), so the sniff sees the records only.
 * @param text - the preview text
 * @param comments - the comment characters
 * @returns the text from the first non-comment line on
 */
function stripLeadingComments(text: string, comments: readonly string[]): string {
    if (comments.length === 0) {
        return text;
    }
    let at = 0;
    while (at < text.length && comments.includes(text[at])) {
        const lf = text.indexOf("\n", at);
        const cr = text.indexOf("\r", at);
        const end = Math.min(lf < 0 ? Infinity : lf, cr < 0 ? Infinity : cr);
        if (end === Infinity) {
            return "";
        }
        at = end + 1;
        if (text[at - 1] === "\r" && text[at] === "\n") {
            at++;
        }
    }
    return at === 0 ? text : text.slice(at);
}

/**
 * Sniff the delimiter of a text: every candidate is tried over the first rows and the one whose
 * field count is most consistent across rows wins (ties broken by the higher field count); a
 * candidate that yields fewer than two fields per row on average is never chosen. `null` when no
 * candidate qualifies (a single-column file).
 * @param text - the preview text
 * @param newline - the sniffed line terminator (unused by the splitter, which reads every kind;
 * kept for callers that sniffed it)
 * @param candidates - the delimiters to try, in priority order
 * @param quote - the quote character
 * @returns the delimiter, or null
 */
export function sniffDelimiter(
    text: string,
    newline: "\n" | "\r" = "\n",
    candidates: readonly string[] = DELIMITER_CANDIDATES,
    quote = '"',
): string | null {
    let best: string | null = null;
    let bestDelta = Infinity;
    let bestAverage = 0;
    const terminated = text.endsWith("\n") || text.endsWith("\r");
    for (const delimiter of candidates) {
        if (delimiter === quote || delimiter === newline) {
            continue;
        }
        let rows = splitRecords(text, delimiter, quote, PREVIEW_ROWS).filter((row) => !isBlankRow(row));
        if (rows.length > 1 && !terminated) {
            // the preview is a prefix of the file: its last row may be cut short
            rows = rows.slice(0, -1);
        }
        if (rows.length === 0) {
            continue;
        }
        let total = 0;
        let delta = 0;
        for (let i = 0; i < rows.length; i++) {
            const count = rows[i].length;
            total += count;
            if (i > 0) {
                delta += Math.abs(count - rows[i - 1].length);
            }
        }
        const average = total / rows.length;
        if (average > 1.99 && (delta < bestDelta || (delta === bestDelta && average > bestAverage))) {
            best = delimiter;
            bestDelta = delta;
            bestAverage = average;
        }
    }
    return best;
}

/**
 * Whether a parsed row is a blank line: one cell holding only whitespace.
 * @param row - the row
 * @returns true for a blank line
 */
function isBlankRow(row: readonly string[]): boolean {
    return row.length === 1 && row[0].trim().length === 0;
}

/**
 * Reads the records of one input. Iterate with `for await (const count of reader)`: each
 * iteration fills `reader.cells[0..count)` and `reader.quoted[0..count)` and sets `reader.line`
 * to the 1-based line the record started on. Blank lines (an empty unquoted record of one field)
 * are skipped. The arrays are reused between records. When the syntax gives no delimiter, the
 * first rows (PREVIEW_ROWS, or PREVIEW_CHARS characters) are buffered and the delimiter sniffed
 * from them before the first record is yielded.
 */
export class RecordReader implements AsyncIterable<number> {
    /** The cells of the record most recently yielded; only the first `count` entries are valid. */
    readonly cells: string[] = [];

    /** Whether each cell of the record most recently yielded was quoted. */
    readonly quoted: boolean[] = [];

    /** The leading comment lines (without their line breaks), in order; empty without `syntax.comments`. */
    readonly leadingComments: string[] = [];

    private readonly input: ImportInput;

    private readonly report: ImportReportBuilder;

    private readonly readOptions: ReadOptions;

    /** The delimiter (null while unsniffed), quote and comment characters. */
    readonly syntax: RecordSyntax;

    private delimiterText: string | null;

    private recordLine = 0;

    /**
     * Create a reader; nothing is read until iteration starts.
     * @param input - the input (or an async iterable of already decoded text chunks)
     * @param report - the report decode and quoting errors are recorded in
     * @param syntax - the delimiter (null to sniff) and quote (already checked)
     * @param readOptions - cancellation and progress
     */
    constructor(input: ImportInput, report: ImportReportBuilder, syntax: RecordSyntax, readOptions: ReadOptions) {
        this.input = input;
        this.report = report;
        this.readOptions = readOptions;
        this.syntax = syntax;
        this.delimiterText = syntax.delimiter;
    }

    /**
     * The 1-based line the most recently yielded record started on.
     * @returns the line number; 0 before the first record
     */
    get line(): number {
        return this.recordLine;
    }

    /**
     * The delimiter in use.
     * @returns the given or sniffed delimiter; null before the sniff ran
     */
    get delimiter(): string | null {
        return this.delimiterText;
    }

    /**
     * Iterate the records.
     * @yields the number of cells of each record
     * @returns nothing
     */
    async *[Symbol.asyncIterator](): AsyncGenerator<number, void, undefined> {
        const scanner = new RecordScanner(this);
        for await (const chunk of this.chunks()) {
            // the first chunk arrives after the sniff (or with the given delimiter)
            scanner.setDelimiter((this.delimiterText ?? ",").charCodeAt(0));
            let i = 0;
            while (i < chunk.length) {
                i = scanner.scan(chunk, i);
                if (scanner.ready) {
                    scanner.ready = false;
                    this.recordLine = scanner.recordLine;
                    yield scanner.count;
                    scanner.count = 0;
                }
            }
            scanner.endChunk(chunk);
        }
        if (scanner.finish()) {
            this.recordLine = scanner.recordLine;
            yield scanner.count;
        }
    }

    /**
     * Record a text-after-quote error (fatal).
     * @param line - the line the field started on
     * @returns never
     */
    badQuote(line: number): never {
        return this.report.fail(BAD_QUOTE_CODE, `line ${line}: text after the closing quote of a quoted field`, {
            line,
        });
    }

    /**
     * Record an unclosed-quote error (fatal).
     * @param line - the line the field started on
     * @returns never
     */
    unclosedQuote(line: number): never {
        return this.report.fail(UNCLOSED_QUOTE_CODE, `a quoted field starting on line ${line} is never closed`, {
            line,
        });
    }

    /**
     * The decoded text chunks of the input; when the delimiter is to be sniffed, the first rows
     * are buffered (bounded by PREVIEW_ROWS line breaks or PREVIEW_CHARS characters), the sniff
     * runs, and the buffered text is yielded as one chunk before the rest streams through.
     * @yields the text chunks
     * @returns nothing
     */
    private async *chunks(): AsyncGenerator<string, void, undefined> {
        const source = textChunks(this.input, this.report, this.readOptions);
        if (this.delimiterText !== null) {
            yield* source;
            return;
        }
        const pieces: string[] = [];
        let length = 0;
        let breaks = 0;
        let lastWasCr = false;
        let sniffed = false;
        for await (const chunk of source) {
            if (sniffed) {
                yield chunk;
                continue;
            }
            pieces.push(chunk);
            length += chunk.length;
            for (let i = 0; i < chunk.length && breaks < PREVIEW_ROWS; i++) {
                const c = chunk.charCodeAt(i);
                if (c === LF) {
                    if (!lastWasCr) {
                        breaks++;
                    }
                } else if (c === CR) {
                    breaks++;
                }
                lastWasCr = c === CR;
            }
            if (breaks < PREVIEW_ROWS && length < PREVIEW_CHARS) {
                continue;
            }
            const text = pieces.join("");
            pieces.length = 0;
            this.sniff(text);
            sniffed = true;
            yield text;
        }
        if (!sniffed) {
            const text = pieces.join("");
            if (text.length > 0) {
                this.sniff(text);
                yield text;
            }
        }
    }

    /**
     * Sniff the delimiter from the preview text and record it.
     * @param text - the preview
     */
    private sniff(text: string): void {
        const candidates = this.syntax.candidates ?? DELIMITER_CANDIDATES;
        // the sniff is a heuristic over the first rows: a single chunk holding a huge quoted cell
        // is capped so the candidate scans stay bounded
        const body = stripLeadingComments(text.slice(0, PREVIEW_CHARS), this.syntax.comments ?? []);
        this.delimiterText = sniffDelimiter(body, sniffNewline(body), candidates, this.syntax.quote) ?? candidates[0];
    }
}

/**
 * The record state machine of RecordReader, resumable at every record end so the reader can
 * yield a record before the next one overwrites the shared cell arrays. Every state lives on the
 * instance; `scan()` runs the per-character loop over one chunk from a position and returns where
 * it stopped (after the character that completed a record, with `ready` set, or the chunk end).
 */
class RecordScanner {
    /** Whether a record just completed. */
    ready = false;

    /** The cell count of the record in progress (the completed one while `ready`). */
    count = 0;

    /** The line the completed record started on (valid while `ready`, and after finish()). */
    recordLine = 1;

    /** The line the record in progress started on. */
    private startLine = 1;

    private readonly reader: RecordReader;

    private readonly quoteCode: number;

    private readonly quoteText: string;

    private readonly commentCodes: readonly number[];

    private delimiterCode = -1;

    private state: State = START;

    private field = "";

    private fieldQuoted = false;

    private line = 1;

    private lastWasCr = false;

    /** Whether no record has started yet (comment lines are recognised until one does). */
    private leading: boolean;

    private comment = "";

    /** The start of the unconsumed run of the current chunk that belongs to the field or comment. */
    private segment = 0;

    /**
     * Create a scanner over a reader's cell arrays.
     * @param reader - the reader
     */
    constructor(reader: RecordReader) {
        this.reader = reader;
        this.quoteCode = reader.syntax.quote.charCodeAt(0);
        this.quoteText = reader.syntax.quote;
        this.commentCodes = (reader.syntax.comments ?? []).map((c) => c.charCodeAt(0));
        this.leading = this.commentCodes.length > 0;
    }

    /**
     * Set the delimiter once it is known (before the first chunk is scanned).
     * @param code - the delimiter's character code
     */
    setDelimiter(code: number): void {
        if (this.delimiterCode < 0) {
            this.delimiterCode = code;
        }
    }

    /**
     * Scan one chunk from a position until a record completes or the chunk ends.
     * @param chunk - the chunk
     * @param from - where to start
     * @returns the position after the last character consumed
     */
    scan(chunk: string, from: number): number {
        const { quoteCode, delimiterCode } = this;
        const n = chunk.length;
        let i = from;
        this.segment = from;
        while (i < n) {
            const c = chunk.charCodeAt(i);
            if (this.state === COMMENT) {
                if (c === LF || c === CR) {
                    this.comment += chunk.slice(this.segment, i);
                    this.reader.leadingComments.push(this.comment);
                    this.comment = "";
                    this.state = START;
                    this.endLine(c);
                    this.segment = i + 1;
                }
                i++;
                continue;
            }
            if (this.state === QUOTED) {
                // jump to the next quote; the run in between is field text whose line breaks are counted
                const q = chunk.indexOf(this.quoteText, i);
                const end = q < 0 ? n : q;
                if (end > i) {
                    this.countLineBreaks(chunk, i, end);
                }
                if (q < 0) {
                    i = n;
                    break;
                }
                this.field += chunk.slice(this.segment, q);
                this.state = CLOSING;
                this.lastWasCr = false;
                i = q + 1;
                continue;
            }
            if (this.state === CLOSING) {
                if (c === quoteCode) {
                    this.field += this.quoteText;
                    this.segment = i + 1;
                    this.state = QUOTED;
                    this.lastWasCr = false;
                    i++;
                    continue;
                }
                if (c !== delimiterCode && c !== LF && c !== CR) {
                    this.reader.badQuote(this.startLine);
                }
                this.state = AFTER_QUOTED;
                this.segment = i;
                // fall through to the delimiter / line-end handling below without consuming c
            }
            if (c === delimiterCode || c === LF || c === CR) {
                if (c === LF && this.lastWasCr) {
                    // the second half of a CRLF already ended the record
                    this.lastWasCr = false;
                    this.segment = i + 1;
                    i++;
                    continue;
                }
                if (this.state === UNQUOTED) {
                    this.field += chunk.slice(this.segment, i);
                }
                if (c === delimiterCode) {
                    this.push();
                    this.state = START;
                    this.lastWasCr = false;
                    this.segment = i + 1;
                    i++;
                    continue;
                }
                const blank = this.state === START && this.count === 0;
                if (!blank) {
                    this.push();
                    this.ready = true;
                    this.recordLine = this.startLine;
                }
                this.state = START;
                this.endLine(c);
                this.segment = i + 1;
                i++;
                if (this.ready) {
                    return i;
                }
                continue;
            }
            this.lastWasCr = false;
            if (this.state === START) {
                if (this.leading && this.count === 0 && this.commentCodes.includes(c)) {
                    this.state = COMMENT;
                    this.segment = i;
                    i++;
                    continue;
                }
                this.leading = false;
                if (c === quoteCode) {
                    this.state = QUOTED;
                    this.fieldQuoted = true;
                    this.segment = i + 1;
                } else {
                    this.state = UNQUOTED;
                    this.segment = i;
                }
            }
            i++;
        }
        return i;
    }

    /**
     * Keep the tail of a chunk that belongs to an open field or comment.
     * @param chunk - the chunk just scanned to its end
     */
    endChunk(chunk: string): void {
        const n = chunk.length;
        if (this.segment >= n) {
            return;
        }
        if (this.state === COMMENT) {
            this.comment += chunk.slice(this.segment, n);
        } else if (this.state === QUOTED || this.state === UNQUOTED) {
            this.field += chunk.slice(this.segment, n);
        }
        this.segment = n;
    }

    /**
     * The end of the input: an open comment is kept, an open quote is fatal, an open record
     * (no trailing line break) is completed.
     * @returns true when a last record is ready
     */
    finish(): boolean {
        const { state } = this;
        if (state === COMMENT) {
            this.reader.leadingComments.push(this.comment);
            return false;
        }
        if (state === QUOTED) {
            this.reader.unclosedQuote(this.startLine);
        }
        if (state !== START || this.count > 0) {
            this.push();
            this.recordLine = this.startLine;
            return true;
        }
        return false;
    }

    /**
     * Count the line breaks of a run of quoted text (a CRLF counts once, like the record loop).
     * @param chunk - the chunk
     * @param from - the first index of the run
     * @param to - the index after the run
     */
    private countLineBreaks(chunk: string, from: number, to: number): void {
        let { line, lastWasCr } = this;
        for (let i = from; i < to; i++) {
            const c = chunk.charCodeAt(i);
            if (c === LF) {
                if (!lastWasCr) {
                    line++;
                }
                lastWasCr = false;
            } else if (c === CR) {
                line++;
                lastWasCr = true;
            } else {
                lastWasCr = false;
            }
        }
        this.line = line;
        this.lastWasCr = lastWasCr;
    }

    /** Store the field in progress as the next cell. */
    private push(): void {
        const { reader, count } = this;
        reader.cells[count] = this.field;
        reader.quoted[count] = this.fieldQuoted;
        this.count = count + 1;
        this.field = "";
        this.fieldQuoted = false;
    }

    /**
     * Count a line break that ends a record or a comment line.
     * @param c - the LF or CR
     */
    private endLine(c: number): void {
        if (!(c === LF && this.lastWasCr)) {
            this.line++;
        }
        this.startLine = this.line;
        this.lastWasCr = c === CR;
    }
}

/** What the CSV importer's reader needs besides the input. */
export interface CsvReaderOptions extends ReadOptions {
    /** The delimiter; null or undefined sniffs it from the preview. */
    readonly delimiter?: string | null | undefined;
    /** The characters opening a leading comment line (RecordSyntax.comments); none by default. */
    readonly comments?: readonly string[] | undefined;
}

/**
 * Records of a CSV input, one string array per row, streamed: iterate with
 * `for await (const row of reader)` and read `reader.line` for the 1-based line the row starts
 * on. The delimiter is known after the first row (`reader.delimiter`). Rows come out as fresh
 * arrays of cell texts exactly as written (no trimming, no typing); `reader.quoted` tells, for
 * the row just yielded, which cells were quoted (a quoted empty cell is the empty string, an
 * unquoted one is "not set"). Blank lines and whitespace-only single-cell lines are skipped.
 */
export class CsvRecordReader implements AsyncIterable<string[]> {
    private readonly inner: RecordReader;

    /**
     * Create a reader; nothing is read until iteration starts.
     * @param input - the input
     * @param report - the report parse errors are recorded in
     * @param options - the delimiter, cancellation and progress
     */
    constructor(input: ImportInput, report: ImportReportBuilder, options: CsvReaderOptions = {}) {
        this.inner = new RecordReader(
            input,
            report,
            { delimiter: options.delimiter ?? null, quote: '"', comments: options.comments },
            { signal: options.signal, onProgress: options.onProgress },
        );
    }

    /**
     * The leading comment lines the reader skipped (SNAP `#` lines, the KONECT `%` header), in order.
     * @returns the lines without their line breaks
     */
    get leadingComments(): readonly string[] {
        return this.inner.leadingComments;
    }

    /**
     * The 1-based line the most recently yielded row starts on.
     * @returns the line; 0 before the first row
     */
    get line(): number {
        return this.inner.line;
    }

    /**
     * The delimiter in use.
     * @returns the given or sniffed delimiter; null before the sniff ran
     */
    get delimiter(): string | null {
        return this.inner.delimiter;
    }

    /**
     * Whether each cell of the row most recently yielded was quoted.
     * @returns the flags (valid for the first `row.length` entries)
     */
    get quoted(): readonly boolean[] {
        return this.inner.quoted;
    }

    /**
     * Iterate the rows.
     * @yields one row at a time as its cell texts
     * @returns nothing
     */
    async *[Symbol.asyncIterator](): AsyncGenerator<string[], void, undefined> {
        const { inner } = this;
        for await (const count of inner) {
            if (count === 1 && !inner.quoted[0] && inner.cells[0].trim().length === 0) {
                continue;
            }
            yield inner.cells.slice(0, count);
        }
    }
}
