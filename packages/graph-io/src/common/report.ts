/**
 * The ImportReport builder shared by every importer (design section 8.6): issues by category and
 * severity, the element counts, loss notes, the error limit with truncation, and the ImportError
 * that carries the partial report when the importer aborts.
 *
 * Layering (design section 8.6): the builder throws on the first hard error; the importer catches
 * per element, records an ImportIssue through this class, skips the element and continues until the
 * error limit, then aborts with E_IMPORT. `recordError()` is that catch: it turns a thrown
 * GraphFormatError into an issue with the category its code implies and re-throws everything else
 * (an ImportError, an abort reason, a programming error) unchanged.
 */

import { GraphFormatError, type GraphFormatErrorCode } from "@graphty/graph-format";

import { ImportError, type ImportIssue, type ImportReport, type IssueCategory, type LossNote } from "../types.js";

/**
 * Where an issue was found: the 1-based line and the element (id or attribute name) when known.
 * Consumed by the per-format importers and exporters under src/formats.
 * @public
 */
export interface IssueLocation {
    /** The 1-based source line, when known. */
    readonly line?: number | null | undefined;
    /** The element the issue is about, when known. */
    readonly element?: string | null | undefined;
}

/**
 * The mutable element counters of a report in progress; importers increment them directly in hot loops.
 * Consumed by the per-format importers and exporters under src/formats.
 * @public
 */
export interface MutableCounts {
    /** Nodes pushed. */
    nodes: number;
    /** Logical edges pushed (both halves of an expanded edge count). */
    edges: number;
    /** Nodes skipped after an error. */
    skippedNodes: number;
    /** Edges skipped after an error. */
    skippedEdges: number;
    /** Source edges expanded into two logical edges (design section 3.6). */
    expandedMixed: number;
}

/**
 * The category a thrown GraphFormatError maps to when an importer catches it per element: the codes
 * that mean "the file referenced something it never declared" are missing values, a refused
 * direction change is a coercion, size and feature limits are unsupported, everything else is a
 * validation error of the element.
 */
const CATEGORY_BY_CODE: Readonly<Partial<Record<GraphFormatErrorCode, IssueCategory>>> = {
    E_UNKNOWN_NODE: "missing-value",
    E_UNKNOWN_COLUMN: "missing-value",
    E_DIRECTED: "coercion",
    E_TOO_LARGE: "unsupported",
    E_UNSUPPORTED: "unsupported",
    E_GPU_INELIGIBLE: "unsupported",
};

/**
 * The code a report used to give a thrown value that was not a GraphFormatError. Since audit round
 * 1 such a value (a TypeError, a RangeError: a bug in an importer or a sink, never a defect of the
 * input) propagates out of import() untouched, so no importer records this code any more; the
 * constant stays for consumers that switch on the codes of earlier reports.
 */
export const PARSE_ERROR_CODE = "E_PARSE";

/**
 * Accumulates an ImportReport while an importer runs. Errors count toward the error limit; the
 * error that takes the count beyond the limit is still recorded, `truncated` is set, and an
 * ImportError carrying the report so far is thrown (design section 8.4: "beyond it the importer
 * aborts with E_IMPORT"). Warnings never abort.
 */
export class ImportReportBuilder {
    /** The importer's format name. */
    readonly format: string;

    /** The error limit the report was created with. */
    readonly errorLimit: number;

    /** The element counters; importers increment them directly. */
    readonly counts: MutableCounts = { nodes: 0, edges: 0, skippedNodes: 0, skippedEdges: 0, expandedMixed: 0 };

    private readonly issueList: ImportIssue[] = [];

    private readonly lossList: LossNote[] = [];

    private readonly onceCodes = new Set<string>();

    private errors = 0;

    private warnings = 0;

    private truncatedFlag = false;

    private readonly startedAt: number;

    /**
     * Create a builder for one import call.
     * @param format - the importer's format name
     * @param errorLimit - errors tolerated before the import aborts; Infinity for no limit
     */
    constructor(format: string, errorLimit: number) {
        this.format = format;
        this.errorLimit = errorLimit;
        this.startedAt = now();
    }

    /**
     * Issues recorded so far, in order.
     * @returns the live list (not copied; finish() copies)
     */
    get issues(): readonly ImportIssue[] {
        return this.issueList;
    }

    /**
     * Issues with severity "error".
     * @returns the count
     */
    get errorCount(): number {
        return this.errors;
    }

    /**
     * Issues with severity "warning".
     * @returns the count
     */
    get warningCount(): number {
        return this.warnings;
    }

    /**
     * Whether the error limit was exceeded.
     * @returns true once an error beyond the limit was recorded
     */
    get truncated(): boolean {
        return this.truncatedFlag;
    }

    /**
     * Loss notes recorded so far.
     * @returns the live list
     */
    get lossy(): readonly LossNote[] {
        return this.lossList;
    }

    /**
     * Record an error. When the count goes beyond the error limit the report is marked truncated
     * and an ImportError carrying it is thrown; the importer does not catch that.
     * @param category - the issue category
     * @param code - a stable code such as "E_UNKNOWN_NODE"
     * @param message - a plain-ASCII message
     * @param where - the line and element, when known
     * @returns the recorded issue
     */
    error(category: IssueCategory, code: string, message: string, where?: IssueLocation): ImportIssue {
        const issue = makeIssue(category, "error", code, message, where);
        this.issueList.push(issue);
        this.errors++;
        if (this.errors > this.errorLimit) {
            this.truncatedFlag = true;
            throw this.abort(`error limit of ${this.errorLimit} exceeded: ${message}`, {
                code,
                limit: this.errorLimit,
            });
        }
        return issue;
    }

    /**
     * Record a warning; warnings never count toward the limit.
     * @param category - the issue category
     * @param code - a stable code such as "W_WIDENED"
     * @param message - a plain-ASCII message
     * @param where - the line and element, when known
     * @returns the recorded issue
     */
    warning(category: IssueCategory, code: string, message: string, where?: IssueLocation): ImportIssue {
        const issue = makeIssue(category, "warning", code, message, where);
        this.issueList.push(issue);
        this.warnings++;
        return issue;
    }

    /**
     * Record a warning the first time its key is seen and ignore later repeats, for conditions
     * that would otherwise produce one warning per row.
     * @param category - the issue category
     * @param code - the stable code
     * @param message - a plain-ASCII message
     * @param where - the line and element, when known
     * @param key - what repeats are deduplicated on; the code itself by default (pass `code:column`
     * to warn once per column)
     * @returns the recorded issue, or null when the key was already recorded
     */
    warnOnce(
        category: IssueCategory,
        code: string,
        message: string,
        where?: IssueLocation,
        key: string = code,
    ): ImportIssue | null {
        if (this.onceCodes.has(key)) {
            return null;
        }
        this.onceCodes.add(key);
        return this.warning(category, code, message, where);
    }

    /**
     * Record something the importer could not represent.
     * @param code - a stable code
     * @param message - a plain-ASCII message
     * @param column - the affected column, or null
     * @param count - the affected count, or null
     * @returns the recorded note
     */
    loss(code: string, message: string, column: string | null = null, count: number | null = null): LossNote {
        const note: LossNote = Object.freeze({ code, message, column, count });
        this.lossList.push(note);
        return note;
    }

    /**
     * The per-element catch of design section 8.6: record a thrown GraphFormatError as an error
     * issue with its code and the category the code implies. Anything else is re-thrown unchanged:
     * an ImportError (the import already aborted), an abort reason, and any other thrown value (a
     * TypeError or RangeError is a programming error in the importer or the sink, never a defect of
     * the input, so it must surface as an exception rather than as an issue blamed on the file).
     * @param err - the thrown value
     * @param where - the line and element, when known
     * @returns the recorded issue
     */
    recordError(err: unknown, where?: IssueLocation): ImportIssue {
        if (err instanceof GraphFormatError && !(err instanceof ImportError)) {
            const category = CATEGORY_BY_CODE[err.code] ?? "validation-error";
            return this.error(category, err.code, err.message, where);
        }
        throw err;
    }

    /**
     * Build the ImportError that aborts the import, carrying the report so far. The caller throws
     * it; this method only constructs it so it can be used in expression position.
     * @param message - a plain-ASCII message
     * @param details - optional machine-readable context
     * @returns the error to throw
     */
    abort(message: string, details?: Readonly<Record<string, unknown>>): ImportError {
        return new ImportError(message, this.finish(), details);
    }

    /**
     * Record a fatal parse error and abort: the issue is recorded (even beyond the error limit)
     * and an ImportError carrying the report is thrown.
     * @param code - a stable code such as "E_INVALID_UTF8"
     * @param message - a plain-ASCII message
     * @param where - the line and element, when known
     * @param details - optional machine-readable context for the ImportError
     */
    fail(code: string, message: string, where?: IssueLocation, details?: Readonly<Record<string, unknown>>): never {
        const issue = makeIssue("parse-error", "error", code, message, where);
        this.issueList.push(issue);
        this.errors++;
        if (this.errors > this.errorLimit) {
            this.truncatedFlag = true;
        }
        throw this.abort(message, { code, ...details });
    }

    /**
     * The report as it stands: a frozen snapshot with the parse duration so far. Can be called more
     * than once; each call reflects everything recorded up to that point.
     * @returns the report
     */
    finish(): ImportReport {
        const { nodes, edges, skippedNodes, skippedEdges, expandedMixed } = this.counts;
        return Object.freeze({
            format: this.format,
            counts: Object.freeze({ nodes, edges, skippedNodes, skippedEdges, expandedMixed }),
            issues: Object.freeze([...this.issueList]),
            errorCount: this.errors,
            warningCount: this.warnings,
            truncated: this.truncatedFlag,
            lossy: Object.freeze([...this.lossList]),
            durationMs: now() - this.startedAt,
        });
    }
}

/**
 * Build a frozen ImportIssue.
 * @param category - the category
 * @param severity - error or warning
 * @param code - the stable code
 * @param message - the message
 * @param where - the location, when known
 * @returns the issue
 */
function makeIssue(
    category: IssueCategory,
    severity: "error" | "warning",
    code: string,
    message: string,
    where: IssueLocation | undefined,
): ImportIssue {
    return Object.freeze({
        category,
        severity,
        code,
        message,
        line: where?.line ?? null,
        element: where?.element ?? null,
    });
}

/**
 * Whether a thrown value is an abort reason (the DOMException or Error named "AbortError" that
 * AbortSignal.reason holds), which an importer must let through untouched.
 * @param err - the thrown value
 * @returns true for an AbortError
 */
export function isAbortError(err: unknown): boolean {
    return typeof err === "object" && err !== null && (err as { name?: unknown }).name === "AbortError";
}

/**
 * The message of a thrown value: an Error's message, a string as is, anything else described by type.
 * @param err - the thrown value
 * @returns a plain message
 */
export function messageOf(err: unknown): string {
    if (err instanceof Error) {
        return err.message;
    }
    if (typeof err === "string") {
        return err;
    }
    return `non-error thrown (${typeof err})`;
}

/**
 * A monotonic millisecond clock: performance.now() where it exists, Date.now() otherwise.
 * @returns milliseconds
 */
function now(): number {
    return typeof performance === "object" ? performance.now() : Date.now();
}
