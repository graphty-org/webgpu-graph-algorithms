/**
 * Attribute declaration shared by the typed importers (design sections 5.1, 5.5 and 5.6): turn a
 * GEXF `<attribute>`, a GraphML `<key>` or a Neo4j header column into a ColumnDecl with the right
 * dtype, default, options and origin metadata; parse values (scalars and lists) by the declaration;
 * derive the deterministic `<name>#<origin.id>` name when a declared name collides; and the dict
 * heuristic of design section 5.4.
 *
 * Naming rule (design section 5.6): a column's name is the GEXF `title` / GraphML `attr.name`,
 * falling back to the attribute `id` when the title is absent; `origin.id` always keeps the source
 * id and `origin.title` the title when it differs from the resolved name.
 */

import {
    type ColumnDecl,
    type ColumnHandle,
    type ColumnRole,
    GraphFormatError,
    type GraphSink,
    INVALID_INDEX,
} from "@graphty/graph-format";

import { type IssueCategory } from "../types.js";
import {
    BAD_DEFAULT_CODE,
    BAD_OPTIONS_CODE,
    COLUMN_RENAMED_CODE,
    PRECISION_CODE,
    ROLE_TAKEN_CODE,
    UNKNOWN_ATTR_TYPE_CODE,
} from "./codes.js";
import {
    type DeclaredTypeSpec,
    type DeclaringFormat,
    mapDeclaredType,
    parseScalarText,
    stringSpec,
} from "./declared-types.js";
import { type ListSyntax, splitListText } from "./lists.js";
import { type ImportReportBuilder, type IssueLocation } from "./report.js";
import { parseTemporal, type TemporalValue, timeTextCompanion } from "./temporal.js";

/**
 * What an importer knows about a declared attribute.
 * Consumed by the per-format importers and exporters under src/formats.
 * @public
 */
export interface AttributeDeclarationInput {
    /** The declaring format. */
    readonly format: DeclaringFormat;
    /** The attribute id (GEXF `id`, GraphML key `id`, Neo4j header name); null when the format has none. */
    readonly id: string | null;
    /** The GEXF `title` / GraphML `attr.name`; null when absent (yFiles keys omit it). */
    readonly title: string | null;
    /** The declared type text; null for an untyped declaration (stored as string). */
    readonly type: string | null;
    /** The declaring namespace ("viz", "yfiles", "neo4j"), or null. */
    readonly namespace?: string | null | undefined;
    /** The `<default>` text, or null. */
    readonly defaultText?: string | null | undefined;
    /** The GEXF `<options>` text, or null. */
    readonly optionsText?: string | null | undefined;
    /** The list syntax of defaults, options and values; "gexf" by default. */
    readonly listSyntax?: ListSyntax | undefined;
    /** A role to give the column, or null. */
    readonly role?: ColumnRole | null | undefined;
    /** Whether the attribute is dynamic (GEXF mode="dynamic"). */
    readonly dynamic?: boolean | undefined;
    /** The importer's `long` option. */
    readonly long: "f64" | "string";
    /** Whether a column name is already taken in the target table; drives the `#id` rename. */
    readonly taken?: ((name: string) => boolean) | undefined;
}

/**
 * A non-fatal problem with a declaration, for the importer to record as a warning.
 * Consumed by the per-format importers and exporters under src/formats.
 * @public
 */
export interface DeclarationIssue {
    /** The issue category. */
    readonly category: IssueCategory;
    /** The stable code. */
    readonly code: string;
    /** A plain-ASCII message. */
    readonly message: string;
}

/**
 * A declared attribute: the ColumnDecl to push, the spec to parse values with, and what to report.
 * Consumed by the per-format importers and exporters under src/formats.
 * @public
 */
export interface DeclaredAttribute {
    /** The declaration for declareNodeColumn / declareEdgeColumn. */
    readonly decl: ColumnDecl;
    /** How values are parsed. */
    readonly spec: DeclaredTypeSpec;
    /** The list syntax values are split with. */
    readonly listSyntax: ListSyntax;
    /** The companion text column of a temporal attribute (design section 5.1), or null. */
    readonly companion: ColumnDecl | null;
    /** Whether the name received the `#id` suffix. */
    readonly renamed: boolean;
    /** Problems the importer records as warnings (unknown type, unparsable default or options). */
    readonly issues: readonly DeclarationIssue[];
}

/** Issue code: the declared type is not one the format defines; the column is kept as string. */
export const UNKNOWN_TYPE_CODE = UNKNOWN_ATTR_TYPE_CODE;
/** Issue code: the column was renamed `<name>#<id>` because the name was taken (design section 5.6). */
export const RENAMED_CODE = COLUMN_RENAMED_CODE;

export { BAD_DEFAULT_CODE, BAD_OPTIONS_CODE, PRECISION_CODE, ROLE_TAKEN_CODE };

/** Rows sampled before the dict heuristic decides (design section 5.4). */
export const DICT_SAMPLE_ROWS = 1024;

/**
 * Build the column declaration of a declared attribute.
 * @param input - what the file declares
 * @returns the declaration, the value spec and any warnings to record
 */
export function declareAttribute(input: AttributeDeclarationInput): DeclaredAttribute {
    const issues: DeclarationIssue[] = [];
    const listSyntax = input.listSyntax ?? "gexf";
    let spec: DeclaredTypeSpec;
    if (input.type === null) {
        spec = stringSpec(null);
    } else {
        const mapped = mapDeclaredType(input.format, input.type, input.long);
        if (mapped === null) {
            spec = stringSpec(input.type);
            issues.push({
                category: "unsupported",
                code: UNKNOWN_TYPE_CODE,
                message: `${input.format} type "${input.type}" is not supported; values kept as text`,
            });
        } else {
            spec = mapped;
        }
    }
    const { name, renamed } = resolveName(input);
    if (renamed) {
        issues.push({
            category: "coercion",
            code: RENAMED_CODE,
            message: `attribute "${input.title ?? input.id ?? ""}" renamed to "${name}": the name was taken`,
        });
    }
    const decl: ColumnDecl = {
        name,
        dtype: spec.dtype,
        nullable: true,
        origin: {
            format: input.format,
            id: input.id,
            title: input.title !== null && input.title !== name ? input.title : null,
            type: input.type,
            namespace: input.namespace ?? null,
        },
    };
    if (spec.list) {
        decl.itemDtype = spec.itemDtype ?? "string";
    }
    if (input.role !== undefined && input.role !== null) {
        decl.role = input.role;
    }
    if (input.dynamic === true) {
        decl.dynamic = true;
    }
    const optionsText = input.optionsText ?? null;
    if (optionsText !== null) {
        try {
            const options = splitListText(optionsText, listSyntax).map((item) => parseItem(item, spec));
            decl.options = options;
            if (spec.dtype === "string" && !spec.list && options.every((o) => typeof o === "string")) {
                decl.dtype = "dict";
                spec = { ...spec, dtype: "dict" };
            }
        } catch (err) {
            issues.push({
                category: "validation-error",
                code: BAD_OPTIONS_CODE,
                message: `options "${optionsText}" of attribute "${name}" do not parse as ${spec.declared}: ${messageOf(err)}`,
            });
        }
    }
    const defaultText = input.defaultText ?? null;
    if (defaultText !== null) {
        try {
            decl.default = parseDeclaredValue(defaultText, spec, listSyntax);
        } catch (err) {
            issues.push({
                category: "validation-error",
                code: BAD_DEFAULT_CODE,
                message: `default "${defaultText}" of attribute "${name}" does not parse as ${spec.declared}: ${messageOf(err)}`,
            });
        }
    }
    const companion = spec.temporal !== null && !spec.list ? timeTextCompanion(name) : null;
    return { decl, spec, listSyntax, companion, renamed, issues };
}

/**
 * Parse a value text by a declaration: a scalar, or a list split by the syntax with each item
 * parsed by the item kind.
 * @param text - the value text
 * @param spec - the declared type
 * @param listSyntax - the list syntax
 * @returns the value to push (a boolean, a number, a string, a JSON value or an array)
 */
export function parseDeclaredValue(text: string, spec: DeclaredTypeSpec, listSyntax: ListSyntax): unknown {
    if (spec.list) {
        return splitListText(text, listSyntax).map((item) => parseItem(item, spec));
    }
    return parseItem(text, spec);
}

/**
 * Parse a temporal value text by a declaration, keeping the source text when its canonical form
 * differs (for the companion column).
 * @param text - the value text
 * @param spec - a scalar temporal spec
 * @returns the value and the text to keep, or null text
 */
export function parseDeclaredTemporal(text: string, spec: DeclaredTypeSpec): TemporalValue {
    if (spec.temporal === null || spec.list) {
        throw new GraphFormatError("E_COLUMN_TYPE", `"${spec.declared}" is not a scalar temporal type`, {
            declared: spec.declared,
        });
    }
    return parseTemporal(text.trim(), spec.temporal);
}

const INTEGER_TEXT = /^[+-]?[0-9]+$/;

/**
 * Whether a long value's text lost precision when parsed to f64 (design section 5.1: a value with
 * |v| > 2^53 is stored as the nearest f64 and recorded as a `precision` issue). Decided from the
 * text, since the parsed number alone cannot tell 2^53 from 2^53 + 1.
 * @param spec - the declared type
 * @param text - the value text (trimmed or not)
 * @returns true when the importer should record a `precision` issue
 */
export function losesPrecision(spec: DeclaredTypeSpec, text: string): boolean {
    if (!spec.precision) {
        return false;
    }
    const trimmed = text.trim();
    if (!INTEGER_TEXT.test(trimmed)) {
        return false;
    }
    const n = Number(trimmed);
    return !Number.isFinite(n) || BigInt(trimmed) !== BigInt(n);
}

/**
 * Declare a column on the sink for one domain.
 * @param sink - the sink
 * @param domain - node or edge
 * @param decl - the declaration
 * @returns the column handle
 */
export function declareOn(sink: GraphSink, domain: "node" | "edge", decl: ColumnDecl): ColumnHandle {
    return domain === "node" ? sink.declareNodeColumn(decl) : sink.declareEdgeColumn(decl);
}

/**
 * The outcome of declareResolved(): the handle and the declaration as it was applied.
 * Consumed by the per-format importers under src/formats.
 * @public
 */
export interface ResolvedDeclaration {
    /** The column handle. */
    readonly handle: ColumnHandle;
    /** The declaration as applied (its name and role may differ from the request). */
    readonly decl: ColumnDecl;
    /** Whether the column was renamed `<name>#<id>` (W_COLUMN_RENAMED recorded). */
    readonly renamed: boolean;
    /** Whether the role was dropped because the table already holds it (W_ROLE_TAKEN recorded). */
    readonly roleDropped: boolean;
}

/**
 * The io rule of design section 5.6 in one place, so every importer agrees: declare a column on
 * the sink and, when the sink already holds the name with another shape (an earlier import, a
 * caller's column), rename it `<name>#<origin.id>` (a counter when the source has no id) and
 * record W_COLUMN_RENAMED (coercion); when the table already holds the role (design section 5.5:
 * at most one column per role), declare without the role and record W_ROLE_TAKEN (coercion). The
 * same shape under the same name returns the existing handle, as the sink does.
 * @param sink - the sink
 * @param domain - node or edge
 * @param decl - the wanted declaration
 * @param report - the report the coercion issues are recorded in
 * @param where - the line and element, when known
 * @returns the handle and the declaration as applied
 */
export function declareResolved(
    sink: GraphSink,
    domain: "node" | "edge",
    decl: ColumnDecl,
    report: ImportReportBuilder,
    where?: IssueLocation,
): ResolvedDeclaration {
    let current = decl;
    let renamed = false;
    let roleDropped = false;
    for (;;) {
        try {
            return { handle: declareOn(sink, domain, current), decl: current, renamed, roleDropped };
        } catch (err) {
            if (!(err instanceof GraphFormatError)) {
                throw err;
            }
            if (err.code === "E_COLUMN_EXISTS" && !renamed) {
                const taken = takenIn(sink, domain);
                const id = current.origin?.id ?? null;
                const name = uniqueColumnName(current.name, id, taken);
                report.warning(
                    "coercion",
                    COLUMN_RENAMED_CODE,
                    `${domain} column "${current.name}" renamed to "${name}": the sink already holds that name with another shape`,
                    { ...where, element: where?.element ?? current.name },
                );
                current = { ...current, name };
                renamed = true;
                continue;
            }
            if (err.code === "E_DUPLICATE_ROLE" && current.role !== undefined && !roleDropped) {
                const holder = typeof err.details.holder === "string" ? err.details.holder : "another column";
                report.warning(
                    "coercion",
                    ROLE_TAKEN_CODE,
                    `${domain} column "${current.name}" loses the ${current.role} role: "${holder}" already holds it`,
                    { ...where, element: where?.element ?? current.name },
                );
                const { role: _role, ...withoutRole } = current;
                current = withoutRole;
                roleDropped = true;
                continue;
            }
            throw err;
        }
    }
}

/**
 * Declare the companion text column of a temporal column (design section 5.1). A table holds at
 * most one column per role, so when another companion already carries `timeText` this one is
 * declared without the role (reported once per table); exporters find every companion through
 * `extra.for`.
 * @param sink - the sink
 * @param domain - node or edge
 * @param companion - the declaration from declareAttribute()
 * @param report - the report the role loss is recorded in; when omitted the loss is silent
 * @returns the column handle
 */
export function declareCompanion(
    sink: GraphSink,
    domain: "node" | "edge",
    companion: ColumnDecl,
    report?: ImportReportBuilder,
): ColumnHandle {
    try {
        return declareOn(sink, domain, companion);
    } catch (err) {
        if (err instanceof GraphFormatError && err.code === "E_DUPLICATE_ROLE") {
            report?.warnOnce(
                "coercion",
                ROLE_TAKEN_CODE,
                `${domain} column "${companion.name}" is declared without the timeText role: another companion holds it (exporters use extra.for)`,
                { element: companion.name },
            );
            const { role: _role, ...withoutRole } = companion;
            return declareOn(sink, domain, withoutRole);
        }
        throw err;
    }
}

/**
 * Whether a column name is taken in a sink's table, the usual `taken` callback.
 * @param sink - the sink
 * @param domain - node or edge
 * @returns a predicate over column names
 */
export function takenIn(sink: GraphSink, domain: "node" | "edge"): (name: string) => boolean {
    return (name: string): boolean =>
        (domain === "node" ? sink.nodeColumn(name) : sink.edgeColumn(name)) !== INVALID_INDEX;
}

/**
 * The deterministic name of a declared attribute whose preferred name is taken (design section
 * 5.6): `<name>#<origin.id>`, and `<name>#<origin.id>#2`, `#3`... should that be taken as well.
 * @param name - the preferred name
 * @param id - the source id (the attribute id or key id); null falls back to a counter
 * @param taken - whether a name is already used in the table
 * @returns a free name
 */
export function uniqueColumnName(name: string, id: string | null, taken: (name: string) => boolean): string {
    if (!taken(name)) {
        return name;
    }
    const base = id === null ? name : `${name}#${id}`;
    if (id !== null && !taken(base)) {
        return base;
    }
    for (let n = 2; ; n++) {
        const candidate = `${base}#${n}`;
        if (!taken(candidate)) {
            return candidate;
        }
    }
}

/**
 * The dict heuristic of design section 5.4 for an untyped string column: observe values until the
 * sample is full; the column is a dict when the distinct count stays below half the rows, a string
 * otherwise. The importer buffers the sampled values and declares the column on the decision.
 */
export class DictHeuristic {
    private readonly distinct = new Set<string>();

    private rowsSeen = 0;

    private readonly sampleRows: number;

    /**
     * Create a heuristic.
     * @param sampleRows - rows observed before deciding; DICT_SAMPLE_ROWS by default
     */
    constructor(sampleRows: number = DICT_SAMPLE_ROWS) {
        this.sampleRows = sampleRows;
    }

    /**
     * Rows observed so far.
     * @returns the count
     */
    get rows(): number {
        return this.rowsSeen;
    }

    /**
     * Distinct values observed so far.
     * @returns the count
     */
    get distinctCount(): number {
        return this.distinct.size;
    }

    /**
     * Whether the sample is full.
     * @returns true once sampleRows values were observed
     */
    get decided(): boolean {
        return this.rowsSeen >= this.sampleRows;
    }

    /**
     * Observe one value.
     * @param value - the value text
     * @returns true when this observation completed the sample
     */
    observe(value: string): boolean {
        this.rowsSeen++;
        if (this.distinct.size < this.rowsSeen) {
            this.distinct.add(value);
        }
        return this.rowsSeen === this.sampleRows;
    }

    /**
     * The dtype the sample suggests: dict when the distinct count is below rows / 2 (and at least
     * one row was seen), string otherwise. Valid before the sample is full (end of input).
     * @returns "dict" or "string"
     */
    decide(): "dict" | "string" {
        return this.rowsSeen > 0 && this.distinct.size * 2 < this.rowsSeen ? "dict" : "string";
    }
}

/**
 * The resolved name of a declaration and whether it was renamed.
 * @param input - the declaration input
 * @returns the name and the rename flag; E_COLUMN_TYPE when neither a title nor an id exists
 */
function resolveName(input: AttributeDeclarationInput): { name: string; renamed: boolean } {
    const preferred = input.title !== null && input.title.length > 0 ? input.title : input.id;
    if (preferred === null || preferred.length === 0) {
        throw new GraphFormatError("E_COLUMN_TYPE", "an attribute declaration needs a title or an id", {
            field: "name",
            format: input.format,
        });
    }
    if (input.taken === undefined) {
        return { name: preferred, renamed: false };
    }
    const name = uniqueColumnName(preferred, input.id, input.taken);
    return { name, renamed: name !== preferred };
}

/**
 * Parse one scalar value or list item by a spec.
 * @param text - the item text
 * @param spec - the declared type
 * @returns the parsed value
 */
function parseItem(text: string, spec: DeclaredTypeSpec): unknown {
    return parseScalarText(text, spec.kind, spec.temporal);
}

/**
 * The message of a caught value.
 * @param err - the thrown value
 * @returns the message
 */
function messageOf(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
}
