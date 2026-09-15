/**
 * The DOT / Graphviz importer (design sections 8.4, 4.1, 5.1; research note 07 section 2.4). A
 * recursive-descent parser over the DOT grammar pushes nodes and edges into the sink as they are
 * mentioned, in first-mention order:
 *
 * - `graph` / `digraph` set the direction through the DirectionResolver (rule 1 of 8.4); `strict`
 *   merges parallel edges as cgraph does and is recorded in `meta.extra.dot.strict`; the graph name
 *   becomes `meta.name`.
 * - node, edge and graph attribute statements (`node [..]`, `edge [..]`, `graph [..]`, `ID = ID`)
 *   are stateful, scoped defaults applied to the elements created after them in the same or a
 *   nested subgraph; every element receives its effective attributes as plain cells, so the
 *   exporter writes them back explicitly.
 * - edge chains `a -> b -> c` and subgraph endpoints `{a b} -> c` expand to one edge per pair, in
 *   cgraph order; a port on an edge endpoint is kept in the `graphty.sourcePort` /
 *   `graphty.targetPort` edge columns (roles sourcePort / targetPort) and never part of the id; a
 *   port on a node statement has no meaning and is dropped with a warning.
 * - a subgraph named `cluster*` (or carrying `cluster=true`) becomes a container NODE whose id is
 *   the cluster name (design section 5.10: containment is the `parent` role, never an adjacency
 *   edge); members get `graphty.parent` = the container's index, the container's own attributes are
 *   its node cells, and `graphty.cluster` = true marks it. A plain node and a cluster of the same
 *   name (fdp's cluster edges) merge into that one node with a warning. Other subgraphs are
 *   transparent grouping; their attributes are reported as dropped.
 * - attribute values are ID strings inferred per column by the sink under the 5.1 grammar, except
 *   `label` (text, role label), `pos` on a node (the position role column, f32 x3, design section
 *   5.2), `weight` (THE weight, `weightFrom`) and `key` (cgraph's edge identity, the edge id role).
 * - ids are coerced with the common rule (`canonical` by default: `1` and `"1"` are the same node,
 *   as the DOT grammar says).
 *
 * The whole text is read first (design section 8.4 allows it for DOT). A grammar violation is
 * fatal, as it is for Graphviz itself: the import aborts with ImportError (code E_DOT_SYNTAX)
 * carrying the partial report. Errors the sink raises for one element are recorded and the element
 * is skipped (section 8.6).
 */

import {
    type ColumnDecl,
    type ColumnHandle,
    GraphFormatError,
    type GraphSink,
    INVALID_INDEX,
    type NodeId,
} from "@graphty/graph-format";

import { declareResolved } from "../../common/attributes.js";
import {
    COLUMN_RENAMED_CODE,
    DIRECTION_FORCED_CODE,
    DIRECTION_REFUSED_CODE,
    EMPTY_INPUT_CODE,
    ID_MERGED_CODE,
    INVALID_UTF8_CODE,
    MIXED_DIRECTION_CODE,
    MULTIPLE_GRAPHS_CODE,
    OPTION_IGNORED_CODE,
    ROLE_TAKEN_CODE,
    SINK_OPTION_CODE,
    SYNTAX_CODE,
} from "../../common/codes.js";
import { DirectionResolver, type EdgeKind } from "../../common/direction.js";
import { IdCoercer } from "../../common/ids.js";
import { readText, throwIfAborted } from "../../common/input.js";
import {
    reportSinkOptions,
    reportUnusedOptions,
    type ResolvedImportOptions,
    resolveImportOptions,
} from "../../common/options.js";
import { ImportReportBuilder } from "../../common/report.js";
import { parseTextCell, TextCellWriter, WIDENING_UNSUPPORTED_CODE } from "../../common/text.js";
import { parseWeightText } from "../../common/weights.js";
import { type CommonImportOptions, type GraphImporter, type ImportInput, type ImportReport } from "../../types.js";
import {
    CLUSTER_COLUMN,
    DOT_FORMAT,
    DOT_ORIGIN,
    KEY_ATTRIBUTE,
    LABEL_ATTRIBUTE,
    PARENT_COLUMN,
    PIN_ATTRIBUTE,
    POS_ATTRIBUTE,
    SOURCE_PORT_COLUMN,
    TARGET_PORT_COLUMN,
} from "./names.js";
import { DotSyntaxError, type DotToken, DotTokenizer } from "./tokenizer.js";

/** The DOT importer's format-specific options. */
export interface DotImportOptions {
    /**
     * What an edge operator that contradicts the graph keyword means (`--` in a digraph, `->` in a
     * graph; a syntax error for Graphviz): "operator" (default) reads the edge with the operator's
     * direction and resolves it per onMixedDirection, with a warning; "header" reads it with the
     * graph's direction, with a warning; "error" aborts the import as Graphviz does.
     */
    mismatchedEdgeOperator?: "operator" | "header" | "error" | undefined;
}

/**
 * The issue codes the DOT importer records (design section 8.6), by name: the codes shared with
 * the other importers (src/common/codes.ts) and the DOT-specific ones. A key is the code without
 * its severity and format prefixes.
 */
export const DOT_ISSUE = Object.freeze({
    /** A grammar violation; fatal. */
    SYNTAX: SYNTAX_CODE,
    /** The input holds no graph at all (empty or only comments); fatal. */
    EMPTY_INPUT: EMPTY_INPUT_CODE,
    /** The input holds invalid UTF-8 (fatal). */
    INVALID_UTF8: INVALID_UTF8_CODE,
    /** Subgraphs or braces nested deeper than the parser's limit; fatal. */
    NESTING: "E_DOT_NESTING",
    /** An edge operator contradicting the graph keyword (warning under "operator" / "header"). */
    EDGE_OPERATOR: "W_DOT_EDGE_OPERATOR",
    /** A second graph in the same input; only the first is read. */
    MULTIPLE_GRAPHS: MULTIPLE_GRAPHS_CODE,
    /** A badly delimited numeral (`1e3`) split into two tokens, as Graphviz does with a warning. */
    NUMERAL_AMBIGUITY: "W_DOT_NUMERAL_AMBIGUITY",
    /** Attributes of a subgraph that is not a cluster (rank=same and the like) cannot be represented. */
    SUBGRAPH_ATTRIBUTES_DROPPED: "W_DOT_SUBGRAPH_ATTRIBUTES_DROPPED",
    /** A port on a node statement has no meaning and was dropped. */
    NODE_PORT_DROPPED: "W_DOT_NODE_PORT_DROPPED",
    /** A plain node and a cluster share a name and were merged into one container node. */
    CLUSTER_NODE_MERGED: "W_DOT_CLUSTER_NODE_MERGED",
    /** A node mentioned in two unrelated clusters keeps the first. */
    CLUSTER_CONFLICT: "W_DOT_CLUSTER_CONFLICT",
    /** A node `pos` that is not a point; the value was dropped. */
    BAD_POS: "W_DOT_BAD_POS",
    /** A parallel edge merged into an earlier one under `strict`. */
    STRICT_MERGED: "W_DOT_STRICT_MERGED",
    /** An edge merged into an earlier one with the same endpoints and `key`. */
    KEY_MERGED: "W_DOT_KEY_MERGED",
    /** A role (label, id, position, ...) was already taken in the caller's sink; the column was declared without it. */
    ROLE_TAKEN: ROLE_TAKEN_CODE,
    /** A column of another shape exists in the caller's sink under a name the importer declares; renamed `<name>#<id>`. */
    COLUMN_RENAMED: COLUMN_RENAMED_CODE,
    /** A common option the format has no use for was given a non-default value. */
    OPTION_IGNORED: OPTION_IGNORED_CODE,
    /** Two distinct id texts merged under ids: "number". */
    ID_MERGED: ID_MERGED_CODE,
    /** A builder-policy option the sink does not honour. */
    SINK_OPTION: SINK_OPTION_CODE,
    /** The sink refused the file's direction. */
    DIRECTION_REFUSED: DIRECTION_REFUSED_CODE,
    /** Edges forced to the policy's direction. */
    DIRECTION_FORCED: DIRECTION_FORCED_CODE,
    /** A mixed file under onMixedDirection "error" (fatal). */
    MIXED_DIRECTION: MIXED_DIRECTION_CODE,
    /** A text column the sink could not widen to the dtype its cells imply. */
    WIDENING_UNSUPPORTED: WIDENING_UNSUPPORTED_CODE,
});

/** The common options the DOT importer reads (the rest is reported by reportUnusedOptions). */
const USED_OPTIONS: ReadonlySet<keyof CommonImportOptions> = new Set<keyof CommonImportOptions>([
    "ids",
    "addMissingNodes",
    "duplicateEdges",
    "selfLoops",
    "onMixedDirection",
    "weightFrom",
    "weightDtype",
    "errorLimit",
    "signal",
    "onProgress",
]);

/** The deepest nesting of subgraphs and braces the parser accepts (each level is one stack frame). */
const MAX_NESTING = 1024;

const EXTENSIONS: readonly string[] = Object.freeze([".dot", ".gv"]);
const MIME_TYPES: readonly string[] = Object.freeze(["text/vnd.graphviz"]);
const CLUSTER_PREFIX = "cluster";
const CLUSTER_ATTRIBUTE = "cluster";
const STATEMENTS_PER_ABORT_CHECK = 64;
const MAX_ANCESTOR_WALK = 4096;

const DOT_HEADER = /^\s*(strict\s+)?(di)?graph\b/i;
const TRUE_TEXTS: ReadonlySet<string> = new Set(["true", "yes", "1"]);
const POINT_TEXT =
    /^\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*,\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)(?:\s*,\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?))?\s*(!?)\s*$/;

/** An attribute as written: name, value text and the line of the assignment. */
interface DotAttribute {
    readonly name: string;
    readonly value: string;
    readonly line: number;
}

/** Where an edge issue is recorded: the statement's line and the edge's description. */
interface EdgeWhere {
    readonly line: number;
    readonly element: string;
}

/** One endpoint of an edge chain: a mentioned node and its port. */
interface Endpoint {
    readonly id: NodeId;
    readonly index: number;
    readonly port: string | null;
}

/** A lexical scope: the root graph or one subgraph. */
interface Scope {
    readonly root: boolean;
    /** The subgraph name, null for the root and for an anonymous subgraph. */
    readonly name: string | null;
    readonly line: number;
    /** Defaults for nodes created in this scope, copied from the enclosing scope on entry. */
    readonly nodeDefaults: Map<string, string>;
    /** Defaults for edges created in this scope. */
    readonly edgeDefaults: Map<string, string>;
    /** Node indices mentioned in this scope (nested scopes included), in first-mention order. */
    readonly members: number[];
    readonly memberSet: Set<number>;
    /** Container nodes of clusters closed inside this scope. */
    readonly containers: number[];
    /** Subgraph attributes buffered until the subgraph closes (the root applies them at once). */
    readonly attributes: Map<string, DotAttribute>;
    /** Whether the subgraph is a cluster (by name or by `cluster=true`). */
    cluster: boolean;
    /** The container node index once created, or INVALID_INDEX. */
    container: number;
}

/**
 * The importer plugin for DOT / Graphviz text (design section 12.4).
 */
export const dotImporter: GraphImporter<DotImportOptions> = Object.freeze({
    format: DOT_FORMAT,
    extensions: EXTENSIONS,
    mimeTypes: MIME_TYPES,

    /**
     * Confidence that the head of an input is DOT: the `[strict] graph | digraph` header after
     * optional comments.
     * @param head - the first bytes of the input
     * @returns 0.95 for a header followed by `{`, 0.8 for a header alone, 0 otherwise
     */
    sniff(head: Uint8Array): number {
        const text = stripLeadingComments(new TextDecoder("utf-8").decode(head));
        const match = DOT_HEADER.exec(text);
        if (match === null) {
            return 0;
        }
        return text.includes("{") ? 0.95 : 0.8;
    },

    /**
     * Read a DOT document into the sink.
     * @param input - the text, bytes or stream
     * @param sink - the sink to push into
     * @param options - format-specific and common options
     * @returns the import report; ImportError (E_IMPORT) on a syntax error or beyond the error limit
     */
    async import(
        input: ImportInput,
        sink: GraphSink,
        options?: DotImportOptions & CommonImportOptions,
    ): Promise<ImportReport> {
        const resolved = resolveImportOptions(options, {
            ids: "canonical",
            defaultDirected: true,
            weightFrom: "weight",
        });
        const mismatch = mismatchOption(options?.mismatchedEdgeOperator);
        const report = new ImportReportBuilder(DOT_FORMAT, resolved.errorLimit);
        reportUnusedOptions(options, report, USED_OPTIONS);
        reportSinkOptions(sink, options, report);
        const text = await readText(input, report, resolved);
        const parser = new DotParser(text, sink, report, resolved, mismatch);
        try {
            parser.parse();
        } catch (err) {
            if (err instanceof DotSyntaxError) {
                report.fail(SYNTAX_CODE, err.message, { line: err.line }, { line: err.line });
            }
            throw err;
        }
        // an abort raised during the last few statements (after the last periodic check) still rejects
        throwIfAborted(resolved.signal);
        return report.finish();
    },
});

/**
 * Resolve the mismatchedEdgeOperator option.
 * @param value - the caller's value
 * @returns the value or the default; E_UNSUPPORTED for anything else
 */
function mismatchOption(value: unknown): "operator" | "header" | "error" {
    if (value === undefined) {
        return "operator";
    }
    if (value === "operator" || value === "header" || value === "error") {
        return value;
    }
    throw new GraphFormatError(
        "E_UNSUPPORTED",
        `option mismatchedEdgeOperator: ${JSON.stringify(value)} is not one of "operator", "header", "error"`,
        { option: "mismatchedEdgeOperator", found: value, supported: ["operator", "header", "error"] },
    );
}

/**
 * Remove leading whitespace and comments from a head sample so the header regex sees the keyword.
 * @param text - the decoded head
 * @returns the text from the first non-comment character
 */
function stripLeadingComments(text: string): string {
    let rest = text;
    for (;;) {
        const trimmed = rest.replace(/^\s+/, "");
        if (trimmed.startsWith("//") || trimmed.startsWith("#")) {
            const nl = trimmed.search(/[\r\n]/);
            if (nl < 0) {
                return "";
            }
            rest = trimmed.slice(nl);
            continue;
        }
        if (trimmed.startsWith("/*")) {
            const end = trimmed.indexOf("*/");
            if (end < 0) {
                return "";
            }
            rest = trimmed.slice(end + 2);
            continue;
        }
        return trimmed;
    }
}

/**
 * Whether a bare token is a given keyword (keywords are case-insensitive; a quoted id never is).
 * @param token - the token
 * @param keyword - the lower-case keyword
 * @returns true for a match
 */
function isKeyword(token: DotToken, keyword: string): boolean {
    return token.kind === "id" && !token.quoted && !token.html && token.text.toLowerCase() === keyword;
}

/**
 * Whether a token is a given punctuation.
 * @param token - the token
 * @param text - the punctuation
 * @returns true for a match
 */
function isPunct(token: DotToken, text: string): boolean {
    return token.kind === "punct" && token.text === text;
}

/**
 * Whether a token is an edge operator.
 * @param token - the token
 * @returns true for `->` or `--`
 */
function isEdgeOp(token: DotToken): boolean {
    return token.kind === "punct" && (token.text === "->" || token.text === "--");
}

/**
 * A short description of a token for syntax error messages.
 * @param token - the token
 * @returns `end of input`, or the token text in quotes
 */
function describeToken(token: DotToken): string {
    if (token.kind === "eof") {
        return "end of input";
    }
    return JSON.stringify(token.text.length > 40 ? `${token.text.slice(0, 40)}...` : token.text);
}

/**
 * Whether a subgraph name marks a cluster.
 * @param name - the name, or null
 * @returns true when it starts with `cluster`
 */
function isClusterName(name: string | null): boolean {
    return name !== null && name.startsWith(CLUSTER_PREFIX);
}

/**
 * The parser and pusher for one import call.
 */
class DotParser {
    private readonly lexer: DotTokenizer;

    private readonly sink: GraphSink;

    private readonly report: ImportReportBuilder;

    private readonly options: ResolvedImportOptions;

    private readonly mismatch: "operator" | "header" | "error";

    private readonly ids: IdCoercer;

    private readonly resolver: DirectionResolver;

    private directed = true;

    private strict = false;

    private statements = 0;

    /** The id of every node mentioned so far, by index (the sink has no idOf). */
    private readonly idOf = new Map<number, NodeId>();

    /** Container nodes this import created or adopted. */
    private readonly containers = new Set<number>();

    /** The parent assigned to a node by this import. */
    private readonly parentOf = new Map<number, number>();

    /** Nodes already warned about a cluster conflict. */
    private readonly conflictWarned = new Set<number>();

    /** Under `strict`: the edge index of every (source, target) pair pushed. */
    private readonly strictEdges = new Map<string, number>();

    /** The edge index of every (source, target, key) triple pushed. */
    private readonly keyedEdges = new Map<string, number>();

    /** Columns of the caller's sink adopted under a name the importer declares with another shape; values are inferred for them. */

    private nodeLabelHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private edgeLabelHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private positionHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private clusterHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private parentHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private keyHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private sourcePortHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private targetPortHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    /** The current subgraph nesting depth. */
    private depth = 0;

    /** The inferred attribute columns by name, per domain (the 5.1 text grammar per column). */
    private readonly nodeWriters = new Map<string, TextCellWriter>();

    private readonly edgeWriters = new Map<string, TextCellWriter>();

    /**
     * Create a parser over one document.
     * @param text - the DOT text
     * @param sink - the sink
     * @param report - the report
     * @param options - the resolved common options
     * @param mismatch - the resolved mismatchedEdgeOperator option
     */
    constructor(
        text: string,
        sink: GraphSink,
        report: ImportReportBuilder,
        options: ResolvedImportOptions,
        mismatch: "operator" | "header" | "error",
    ) {
        this.lexer = new DotTokenizer(text, (numeral, line) => {
            report.warning(
                "validation-error",
                DOT_ISSUE.NUMERAL_AMBIGUITY,
                `badly delimited number ${JSON.stringify(numeral)} splits into two tokens (Graphviz warns the same)`,
                { line, element: numeral },
            );
        });
        this.sink = sink;
        this.report = report;
        this.options = options;
        this.mismatch = mismatch;
        this.ids = new IdCoercer(options.ids);
        this.resolver = new DirectionResolver(sink, report, options.onMixedDirection);
    }

    /**
     * Parse the whole document: `[strict] (graph | digraph) [ID] { stmt_list }`.
     */
    parse(): void {
        const { lexer } = this;
        let token = lexer.next();
        if (token.kind === "eof") {
            this.report.fail(EMPTY_INPUT_CODE, "the input holds no graph (empty or only comments)", {
                line: token.line,
            });
        }
        if (isKeyword(token, "strict")) {
            this.strict = true;
            token = lexer.next();
        }
        if (isKeyword(token, "digraph")) {
            this.directed = true;
        } else if (isKeyword(token, "graph")) {
            this.directed = false;
        } else {
            throw new DotSyntaxError(`expected "graph" or "digraph", found ${describeToken(token)}`, token.line);
        }
        const headerLine = token.line;
        token = lexer.next();
        let name: string | null = null;
        if (token.kind === "id") {
            name = token.text;
            token = lexer.next();
        }
        if (!isPunct(token, "{")) {
            throw new DotSyntaxError(`expected "{" after the graph header, found ${describeToken(token)}`, token.line);
        }
        this.resolver.setHeader(this.directed, { line: headerLine });
        this.sink.setMeta({
            name,
            sourceFormat: DOT_FORMAT,
            ...(this.strict ? { extra: { dot: { strict: true } } } : {}),
        });
        const root = this.newScope(null, headerLine, true, null);
        this.statementList(root);
        const trailing = lexer.next();
        if (trailing.kind !== "eof") {
            this.report.warning(
                "unsupported",
                MULTIPLE_GRAPHS_CODE,
                `content after the closing brace of the graph (${describeToken(trailing)}) was not read; one graph per input`,
                { line: trailing.line },
            );
        }
    }

    /**
     * Create a scope.
     * @param parent - the enclosing scope, or null for the root
     * @param line - the line the scope opens on
     * @param root - whether this is the root graph
     * @param name - the subgraph name, or null
     * @returns the scope
     */
    private newScope(parent: Scope | null, line: number, root: boolean, name: string | null): Scope {
        return {
            root,
            name,
            line,
            nodeDefaults: new Map(parent?.nodeDefaults),
            edgeDefaults: new Map(parent?.edgeDefaults),
            members: [],
            memberSet: new Set(),
            containers: [],
            attributes: new Map(),
            cluster: false,
            container: INVALID_INDEX,
        };
    }

    /**
     * Parse `stmt_list }` for a scope whose `{` was consumed.
     * @param scope - the scope
     */
    private statementList(scope: Scope): void {
        const { lexer } = this;
        for (;;) {
            const token = lexer.peek();
            if (isPunct(token, "}")) {
                lexer.next();
                return;
            }
            if (token.kind === "eof") {
                throw new DotSyntaxError(
                    scope.root
                        ? 'unexpected end of input: missing "}" closing the graph'
                        : `unexpected end of input: missing "}" closing the subgraph opened on line ${scope.line}`,
                    token.line,
                );
            }
            if (isPunct(token, ";")) {
                lexer.next();
                continue;
            }
            this.statement(scope);
            if (++this.statements % STATEMENTS_PER_ABORT_CHECK === 0) {
                throwIfAborted(this.options.signal);
            }
        }
    }

    /**
     * Parse one statement of a scope.
     * @param scope - the scope
     */
    private statement(scope: Scope): void {
        const { lexer } = this;
        const token = lexer.peek();
        if (token.kind === "punct") {
            if (token.text === "{") {
                const group = this.subgraph(scope);
                this.maybeEdgeStatement(scope, group, token.line);
                return;
            }
            throw new DotSyntaxError(`unexpected ${describeToken(token)} at the start of a statement`, token.line);
        }
        if (!token.quoted && !token.html) {
            const keyword = token.text.toLowerCase();
            if (keyword === "node" || keyword === "edge" || keyword === "graph") {
                lexer.next();
                this.attributeStatement(scope, keyword, token.line);
                return;
            }
            if (keyword === "subgraph") {
                const group = this.subgraph(scope);
                this.maybeEdgeStatement(scope, group, token.line);
                return;
            }
            if (keyword === "digraph" || keyword === "strict") {
                throw new DotSyntaxError(`unexpected keyword ${describeToken(token)} inside a graph`, token.line);
            }
        }
        // ID '=' ID, a node statement, or an edge statement starting with a node
        const id = this.identifier();
        if (isPunct(lexer.peek(), "=")) {
            lexer.next();
            const value = this.identifier();
            this.scopeAttribute(scope, { name: id.text, value: value.text, line: id.line });
            return;
        }
        const port = this.port();
        if (isEdgeOp(lexer.peek())) {
            const endpoint = this.mentionEndpoint(scope, id, port);
            this.edgeStatement(scope, endpoint === null ? [] : [endpoint], id.line);
            return;
        }
        this.nodeStatement(scope, id, port);
    }

    /**
     * Read an ID, concatenating quoted strings joined by `+`.
     * @returns the id token (the concatenation keeps the first token's line and quoted flag)
     */
    private identifier(): DotToken {
        const { lexer } = this;
        const token = lexer.next();
        if (token.kind !== "id") {
            throw new DotSyntaxError(`expected an identifier, found ${describeToken(token)}`, token.line);
        }
        if (!token.quoted || !isPunct(lexer.peek(), "+")) {
            return token;
        }
        let { text } = token;
        while (isPunct(lexer.peek(), "+")) {
            const plus = lexer.next();
            const more = lexer.next();
            if (more.kind !== "id" || !more.quoted) {
                throw new DotSyntaxError(
                    `expected a quoted string after "+", found ${describeToken(more)}`,
                    more.kind === "eof" ? plus.line : more.line,
                );
            }
            text += more.text;
        }
        return { kind: "id", text, quoted: true, html: false, line: token.line };
    }

    /**
     * Read an optional port after a node id: `: ID [ : compass ]` or `: compass`.
     * @returns the port text (`f0`, `f0:n`, `n`), or null when there is none
     */
    private port(): string | null {
        const { lexer } = this;
        if (!isPunct(lexer.peek(), ":")) {
            return null;
        }
        lexer.next();
        let { text } = this.identifier();
        if (isPunct(lexer.peek(), ":")) {
            lexer.next();
            text += `:${this.identifier().text}`;
        }
        return text;
    }

    /**
     * Parse `[subgraph [ID]] { stmt_list }` and return its member nodes for use as an edge endpoint.
     * @param parent - the enclosing scope
     * @returns the endpoints (every node mentioned in the subgraph, in first-mention order)
     */
    private subgraph(parent: Scope): Endpoint[] {
        const { lexer } = this;
        let token = lexer.next();
        let name: string | null = null;
        if (isKeyword(token, "subgraph")) {
            token = lexer.next();
            if (token.kind === "id") {
                name = token.text;
                token = lexer.next();
            }
        }
        if (!isPunct(token, "{")) {
            throw new DotSyntaxError(`expected "{" to open a subgraph, found ${describeToken(token)}`, token.line);
        }
        if (++this.depth > MAX_NESTING) {
            this.report.fail(
                DOT_ISSUE.NESTING,
                `subgraphs nested deeper than ${MAX_NESTING} levels (the parser recurses per level)`,
                { line: token.line },
            );
        }
        const scope = this.newScope(parent, token.line, false, name);
        if (isClusterName(name)) {
            scope.cluster = true;
            this.ensureContainer(scope);
        }
        this.statementList(scope);
        this.depth--;
        this.closeScope(scope, parent);
        return scope.members.map((index) => ({ id: this.idOf.get(index) ?? index, index, port: null }));
    }

    /**
     * Finish a subgraph: apply its attributes to the container node when it is a cluster (or report
     * them dropped), assign parents to its members, and propagate members and containers upward.
     * @param scope - the closed scope
     * @param parent - the enclosing scope
     */
    private closeScope(scope: Scope, parent: Scope): void {
        const { container } = scope;
        if (container !== INVALID_INDEX) {
            for (const attribute of scope.attributes.values()) {
                if (attribute.name !== CLUSTER_ATTRIBUTE) {
                    this.setNodeAttribute(container, attribute.name, attribute.value, attribute.line);
                }
            }
            for (const nested of scope.containers) {
                if (nested !== container && !this.parentOf.has(nested)) {
                    this.setParent(nested, container);
                }
            }
            for (const member of scope.members) {
                if (member === container) {
                    continue;
                }
                const existing = this.parentOf.get(member);
                if (existing === undefined) {
                    this.setParent(member, container);
                } else if (existing !== container && !this.isAncestor(container, existing)) {
                    this.clusterConflict(member, existing, container, scope.line);
                }
            }
            parent.containers.push(container);
        } else {
            if (scope.attributes.size > 0) {
                const names = [...scope.attributes.keys()].join(", ");
                this.report.warning(
                    "unsupported",
                    DOT_ISSUE.SUBGRAPH_ATTRIBUTES_DROPPED,
                    `subgraph ${scope.name === null ? "(anonymous)" : JSON.stringify(scope.name)} is not a cluster; its attributes (${names}) cannot be represented and were dropped`,
                    { line: scope.line, element: scope.name },
                );
            }
            for (const nested of scope.containers) {
                parent.containers.push(nested);
            }
        }
        if (!parent.root) {
            for (const member of scope.members) {
                if (!parent.memberSet.has(member)) {
                    parent.memberSet.add(member);
                    parent.members.push(member);
                }
            }
        }
    }

    /**
     * Whether `candidate` is an ancestor of `node` through the parents assigned so far.
     * @param candidate - the possible ancestor
     * @param node - the node whose chain is walked
     * @returns true when candidate is reached
     */
    private isAncestor(candidate: number, node: number): boolean {
        let current: number | undefined = node;
        for (let steps = 0; current !== undefined && steps < MAX_ANCESTOR_WALK; steps++) {
            if (current === candidate) {
                return true;
            }
            current = this.parentOf.get(current);
        }
        return false;
    }

    /**
     * Record a node mentioned in two unrelated clusters (once per node).
     * @param member - the node
     * @param existing - its parent
     * @param container - the cluster it was also mentioned in
     * @param line - the line of the losing cluster
     */
    private clusterConflict(member: number, existing: number, container: number, line: number): void {
        if (this.conflictWarned.has(member)) {
            return;
        }
        this.conflictWarned.add(member);
        const id = this.idOf.get(member) ?? member;
        this.report.warning(
            "coercion",
            DOT_ISSUE.CLUSTER_CONFLICT,
            `node ${JSON.stringify(id)} is in cluster ${JSON.stringify(this.idOf.get(existing) ?? existing)} and in cluster ${JSON.stringify(this.idOf.get(container) ?? container)}; the first is kept`,
            { line, element: String(id) },
        );
    }

    /**
     * Assign a parent.
     * @param node - the node index
     * @param container - the container index
     */
    private setParent(node: number, container: number): void {
        if (node === container) {
            return;
        }
        try {
            this.sink.setNodeValue(this.parentColumn(), node, container);
            this.parentOf.set(node, container);
        } catch (err) {
            this.report.recordError(err, { element: String(this.idOf.get(node) ?? node) });
        }
    }

    /**
     * Create (or adopt) the container node of a cluster scope.
     * @param scope - the cluster scope
     */
    private ensureContainer(scope: Scope): void {
        if (scope.container !== INVALID_INDEX || scope.name === null) {
            return;
        }
        const id = this.coerceId(scope.name, scope.line);
        if (id === null) {
            return;
        }
        let index = this.sink.indexOf(id);
        try {
            if (index === INVALID_INDEX) {
                index = this.sink.addNode(id);
                this.report.counts.nodes++;
            } else if (!this.containers.has(index)) {
                this.report.warning(
                    "coercion",
                    DOT_ISSUE.CLUSTER_NODE_MERGED,
                    `cluster ${JSON.stringify(scope.name)} and the node of the same name were merged into one container node`,
                    { line: scope.line, element: scope.name },
                );
            }
            this.idOf.set(index, id);
            if (!this.containers.has(index)) {
                this.containers.add(index);
                this.sink.setNodeValue(this.clusterColumn(), index, true);
            }
            scope.container = index;
        } catch (err) {
            this.report.recordError(err, { line: scope.line, element: scope.name });
            if (index === INVALID_INDEX) {
                this.report.counts.skippedNodes++;
            }
        }
    }

    /**
     * Apply an `ID = ID` or `graph [..]` attribute: the graph table at the root, a buffered
     * subgraph attribute otherwise (`cluster=true` turns the subgraph into a cluster).
     * @param scope - the scope
     * @param attribute - the attribute
     */
    private scopeAttribute(scope: Scope, attribute: DotAttribute): void {
        if (scope.root) {
            this.setGraphAttribute(attribute);
            return;
        }
        scope.attributes.set(attribute.name, attribute);
        if (attribute.name === CLUSTER_ATTRIBUTE && TRUE_TEXTS.has(attribute.value.trim().toLowerCase())) {
            scope.cluster = true;
            this.ensureContainer(scope);
        }
    }

    /**
     * Parse `node|edge|graph attr_list` after the keyword.
     * @param scope - the scope
     * @param keyword - which defaults are set
     * @param line - the keyword's line
     */
    private attributeStatement(scope: Scope, keyword: string, line: number): void {
        if (!isPunct(this.lexer.peek(), "[")) {
            const found = this.lexer.peek();
            throw new DotSyntaxError(`expected "[" after "${keyword}", found ${describeToken(found)}`, found.line);
        }
        const attributes = this.attributeList();
        switch (keyword) {
            case "node":
                for (const a of attributes) {
                    scope.nodeDefaults.set(a.name, a.value);
                }
                break;
            case "edge":
                for (const a of attributes) {
                    scope.edgeDefaults.set(a.name, a.value);
                }
                break;
            case "graph":
                for (const a of attributes) {
                    this.scopeAttribute(scope, a);
                }
                break;
            default:
                throw new DotSyntaxError(`unknown attribute statement "${keyword}"`, line);
        }
    }

    /**
     * Parse one or more `[ a_list ]` groups.
     * @returns the attributes in order (a repeated name keeps its last value at application time)
     */
    private attributeList(): DotAttribute[] {
        const { lexer } = this;
        const out: DotAttribute[] = [];
        while (isPunct(lexer.peek(), "[")) {
            lexer.next();
            for (;;) {
                const token = lexer.peek();
                if (isPunct(token, "]")) {
                    lexer.next();
                    break;
                }
                if (isPunct(token, ";") || isPunct(token, ",")) {
                    lexer.next();
                    continue;
                }
                const name = this.identifier();
                const eq = lexer.next();
                if (!isPunct(eq, "=")) {
                    throw new DotSyntaxError(
                        `expected "=" after attribute name ${JSON.stringify(name.text)}, found ${describeToken(eq)}`,
                        eq.line,
                    );
                }
                const value = this.identifier();
                out.push({ name: name.text, value: value.text, line: name.line });
            }
        }
        return out;
    }

    /**
     * Parse a node statement after its id and port: an optional attribute list, then apply.
     * @param scope - the scope
     * @param id - the id token
     * @param port - the port, if any (dropped with a warning)
     */
    private nodeStatement(scope: Scope, id: DotToken, port: string | null): void {
        const attributes = this.attributeList();
        const endpoint = this.mentionEndpoint(scope, id, null);
        if (endpoint === null) {
            return;
        }
        if (port !== null) {
            this.report.warning(
                "unsupported",
                DOT_ISSUE.NODE_PORT_DROPPED,
                `port ${JSON.stringify(port)} on the node statement of ${JSON.stringify(id.text)} has no meaning and was dropped`,
                { line: id.line, element: id.text },
            );
        }
        for (const attribute of attributes) {
            this.setNodeAttribute(endpoint.index, attribute.name, attribute.value, attribute.line);
        }
    }

    /**
     * After a subgraph statement: continue as an edge statement when an edge operator follows.
     * @param scope - the scope
     * @param group - the subgraph's members
     * @param line - the statement's line
     */
    private maybeEdgeStatement(scope: Scope, group: Endpoint[], line: number): void {
        if (isEdgeOp(this.lexer.peek())) {
            this.edgeStatement(scope, group, line);
        }
    }

    /**
     * Parse `edgeRHS [attr_list]` after the first endpoint group and push the edges.
     * @param scope - the scope
     * @param first - the first group (empty when its node could not be created)
     * @param line - the statement's line
     */
    private edgeStatement(scope: Scope, first: Endpoint[], line: number): void {
        const { lexer } = this;
        const groups: Endpoint[][] = [first];
        const kinds: EdgeKind[] = [];
        while (isEdgeOp(lexer.peek())) {
            const op = lexer.next();
            kinds.push(this.edgeKind(op));
            const next = lexer.peek();
            if (isPunct(next, "{") || isKeyword(next, "subgraph")) {
                groups.push(this.subgraph(scope));
            } else {
                const id = this.identifier();
                const port = this.port();
                const endpoint = this.mentionEndpoint(scope, id, port);
                groups.push(endpoint === null ? [] : [endpoint]);
            }
        }
        const attributes = this.attributeList();
        for (let k = 0; k < kinds.length; k++) {
            for (const source of groups[k]) {
                for (const target of groups[k + 1]) {
                    this.pushEdge(scope, source, target, kinds[k], attributes, line);
                }
            }
        }
    }

    /**
     * The direction of an edge from its operator, checked against the graph keyword.
     * @param op - the operator token
     * @returns the edge kind
     */
    private edgeKind(op: DotToken): EdgeKind {
        const operatorDirected = op.text === "->";
        if (operatorDirected === this.directed) {
            return operatorDirected ? "directed" : "undirected";
        }
        const message = `edge operator "${op.text}" in a ${this.directed ? "digraph" : "graph"}`;
        switch (this.mismatch) {
            case "error":
                throw new DotSyntaxError(`${message} (mismatchedEdgeOperator: "error")`, op.line);
            case "header":
                this.report.warnOnce(
                    "coercion",
                    DOT_ISSUE.EDGE_OPERATOR,
                    `${message}; read with the graph's direction (mismatchedEdgeOperator: "header")`,
                    { line: op.line },
                );
                return this.directed ? "directed" : "undirected";
            case "operator":
                this.report.warnOnce(
                    "coercion",
                    DOT_ISSUE.EDGE_OPERATOR,
                    `${message}; read with the operator's direction and resolved per onMixedDirection`,
                    { line: op.line },
                );
                return operatorDirected ? "directed" : "undirected";
            default: {
                const name: string = this.mismatch;
                throw new GraphFormatError("E_UNSUPPORTED", `unknown mismatchedEdgeOperator ${name}`, { found: name });
            }
        }
    }

    /**
     * Coerce an id text, recording a merge under ids: "number".
     * @param text - the id text
     * @param line - the line, for issues
     * @returns the id, or null when the text was rejected (recorded, node skipped)
     */
    private coerceId(text: string, line: number): NodeId | null {
        try {
            const id = this.ids.text(text);
            const merge = this.ids.lastMerge;
            if (merge !== null) {
                this.report.warning(
                    "coercion",
                    DOT_ISSUE.ID_MERGED,
                    `id text ${JSON.stringify(merge.text)} merged with ${JSON.stringify(merge.previousText)} as ${merge.id} under ids: "number"`,
                    { line, element: text },
                );
            }
            return id;
        } catch (err) {
            this.report.recordError(err, { line, element: text });
            this.report.counts.skippedNodes++;
            return null;
        }
    }

    /**
     * Mention a node: create it on first mention (applying the scope's node defaults), record it as
     * a member of the scope, and return it as an endpoint.
     * @param scope - the scope
     * @param id - the id token
     * @param port - the endpoint's port, or null
     * @returns the endpoint, or null when the node could not be created (recorded)
     */
    private mentionEndpoint(scope: Scope, id: DotToken, port: string | null): Endpoint | null {
        const nodeId = this.coerceId(id.text, id.line);
        if (nodeId === null) {
            return null;
        }
        const { sink } = this;
        let index = sink.indexOf(nodeId);
        if (index === INVALID_INDEX) {
            try {
                index = sink.addNode(nodeId);
            } catch (err) {
                this.report.recordError(err, { line: id.line, element: id.text });
                this.report.counts.skippedNodes++;
                return null;
            }
            this.report.counts.nodes++;
            this.idOf.set(index, nodeId);
            for (const [name, value] of scope.nodeDefaults) {
                this.setNodeAttribute(index, name, value, id.line);
            }
        } else if (!this.idOf.has(index)) {
            this.idOf.set(index, nodeId);
        }
        if (!scope.root && !scope.memberSet.has(index)) {
            scope.memberSet.add(index);
            scope.members.push(index);
        }
        return { id: nodeId, index, port };
    }

    /**
     * Push one edge with its effective attributes (scope defaults overridden by the statement's).
     * @param scope - the scope
     * @param source - the source endpoint
     * @param target - the target endpoint
     * @param kind - the edge's direction
     * @param attributes - the statement's attributes
     * @param line - the statement's line
     */
    private pushEdge(
        scope: Scope,
        source: Endpoint,
        target: Endpoint,
        kind: EdgeKind,
        attributes: readonly DotAttribute[],
        line: number,
    ): void {
        const effective = new Map(scope.edgeDefaults);
        for (const a of attributes) {
            effective.set(a.name, a.value);
        }
        const element = `${String(source.id)} ${kind === "directed" ? "->" : "--"} ${String(target.id)}`;
        const where: EdgeWhere = { line, element };
        const { weightFrom } = this.options;
        let weight: number | undefined;
        const weightText = weightFrom === null ? undefined : effective.get(weightFrom);
        if (weightText !== undefined) {
            try {
                weight = parseWeightText(weightText);
            } catch (err) {
                this.report.recordError(err, where);
                this.report.counts.skippedEdges++;
                return;
            }
        }
        const key = effective.get(KEY_ATTRIBUTE);
        const dedupeKey = this.dedupeKey(source.index, target.index, kind, key);
        if (dedupeKey !== null) {
            const existing = (this.strict ? this.strictEdges : this.keyedEdges).get(dedupeKey);
            if (existing !== undefined) {
                this.mergeEdge(existing, weight, effective, where, key);
                this.setPorts(existing, source, target, where);
                return;
            }
        }
        const { sink } = this;
        const before = sink.edgeCount;
        let e: number;
        try {
            e = this.resolver.addEdge(source.id, target.id, kind, weight, where);
        } catch (err) {
            this.report.recordError(err, where);
            this.report.counts.skippedEdges++;
            return;
        }
        this.report.counts.edges += sink.edgeCount - before;
        if (dedupeKey !== null) {
            (this.strict ? this.strictEdges : this.keyedEdges).set(dedupeKey, e);
        }
        for (const [name, value] of effective) {
            if (name !== weightFrom) {
                this.setEdgeAttribute(e, name, value, line, element);
            }
        }
        this.setPorts(e, source, target, where);
    }

    /**
     * Write the endpoints' ports of an edge, when they have any.
     * @param e - the edge index
     * @param source - the source endpoint
     * @param target - the target endpoint
     * @param where - the line and element
     */
    private setPorts(e: number, source: Endpoint, target: Endpoint, where: EdgeWhere): void {
        if (source.port !== null) {
            this.setPort(e, "source", source.port, where);
        }
        if (target.port !== null) {
            this.setPort(e, "target", target.port, where);
        }
    }

    /**
     * The key under which an edge is merged with an earlier one: every (source, target) pair under
     * `strict` (unordered for an undirected edge), else the (source, target, key) triple of an edge
     * carrying a `key` attribute (cgraph's edge identity).
     * @param source - the source index
     * @param target - the target index
     * @param kind - the edge's direction
     * @param key - the `key` attribute, or undefined
     * @returns the dedupe key, or null when the edge is never merged
     */
    private dedupeKey(source: number, target: number, kind: EdgeKind, key: string | undefined): string | null {
        const ordered = kind === "undirected" && target < source ? `${target}>${source}` : `${source}>${target}`;
        if (this.strict) {
            return ordered;
        }
        return key === undefined ? null : `${ordered}#${key}`;
    }

    /**
     * Merge a repeated edge into the earlier one: its attributes overwrite, an explicit weight too.
     * @param e - the existing edge index
     * @param weight - the repeated edge's weight, or undefined
     * @param effective - the repeated edge's effective attributes
     * @param where - the line and element
     * @param key - the `key` attribute when the merge is by key
     */
    private mergeEdge(
        e: number,
        weight: number | undefined,
        effective: ReadonlyMap<string, string>,
        where: EdgeWhere,
        key: string | undefined,
    ): void {
        const byKey = !this.strict && key !== undefined;
        this.report.warning(
            "merged",
            byKey ? DOT_ISSUE.KEY_MERGED : DOT_ISSUE.STRICT_MERGED,
            byKey
                ? `edge ${where.element} with key ${JSON.stringify(key)} repeats an earlier edge; attributes merged`
                : `parallel edge ${where.element} merged into the earlier one (strict graph)`,
            where,
        );
        try {
            if (weight !== undefined) {
                this.sink.setEdgeWeight(e, weight);
            }
        } catch (err) {
            this.report.recordError(err, where);
        }
        for (const [name, value] of effective) {
            if (name !== this.options.weightFrom) {
                this.setEdgeAttribute(e, name, value, where.line, where.element);
            }
        }
    }

    /**
     * Write one node attribute: `label` to the label column, `pos` to the position column,
     * anything else as an inferred cell.
     * @param index - the node index
     * @param name - the attribute name
     * @param value - the value text
     * @param line - the line, for issues
     */
    private setNodeAttribute(index: number, name: string, value: string, line: number): void {
        const element = String(this.idOf.get(index) ?? index);
        try {
            if (name === LABEL_ATTRIBUTE) {
                this.sink.setNodeValue(this.nodeLabel(), index, value);
            } else if (name === POS_ATTRIBUTE) {
                this.setPosition(index, value, line, element);
            } else {
                this.textWriter("node", name).write(index, value);
            }
        } catch (err) {
            this.report.recordError(err, { line, element });
        }
    }

    /**
     * Write one edge attribute: `label` to the label column, `key` to the edge id column, anything
     * else as an inferred cell.
     * @param e - the edge index
     * @param name - the attribute name
     * @param value - the value text
     * @param line - the line, for issues
     * @param element - the edge description, for issues
     */
    private setEdgeAttribute(e: number, name: string, value: string, line: number, element: string): void {
        try {
            if (name === LABEL_ATTRIBUTE) {
                this.setEdgeText(this.edgeLabel(), e, value);
            } else if (name === KEY_ATTRIBUTE) {
                this.setEdgeText(this.keyColumn(), e, value);
            } else {
                this.textWriter("edge", name).write(e, value);
            }
        } catch (err) {
            this.report.recordError(err, { line, element });
        }
    }

    /**
     * Write a graph attribute (the root's `ID = ID` and `graph [..]`).
     * @param attribute - the attribute
     */
    private setGraphAttribute(attribute: DotAttribute): void {
        try {
            if (attribute.name === LABEL_ATTRIBUTE) {
                this.sink.setGraphValue(attribute.name, attribute.value, { dtype: "string", origin: DOT_ORIGIN });
            } else {
                this.sink.setGraphValue(attribute.name, parseTextCell(attribute.value), { origin: DOT_ORIGIN });
            }
        } catch (err) {
            this.report.recordError(err, { line: attribute.line, element: attribute.name });
        }
    }

    /**
     * Write a node's `pos`: `x,y[,z][!]` into the position column, the `!` as `pin` = true.
     * @param index - the node index
     * @param text - the pos text
     * @param line - the line
     * @param element - the node id text
     */
    private setPosition(index: number, text: string, line: number, element: string): void {
        const match = POINT_TEXT.exec(text);
        if (match === null) {
            this.report.warning(
                "validation-error",
                DOT_ISSUE.BAD_POS,
                `pos ${JSON.stringify(text)} is not a point "x,y[,z][!]"; dropped`,
                { line, element },
            );
            return;
        }
        const x = Number(match[1]);
        const y = Number(match[2]);
        const z = match[3] === undefined ? 0 : Number(match[3]);
        const dims = match[3] === undefined ? 2 : 3;
        this.sink.setNodeValue(this.positionColumn(dims), index, [x, y, z]);
        if (match[4] === "!") {
            this.sink.setNodeValue(PIN_ATTRIBUTE, index, true);
        }
    }

    /**
     * Write an endpoint's port into the source / target port column.
     * @param e - the edge index
     * @param side - which endpoint
     * @param port - the port text
     * @param where - the line and element
     */
    private setPort(e: number, side: "source" | "target", port: string, where: EdgeWhere): void {
        try {
            this.setEdgeText(side === "source" ? this.sourcePortColumn() : this.targetPortColumn(), e, port);
        } catch (err) {
            this.report.recordError(err, where);
        }
    }

    /**
     * Write a text cell into one of the importer's edge text columns.
     * @param handle - the column
     * @param e - the edge index
     * @param text - the text
     */
    private setEdgeText(handle: ColumnHandle, e: number, text: string): void {
        this.sink.setEdgeValue(handle, e, text);
    }

    // ============================================================ lazily declared columns

    /**
     * The node label column (string, role label).
     * @returns the handle
     */
    private nodeLabel(): ColumnHandle {
        if (this.nodeLabelHandle === INVALID_INDEX) {
            this.nodeLabelHandle = this.declare("node", {
                name: LABEL_ATTRIBUTE,
                dtype: "string",
                nullable: true,
                role: "label",
                origin: DOT_ORIGIN,
            });
        }
        return this.nodeLabelHandle;
    }

    /**
     * The edge label column (string, role label).
     * @returns the handle
     */
    private edgeLabel(): ColumnHandle {
        if (this.edgeLabelHandle === INVALID_INDEX) {
            this.edgeLabelHandle = this.declare("edge", {
                name: LABEL_ATTRIBUTE,
                dtype: "string",
                nullable: true,
                role: "label",
                origin: DOT_ORIGIN,
            });
        }
        return this.edgeLabelHandle;
    }

    /**
     * The position column (f32 x3, role position, design section 5.2), declared on the first `pos`.
     * @param dims - the dimensions of the first value, recorded in extra.sourceDims
     * @returns the handle
     */
    private positionColumn(dims: 2 | 3): ColumnHandle {
        if (this.positionHandle === INVALID_INDEX) {
            this.positionHandle = this.declare("node", {
                name: POS_ATTRIBUTE,
                dtype: "f32",
                components: 3,
                nullable: true,
                mutable: true,
                role: "position",
                origin: { ...DOT_ORIGIN, type: "point" },
                extra: { sourceDims: dims, units: "file" },
            });
        }
        return this.positionHandle;
    }

    /**
     * The cluster marker column (bool).
     * @returns the handle
     */
    private clusterColumn(): ColumnHandle {
        if (this.clusterHandle === INVALID_INDEX) {
            this.clusterHandle = this.declare("node", {
                name: CLUSTER_COLUMN,
                dtype: "bool",
                nullable: true,
                origin: DOT_ORIGIN,
            });
        }
        return this.clusterHandle;
    }

    /**
     * The parent column (u32, role parent, refersTo node).
     * @returns the handle
     */
    private parentColumn(): ColumnHandle {
        if (this.parentHandle === INVALID_INDEX) {
            this.parentHandle = this.declare("node", {
                name: PARENT_COLUMN,
                dtype: "u32",
                nullable: true,
                role: "parent",
                refersTo: "node",
                origin: DOT_ORIGIN,
            });
        }
        return this.parentHandle;
    }

    /**
     * The edge key column (string, role id): cgraph's edge identity within a source / target pair.
     * @returns the handle
     */
    private keyColumn(): ColumnHandle {
        if (this.keyHandle === INVALID_INDEX) {
            this.keyHandle = this.declare("edge", {
                name: KEY_ATTRIBUTE,
                dtype: "string",
                nullable: true,
                role: "id",
                origin: { ...DOT_ORIGIN, id: KEY_ATTRIBUTE },
            });
        }
        return this.keyHandle;
    }

    /**
     * The source port column (string, role sourcePort).
     * @returns the handle
     */
    private sourcePortColumn(): ColumnHandle {
        if (this.sourcePortHandle === INVALID_INDEX) {
            this.sourcePortHandle = this.declare("edge", {
                name: SOURCE_PORT_COLUMN,
                dtype: "string",
                nullable: true,
                role: "sourcePort",
                origin: DOT_ORIGIN,
            });
        }
        return this.sourcePortHandle;
    }

    /**
     * The target port column (string, role targetPort).
     * @returns the handle
     */
    private targetPortColumn(): ColumnHandle {
        if (this.targetPortHandle === INVALID_INDEX) {
            this.targetPortHandle = this.declare("edge", {
                name: TARGET_PORT_COLUMN,
                dtype: "string",
                nullable: true,
                role: "targetPort",
                origin: DOT_ORIGIN,
            });
        }
        return this.targetPortHandle;
    }

    /**
     * The inferred-column writer of an attribute name in a domain (design section 5.1: the column's
     * dtype follows the text grammar per column; a caller's column of the name receives parsed
     * values through the sink's own inference).
     * @param domain - node or edge
     * @param name - the attribute name
     * @returns the writer
     */
    private textWriter(domain: "node" | "edge", name: string): TextCellWriter {
        const writers = domain === "node" ? this.nodeWriters : this.edgeWriters;
        let writer = writers.get(name);
        if (writer === undefined) {
            writer = new TextCellWriter(name, domain, this.sink, this.report);
            writers.set(name, writer);
        }
        return writer;
    }

    /**
     * Declare one of the importer's columns on the sink through the shared design section 5.6
     * rule: a caller's sink that already holds the role gets the column without it (reported), one
     * that holds the name with another shape gets it renamed `<name>#<id>` (reported).
     * @param domain - node or edge
     * @param decl - the declaration
     * @returns the handle
     */
    private declare(domain: "node" | "edge", decl: ColumnDecl): ColumnHandle {
        const withId: ColumnDecl =
            decl.origin?.id === undefined ? { ...decl, origin: { ...decl.origin, id: decl.name } } : decl;
        return declareResolved(this.sink, domain, withId, this.report, { element: decl.name }).handle;
    }
}
