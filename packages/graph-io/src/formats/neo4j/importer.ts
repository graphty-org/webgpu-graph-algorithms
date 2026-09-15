/**
 * The Neo4j importer (design sections 8.4 and 5.1, research note 07 sections 2.6 and 2.10): reads
 * neo4j-admin import CSV -- node tables with `:ID`, `:LABEL` and typed property columns,
 * relationship tables with `:START_ID`, `:END_ID`, `:TYPE` and typed property columns -- and pushes
 * scalars into the sink one record at a time. Input is one or more files: the primary input plus the
 * `nodes` and `relationships` option inputs, each read in order; every input holds one or more
 * sections, each starting with its own header row (the single-file convention of graphty-element's
 * CSVDataSource, where a node table and a relationship table follow each other in one file).
 *
 * Mapping (design section 5.1 and decision Q27):
 * - Property columns are declared up front from the header types (`int` -> i32, `long` -> f64
 *   with a precision issue beyond 2^53, `float` -> f32, `double` -> f64, `boolean` -> bool,
 *   `string` / `char` / `duration` -> string, temporal types -> f64 milliseconds with a `.text`
 *   companion when the source text is not canonical, `point` -> json, `type[]` -> list); an
 *   untyped column is a string property. An unquoted empty cell is "property not set"; a quoted
 *   empty cell is an empty string (or an empty list), as neo4j-admin stores it by default.
 * - `:LABEL` becomes the node list-of-dict column `labels` (role `labels`); `:TYPE` the edge dict
 *   column `type` (role `kind`); an id space `:ID(Space)` the node dict column `idSpace` (role
 *   `idSpace`), and a stored id `name:ID(Space)` also a string node column `name` whose
 *   `origin.namespace` is the space. A property whose name collides with one of those is renamed
 *   `<name>#<name>` (design section 5.6).
 * - Ids are text cells coerced by the `ids` option ("canonical" by default, so `1` is the number 1
 *   and `007` stays a string; integers beyond 2^53 stay strings). Relationships are always directed
 *   ("In Neo4j, all relationships have a direction"), so the sink is set directed before the first
 *   edge and `onMixedDirection: "undirected"` is the way to read a file as undirected.
 * - `:IGNORE` columns are skipped and counted in a loss note.
 */

import {
    type ColumnDecl,
    type ColumnHandle,
    GraphFormatError,
    type GraphSink,
    INVALID_INDEX,
    type NodeId,
} from "@graphty/graph-format";

import {
    type AttributeDeclarationInput,
    declareAttribute,
    declareCompanion,
    type DeclaredAttribute,
    declareOn,
    losesPrecision,
    parseDeclaredTemporal,
    parseDeclaredValue,
    PRECISION_CODE,
    RENAMED_CODE,
    takenIn,
    uniqueColumnName,
} from "../../common/attributes.js";
import {
    DUPLICATE_NODE_CODE as SHARED_DUPLICATE_NODE_CODE,
    MISSING_ENDPOINT_CODE as SHARED_MISSING_ENDPOINT_CODE,
    MISSING_ID_CODE as SHARED_MISSING_ID_CODE,
    ROLE_TAKEN_CODE as SHARED_ROLE_TAKEN_CODE,
} from "../../common/codes.js";
import { type DeclaredTypeSpec } from "../../common/declared-types.js";
import { DirectionResolver } from "../../common/direction.js";
import { ID_MERGED_CODE as SHARED_ID_MERGED_CODE, IdCoercer } from "../../common/ids.js";
import { inputLength, isImportInput, type ReadOptions, throwIfAborted } from "../../common/input.js";
import { type ListSyntax, splitListText } from "../../common/lists.js";
import {
    reportSinkOptions,
    reportUnusedOptions,
    type ResolvedImportOptions,
    resolveImportOptions,
} from "../../common/options.js";
import { ImportReportBuilder } from "../../common/report.js";
import { parseWeightText } from "../../common/weights.js";
import {
    type CommonImportOptions,
    type GraphImporter,
    ImportError,
    type ImportInput,
    type ImportReport,
} from "../../types.js";
import { checkRecordSyntax, RecordReader, type RecordSyntax } from "../csv/records.js";
import { type FieldKind, type HeaderField, isHeaderRecord, parseHeaderField } from "./header.js";

/** The format-specific options of the Neo4j importer. */
export interface Neo4jImportOptions {
    /** Further node files, each with its own header row(s); read after the primary input. */
    nodes?: ImportInput | readonly ImportInput[] | undefined;
    /** Relationship files, each with its own header row(s); read after the node files. */
    relationships?: ImportInput | readonly ImportInput[] | undefined;
    /**
     * The field delimiter (neo4j-admin `--delimiter`); one character. When absent it is sniffed
     * from the first rows between "," and a tab, so a `.tsv` file needs no option.
     */
    delimiter?: string | undefined;
    /** The array delimiter of list values and `:LABEL` cells (neo4j-admin `--array-delimiter`); ";" by default. */
    arrayDelimiter?: ";" | "," | "|" | undefined;
    /** The quote character (neo4j-admin `--quote`); one character; a double quote by default. */
    quote?: string | undefined;
}

/** The name of the node list column holding `:LABEL` values. */
export const LABELS_COLUMN = "labels";

/** The name of the edge dict column holding `:TYPE` values. */
export const TYPE_COLUMN = "type";

/** The name of the node dict column holding the id space of `:ID(Space)`. */
export const ID_SPACE_COLUMN = "idSpace";

/** Issue code: a header row (or a whole section) is malformed; the import aborts. */
export const HEADER_CODE = "E_NEO4J_HEADER";

/** Issue code: a row has a different number of cells than its header. */
export const COLUMN_COUNT_CODE = "E_NEO4J_COLUMN_COUNT";

/** Issue code: a node row has an unquoted empty `:ID` cell (a quoted empty cell is the id ""). */
export const MISSING_ID_CODE = SHARED_MISSING_ID_CODE;

/** Issue code: a relationship row has an unquoted empty `:START_ID` or `:END_ID` cell. */
export const MISSING_ENDPOINT_CODE = SHARED_MISSING_ENDPOINT_CODE;

/** Issue code: a node id was declared twice (same id space); the later row's properties win. */
export const DUPLICATE_NODE_CODE = SHARED_DUPLICATE_NODE_CODE;

/** Issue code: a node id was declared in two id spaces; the core has one id space and the later row is skipped. */
export const ID_SPACE_COLLISION_CODE = "E_NEO4J_ID_SPACE_COLLISION";

/** Issue code: two different id cells became one id under `ids: "number"`. */
export const ID_MERGED_CODE = SHARED_ID_MERGED_CODE;

/** Issue code: a header brace option the importer does not act on. */
export const HEADER_OPTION_CODE = "W_NEO4J_HEADER_OPTION_IGNORED";

/** Issue code: a reserved column (labels / type / idSpace) lost its role because the sink already holds it. */
export const ROLE_TAKEN_CODE = SHARED_ROLE_TAKEN_CODE;

/** Loss code: `:IGNORE` columns were skipped. */
export const IGNORED_COLUMNS_LOSS = "W_NEO4J_IGNORED_COLUMNS";

/** The common options the Neo4j importer reads (the rest is reported by reportUnusedOptions). */
const USED_OPTIONS: ReadonlySet<keyof CommonImportOptions> = new Set<keyof CommonImportOptions>([
    "ids",
    "addMissingNodes",
    "duplicateEdges",
    "selfLoops",
    "onMixedDirection",
    "weightFrom",
    "weightDtype",
    "long",
    "errorLimit",
    "signal",
    "onProgress",
]);

const NEO4J = "neo4j";

/** Rows between two checks of the cancellation signal (a whole string input is one chunk). */
const ABORT_CHECK_INTERVAL = 64;

const LABELS_DECL: ColumnDecl = {
    name: LABELS_COLUMN,
    dtype: "list",
    itemDtype: "dict",
    nullable: true,
    role: "labels",
    origin: { format: NEO4J, id: ":LABEL", title: null, type: "LABEL", namespace: null },
};

const TYPE_DECL: ColumnDecl = {
    name: TYPE_COLUMN,
    dtype: "dict",
    nullable: true,
    role: "kind",
    origin: { format: NEO4J, id: ":TYPE", title: null, type: "TYPE", namespace: null },
};

const ID_SPACE_DECL: ColumnDecl = {
    name: ID_SPACE_COLUMN,
    dtype: "dict",
    nullable: true,
    role: "idSpace",
    origin: { format: NEO4J, id: ":ID", title: null, type: "ID", namespace: null },
};

const ARRAY_DELIMITERS: Readonly<Record<string, ListSyntax>> = { ";": "semicolon", ",": "comma", "|": "pipe" };

/** The resolved format-specific options. */
interface ResolvedNeo4jOptions {
    readonly nodes: readonly ImportInput[];
    readonly relationships: readonly ImportInput[];
    readonly syntax: RecordSyntax;
    readonly listSyntax: ListSyntax;
}

/** One property column of a section. */
interface PropertySlot {
    /** The cell index. */
    readonly cell: number;
    /** The column name (after any rename). */
    readonly name: string;
    /** The column handle. */
    readonly handle: ColumnHandle;
    /** How the cell text is parsed. */
    readonly spec: DeclaredTypeSpec;
    /** The declaration of the companion text column of a temporal property, or null. */
    readonly companionDecl: ColumnDecl | null;
    /** The companion's handle once a value needed it (design section 5.1); INVALID_INDEX before. */
    companion: ColumnHandle;
}

/** A node section: the header interpreted. */
interface NodeSection {
    readonly kind: "node";
    readonly width: number;
    readonly idCell: number;
    /** The column holding the id as a property (`name:ID`), or INVALID_INDEX. */
    readonly idHandle: ColumnHandle;
    readonly space: string | null;
    readonly spaceCode: number;
    readonly labelCells: readonly number[];
    readonly extraLabels: readonly string[];
    readonly properties: readonly PropertySlot[];
}

/** A relationship section: the header interpreted. */
interface RelationshipSection {
    readonly kind: "relationship";
    readonly width: number;
    readonly startCell: number;
    readonly endCell: number;
    /** The `:TYPE` cell, or -1. */
    readonly typeCell: number;
    /** The cell of the property named by `weightFrom`, or -1. */
    readonly weightCell: number;
    readonly properties: readonly PropertySlot[];
}

type Section = NodeSection | RelationshipSection;

/** The delimiters sniffed between when none is given: neo4j-admin's default and the TSV tab. */
const NEO4J_DELIMITER_CANDIDATES: readonly string[] = Object.freeze([",", "\t"]);

/**
 * Resolve the format-specific options.
 * @param options - the caller's options
 * @returns the resolved options; E_UNSUPPORTED for an invalid value
 */
function resolveNeo4jOptions(options: Neo4jImportOptions | undefined): ResolvedNeo4jOptions {
    const o: Neo4jImportOptions = options ?? {};
    const arrayDelimiter = o.arrayDelimiter ?? ";";
    const listSyntax = ARRAY_DELIMITERS[arrayDelimiter];
    if (typeof arrayDelimiter !== "string" || listSyntax === undefined) {
        throw new GraphFormatError(
            "E_UNSUPPORTED",
            `option arrayDelimiter: ${JSON.stringify(arrayDelimiter)} is not one of ";", ",", "|"`,
            { option: "arrayDelimiter", found: arrayDelimiter },
        );
    }
    const syntax = checkRecordSyntax({
        delimiter: o.delimiter ?? null,
        quote: o.quote ?? '"',
        candidates: NEO4J_DELIMITER_CANDIDATES.filter((d) => d !== arrayDelimiter),
    });
    if (syntax.delimiter === arrayDelimiter) {
        throw new GraphFormatError("E_UNSUPPORTED", "options delimiter and arrayDelimiter must differ", {
            option: "arrayDelimiter",
            found: arrayDelimiter,
        });
    }
    return {
        nodes: inputList("nodes", o.nodes),
        relationships: inputList("relationships", o.relationships),
        syntax,
        listSyntax,
    };
}

/**
 * Normalise an input-list option.
 * @param name - the option name
 * @param value - one input, a list of inputs, or undefined
 * @returns the inputs; E_UNSUPPORTED for anything else
 */
function inputList(name: string, value: unknown): readonly ImportInput[] {
    if (value === undefined || value === null) {
        return [];
    }
    const list: unknown[] = Array.isArray(value) ? (value as unknown[]) : [value];
    for (const item of list) {
        if (!isImportInput(item)) {
            throw new GraphFormatError("E_UNSUPPORTED", `option ${name}: an entry is not an ImportInput`, {
                option: name,
                found: typeof item,
            });
        }
    }
    return list as ImportInput[];
}

/**
 * Which nodes were declared by a node row and in which id space, by node index, so a repeated id
 * is reported (a duplicate in one space, a collision across spaces).
 */
class NodeRegistry {
    private codes = new Uint32Array(1024);

    private readonly spaces = new Map<string | null, number>();

    /**
     * The code of an id space (1 for "no space").
     * @param space - the space name or null
     * @returns a code >= 1
     */
    codeOf(space: string | null): number {
        let code = this.spaces.get(space);
        if (code === undefined) {
            code = this.spaces.size + 1;
            this.spaces.set(space, code);
        }
        return code;
    }

    /**
     * Record that a node row declared a node.
     * @param index - the node index
     * @param code - the space code
     * @returns "new" for a first declaration, "duplicate" for a repeat in the same space, "collision" across spaces
     */
    declare(index: number, code: number): "new" | "duplicate" | "collision" {
        if (index >= this.codes.length) {
            let size = this.codes.length * 2;
            while (size <= index) {
                size *= 2;
            }
            const grown = new Uint32Array(size);
            grown.set(this.codes);
            this.codes = grown;
        }
        const previous = this.codes[index];
        if (previous === 0) {
            this.codes[index] = code;
            return "new";
        }
        return previous === code ? "duplicate" : "collision";
    }
}

/**
 * Byte progress over several inputs as one sequence: each input's progress is offset by the bytes
 * of the inputs before it, and the total is known only when every input is in memory.
 */
class ProgressTracker {
    private readonly callback: ((bytesDone: number, bytesTotal?: number) => void) | null;

    private readonly total: number | undefined;

    private offset = 0;

    private lastDone = 0;

    /**
     * Create a tracker.
     * @param callback - the caller's onProgress, or null
     * @param inputs - every input in reading order
     */
    constructor(callback: ((bytesDone: number, bytesTotal?: number) => void) | null, inputs: readonly ImportInput[]) {
        this.callback = callback;
        let total: number | undefined = 0;
        for (const input of inputs) {
            const length = inputLength(input);
            if (length === null) {
                total = undefined;
                break;
            }
            total += length;
        }
        this.total = total;
    }

    /**
     * The read options for one input.
     * @param signal - the cancellation signal
     * @returns options whose onProgress reports cumulative bytes
     */
    optionsFor(signal: AbortSignal | null): ReadOptions {
        const { callback } = this;
        if (callback === null) {
            return { signal };
        }
        return {
            signal,
            onProgress: (done: number): void => {
                this.lastDone = done;
                callback(this.offset + done, this.total);
            },
        };
    }

    /** Move the offset past the input just finished. */
    finishInput(): void {
        this.offset += this.lastDone;
        this.lastDone = 0;
    }
}

/**
 * The state of one import call: the sink, the report, the resolved options, the reserved column
 * handles and the per-section row handlers.
 */
class Neo4jImportSession {
    private readonly sink: GraphSink;

    private readonly report: ImportReportBuilder;

    private readonly common: ResolvedImportOptions;

    private readonly options: ResolvedNeo4jOptions;

    private readonly coercer: IdCoercer;

    private readonly direction: DirectionResolver;

    private readonly registry = new NodeRegistry();

    private labelsHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private typeHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private idSpaceHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;

    private ignoredColumns = 0;

    /** Whether the sink's direction was set (before the first relationship, design section 8.4 rule 1). */
    private headerSet = false;

    /** Scratch: the parsed value of every cell of the current row. */
    private values: unknown[] = [];

    /** Scratch: the companion text of every cell of the current row. */
    private texts: (string | null)[] = [];

    /** Scratch: the property slots of the current row that lost precision. */
    private readonly precisionSlots: PropertySlot[] = [];

    /**
     * Create a session.
     * @param sink - the sink
     * @param report - the report
     * @param common - the resolved common options
     * @param options - the resolved format options
     */
    constructor(
        sink: GraphSink,
        report: ImportReportBuilder,
        common: ResolvedImportOptions,
        options: ResolvedNeo4jOptions,
    ) {
        this.sink = sink;
        this.report = report;
        this.common = common;
        this.options = options;
        this.coercer = new IdCoercer(common.ids);
        this.direction = new DirectionResolver(sink, report, common.onMixedDirection);
    }

    /**
     * Read every input.
     * @param inputs - the inputs in reading order
     */
    async run(inputs: readonly ImportInput[]): Promise<void> {
        const progress = new ProgressTracker(this.common.onProgress, inputs);
        for (const input of inputs) {
            await this.readInput(input, progress.optionsFor(this.common.signal));
            progress.finishInput();
        }
        if (this.ignoredColumns > 0) {
            this.report.loss(
                IGNORED_COLUMNS_LOSS,
                `${this.ignoredColumns} :IGNORE column(s) were skipped as the header instructs`,
                null,
                this.ignoredColumns,
            );
        }
    }

    /**
     * Read one input: a header row, then data rows until the next header row.
     * @param input - the input
     * @param readOptions - cancellation and progress
     */
    private async readInput(input: ImportInput, readOptions: ReadOptions): Promise<void> {
        const reader = new RecordReader(input, this.report, this.options.syntax, readOptions);
        let section: Section | null = null;
        let sinceCheck = 0;
        for await (const count of reader) {
            if (section === null || isHeaderRecord(reader.cells, count)) {
                section = this.declareSection(reader, count);
                continue;
            }
            if (section.kind === "node") {
                this.nodeRow(section, reader, count);
            } else {
                this.relationshipRow(section, reader, count);
            }
            if (++sinceCheck >= ABORT_CHECK_INTERVAL) {
                sinceCheck = 0;
                throwIfAborted(readOptions.signal);
            }
        }
        if (section === null) {
            this.report.fail(HEADER_CODE, "the input has no header row", { line: 1 });
        }
    }

    /**
     * Interpret a header row: parse every cell, check the section shape, declare its columns.
     * @param reader - the reader positioned on the header
     * @param count - the number of header cells
     * @returns the section
     */
    private declareSection(reader: RecordReader, count: number): Section {
        const { line } = reader;
        const fields: HeaderField[] = [];
        try {
            for (let i = 0; i < count; i++) {
                fields.push(parseHeaderField(reader.cells[i]));
            }
            const kinds = new Map<FieldKind, number[]>();
            fields.forEach((field, i) => {
                const list = kinds.get(field.kind);
                if (list === undefined) {
                    kinds.set(field.kind, [i]);
                } else {
                    list.push(i);
                }
            });
            const ids = kinds.get("ID") ?? [];
            const starts = kinds.get("START_ID") ?? [];
            const ends = kinds.get("END_ID") ?? [];
            const isNode = ids.length > 0;
            const isRelationship = starts.length > 0 || ends.length > 0;
            if (isNode && isRelationship) {
                throw headerError("a header mixes :ID with :START_ID / :END_ID");
            }
            if (!isNode && !isRelationship) {
                throw headerError(
                    "a header needs an :ID column (nodes) or :START_ID and :END_ID columns (relationships)",
                );
            }
            checkPropertyNames(fields);
            this.ignoredColumns += (kinds.get("IGNORE") ?? []).length;
            if (isNode) {
                if (ids.length > 1) {
                    throw headerError("a node header has more than one :ID column");
                }
                if (kinds.has("TYPE")) {
                    throw headerError("a node header cannot have a :TYPE column");
                }
                return this.declareNodeSection(fields, ids[0], kinds.get("LABEL") ?? [], line);
            }
            if (starts.length !== 1 || ends.length !== 1) {
                throw headerError("a relationship header needs exactly one :START_ID and one :END_ID column");
            }
            if (kinds.has("LABEL")) {
                throw headerError("a relationship header cannot have a :LABEL column");
            }
            const types = kinds.get("TYPE") ?? [];
            if (types.length > 1) {
                throw headerError("a relationship header has more than one :TYPE column");
            }
            return this.declareRelationshipSection(
                fields,
                starts[0],
                ends[0],
                types.length === 1 ? types[0] : -1,
                line,
            );
        } catch (err) {
            if (err instanceof GraphFormatError && !(err instanceof ImportError)) {
                this.report.fail(HEADER_CODE, `line ${line}: ${err.message}`, { line }, { cause: err.code });
            }
            throw err;
        }
    }

    /**
     * Declare the columns of a node section.
     * @param fields - the parsed header
     * @param idCell - the `:ID` cell
     * @param labelCells - the `:LABEL` cells
     * @param line - the header line
     * @returns the section
     */
    private declareNodeSection(
        fields: readonly HeaderField[],
        idCell: number,
        labelCells: readonly number[],
        line: number,
    ): NodeSection {
        const idField = fields[idCell];
        const extraLabels: string[] = [];
        for (const field of fields) {
            for (const [key, value] of field.options) {
                if (key === "label" && field.kind === "ID") {
                    extraLabels.push(value);
                } else {
                    this.report.warning(
                        "unsupported",
                        HEADER_OPTION_CODE,
                        `header option ${key}:${value} of "${field.text}" is ignored`,
                        { line, element: field.text },
                    );
                }
            }
        }
        if (labelCells.length > 0 || extraLabels.length > 0) {
            this.ensureLabels();
        }
        if (idField.space !== null) {
            this.ensureIdSpace();
        }
        let idHandle: ColumnHandle = INVALID_INDEX as ColumnHandle;
        if (idField.name.length > 0) {
            idHandle = this.declareProperty(
                "node",
                { ...idField, type: null },
                idCell,
                { origin: { format: NEO4J, id: idField.name, title: null, type: "ID", namespace: idField.space } },
                line,
            ).handle;
        }
        const properties = this.declareProperties("node", fields, line, null);
        return {
            kind: "node",
            width: fields.length,
            idCell,
            idHandle,
            space: idField.space,
            spaceCode: this.registry.codeOf(idField.space),
            labelCells,
            extraLabels,
            properties,
        };
    }

    /**
     * Declare the columns of a relationship section.
     * @param fields - the parsed header
     * @param startCell - the `:START_ID` cell
     * @param endCell - the `:END_ID` cell
     * @param typeCell - the `:TYPE` cell, or -1
     * @param line - the header line
     * @returns the section
     */
    private declareRelationshipSection(
        fields: readonly HeaderField[],
        startCell: number,
        endCell: number,
        typeCell: number,
        line: number,
    ): RelationshipSection {
        for (const field of fields) {
            for (const [key, value] of field.options) {
                this.report.warning(
                    "unsupported",
                    HEADER_OPTION_CODE,
                    `header option ${key}:${value} of "${field.text}" is ignored`,
                    { line, element: field.text },
                );
            }
        }
        if (typeCell >= 0) {
            this.ensureType();
        }
        const { weightFrom } = this.common;
        let weightCell = -1;
        if (weightFrom !== null) {
            weightCell = fields.findIndex((field) => field.kind === "PROPERTY" && field.name === weightFrom);
        }
        const properties = this.declareProperties("edge", fields, line, weightCell);
        return { kind: "relationship", width: fields.length, startCell, endCell, typeCell, weightCell, properties };
    }

    /**
     * Declare every PROPERTY field of a header.
     * @param domain - node or edge
     * @param fields - the parsed header
     * @param line - the header line
     * @param skipCell - a cell to leave undeclared (the weight), or null
     * @returns the property slots
     */
    private declareProperties(
        domain: "node" | "edge",
        fields: readonly HeaderField[],
        line: number,
        skipCell: number | null,
    ): PropertySlot[] {
        const slots: PropertySlot[] = [];
        fields.forEach((field, cell) => {
            if (field.kind !== "PROPERTY" || cell === skipCell) {
                return;
            }
            slots.push(this.declareProperty(domain, field, cell, {}, line));
        });
        const width = fields.length;
        if (this.values.length < width) {
            this.values = new Array<unknown>(width);
            this.texts = new Array<string | null>(width);
        }
        return slots;
    }

    /**
     * Declare one property column on the sink: the same name and shape again shares the column;
     * a different shape under the same name is renamed `<name>#<name>` (design section 5.6).
     * @param domain - node or edge
     * @param field - the header field
     * @param cell - the field's cell index
     * @param patch - declaration fields to override (the id property's origin)
     * @param line - the header line
     * @returns the slot
     */
    private declareProperty(
        domain: "node" | "edge",
        field: HeaderField,
        cell: number,
        patch: Partial<ColumnDecl>,
        line: number,
    ): PropertySlot {
        const { sink, report } = this;
        const { listSyntax } = this.options;
        const input: AttributeDeclarationInput = {
            format: NEO4J,
            id: field.name,
            title: null,
            type: field.type,
            namespace: null,
            listSyntax,
            long: this.common.long,
        };
        let declared: DeclaredAttribute = declareAttribute(input);
        let decl: ColumnDecl = { ...declared.decl, ...patch };
        let handle: ColumnHandle;
        try {
            handle = declareOn(sink, domain, decl);
        } catch (err) {
            if (!(err instanceof GraphFormatError) || err.code !== "E_COLUMN_EXISTS") {
                throw err;
            }
            declared = declareAttribute({ ...input, taken: takenIn(sink, domain) });
            decl = { ...declared.decl, ...patch, name: declared.decl.name };
            handle = declareOn(sink, domain, decl);
        }
        for (const issue of declared.issues) {
            report.warning(issue.category, issue.code, issue.message, { line, element: field.text });
        }
        return {
            cell,
            name: decl.name,
            handle,
            spec: declared.spec,
            companionDecl: declared.companion,
            companion: INVALID_INDEX as ColumnHandle,
        };
    }

    /** Declare (or adopt) the labels column. */
    private ensureLabels(): void {
        if (this.labelsHandle === INVALID_INDEX) {
            this.labelsHandle = this.declareReserved("node", LABELS_DECL);
        }
    }

    /** Declare (or adopt) the relationship type column. */
    private ensureType(): void {
        if (this.typeHandle === INVALID_INDEX) {
            this.typeHandle = this.declareReserved("edge", TYPE_DECL);
        }
    }

    /** Declare (or adopt) the id space column. */
    private ensureIdSpace(): void {
        if (this.idSpaceHandle === INVALID_INDEX) {
            this.idSpaceHandle = this.declareReserved("node", ID_SPACE_DECL);
        }
    }

    /**
     * Declare a reserved column: the same shape again shares it; a name taken by another shape is
     * renamed `<name>#<origin.id>` and reported; a role the sink already holds elsewhere is dropped
     * and reported.
     * @param domain - node or edge
     * @param decl - the reserved declaration
     * @returns the handle
     */
    private declareReserved(domain: "node" | "edge", decl: ColumnDecl): ColumnHandle {
        const { sink, report } = this;
        let current = decl;
        for (;;) {
            try {
                return declareOn(sink, domain, current);
            } catch (err) {
                if (!(err instanceof GraphFormatError)) {
                    throw err;
                }
                if (err.code === "E_COLUMN_EXISTS") {
                    const name = uniqueColumnName(current.name, current.origin?.id ?? null, takenIn(sink, domain));
                    report.warning(
                        "coercion",
                        RENAMED_CODE,
                        `column "${current.name}" renamed to "${name}": the name was taken`,
                        { element: current.name },
                    );
                    current = { ...current, name };
                } else if (err.code === "E_DUPLICATE_ROLE") {
                    report.warning(
                        "coercion",
                        ROLE_TAKEN_CODE,
                        `column "${current.name}" declared without role "${String(current.role)}": the sink already holds that role`,
                        { element: current.name },
                    );
                    const { role: _role, ...withoutRole } = current;
                    current = withoutRole;
                } else {
                    throw err;
                }
            }
        }
    }

    /**
     * Push one node row.
     * @param section - the section
     * @param reader - the reader positioned on the row
     * @param count - the row's cell count
     */
    private nodeRow(section: NodeSection, reader: RecordReader, count: number): void {
        const { sink, report } = this;
        const { cells, quoted, line } = reader;
        if (count !== section.width) {
            report.error(
                "validation-error",
                COLUMN_COUNT_CODE,
                `row has ${count} cell(s) but the header has ${section.width}`,
                { line },
            );
            report.counts.skippedNodes++;
            return;
        }
        const idText = cells[section.idCell];
        if (idText.length === 0 && !quoted[section.idCell]) {
            // a quoted empty cell is the id "" (neo4j-admin: an empty quoted field is an empty string)
            report.error("missing-value", MISSING_ID_CODE, "empty :ID cell", { line });
            report.counts.skippedNodes++;
            return;
        }
        const id = this.coerceId(idText, line);
        if (id === null || !this.parseProperties(section.properties, cells, quoted, line, idText)) {
            report.counts.skippedNodes++;
            return;
        }
        let labels: string[] | undefined;
        if (section.labelCells.length > 0 || section.extraLabels.length > 0) {
            labels = this.labelsOf(section, cells, quoted);
        }
        const index = sink.addNode(id);
        const status = this.registry.declare(index, section.spaceCode);
        if (status === "collision") {
            report.error(
                "validation-error",
                ID_SPACE_COLLISION_CODE,
                `node ${idText} is declared in id space ${section.space ?? "(none)"} and in another id space; the core has one id space and this row is skipped`,
                { line, element: idText },
            );
            report.counts.skippedNodes++;
            return;
        }
        if (status === "duplicate") {
            report.warning(
                "merged",
                DUPLICATE_NODE_CODE,
                `node ${idText} is declared twice; the later properties win`,
                {
                    line,
                    element: idText,
                },
            );
        }
        if (section.idHandle !== INVALID_INDEX) {
            sink.setNodeValue(section.idHandle, index, idText);
        }
        if (section.space !== null) {
            sink.setNodeValue(this.idSpaceHandle, index, section.space);
        }
        if (labels !== undefined) {
            sink.setNodeValue(this.labelsHandle, index, labels);
        }
        this.writeProperties("node", section.properties, index);
        this.reportPrecision(line, idText);
        report.counts.nodes++;
    }

    /**
     * Push one relationship row.
     * @param section - the section
     * @param reader - the reader positioned on the row
     * @param count - the row's cell count
     */
    private relationshipRow(section: RelationshipSection, reader: RecordReader, count: number): void {
        const { sink, report } = this;
        const { cells, quoted, line } = reader;
        if (!this.headerSet) {
            // a Neo4j file is directed by definition ("all relationships have a direction"); the sink's
            // direction is set once, before the first relationship, so a node-only file leaves it alone
            this.headerSet = true;
            this.direction.setHeader(true, { line });
        }
        if (count !== section.width) {
            report.error(
                "validation-error",
                COLUMN_COUNT_CODE,
                `row has ${count} cell(s) but the header has ${section.width}`,
                { line },
            );
            report.counts.skippedEdges++;
            return;
        }
        const startText = cells[section.startCell];
        const endText = cells[section.endCell];
        const startMissing = startText.length === 0 && !quoted[section.startCell];
        if (startMissing || (endText.length === 0 && !quoted[section.endCell])) {
            report.error(
                "missing-value",
                MISSING_ENDPOINT_CODE,
                `empty ${startMissing ? ":START_ID" : ":END_ID"} cell`,
                {
                    line,
                },
            );
            report.counts.skippedEdges++;
            return;
        }
        const element = `${startText}->${endText}`;
        const source = this.coerceId(startText, line);
        const target = source === null ? null : this.coerceId(endText, line);
        if (source === null || target === null) {
            report.counts.skippedEdges++;
            return;
        }
        let weight: number | undefined;
        if (section.weightCell >= 0) {
            try {
                weight = parseWeightText(cells[section.weightCell]);
            } catch (err) {
                report.recordError(err, { line, element });
                report.counts.skippedEdges++;
                return;
            }
        }
        if (!this.parseProperties(section.properties, cells, quoted, line, element)) {
            report.counts.skippedEdges++;
            return;
        }
        let edge: number;
        try {
            edge = this.direction.addEdge(source, target, "directed", weight, { line, element });
        } catch (err) {
            report.recordError(err, { line, element });
            report.counts.skippedEdges++;
            return;
        }
        if (section.typeCell >= 0) {
            const type = cells[section.typeCell];
            if (type.length > 0) {
                sink.setEdgeValue(this.typeHandle, edge, type);
            }
        }
        this.writeProperties("edge", section.properties, edge);
        this.reportPrecision(line, element);
        report.counts.edges++;
    }

    /**
     * Coerce an id cell, reporting a merge under `ids: "number"` and an invalid id as an error.
     * @param text - the cell
     * @param line - the row's line
     * @returns the id, or null when the cell was rejected (the error is recorded)
     */
    private coerceId(text: string, line: number): NodeId | null {
        const { coercer, report } = this;
        let id: NodeId;
        try {
            id = coercer.text(text);
        } catch (err) {
            report.recordError(err, { line, element: text });
            return null;
        }
        const merge = coercer.lastMerge;
        if (merge !== null) {
            report.warning(
                "coercion",
                ID_MERGED_CODE,
                `id "${merge.text}" merged with "${merge.previousText}" as ${merge.id} under ids: "number"`,
                { line, element: text },
            );
        }
        return id;
    }

    /**
     * Parse the property cells of a row into the scratch arrays.
     * @param slots - the section's property slots
     * @param cells - the row's cells
     * @param quoted - whether each cell was quoted
     * @param line - the row's line
     * @param element - the row's element name for issues
     * @returns true when every cell parsed; false after recording the first error
     */
    private parseProperties(
        slots: readonly PropertySlot[],
        cells: readonly string[],
        quoted: readonly boolean[],
        line: number,
        element: string,
    ): boolean {
        const { values, texts, precisionSlots } = this;
        precisionSlots.length = 0;
        for (const slot of slots) {
            const text = cells[slot.cell];
            texts[slot.cell] = null;
            if (text.length === 0) {
                values[slot.cell] = quoted[slot.cell] ? emptyValue(slot.spec) : undefined;
                continue;
            }
            const { spec } = slot;
            try {
                if (spec.temporal !== null && !spec.list) {
                    const parsed = parseDeclaredTemporal(text, spec);
                    values[slot.cell] = parsed.value;
                    texts[slot.cell] = parsed.text;
                } else {
                    values[slot.cell] = parseDeclaredValue(text, spec, this.options.listSyntax);
                }
            } catch (err) {
                this.report.recordError(err, { line, element: `${element} ${slot.name}` });
                return false;
            }
            if (spec.precision && losesPrecision(spec, text)) {
                precisionSlots.push(slot);
            }
        }
        return true;
    }

    /**
     * Write the parsed property values of a row.
     * @param domain - node or edge
     * @param slots - the section's property slots
     * @param row - the node or edge index
     */
    private writeProperties(domain: "node" | "edge", slots: readonly PropertySlot[], row: number): void {
        const { sink, values, texts } = this;
        for (const slot of slots) {
            const value = values[slot.cell];
            if (value === undefined) {
                continue;
            }
            const text = texts[slot.cell];
            if (text !== null && slot.companion === INVALID_INDEX && slot.companionDecl !== null) {
                slot.companion = declareCompanion(sink, domain, slot.companionDecl);
            }
            if (domain === "node") {
                sink.setNodeValue(slot.handle, row, value);
                if (text !== null) {
                    sink.setNodeValue(slot.companion, row, text);
                }
            } else {
                sink.setEdgeValue(slot.handle, row, value);
                if (text !== null) {
                    sink.setEdgeValue(slot.companion, row, text);
                }
            }
        }
    }

    /**
     * Report the precision losses of the row just accepted.
     * @param line - the row's line
     * @param element - the row's element name
     */
    private reportPrecision(line: number, element: string): void {
        for (const slot of this.precisionSlots) {
            this.report.warning(
                "precision",
                PRECISION_CODE,
                `long value of "${slot.name}" is beyond 2^53 and was stored as the nearest f64`,
                { line, element },
            );
        }
        this.precisionSlots.length = 0;
    }

    /**
     * The labels of a node row: every `:LABEL` cell split by the array delimiter (empty items
     * dropped), plus the header's `{label:...}` options.
     * @param section - the section
     * @param cells - the row's cells
     * @param quoted - whether each cell was quoted
     * @returns the labels, or undefined when every label cell is unset and no extra label exists
     */
    private labelsOf(section: NodeSection, cells: readonly string[], quoted: readonly boolean[]): string[] | undefined {
        let labels: string[] | undefined;
        for (const cell of section.labelCells) {
            const text = cells[cell];
            if (text.length === 0 && !quoted[cell]) {
                continue;
            }
            if (labels === undefined) {
                labels = [];
            }
            for (const item of splitListText(text, this.options.listSyntax)) {
                if (item.length > 0) {
                    labels.push(item);
                }
            }
        }
        if (section.extraLabels.length > 0) {
            if (labels === undefined) {
                labels = [];
            }
            labels.push(...section.extraLabels);
        }
        return labels;
    }
}

/**
 * The value of a quoted empty cell: an empty string for a string-kind property, an empty list for
 * a list property, unset for anything else (neo4j-admin's default `--ignore-empty-strings=false`).
 * @param spec - the property's declared type
 * @returns the value, or undefined for unset
 */
function emptyValue(spec: DeclaredTypeSpec): unknown {
    if (spec.list) {
        return [];
    }
    if (spec.kind === "string" || spec.kind === "duration") {
        return "";
    }
    return undefined;
}

/**
 * Check that no two PROPERTY (or stored-id) fields of one header share a name.
 * @param fields - the parsed header
 */
function checkPropertyNames(fields: readonly HeaderField[]): void {
    const seen = new Set<string>();
    for (const field of fields) {
        if (field.name.length === 0 || field.kind === "IGNORE") {
            continue;
        }
        if (seen.has(field.name)) {
            throw headerError(`property "${field.name}" is declared twice`);
        }
        seen.add(field.name);
    }
}

/**
 * The error of a malformed section header.
 * @param reason - why
 * @returns the error
 */
function headerError(reason: string): GraphFormatError {
    return new GraphFormatError("E_UNSUPPORTED", reason, { reason: "header" });
}

/**
 * Confidence that a head of bytes is a neo4j-admin CSV: the first line holds an `:ID`,
 * `:START_ID` or `:END_ID` header cell.
 * @param head - the first bytes of the input
 * @returns 0.95 for a Neo4j header, 0 otherwise
 */
function sniffNeo4j(head: Uint8Array): number {
    const text = new TextDecoder("utf-8", { fatal: false, ignoreBOM: false }).decode(head);
    const end = text.search(/[\r\n]/);
    const first = end < 0 ? text : text.slice(0, end);
    const cells = first.split(/[,;|\t]/);
    return isHeaderRecord(cells, cells.length) ? 0.95 : 0;
}

/** The Neo4j importer. */
export const neo4jImporter: GraphImporter<Neo4jImportOptions> = Object.freeze({
    format: NEO4J,
    extensions: Object.freeze([".csv", ".tsv"]),
    mimeTypes: Object.freeze(["text/csv", "text/tab-separated-values"]),
    sniff: sniffNeo4j,
    /**
     * Read one or more neo4j-admin CSV inputs into the sink.
     * @param input - the primary input (a node table, a relationship table, or sections of both)
     * @param sink - the sink
     * @param options - format-specific and common options
     * @returns the report; ImportError beyond the error limit or on a malformed header
     */
    async import(
        input: ImportInput,
        sink: GraphSink,
        options?: Neo4jImportOptions & CommonImportOptions,
    ): Promise<ImportReport> {
        const common = resolveImportOptions(options, { ids: "canonical", defaultDirected: true, weightFrom: "weight" });
        const format = resolveNeo4jOptions(options);
        const report = new ImportReportBuilder(NEO4J, common.errorLimit);
        reportSinkOptions(sink, options, report);
        // nodeIdFrom and defaultDirected are among the reported options: Neo4j ids are the :ID
        // column and every relationship is directed
        reportUnusedOptions(options, report, USED_OPTIONS);
        const session = new Neo4jImportSession(sink, report, common, format);
        await session.run([input, ...format.nodes, ...format.relationships]);
        throwIfAborted(common.signal);
        return report.finish();
    },
});
