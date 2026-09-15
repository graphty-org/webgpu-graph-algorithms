/**
 * The Neo4j exporter (design section 8.5, research note 07 section 9): writes a snapshot as
 * neo4j-admin import CSV -- node sections with `:ID`, `:LABEL` and typed property columns,
 * relationship sections with `:START_ID`, `:END_ID`, `:TYPE`, the weight and typed property
 * columns -- as one document (node sections first) or as the node or relationship part alone.
 *
 * What a property graph cannot carry is reported by check() before anything is written: an
 * undirected snapshot (every edge becomes a directed relationship), expanded mixed direction (per
 * `onMixedDirection`), edge ids, hierarchy, element lifetimes, graph attributes, nested json (points
 * excepted), multi-component columns (flattened to arrays), u32 / u8 columns (written as `long` /
 * `int`), ids whose text would re-import as another type or collide, and list items that contain
 * the array delimiter.
 *
 * Column headers restore the declared Neo4j type from `origin.type` when it is compatible with the
 * column's dtype (`int`, `byte`, `short`, `long`, `char`, `duration`, the temporal types, `point`,
 * `type[]`), and derive it from the dtype otherwise. Temporal columns write their `.text`
 * companion when set and the canonical ISO form otherwise. Nodes are grouped into sections by
 * (id space, stored-id column) in index order, so a re-import restores the same node order;
 * relationships likewise by (start space, end space).
 */

import {
    type Column,
    type ColumnMeta,
    type Dtype,
    GraphFormatError,
    type GraphSnapshot,
    type NodeId,
    type ScalarDtype,
} from "@graphty/graph-format";

import { ID_TEXT_COLLISION_CODE, ID_TEXT_TYPE_CODE } from "../../common/codes.js";
import { type DeclaredTypeSpec, mapDeclaredType } from "../../common/declared-types.js";
import { type PairFolding, pairFolding } from "../../common/direction.js";
import { quoteCsvCell } from "../../common/escape.js";
import { capabilities, checkCapabilities, LOSS } from "../../common/export.js";
import { formatF32, formatF64, formatInteger } from "../../common/format.js";
import { isCanonicalIntegerText } from "../../common/ids.js";
import { type ResolvedExportOptions, resolveExportOptions } from "../../common/options.js";
import { formatTemporal, type TemporalKind } from "../../common/temporal.js";
import { type ExplicitWeights, explicitWeights } from "../../common/weights.js";
import { encodeChunks, joinText } from "../../common/writer.js";
import { type CommonExportOptions, type ExportCapabilities, type GraphExporter, type LossNote } from "../../types.js";
import { checkRecordSyntax, type RecordSyntax } from "../csv/records.js";
import { formatHeaderField } from "./header.js";
import { ID_SPACE_COLUMN, LABELS_COLUMN, TYPE_COLUMN } from "./importer.js";

/** The format-specific options of the Neo4j exporter. */
export interface Neo4jExportOptions {
    /** Which tables to write: both (node sections first; default), the node sections or the relationship sections. */
    part?: "all" | "nodes" | "relationships" | undefined;
    /** The field delimiter; one character; "," by default. */
    delimiter?: string | undefined;
    /** The array delimiter of list values and `:LABEL` cells; ";" by default. */
    arrayDelimiter?: ";" | "," | "|" | undefined;
    /** The quote character; one character; a double quote by default. */
    quote?: string | undefined;
    /**
     * The property that receives explicit edge weights (`<name>:double`); "weight" by default (the
     * importer's `weightFrom` default); null writes no weights.
     */
    weightColumn?: string | null | undefined;
    /**
     * The property name of the `:ID` column for nodes that have no stored-id column of their own
     * (`<name>:ID`); null (default) writes a bare `:ID`.
     */
    idColumn?: string | null | undefined;
}

/** Loss code: undirected edges (an undirected snapshot, or the folded pairs of a mixed one) written as directed relationships. */
export const UNDIRECTED_LOSS = "W_NEO4J_UNDIRECTED_AS_DIRECTED";

/** Loss code: node ids whose text re-imports as another type under the canonical id rule. */
export const ID_TEXT_TYPE_LOSS = ID_TEXT_TYPE_CODE;

/** Loss code: two node ids share one text; export() throws E_INVALID_ID. */
export const ID_TEXT_COLLISION_LOSS = ID_TEXT_COLLISION_CODE;

/** Loss code: an edge property column already uses the weight column name; export() throws E_COLUMN_EXISTS. */
export const WEIGHT_COLUMN_TAKEN_LOSS = "E_NEO4J_WEIGHT_COLUMN_TAKEN";

/** Loss code: a node property column already uses the idColumn name; export() throws E_COLUMN_EXISTS. */
export const ID_COLUMN_TAKEN_LOSS = "E_NEO4J_ID_COLUMN_TAKEN";

/** Loss code: a node has more than one stored-id column set; only the first is written. */
export const MULTIPLE_ID_PROPERTIES_LOSS = "W_NEO4J_MULTIPLE_ID_PROPERTIES";

/** Loss code: a declared integer type holds non-integral values and is written as double. */
export const DECLARED_TYPE_CHANGED_LOSS = "W_NEO4J_DECLARED_TYPE_CHANGED";

/** Loss code: a list item contains the array delimiter, which Neo4j cannot escape. */
export const ARRAY_DELIMITER_LOSS = "W_NEO4J_ARRAY_DELIMITER";

/** The roles Neo4j has a slot for: `:TYPE`, `:LABEL` and the id space of `:ID(Space)`. */
const SLOT_ROLES: ReadonlySet<string> = new Set(["kind", "labels", "idSpace"]);

const NEO4J = "neo4j";
const ID_TYPE = "ID";

/** The weight property the importer reads by default (its weightFrom default). */
const DEFAULT_WEIGHT_COLUMN = "weight";

/** The names the importer gives the slot columns, for the name-change notes. */
const ROLE_NAMES: Readonly<Record<string, string>> = Object.freeze({
    kind: TYPE_COLUMN,
    labels: LABELS_COLUMN,
    idSpace: ID_SPACE_COLUMN,
});

/**
 * The capabilities of neo4j-admin CSV (research note 07 section 9): declared scalar types and
 * arrays; a dict column reads back as string (the header has no enumeration type); a position or
 * visual column is written as a plain property (a point for a 2- or 3-component position) and
 * reads back without its role.
 */
export const NEO4J_CAPABILITIES: ExportCapabilities = capabilities({
    mixedDirection: false,
    multiEdges: true,
    selfLoops: true,
    edgeIds: "none",
    idCharset: "any",
    dtypes: ["f32", "f64", "i32", "bool", "string"],
    components: false,
    lists: true,
    json: false,
    defaults: false,
    options: false,
    hierarchy: false,
    temporal: "none",
    graphAttributes: false,
    positions: false,
    viz: false,
});

/** Roles whose columns are never written as properties. */
const SKIPPED_ROLES: ReadonlySet<string> = new Set([
    "directed",
    "pair",
    "mutual",
    "weight",
    "timeText",
    "originalId",
    "parent",
    "parents",
    "start",
    "end",
    "timestamp",
    "timestamps",
    "spells",
    "open",
]);

/** Notes about relationships only, dropped when only nodes are written. */
const EDGE_NOTE_CODES: ReadonlySet<string> = new Set([
    LOSS.MIXED_DIRECTION,
    LOSS.MIXED_DIRECTION_ERROR,
    LOSS.MULTI_EDGES,
    LOSS.SELF_LOOPS,
    LOSS.EDGE_IDS_GENERATED,
    LOSS.EDGE_IDS_DROPPED,
    UNDIRECTED_LOSS,
    WEIGHT_COLUMN_TAKEN_LOSS,
]);

/** Notes about nodes only, dropped when only relationships are written. */
const NODE_NOTE_CODES: ReadonlySet<string> = new Set([
    LOSS.ID_MANGLED,
    LOSS.ID_CHARSET,
    LOSS.ID_RENUMBERED,
    ID_TEXT_TYPE_LOSS,
    ID_COLUMN_TAKEN_LOSS,
    MULTIPLE_ID_PROPERTIES_LOSS,
]);

const ARRAY_SYNTAXES: ReadonlySet<string> = new Set([";", ",", "|"]);
const PARTS: ReadonlySet<string> = new Set(["all", "nodes", "relationships"]);

/** The resolved format options. */
interface ResolvedNeo4jExportOptions {
    readonly part: "all" | "nodes" | "relationships";
    readonly syntax: RecordSyntax & { readonly delimiter: string };
    readonly arrayDelimiter: string;
    readonly weightColumn: string | null;
    readonly idColumn: string | null;
}

/** How one cell's value is written. */
type CellKind = "integer" | "float" | "double" | "boolean" | "string" | "temporal" | "point" | "json";

/** The Neo4j type and cell kind of one scalar (a column or a list item). */
interface ScalarPlan {
    readonly type: string;
    readonly kind: CellKind;
    readonly temporal: TemporalKind | null;
}

/** One property column as it will be written. */
interface ColumnPlan {
    readonly column: Column;
    readonly header: string;
    readonly scalar: ScalarPlan;
    /** A list (or flattened vector) column: items are joined by the array delimiter. */
    readonly list: boolean;
    /** The companion text column of a temporal column, or null. */
    readonly companion: Column | null;
}

/**
 * Resolve the format options.
 * @param options - the caller's options
 * @returns the resolved options; E_UNSUPPORTED for an invalid value
 */
function resolveNeo4jExportOptions(options: Neo4jExportOptions | undefined): ResolvedNeo4jExportOptions {
    const o: Neo4jExportOptions = options ?? {};
    const part = o.part ?? "all";
    if (!PARTS.has(part)) {
        throw new GraphFormatError(
            "E_UNSUPPORTED",
            `option part: ${JSON.stringify(part)} is not "all", "nodes" or "relationships"`,
            {
                option: "part",
                found: part,
            },
        );
    }
    const arrayDelimiter = o.arrayDelimiter ?? ";";
    if (!ARRAY_SYNTAXES.has(arrayDelimiter)) {
        throw new GraphFormatError(
            "E_UNSUPPORTED",
            `option arrayDelimiter: ${JSON.stringify(arrayDelimiter)} is not one of ";", ",", "|"`,
            { option: "arrayDelimiter", found: arrayDelimiter },
        );
    }
    const delimiter = o.delimiter ?? ",";
    const syntax = { ...checkRecordSyntax({ delimiter, quote: o.quote ?? '"' }), delimiter };
    if (syntax.delimiter === arrayDelimiter) {
        throw new GraphFormatError("E_UNSUPPORTED", "options delimiter and arrayDelimiter must differ", {
            option: "arrayDelimiter",
            found: arrayDelimiter,
        });
    }
    const weightColumn = o.weightColumn === undefined ? "weight" : o.weightColumn;
    if (weightColumn !== null && (typeof weightColumn !== "string" || weightColumn.length === 0)) {
        throw new GraphFormatError("E_UNSUPPORTED", "option weightColumn: expected a non-empty name or null", {
            option: "weightColumn",
            found: weightColumn,
        });
    }
    const idColumn = o.idColumn ?? null;
    if (idColumn !== null && (typeof idColumn !== "string" || idColumn.length === 0)) {
        throw new GraphFormatError("E_UNSUPPORTED", "option idColumn: expected a non-empty name or null", {
            option: "idColumn",
            found: idColumn,
        });
    }
    return { part, syntax, arrayDelimiter, weightColumn, idColumn };
}

/**
 * Everything check() and export() share: the column plans, the section keys, the notes and the
 * error export() throws (if any), computed once from the snapshot and the options.
 */
class ExportPlan {
    readonly snapshot: GraphSnapshot;

    readonly options: ResolvedNeo4jExportOptions;

    readonly common: ResolvedExportOptions;

    /** The notes: the generic capability notes first, then the format's own, in that order. */
    readonly notes: LossNote[] = [];

    /** The format's own notes, appended after the generic ones. */
    private readonly ownNotes: LossNote[] = [];

    /** The error export() throws because of the node sections, or null. */
    private fatalNodes: GraphFormatError | null = null;

    /** The error export() throws because of the relationship sections, or null. */
    private fatalEdges: GraphFormatError | null = null;

    /** The error export() throws whatever the part (an id text collision), or null. */
    private fatalAll: GraphFormatError | null = null;

    readonly nodeColumns: ColumnPlan[] = [];

    readonly edgeColumns: ColumnPlan[] = [];

    /** Stored-id node columns (`name:ID`), in declaration order. */
    readonly idColumns: Column[] = [];

    readonly idSpace: Column | null;

    readonly labels: Column | null;

    readonly kind: Column | null;

    readonly weights: ExplicitWeights | null;

    /** The pair folding (design section 3.6): the mirror halves of undirected pairs are not written. */
    readonly folding: PairFolding;

    /**
     * Build the plan.
     * @param snapshot - the snapshot
     * @param options - the resolved format options
     * @param common - the resolved common options
     */
    constructor(snapshot: GraphSnapshot, options: ResolvedNeo4jExportOptions, common: ResolvedExportOptions) {
        this.snapshot = snapshot;
        this.options = options;
        this.common = common;
        this.idSpace = snapshot.nodes.byRole("idSpace");
        this.labels = snapshot.nodes.byRole("labels");
        this.kind = snapshot.edges.byRole("kind");
        for (const column of snapshot.nodes) {
            if (isStoredId(column.meta)) {
                this.idColumns.push(column);
            }
        }
        const generic = checkCapabilities(snapshot, NEO4J_CAPABILITIES, common, {
            roles: SLOT_ROLES,
            roleNames: ROLE_NAMES,
            temporalText: true,
        });
        this.planColumns("node", snapshot.nodes, this.nodeColumns);
        this.planColumns("edge", snapshot.edges, this.edgeColumns);
        this.weights = this.planWeightSource();
        this.folding = pairFolding(snapshot);
        this.checkIds();
        this.checkIdColumn();
        this.checkStoredIds();
        this.directionNotes();
        const slots = new Set([this.kind, this.labels, this.idSpace].flatMap((c) => (c === null ? [] : [c.meta.name])));
        for (const gen of generic) {
            if (gen.code === LOSS.JSON && gen.column !== null && this.writtenAsPoint(gen.column)) {
                continue;
            }
            if (gen.code === LOSS.DTYPE && gen.column !== null && slots.has(gen.column)) {
                // :TYPE, :LABEL and :ID(Space) read back as the dict / list-of-dict columns they were
                continue;
            }
            if (gen.code === LOSS.MIXED_DIRECTION_ERROR && this.fatalEdges === null) {
                this.fatalEdges = new GraphFormatError("E_DIRECTED", gen.message, { reason: "mixed direction" });
            }
            if ((gen.code === LOSS.POSITIONS || gen.code === LOSS.VIZ) && gen.column !== null) {
                // written as a plain property (a point for a position); the role is what is lost
                this.notes.push(
                    Object.freeze({
                        code: gen.code,
                        message: `node column "${gen.column}" is written as a plain property; its role is lost on re-import`,
                        column: gen.column,
                        count: gen.count,
                    }),
                );
                continue;
            }
            this.notes.push(gen);
        }
        this.notes.push(...this.ownNotes);
        this.filterNotesByPart();
    }

    /**
     * The direction notes: an undirected snapshot writes every edge as a directed relationship;
     * a mixed snapshot folds its undirected pairs to one directed relationship each under
     * "directed" or "undirected" (Neo4j has no undirected relationship, so both policies write
     * the same file and the generic W_MIXED_DIRECTION note names the policy); a mutual pair is
     * written as two relationships without its mark.
     */
    private directionNotes(): void {
        const { snapshot, folding } = this;
        if (!snapshot.directed) {
            this.note(
                UNDIRECTED_LOSS,
                `the snapshot is undirected; every edge is written as a directed relationship (${snapshot.edgeCount} edge(s))`,
                null,
                snapshot.edgeCount,
            );
        } else if (this.common.onMixedDirection !== "error") {
            let undirected = 0;
            for (let e = 0; e < snapshot.edgeCount; e++) {
                if (!folding.folded(e) && !folding.sourceDirected(e)) {
                    undirected++;
                }
            }
            if (undirected > 0) {
                this.note(
                    UNDIRECTED_LOSS,
                    `${undirected} undirected edge(s) are written as one directed relationship each (source to target, a pair folded to its primary); Neo4j has no undirected relationship`,
                    null,
                    undirected,
                );
            }
        }
        if (folding.mutualCount > 0) {
            this.note(
                LOSS.MUTUAL_EXPANDED,
                `${folding.mutualCount} mutual pair(s) are written as two directed relationships; the mutual mark is lost`,
                null,
                folding.mutualCount,
            );
        }
    }

    /**
     * Record a note.
     * @param code - the code
     * @param message - the message
     * @param column - the column, or null
     * @param count - the count, or null
     */
    private note(code: string, message: string, column: string | null = null, count: number | null = null): void {
        this.ownNotes.push(Object.freeze({ code, message, column, count }));
    }

    /**
     * Plan the property columns of one table.
     * @param domain - node or edge
     * @param table - the table
     * @param out - receives the plans in declaration order
     */
    private planColumns(domain: "node" | "edge", table: Iterable<Column>, out: ColumnPlan[]): void {
        const all = [...table];
        const names = new Set(all.map((column) => column.meta.name));
        const companionFor = new Map<string, Column>();
        const companions = new Set<Column>();
        for (const column of all) {
            const target = column.meta.extra.for;
            if (column.dtype === "string" && typeof target === "string" && names.has(target)) {
                companionFor.set(target, column);
                companions.add(column);
            }
        }
        for (const column of all) {
            const { meta } = column;
            const { role } = meta;
            if (companions.has(column) || (role !== null && SKIPPED_ROLES.has(role))) {
                continue;
            }
            if (domain === "node" && (isStoredId(meta) || column === this.idSpace || column === this.labels)) {
                continue;
            }
            if (domain === "edge" && (role === "id" || column === this.kind)) {
                continue;
            }
            out.push(this.planColumn(column, companionFor.get(meta.name) ?? null));
        }
    }

    /**
     * Plan one property column: its header type, cell kind and notes.
     * @param column - the column
     * @param companion - its text companion, or null
     * @returns the plan
     */
    private planColumn(column: Column, companion: Column | null): ColumnPlan {
        const { meta } = column;
        const declared = meta.origin?.type ?? null;
        const mapped = declared === null ? null : mapDeclaredType(NEO4J, declared, "f64");
        const label = `${meta.domain} column "${meta.name}"`;
        if (meta.dtype === "list") {
            const itemDtype = meta.itemDtype ?? "string";
            const compatible =
                mapped !== null && mapped.list && mapped.itemDtype === itemDtype && mapped.kind !== "long";
            let scalar = compatible
                ? { type: declaredScalar(declared as string), kind: kindOf(mapped), temporal: mapped.temporal }
                : defaultScalar(itemDtype);
            if (mapped !== null && mapped.list && mapped.kind === "long" && itemDtype === "f64") {
                scalar = this.integralOrDouble(column, declaredScalar(declared as string), label);
            }
            const itemComponents = meta.itemComponents ?? 1;
            if (itemComponents > 1) {
                this.note(
                    LOSS.COMPONENTS,
                    `${label} holds items of ${itemComponents} components; they are flattened into one array`,
                    meta.name,
                    column.length - column.nullCount,
                );
            }
            this.checkArrayItems(column, scalar, label);
            return {
                column,
                header: formatHeaderField(meta.name, `${scalar.type}[]`, null),
                scalar,
                list: true,
                companion,
            };
        }
        let scalar: ScalarPlan;
        const compatible = mapped !== null && !mapped.list && mapped.dtype === meta.dtype;
        if (compatible && mapped.kind === "long") {
            scalar = this.integralOrDouble(column, declared as string, label);
        } else if (compatible) {
            scalar = { type: declared as string, kind: kindOf(mapped), temporal: mapped.temporal };
        } else {
            scalar = defaultScalar(meta.dtype);
        }
        // the generic check treats position and visual columns by role only; their dtype and
        // stride are checked here so the notes match those of any other column
        const visual = meta.role === "position" || (meta.role !== null && isVizRole(meta.role));
        if (visual && meta.dtype === "json" && scalar.kind !== "point") {
            this.note(
                LOSS.JSON,
                `${label} holds nested values; the format has no nested values`,
                meta.name,
                column.length - column.nullCount,
            );
        } else if (visual && meta.dtype !== "json" && !NEO4J_CAPABILITIES.dtypes.includes(meta.dtype)) {
            this.note(
                LOSS.DTYPE,
                `${label} is ${meta.dtype}; the format cannot keep that dtype`,
                meta.name,
                column.length - column.nullCount,
            );
        }
        if (meta.components > 1) {
            if (visual) {
                this.note(
                    LOSS.COMPONENTS,
                    `${label} has ${meta.components} components; the format has no strides`,
                    meta.name,
                    column.length - column.nullCount,
                );
            }
            return {
                column,
                header: formatHeaderField(meta.name, `${scalar.type}[]`, null),
                scalar,
                list: true,
                companion,
            };
        }
        // an untyped Neo4j header (`name` alone) is a string property; restore it as written
        const untyped = declared === null && meta.origin?.format === NEO4J && meta.dtype === "string";
        const header = untyped ? meta.name : formatHeaderField(meta.name, scalar.type, null);
        return { column, header, scalar, list: false, companion };
    }

    /**
     * The plan of an f64 column declared with an integer type: kept when every set value is
     * integral, written as double (with a note) otherwise.
     * @param column - the column
     * @param declared - the declared type text
     * @param label - the column label for messages
     * @returns the scalar plan
     */
    private integralOrDouble(column: Column, declared: string, label: string): ScalarPlan {
        let integral = true;
        if (column.dtype === "f64") {
            for (let r = 0; r < column.length && integral; r++) {
                if (column.isSet(r) && !Number.isInteger(column.data[r])) {
                    integral = false;
                }
            }
        } else if (column.dtype === "list" && column.child.dtype === "f64") {
            const { data } = column.child;
            for (let i = 0; i < data.length && integral; i++) {
                if (!Number.isInteger(data[i])) {
                    integral = false;
                }
            }
        }
        if (integral) {
            return { type: declared, kind: "integer", temporal: null };
        }
        this.note(
            DECLARED_TYPE_CHANGED_LOSS,
            `${label} is declared ${declared} but holds non-integral values; written as double`,
            column.meta.name,
            null,
        );
        return { type: "double", kind: "double", temporal: null };
    }

    /**
     * Count the rows of a list column with an item containing the array delimiter.
     * @param column - the list column
     * @param scalar - the item plan
     * @param label - the column label
     */
    private checkArrayItems(column: Column, scalar: ScalarPlan, label: string): void {
        if (
            column.dtype !== "list" ||
            (scalar.kind !== "string" && scalar.kind !== "json" && scalar.kind !== "point")
        ) {
            return;
        }
        const { arrayDelimiter } = this.options;
        let rows = 0;
        for (let r = 0; r < column.length; r++) {
            if (!column.isSet(r)) {
                continue;
            }
            for (const item of column.sliceOf(r)) {
                if (formatScalar(item, scalar, null).includes(arrayDelimiter)) {
                    rows++;
                    break;
                }
            }
        }
        if (rows > 0) {
            this.note(
                ARRAY_DELIMITER_LOSS,
                `${label}: ${rows} row(s) hold an item containing the array delimiter "${arrayDelimiter}", which Neo4j cannot escape`,
                column.meta.name,
                rows,
            );
        }
    }

    /**
     * The explicit weights (design section 3.7), and the weight-column name check.
     * @returns the weights, or null when no weights are written
     */
    private planWeightSource(): ExplicitWeights | null {
        const { snapshot } = this;
        const name = this.options.weightColumn;
        if (name === null) {
            return null;
        }
        const weights = explicitWeights(snapshot);
        const taken = this.edgeColumns.find((plan) => plan.column.meta.name === name);
        if (taken !== undefined && weights.weighted) {
            this.note(
                WEIGHT_COLUMN_TAKEN_LOSS,
                `edge column "${name}" already exists; explicit weights cannot be written under weightColumn "${name}"`,
                name,
                null,
            );
            if (this.fatalEdges === null) {
                this.fatalEdges = new GraphFormatError(
                    "E_COLUMN_EXISTS",
                    `edge column "${name}" already exists; choose another weightColumn`,
                    { column: name, domain: "edge" },
                );
            }
        } else if (taken !== undefined && name === DEFAULT_WEIGHT_COLUMN) {
            this.note(
                LOSS.WEIGHT_KEY_CLASH,
                `edge column "${name}" is written under the property the importer reads THE weight from (weightFrom "${name}"); it reads back as the weight, not as a column`,
                name,
                taken.column.length - taken.column.nullCount,
            );
        }
        return weights.weighted ? weights : null;
    }

    /** Check the node ids' text forms: type changes under the canonical rule and collisions. */
    private checkIds(): void {
        const { ids } = this.snapshot;
        let typeChanges = 0;
        let collisions = 0;
        switch (ids.kind) {
            case "identity":
            case "dense":
                break;
            case "numeric":
                for (let i = 0; i < ids.size; i++) {
                    if (!Number.isSafeInteger(ids.idOf(i))) {
                        typeChanges++;
                    }
                }
                break;
            case "string":
                for (let i = 0; i < ids.size; i++) {
                    if (isCanonicalIntegerText(String(ids.idOf(i)))) {
                        typeChanges++;
                    }
                }
                break;
            case "mixed": {
                const seen = new Set<string>();
                for (let i = 0; i < ids.size; i++) {
                    const id = ids.idOf(i);
                    const text = String(id);
                    if (typeof id === "number" ? !Number.isSafeInteger(id) : isCanonicalIntegerText(text)) {
                        typeChanges++;
                    }
                    if (seen.has(text)) {
                        collisions++;
                    } else {
                        seen.add(text);
                    }
                }
                break;
            }
            default: {
                const name: string = ids.kind;
                throw new GraphFormatError("E_UNSUPPORTED", `unknown id map kind ${name}`, { kind: name });
            }
        }
        if (typeChanges > 0) {
            this.note(
                ID_TEXT_TYPE_LOSS,
                `${typeChanges} node id(s) re-import as another type under the canonical id rule (a string of canonical integer text, or a non-integer number)`,
                null,
                typeChanges,
            );
        }
        if (collisions > 0) {
            this.note(
                ID_TEXT_COLLISION_LOSS,
                `${collisions} node id(s) share their text with another id (a number and a string); export() will throw`,
                null,
                collisions,
            );
            this.fatalAll = new GraphFormatError(
                "E_INVALID_ID",
                `${collisions} node id(s) share their text with another id; Neo4j ids are text`,
                { reason: "text collision", count: collisions },
            );
        }
    }

    /** Check that the idColumn option does not name an existing node property. */
    private checkIdColumn(): void {
        const name = this.options.idColumn;
        if (name === null) {
            return;
        }
        const taken =
            this.nodeColumns.some((plan) => plan.column.meta.name === name) ||
            this.idColumns.some((column) => column.meta.name === name);
        if (!taken) {
            return;
        }
        this.note(
            ID_COLUMN_TAKEN_LOSS,
            `node column "${name}" already exists; the :ID column cannot be named idColumn "${name}"`,
            name,
            null,
        );
        if (this.fatalNodes === null) {
            this.fatalNodes = new GraphFormatError(
                "E_COLUMN_EXISTS",
                `node column "${name}" already exists; choose another idColumn`,
                { column: name, domain: "node" },
            );
        }
    }

    /** Count the nodes with more than one stored-id column set. */
    private checkStoredIds(): void {
        if (this.idColumns.length < 2) {
            return;
        }
        let count = 0;
        for (let i = 0; i < this.snapshot.nodeCount; i++) {
            let set = 0;
            for (const column of this.idColumns) {
                if (column.isSet(i)) {
                    set++;
                }
            }
            if (set > 1) {
                count++;
            }
        }
        if (count > 0) {
            this.note(
                MULTIPLE_ID_PROPERTIES_LOSS,
                `${count} node(s) have more than one stored-id column set; only the first in declaration order is written`,
                null,
                count,
            );
        }
    }

    /**
     * Whether a json column is written as Neo4j points (so the generic nested-value note does not apply).
     * @param name - the column name
     * @returns true when its plan writes point literals
     */
    private writtenAsPoint(name: string): boolean {
        for (const plan of [...this.nodeColumns, ...this.edgeColumns]) {
            if (plan.column.meta.name === name && plan.scalar.kind === "point") {
                return true;
            }
        }
        return false;
    }

    /** Drop the notes about the part that is not written. */
    private filterNotesByPart(): void {
        const { part } = this.options;
        if (part === "all") {
            return;
        }
        const { snapshot } = this;
        const keep = this.notes.filter((n) => {
            if (part === "nodes") {
                if (EDGE_NOTE_CODES.has(n.code)) {
                    return false;
                }
                return n.column === null || !snapshot.edges.has(n.column) || snapshot.nodes.has(n.column);
            }
            if (NODE_NOTE_CODES.has(n.code)) {
                return false;
            }
            return n.column === null || !snapshot.nodes.has(n.column) || snapshot.edges.has(n.column);
        });
        this.notes.length = 0;
        this.notes.push(...keep);
    }

    /**
     * The error export() throws for the requested part, or null when the export can proceed.
     * @returns the error
     */
    get fatal(): GraphFormatError | null {
        if (this.fatalAll !== null) {
            return this.fatalAll;
        }
        const { part } = this.options;
        if (part !== "relationships" && this.fatalNodes !== null) {
            return this.fatalNodes;
        }
        return part === "nodes" ? null : this.fatalEdges;
    }

    /**
     * The id space of a node.
     * @param index - the node index
     * @returns the space name, or null
     */
    spaceOf(index: number): string | null {
        const { idSpace } = this;
        if (idSpace === null || !idSpace.isSet(index)) {
            return null;
        }
        const value = idSpace.value(index);
        return typeof value === "string" ? value : String(value);
    }

    /**
     * The stored-id column of a node: the first one set in declaration order.
     * @param index - the node index
     * @returns the column, or null
     */
    storedIdOf(index: number): Column | null {
        for (const column of this.idColumns) {
            if (column.isSet(index)) {
                return column;
            }
        }
        return null;
    }

    /**
     * The node sections, then the relationship sections, as CSV lines.
     * @yields one line at a time (with its line break)
     * @returns nothing
     */
    *lines(): Generator<string, void, undefined> {
        if (this.fatal !== null) {
            throw this.fatal;
        }
        const { part } = this.options;
        if (part !== "relationships") {
            yield* this.nodeLines();
        }
        if (part !== "nodes") {
            yield* this.relationshipLines();
        }
    }

    /**
     * The node sections.
     * @yields one line at a time
     * @returns nothing
     */
    private *nodeLines(): Generator<string, void, undefined> {
        const { snapshot, labels, nodeColumns } = this;
        const { ids } = snapshot;
        const { delimiter } = this.options.syntax;
        const propertyHeaders = nodeColumns.map((plan) => plan.header).join(delimiter);
        let sections = 0;
        let sectionSpace: string | null = null;
        let sectionIdName: string | null = null;
        const cells: string[] = [];
        for (let i = 0; i < snapshot.nodeCount; i++) {
            const space = this.spaceOf(i);
            const stored = this.storedIdOf(i);
            const idName = stored === null ? (this.options.idColumn ?? "") : stored.meta.name;
            if (sections === 0 || space !== sectionSpace || idName !== sectionIdName) {
                sections++;
                sectionSpace = space;
                sectionIdName = idName;
                const header = [formatHeaderField(idName, ID_TYPE, space)];
                if (labels !== null) {
                    header.push(":LABEL");
                }
                if (propertyHeaders.length > 0) {
                    header.push(propertyHeaders);
                }
                yield `${header.join(delimiter)}\n`;
            }
            cells.length = 0;
            cells.push(this.cell(idText(ids.idOf(i))));
            if (labels !== null) {
                cells.push(this.cell(this.labelsText(i)));
            }
            for (const plan of nodeColumns) {
                cells.push(this.cell(this.valueText(plan, i)));
            }
            yield `${cells.join(delimiter)}\n`;
        }
        if (sections === 0) {
            // no nodes: one header so the section (and its columns) still exists
            const header = [formatHeaderField(this.options.idColumn ?? "", ID_TYPE, null)];
            if (labels !== null) {
                header.push(":LABEL");
            }
            if (propertyHeaders.length > 0) {
                header.push(propertyHeaders);
            }
            yield `${header.join(delimiter)}\n`;
        }
    }

    /**
     * The relationship sections.
     * @yields one line at a time
     * @returns nothing
     */
    private *relationshipLines(): Generator<string, void, undefined> {
        const { snapshot, kind, weights, edgeColumns, folding } = this;
        const { ids } = snapshot;
        const { delimiter } = this.options.syntax;
        const list = snapshot.edgeList();
        const propertyHeaders = edgeColumns.map((plan) => plan.header).join(delimiter);
        const weightHeader =
            weights === null ? null : formatHeaderField(this.options.weightColumn ?? "weight", "double", null);
        let sections = 0;
        let sectionStart: string | null = null;
        let sectionEnd: string | null = null;
        const cells: string[] = [];
        const headerOf = (startSpace: string | null, endSpace: string | null): string => {
            const header = [formatHeaderField("", "START_ID", startSpace), formatHeaderField("", "END_ID", endSpace)];
            if (kind !== null) {
                header.push(":TYPE");
            }
            if (weightHeader !== null) {
                header.push(weightHeader);
            }
            if (propertyHeaders.length > 0) {
                header.push(propertyHeaders);
            }
            return `${header.join(delimiter)}\n`;
        };
        for (let e = 0; e < snapshot.edgeCount; e++) {
            if (folding.folded(e)) {
                continue;
            }
            const u = list.src[e];
            const v = list.dst[e];
            const startSpace = this.spaceOf(u);
            const endSpace = this.spaceOf(v);
            if (sections === 0 || startSpace !== sectionStart || endSpace !== sectionEnd) {
                sections++;
                sectionStart = startSpace;
                sectionEnd = endSpace;
                yield headerOf(startSpace, endSpace);
            }
            cells.length = 0;
            cells.push(this.cell(idText(ids.idOf(u))), this.cell(idText(ids.idOf(v))));
            if (kind !== null) {
                cells.push(this.cell(kind.isSet(e) ? textOf(kind, e) : null));
            }
            if (weights !== null) {
                cells.push(this.cell(weights.text(e)));
            }
            for (const plan of edgeColumns) {
                cells.push(this.cell(this.valueText(plan, e)));
            }
            yield `${cells.join(delimiter)}\n`;
        }
        if (sections === 0) {
            yield headerOf(null, null);
        }
    }

    /**
     * A CSV cell: empty for an unset value, a quoted empty string for a set empty string, the
     * quoted text otherwise.
     * @param text - the value text, or null for unset
     * @returns the cell as written
     */
    private cell(text: string | null): string {
        if (text === null) {
            return "";
        }
        const { quote } = this.options.syntax;
        if (text.length === 0) {
            return quote + quote;
        }
        return quoteCsvCell(text, this.options.syntax.delimiter);
    }

    /**
     * The `:LABEL` cell of a node.
     * @param index - the node index
     * @returns the labels joined by the array delimiter, or null when unset
     */
    private labelsText(index: number): string | null {
        const { labels } = this;
        if (labels === null || !labels.isSet(index)) {
            return null;
        }
        if (labels.dtype === "list") {
            return labels
                .sliceOf(index)
                .map((item) => String(item))
                .join(this.options.arrayDelimiter);
        }
        return textOf(labels, index);
    }

    /**
     * The text of one property cell.
     * @param plan - the column plan
     * @param row - the row
     * @returns the text, or null when unset
     */
    private valueText(plan: ColumnPlan, row: number): string | null {
        const { column, scalar, companion } = plan;
        if (!column.isSet(row)) {
            return null;
        }
        if (plan.list) {
            const items = column.dtype === "list" ? column.sliceOf(row) : (column.value(row) as ArrayLike<unknown>);
            const texts: string[] = [];
            for (const item of Array.from(items)) {
                // an item of several components (itemComponents > 1) is flattened into the array
                if (Array.isArray(item) || ArrayBuffer.isView(item)) {
                    for (const component of Array.from(item as ArrayLike<unknown>)) {
                        texts.push(formatScalar(component, scalar, null));
                    }
                } else {
                    texts.push(formatScalar(item, scalar, null));
                }
            }
            return texts.join(this.options.arrayDelimiter);
        }
        const text = companion !== null && companion.isSet(row) ? textOf(companion, row) : null;
        return formatScalar(column.dtype === "json" ? column.values[row] : column.value(row), scalar, text);
    }
}

/**
 * Whether a node column is a stored id (`name:ID` on import).
 * @param meta - the column metadata
 * @returns true for a string column declared by the Neo4j importer as the id property
 */
function isStoredId(meta: ColumnMeta): boolean {
    return (
        meta.role === null &&
        meta.dtype === "string" &&
        meta.origin !== null &&
        meta.origin.format === NEO4J &&
        meta.origin.type === ID_TYPE
    );
}

/**
 * Whether a role is one of the visual roles.
 * @param role - the role
 * @returns true for color, size, shape and thickness
 */
function isVizRole(role: string): boolean {
    return role === "color" || role === "size" || role === "shape" || role === "thickness";
}

/**
 * The declared scalar type text of a list declaration (`string[]` -> `string`).
 * @param declared - the declared type text
 * @returns the text without its `[]` suffix
 */
function declaredScalar(declared: string): string {
    const trimmed = declared.trim();
    return trimmed.endsWith("[]") ? trimmed.slice(0, -2) : trimmed;
}

/**
 * The cell kind of a mapped declared type.
 * @param spec - the mapped type
 * @returns the kind
 */
function kindOf(spec: DeclaredTypeSpec): CellKind {
    switch (spec.kind) {
        case "boolean":
            return "boolean";
        case "integer":
        case "long":
            return "integer";
        case "float":
            return "float";
        case "double":
            return "double";
        case "string":
        case "duration":
            return "string";
        case "temporal":
            return "temporal";
        case "point":
            return "point";
        case "json":
            return "json";
        default: {
            const name: string = spec.kind;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown value kind ${name}`, { kind: name });
        }
    }
}

/**
 * The Neo4j type of a dtype without a compatible declaration.
 * @param dtype - the column or item dtype
 * @returns the type and cell kind
 */
function defaultScalar(dtype: Dtype | ScalarDtype): ScalarPlan {
    switch (dtype) {
        case "f32":
            return { type: "float", kind: "float", temporal: null };
        case "f64":
            return { type: "double", kind: "double", temporal: null };
        case "i32":
        case "u8":
            return { type: "int", kind: "integer", temporal: null };
        case "u32":
            return { type: "long", kind: "integer", temporal: null };
        case "bool":
            return { type: "boolean", kind: "boolean", temporal: null };
        case "dict":
        case "string":
            return { type: "string", kind: "string", temporal: null };
        case "json":
            return { type: "string", kind: "json", temporal: null };
        case "list":
            return { type: "string", kind: "json", temporal: null };
        default: {
            const name: string = dtype;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown dtype ${name}`, { dtype: name });
        }
    }
}

/**
 * Format one scalar value for a cell.
 * @param value - the value
 * @param scalar - the plan
 * @param text - the companion text of a temporal value, or null
 * @returns the text
 */
function formatScalar(value: unknown, scalar: ScalarPlan, text: string | null): string {
    switch (scalar.kind) {
        case "integer":
            return typeof value === "number" ? formatInteger(value) : String(value);
        case "float":
            return typeof value === "number" ? formatF32(value) : String(value);
        case "double":
            return typeof value === "number" ? formatF64(value) : String(value);
        case "boolean":
            return value === true ? "true" : "false";
        case "string":
            return typeof value === "string" ? value : String(value);
        case "temporal":
            if (text !== null) {
                return text;
            }
            return typeof value === "number" && scalar.temporal !== null
                ? formatTemporal(value, scalar.temporal)
                : String(value);
        case "point":
            return formatPoint(value);
        case "json":
            return JSON.stringify(value) ?? "";
        default: {
            const name: string = scalar.kind;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown cell kind ${name}`, { kind: name });
        }
    }
}

/**
 * A Neo4j point literal `{x:1.5, y:2, crs:'cartesian'}` from a point object; any other value is
 * written as JSON text.
 * @param value - the json value
 * @returns the literal
 */
function formatPoint(value: unknown): string {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
        return JSON.stringify(value) ?? "";
    }
    const parts: string[] = [];
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
        if (typeof item === "number") {
            parts.push(`${key}:${formatF64(item)}`);
        } else if (typeof item === "string") {
            parts.push(`${key}:'${item.replace(/'/g, "")}'`);
        } else {
            return JSON.stringify(value) ?? "";
        }
    }
    return `{${parts.join(", ")}}`;
}

/**
 * The text of a node id.
 * @param id - the id
 * @returns String(id)
 */
function idText(id: NodeId): string {
    if (typeof id === "string") {
        return id;
    }
    return Number.isInteger(id) ? formatInteger(id) : String(id);
}

/**
 * The text of a set cell of a dict / string (or any scalar) column.
 * @param column - the column
 * @param row - the row
 * @returns the value as text
 */
function textOf(column: Column, row: number): string {
    switch (column.dtype) {
        case "string":
            return column.valueAt(row);
        case "dict":
            return String(column.value(row));
        case "list":
            return column.sliceOf(row).map(String).join(",");
        case "json":
            return JSON.stringify(column.values[row]) ?? "";
        case "bool":
            return column.value(row) === true ? "true" : "false";
        default: {
            const value = column.value(row);
            return typeof value === "number" ? String(value) : Array.from(value as ArrayLike<number>).join(",");
        }
    }
}

/** The Neo4j exporter. */
export const neo4jExporter: GraphExporter<Neo4jExportOptions> = Object.freeze({
    format: NEO4J,
    capabilities: NEO4J_CAPABILITIES,
    /**
     * Pre-flight: what export() would lose.
     * @param snapshot - the snapshot
     * @param options - format-specific and common options
     * @returns the notes
     */
    check(snapshot: GraphSnapshot, options?: Neo4jExportOptions & CommonExportOptions): readonly LossNote[] {
        return Object.freeze([...plan(snapshot, options).notes]);
    },
    /**
     * Write the snapshot as UTF-8 chunks.
     * @param snapshot - the snapshot
     * @param options - format-specific and common options
     * @returns the chunks
     */
    export(snapshot: GraphSnapshot, options?: Neo4jExportOptions & CommonExportOptions): AsyncIterable<Uint8Array> {
        return encodeChunks(plan(snapshot, options).lines());
    },
    /**
     * Write the snapshot as one string.
     * @param snapshot - the snapshot
     * @param options - format-specific and common options
     * @returns the document
     */
    exportToString(snapshot: GraphSnapshot, options?: Neo4jExportOptions & CommonExportOptions): Promise<string> {
        return joinText(plan(snapshot, options).lines());
    },
});

/**
 * Build the export plan of a snapshot.
 * @param snapshot - the snapshot
 * @param options - the caller's options
 * @returns the plan
 */
function plan(snapshot: GraphSnapshot, options: (Neo4jExportOptions & CommonExportOptions) | undefined): ExportPlan {
    return new ExportPlan(snapshot, resolveNeo4jExportOptions(options), resolveExportOptions(options));
}
