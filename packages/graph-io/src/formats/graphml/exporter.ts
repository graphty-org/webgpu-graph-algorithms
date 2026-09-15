/**
 * The GraphML exporter (design section 8.5; research note 07 section 9): `<key>` declarations
 * regenerated from ColumnMeta.origin when the column came from GraphML (key id, attr.name,
 * attr.type, yfiles.type) and derived from the dtype otherwise; node ids sanitised to NMTOKEN
 * (`sanitizeIds: "error"` refuses, `"mangle"` rewrites and keeps the original in a
 * `graphty:originalId` data attribute the importer restores); mixed direction written with
 * `edgedefault` plus per-edge `directed` attributes, folding expanded pairs back through the
 * `pair` / `directed` role columns; explicit weights only (the role-weight column's validity);
 * containment (`parent` role) as nested graphs; yFiles json columns as nested XML again.
 *
 * check() lists every loss before anything is written: the generic capability gaps of
 * checkCapabilities() (dict / u32 / u8 dtypes, lists, components, options, positions, visual and
 * temporal roles, extension tables) plus the GraphML-specific ones (mutual edges written as
 * undirected, json columns without a yfiles origin written as JSON text, multi-parent columns,
 * containment order, ids that change type under the canonical rule, edge ids that read back as
 * strings, roles GraphML cannot express, yfiles values that are not serialisable trees).
 */

import { type Column, GraphFormatError, type GraphSnapshot, INVALID_INDEX } from "@graphty/graph-format";

import { childrenCsr } from "../../children.js";
import { type PairFolding, pairFolding } from "../../common/direction.js";
import { escapeXmlAttribute, escapeXmlText } from "../../common/escape.js";
import {
    capabilities,
    checkCapabilities,
    isNmtoken,
    LOSS,
    mangleNmtoken,
    type SanitizedIds,
    sanitizeIds,
} from "../../common/export.js";
import { formatF32, formatF64, formatInteger } from "../../common/format.js";
import { isCanonicalIntegerText } from "../../common/ids.js";
import { type ResolvedExportOptions, resolveExportOptions } from "../../common/options.js";
import { type ExplicitWeights, explicitWeights } from "../../common/weights.js";
import { encodeChunks, joinText } from "../../common/writer.js";
import { xmlIllegalTextNotes } from "../../common/xml.js";
import { type CommonExportOptions, type ExportCapabilities, type GraphExporter, type LossNote } from "../../types.js";
import {
    EDGE_ID_COLUMN,
    FORMAT,
    GRAPHML_LOSS,
    GRAPHML_NAMESPACE,
    type GraphmlMeta,
    LABEL_COLUMN,
    META_KEY,
    ORIGINAL_ID_ATTRIBUTE,
    PARENT_COLUMN,
    RESERVED_EDGE_NAMES,
    RESERVED_NODE_NAMES,
    SCHEMA_LOCATION,
    SOURCE_PORT_COLUMN,
    TARGET_PORT_COLUMN,
    XSI_NAMESPACE,
    YFILES_NAMESPACE,
} from "./constants.js";
import { treeProblem, writeXmlTree } from "./tree.js";

/** The format-specific options of the GraphML exporter. */
export interface GraphmlExportOptions {
    /** Indent nested elements (default true); false writes one element per line without indentation. */
    pretty?: boolean | undefined;
    /**
     * The top-level `edgedefault`. By default the one the importer recorded in
     * `meta.extra.graphml` (so a mixed file re-imports with the same edge layout), else the
     * snapshot's direction, with the majority direction for a mixed snapshot.
     */
    edgedefault?: "directed" | "undirected" | undefined;
}

/** What GraphML keeps as declared (research note 07 section 9). */
const CAPABILITIES: ExportCapabilities = capabilities({
    mixedDirection: true,
    multiEdges: true,
    selfLoops: true,
    edgeIds: "optional",
    idCharset: "nmtoken",
    dtypes: ["bool", "i32", "f32", "f64", "string"],
    components: false,
    lists: false,
    json: false,
    defaults: true,
    options: false,
    hierarchy: true,
    temporal: "none",
    graphAttributes: true,
    positions: false,
    viz: false,
});

/** Roles handled structurally (never written as a key) or reported by checkCapabilities() and not written. */
const STRUCTURAL_ROLES: ReadonlySet<string> = new Set([
    "directed",
    "pair",
    "mutual",
    "weight",
    "timeText",
    "originalId",
]);
const DROPPED_ROLES: ReadonlySet<string> = new Set([
    "position",
    "color",
    "size",
    "shape",
    "thickness",
    "start",
    "end",
    "timestamp",
    "timestamps",
    "spells",
    "open",
]);

/**
 * The roles GraphML has a slot for (checkCapabilities() reports every other role as lost): the
 * label key, the edge `id`, `sourceport` and `targetport` attributes and nested graphs.
 */
const SLOT_ROLES: ReadonlySet<string> = new Set(["label", "id", "sourcePort", "targetPort", "parent"]);

/** The slot roles written structurally per domain (never as a key); a label is a key titled `label`. */
const SLOT_ROLES_BY_DOMAIN: Readonly<Record<Domain, ReadonlySet<string>>> = {
    graph: new Set(),
    node: new Set(["parent"]),
    edge: new Set(["id", "sourcePort", "targetPort"]),
};

/** The names the importer gives the slot columns (design section 5.6), for the name-change notes. */
const ROLE_NAMES: Readonly<Record<string, string>> = Object.freeze({
    label: LABEL_COLUMN,
    id: EDGE_ID_COLUMN,
    sourcePort: SOURCE_PORT_COLUMN,
    targetPort: TARGET_PORT_COLUMN,
    parent: PARENT_COLUMN,
});

/** The attr.name of the weight key when the weight did not come from GraphML (the importer's weightFrom default). */
const DEFAULT_WEIGHT_NAME = "weight";

/** A note-recording callback. */
type NoteFn = (code: string, message: string, column?: string | null, count?: number | null) => void;

/** The GraphML attr.type values and the dtype each maps to, for restoring origin.type. */
const TYPE_DTYPES: Readonly<Record<string, string>> = {
    boolean: "bool",
    int: "i32",
    long: "f64",
    float: "f32",
    double: "f64",
    string: "string",
};

const I32_MAX = 2147483647;

/** The yfiles.type written for a yfiles json column whose origin records none. */
const DEFAULT_YFILES_TYPES: Readonly<Record<Domain, string>> = {
    node: "nodegraphics",
    edge: "edgegraphics",
    graph: "resources",
};

/** The element domains. */
type Domain = "graph" | "node" | "edge";

/** One column written as a key. */
interface ColumnPlan {
    readonly column: Column;
    readonly domain: Domain;
    /** The key id. */
    keyId: string;
    /** The attr.type, or null for a yfiles key. */
    readonly attrType: string | null;
    /** The attr.name, or null for a yfiles key. */
    readonly attrName: string | null;
    /** The yfiles.type, or null. */
    readonly yfilesType: string | null;
    /** Whether values are written as nested XML trees. */
    readonly yfiles: boolean;
}

/** One `<key>` element (a for="all" key merges up to three column plans). */
interface KeyPlan {
    readonly id: string;
    readonly domains: readonly Domain[];
    readonly attrName: string | null;
    readonly attrType: string | null;
    readonly yfilesType: string | null;
    /** The column whose default and desc are written; null for the weight key. */
    readonly column: Column | null;
}

/** Containment as the exporter writes it. */
interface Hierarchy {
    /** The parent column, or null. */
    readonly parent: Column | null;
    /** Nodes written at the top level, in order (roots, then the roots forced out of cycles). */
    readonly roots: Uint32Array;
    /** CSR of children by parent: childStart[u]..childStart[u+1] over childList. */
    readonly childStart: Uint32Array;
    readonly childList: Uint32Array;
    /** Whether the written order differs from index order. */
    readonly reordered: boolean;
    /** Nodes whose parent chain never reaches a root. */
    readonly unreachable: number;
}

/** Everything check() and export() agree on. */
interface Plan {
    readonly options: ResolvedExportOptions;
    readonly pretty: boolean;
    readonly notes: LossNote[];
    readonly keys: KeyPlan[];
    readonly graphColumns: ColumnPlan[];
    readonly nodeColumns: ColumnPlan[];
    readonly edgeColumns: ColumnPlan[];
    /** The weight key id, attr.type and the explicit weights, or null for an unweighted snapshot. */
    readonly weight: { readonly keyId: string; readonly attrType: string; readonly weights: ExplicitWeights } | null;
    /** The key id of the originalId attribute, when mangled ids are written. */
    readonly originalIdKey: string | null;
    readonly edgedefault: "directed" | "undirected";
    /** The pair folding: mirrors skipped, source directions. */
    readonly folding: PairFolding;
    readonly edgeIdColumn: Column | null;
    readonly sourcePortColumn: Column | null;
    readonly targetPortColumn: Column | null;
    readonly hierarchy: Hierarchy;
    readonly meta: GraphmlMeta | null;
    readonly needsOriginalIdKey: boolean;
}

/** The resolved format-specific options. */
interface FormatOptions {
    /** Whether to indent. */
    readonly pretty: boolean;
    /** The edgedefault override, or null. */
    readonly edgedefault: "directed" | "undirected" | null;
}

/**
 * Resolve the format-specific options.
 * @param options - the caller's options
 * @returns the pretty flag and the edgedefault override
 */
function resolveFormatOptions(options: GraphmlExportOptions | undefined): FormatOptions {
    const pretty = options?.pretty ?? true;
    if (typeof pretty !== "boolean") {
        throw new GraphFormatError("E_UNSUPPORTED", "option pretty must be a boolean", {
            option: "pretty",
            found: typeof pretty,
        });
    }
    const edgedefault = options?.edgedefault ?? null;
    if (edgedefault !== null && edgedefault !== "directed" && edgedefault !== "undirected") {
        throw new GraphFormatError(
            "E_UNSUPPORTED",
            `option edgedefault: ${JSON.stringify(edgedefault)} is not "directed" or "undirected"`,
            { option: "edgedefault", found: edgedefault },
        );
    }
    return { pretty, edgedefault };
}

/**
 * The importer's `meta.extra.graphml`, when the snapshot has one of the expected shape.
 * @param snapshot - the snapshot
 * @returns the meta, or null
 */
function graphmlMetaOf(snapshot: GraphSnapshot): GraphmlMeta | null {
    const value = snapshot.meta.extra[META_KEY];
    if (typeof value !== "object" || value === null) {
        return null;
    }
    const record = value as Record<string, unknown>;
    const graphId = typeof record.graphId === "string" ? record.graphId : null;
    const edgedefault =
        record.edgedefault === "directed" || record.edgedefault === "undirected" ? record.edgedefault : null;
    const namespaces: Record<string, string> = {};
    if (typeof record.namespaces === "object" && record.namespaces !== null) {
        for (const [prefix, uri] of Object.entries(record.namespaces as Record<string, unknown>)) {
            if (typeof uri === "string") {
                namespaces[prefix] = uri;
            }
        }
    }
    return { graphId, edgedefault, namespaces };
}

/**
 * Build the plan: classify every column, assign key ids, decide the edgedefault and the
 * containment order, and collect every loss note.
 * @param snapshot - the snapshot
 * @param options - the resolved common options
 * @param format - the resolved format options
 * @returns the plan
 */
function planExport(snapshot: GraphSnapshot, options: ResolvedExportOptions, format: FormatOptions): Plan {
    // the generic json note is replaced by planTable()'s (yfiles trees are kept, other json is text)
    const notes = checkCapabilities(snapshot, CAPABILITIES, options, {
        roles: SLOT_ROLES,
        roleNames: ROLE_NAMES,
    }).filter((n) => n.code !== LOSS.JSON);
    const note: NoteFn = (code, message, column = null, count = null): void => {
        notes.push(Object.freeze({ code, message, column, count }));
    };
    notes.push(...xmlIllegalTextNotes(snapshot));
    const meta = graphmlMetaOf(snapshot);
    const keyIds = new KeyIds();

    const graphColumns = planTable(snapshot.graph, "graph", notes, note);
    const nodeColumns = planTable(snapshot.nodes, "node", notes, note);
    const edgeColumns = planTable(snapshot.edges, "edge", notes, note);
    const keys = planKeys([...graphColumns, ...nodeColumns, ...edgeColumns], keyIds);
    reservedNameNotes(nodeColumns, RESERVED_NODE_NAMES, note);
    reservedNameNotes(edgeColumns, RESERVED_EDGE_NAMES, note);
    const weight = planWeight(snapshot, edgeColumns, keyIds, keys, note);

    // ids
    const idTypeChanges = countIdTypeChanges(snapshot);
    if (idTypeChanges > 0) {
        note(
            GRAPHML_LOSS.ID_TEXT_TYPE,
            `${idTypeChanges} node id(s) change type when read back under ids: "canonical" (string ids that are integer text, non-integer numbers)`,
            null,
            idTypeChanges,
        );
    }
    const edgeIdColumn = snapshot.edges.byRole("id");
    if (edgeIdColumn !== null) {
        planEdgeIds(snapshot, edgeIdColumn, options, note);
    }
    const unrepresentable = countUnrepresentableNodeIds(snapshot);
    const needsOriginalIdKey = unrepresentable > 0 && options.sanitizeIds === "mangle";
    const originalIdKey = needsOriginalIdKey ? keyIds.next(null) : null;

    // direction: an undirected pair folds to one undirected edge, a mutual pair too (the mark is lost)
    const folding = pairFolding(snapshot, { foldMutual: true });
    if (folding.mutualCount > 0) {
        note(
            GRAPHML_LOSS.MUTUAL_AS_UNDIRECTED,
            `${folding.mutualCount} mutual pair(s) are written as undirected edges; the mutual mark is lost`,
            null,
            folding.mutualCount,
        );
    }
    const edgedefault = planEdgedefault(snapshot, folding, format, meta);

    // containment
    const hierarchy = planHierarchy(snapshot);
    if (hierarchy.reordered) {
        note(
            GRAPHML_LOSS.HIERARCHY_REORDERED,
            "nodes are written in containment order (children nested under their parent); node indices change after a round trip",
            hierarchy.parent?.meta.name ?? null,
        );
    }
    if (hierarchy.unreachable > 0) {
        note(
            GRAPHML_LOSS.PARENT_CYCLE,
            `${hierarchy.unreachable} node(s) whose parent chain never reaches a root are written at the top level`,
            hierarchy.parent?.meta.name ?? null,
            hierarchy.unreachable,
        );
    }

    return {
        options,
        pretty: format.pretty,
        notes,
        keys,
        graphColumns,
        nodeColumns,
        edgeColumns,
        weight,
        originalIdKey,
        edgedefault,
        folding,
        edgeIdColumn,
        sourcePortColumn: snapshot.edges.byRole("sourcePort"),
        targetPortColumn: snapshot.edges.byRole("targetPort"),
        hierarchy,
        meta,
        needsOriginalIdKey,
    };
}

/**
 * A plain column titled like one of the importer's XML-derived columns (`id`, `sourceport`,
 * `targetport` on edges, `parent` on nodes) reads back renamed `<name>#<key id>` (design section
 * 5.6); the note says so.
 * @param plans - the column plans of one domain, key ids assigned
 * @param reserved - the reserved names of that domain
 * @param note - the note recorder
 */
function reservedNameNotes(plans: readonly ColumnPlan[], reserved: ReadonlySet<string>, note: NoteFn): void {
    for (const plan of plans) {
        const { column, attrName, keyId } = plan;
        if (attrName === null || !reserved.has(attrName)) {
            continue;
        }
        note(
            LOSS.COLUMN_NAME_CHANGED,
            `${plan.domain} column "${column.meta.name}" is titled like the importer's ${attrName} column and reads back as "${attrName}#${keyId}"`,
            column.meta.name,
            column.length - column.nullCount,
        );
    }
}

/** The key ids handed out so far: a preferred NMTOKEN when free, else the next free `d<n>`. */
class KeyIds {
    private readonly used = new Set<string>();
    private counter = 0;

    /**
     * Take a key id.
     * @param preferred - the id to keep when it is an NMTOKEN not handed out yet, or null
     * @returns the id
     */
    next(preferred: string | null): string {
        if (preferred !== null && isNmtoken(preferred) && !this.used.has(preferred)) {
            this.used.add(preferred);
            return preferred;
        }
        for (;;) {
            const candidate = `d${this.counter}`;
            this.counter++;
            if (!this.used.has(candidate)) {
                this.used.add(candidate);
                return candidate;
            }
        }
    }
}

/**
 * The `<key>` elements: columns sharing a GraphML origin.id across domains become one for="all"
 * key when their name, type and default agree; every other column is its own key.
 * @param all - the column plans of the three domains, in writing order
 * @param keyIds - the key id allocator
 * @returns the keys, in declaration order
 */
function planKeys(all: readonly ColumnPlan[], keyIds: KeyIds): KeyPlan[] {
    const keys: KeyPlan[] = [];
    const byOrigin = new Map<string, ColumnPlan[]>();
    for (const plan of all) {
        const { origin } = plan.column.meta;
        if (origin !== null && origin.format === FORMAT && origin.id !== null) {
            const group = byOrigin.get(origin.id) ?? [];
            group.push(plan);
            byOrigin.set(origin.id, group);
        }
    }
    const planned = new Set<ColumnPlan>();
    for (const plan of all) {
        if (planned.has(plan)) {
            continue;
        }
        const { origin } = plan.column.meta;
        const group =
            origin !== null && origin.format === FORMAT && origin.id !== null
                ? (byOrigin.get(origin.id) ?? [plan])
                : [plan];
        const shared = group.length > 1 && group.every((other) => other === plan || sameKey(plan, other));
        const members = shared ? group : [plan];
        const id = keyIds.next(origin !== null && origin.format === FORMAT ? origin.id : null);
        for (const member of members) {
            member.keyId = id;
            planned.add(member);
        }
        const domains = members.map((member) => member.domain);
        // a yfiles key carries attr.name only when the column name is not the key id (yEd writes none)
        const yfilesName = plan.column.meta.origin?.title ?? plan.column.meta.name;
        let { attrName } = plan;
        if (plan.yfiles) {
            attrName = yfilesName === id ? null : yfilesName;
        }
        keys.push({
            id,
            domains: domains.length > 1 ? ["graph", "node", "edge"] : domains,
            attrName,
            attrType: plan.attrType,
            yfilesType: plan.yfilesType,
            column: plan.column,
        });
    }
    return keys;
}

/**
 * The weight key of a weighted snapshot (its attr.name and attr.type restored from
 * meta.weightOrigin when the weight came from GraphML), and the note for an edge column the
 * importer would read as THE weight: every edge key titled like the weight key (`weight` by
 * default) is a weight key on import, so such a column reads back as the weight, not as a column.
 * @param snapshot - the snapshot
 * @param edgeColumns - the edge column plans
 * @param keyIds - the key id allocator
 * @param keys - receives the weight key
 * @param note - the note recorder
 * @returns the weight plan, or null for an unweighted snapshot
 */
function planWeight(
    snapshot: GraphSnapshot,
    edgeColumns: readonly ColumnPlan[],
    keyIds: KeyIds,
    keys: KeyPlan[],
    note: NoteFn,
): Plan["weight"] {
    const weights = explicitWeights(snapshot);
    const origin = snapshot.meta.weightOrigin;
    const fromGraphml = weights.weighted && origin !== null && origin.format === FORMAT;
    const attrName = fromGraphml && origin.title !== null ? origin.title : DEFAULT_WEIGHT_NAME;
    for (const plan of edgeColumns) {
        if (plan.attrName === attrName) {
            note(
                LOSS.WEIGHT_KEY_CLASH,
                `edge column "${plan.column.meta.name}" is written as a key titled "${attrName}", which the importer reads as THE weight (weightFrom); it reads back as the weight, not as a column`,
                plan.column.meta.name,
                plan.column.length - plan.column.nullCount,
            );
        }
    }
    if (!weights.weighted) {
        return null;
    }
    let attrType = "double";
    if (fromGraphml && origin.type !== null) {
        const declared = origin.type.trim().toLowerCase();
        if (declared === "float" || declared === "double" || declared === "long" || declared === "int") {
            attrType = declared;
        }
    }
    if ((attrType === "int" || attrType === "long") && !weightsIntegral(snapshot, weights)) {
        // a declared integer weight only survives when every explicit weight is integral
        attrType = "double";
    }
    const id = keyIds.next(fromGraphml ? origin.id : null);
    keys.push({ id, domains: ["edge"], attrName, attrType, yfilesType: null, column: null });
    return { keyId: id, attrType, weights };
}

/**
 * The notes about the edge id column: a numeric column reads back as string; ids outside the
 * NMTOKEN charset are refused or mangled per sanitizeIds.
 * @param snapshot - the snapshot
 * @param column - the edge id column
 * @param options - the resolved options
 * @param note - the note recorder
 */
function planEdgeIds(snapshot: GraphSnapshot, column: Column, options: ResolvedExportOptions, note: NoteFn): void {
    if (column.dtype !== "string" && column.dtype !== "dict") {
        note(
            GRAPHML_LOSS.EDGE_ID_TEXT,
            `edge id column "${column.meta.name}" is ${column.dtype}; GraphML edge ids read back as strings`,
            column.meta.name,
            snapshot.edgeCount - column.nullCount,
        );
    }
    const bad = countBadEdgeIds(column);
    if (bad === 0) {
        return;
    }
    if (options.sanitizeIds === "mangle") {
        note(LOSS.ID_MANGLED, `${bad} edge id(s) outside the nmtoken charset are rewritten`, column.meta.name, bad);
    } else {
        note(
            LOSS.ID_CHARSET,
            `${bad} edge id(s) outside the nmtoken charset; export() will throw unless sanitizeIds is "mangle"`,
            column.meta.name,
            bad,
        );
    }
}

/**
 * The top-level edgedefault: the override, else undirected for an undirected snapshot, else
 * directed when no written edge is undirected, else the one the importer recorded, else the
 * majority direction.
 * @param snapshot - the snapshot
 * @param folding - the pair folding
 * @param format - the resolved format options
 * @param meta - the importer's meta, or null
 * @returns the edgedefault
 */
function planEdgedefault(
    snapshot: GraphSnapshot,
    folding: PairFolding,
    format: FormatOptions,
    meta: GraphmlMeta | null,
): "directed" | "undirected" {
    if (format.edgedefault !== null) {
        return format.edgedefault;
    }
    if (!snapshot.directed) {
        return "undirected";
    }
    let directedCount = 0;
    let undirectedCount = 0;
    for (let e = 0; e < snapshot.edgeCount; e++) {
        if (folding.folded(e)) {
            continue;
        }
        if (isDirectedEdge(folding, e)) {
            directedCount++;
        } else {
            undirectedCount++;
        }
    }
    if (undirectedCount === 0) {
        return "directed";
    }
    const recorded = meta === null ? null : meta.edgedefault;
    if (recorded !== null) {
        return recorded;
    }
    return undirectedCount > directedCount ? "undirected" : "directed";
}

/**
 * Whether two column plans can share one for="all" key.
 * @param a - one plan
 * @param b - another
 * @returns true when name, type and default agree
 */
function sameKey(a: ColumnPlan, b: ColumnPlan): boolean {
    return (
        a.attrName === b.attrName &&
        a.attrType === b.attrType &&
        a.yfilesType === b.yfilesType &&
        a.domain !== b.domain &&
        JSON.stringify(a.column.meta.default ?? null) === JSON.stringify(b.column.meta.default ?? null)
    );
}

/**
 * Classify the columns of one table: which are written as keys and what each loses. A label
 * column is written into the label slot (a key titled `label`, which the importer reads back
 * with the role) unless the table holds another column of that name, in which case it keeps
 * its own title and the role is lost.
 * @param table - the table
 * @param domain - its domain
 * @param notes - the notes so far (the generic name-change note of a label column that cannot take the slot is withdrawn)
 * @param note - the note recorder
 * @returns the plans of the written columns, in declaration order
 */
function planTable(table: Iterable<Column>, domain: Domain, notes: LossNote[], note: NoteFn): ColumnPlan[] {
    const plans: ColumnPlan[] = [];
    const columns = [...table];
    const names = new Set(columns.map((column) => column.meta.name));
    for (const column of columns) {
        const { meta } = column;
        const { role, name } = meta;
        if (role === "parents") {
            note(
                GRAPHML_LOSS.PARENTS_DROPPED,
                `${domain} column "${name}" (parents) cannot be written: nested graphs hold one parent per node`,
                name,
                column.length - column.nullCount,
            );
            continue;
        }
        if (
            role !== null &&
            (STRUCTURAL_ROLES.has(role) || DROPPED_ROLES.has(role) || SLOT_ROLES_BY_DOMAIN[domain].has(role))
        ) {
            continue;
        }
        if (role !== null && role !== "label" && SLOT_ROLES.has(role)) {
            // the slot belongs to another domain (an edge id on a node, a port on a graph)
            withdrawNameChange(notes, name);
            note(
                LOSS.ROLE,
                `${domain} column "${name}" (${role}) is written as a plain attribute; GraphML has no ${role} slot for a ${domain} and the role is lost`,
                name,
                column.length - column.nullCount,
            );
        }
        const yfiles = meta.dtype === "json" && meta.origin?.namespace === "yfiles";
        if (meta.dtype === "json" && !yfiles) {
            note(
                LOSS.JSON,
                `${domain} column "${name}" holds nested values; written as JSON text, which reads back as string`,
                name,
                column.length - column.nullCount,
            );
        }
        if (yfiles) {
            const bad = countBadTrees(column);
            if (bad > 0) {
                note(
                    GRAPHML_LOSS.YFILES_TREE,
                    `${bad} value(s) of yfiles column "${name}" are not XML trees; export() will throw E_COLUMN_TYPE`,
                    name,
                    bad,
                );
            }
        }
        const { origin } = meta;
        const fromGraphml = origin !== null && origin.format === FORMAT;
        let attrName: string | null = null;
        if (!yfiles) {
            attrName = fromGraphml && origin.title !== null ? origin.title : name;
            if (role === "label" && domain !== "graph" && attrName !== LABEL_COLUMN) {
                attrName = labelSlot(column, domain, names, notes, note);
            } else if (role === null && domain !== "graph" && attrName === LABEL_COLUMN) {
                note(
                    LOSS.ROLE_ASSUMED,
                    `${domain} column "${name}" has no role but is titled "${LABEL_COLUMN}", which the importer reads back with the label role`,
                    name,
                    column.length - column.nullCount,
                );
            }
        }
        plans.push({
            column,
            domain,
            keyId: "",
            attrType: yfiles ? null : attrTypeFor(column),
            attrName,
            yfilesType: yfiles ? (origin?.type ?? DEFAULT_YFILES_TYPES[domain]) : null,
            yfiles,
        });
    }
    return plans;
}

/**
 * The title a label column not named `label` is written under: the label slot when the table has
 * no other column of that name (checkCapabilities() has noted the name change), else its own
 * name with the role lost (the name-change note is withdrawn and a role note recorded).
 * @param column - the label column
 * @param domain - its domain
 * @param names - the column names of the table
 * @param notes - the notes so far
 * @param note - the note recorder
 * @returns the attr.name
 */
function labelSlot(
    column: Column,
    domain: Domain,
    names: ReadonlySet<string>,
    notes: LossNote[],
    note: NoteFn,
): string {
    const { name } = column.meta;
    if (!names.has(LABEL_COLUMN)) {
        return LABEL_COLUMN;
    }
    withdrawNameChange(notes, name);
    note(
        LOSS.ROLE,
        `${domain} column "${name}" (label) is written as a plain attribute: the label slot (a key titled "${LABEL_COLUMN}") is taken by column "${LABEL_COLUMN}" and the role is lost`,
        name,
        column.length - column.nullCount,
    );
    return name;
}

/**
 * Withdraw the name-change note checkCapabilities() recorded for a role column that does not
 * take the slot after all.
 * @param notes - the notes so far
 * @param name - the column name
 */
function withdrawNameChange(notes: LossNote[], name: string): void {
    const at = notes.findIndex((n) => n.code === LOSS.COLUMN_NAME_CHANGED && n.column === name);
    if (at >= 0) {
        notes.splice(at, 1);
    }
}

/**
 * The attr.type a column is declared with: the GraphML origin.type when it still describes the
 * column, else the type of the dtype (u32 / u8 as int or long, dict as string, list / json /
 * multi-component as string holding JSON text).
 * @param column - the column
 * @returns the attr.type text
 */
function attrTypeFor(column: Column): string {
    const { meta } = column;
    const { origin } = meta;
    const declared = origin !== null && origin.format === FORMAT && origin.type !== null ? origin.type : null;
    if (meta.components > 1) {
        return "string";
    }
    switch (meta.dtype) {
        case "bool":
            return "boolean";
        case "i32":
            return "int";
        case "f32":
            return "float";
        case "f64": {
            if (declared !== null && declared.trim().toLowerCase() === "long" && columnIntegral(column)) {
                return "long";
            }
            return "double";
        }
        case "u8":
            return "int";
        case "u32":
            return columnMax(column) <= I32_MAX ? "int" : "long";
        case "string":
            if (declared !== null && TYPE_DTYPES[declared.trim().toLowerCase()] === undefined) {
                // an unknown declared type kept as text (W_UNKNOWN_ATTR_TYPE on import) is restored as declared
                return declared;
            }
            return "string";
        case "dict":
        case "list":
        case "json":
            return "string";
        default: {
            const { dtype } = meta;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown dtype ${String(dtype)}`, { dtype });
        }
    }
}

/**
 * Whether every set value of a numeric column is an integer.
 * @param column - an f64 column
 * @returns true when all set values are integral
 */
function columnIntegral(column: Column): boolean {
    if (column.dtype !== "f64") {
        return false;
    }
    const { data } = column;
    for (let r = 0; r < column.length; r++) {
        if (column.isSet(r) && !Number.isInteger(data[r])) {
            return false;
        }
    }
    return true;
}

/**
 * The largest set value of a u32 column.
 * @param column - a u32 column
 * @returns the maximum, 0 for an empty column
 */
function columnMax(column: Column): number {
    if (column.dtype !== "u32") {
        return 0;
    }
    let max = 0;
    const { data } = column;
    for (let r = 0; r < column.length; r++) {
        if (column.isSet(r) && data[r] > max) {
            max = data[r];
        }
    }
    return max;
}

/**
 * Whether every explicit weight is integral (so a declared int / long weight key survives).
 * @param snapshot - the snapshot
 * @param weights - the explicit weights
 * @returns true when all explicit weights are integers
 */
function weightsIntegral(snapshot: GraphSnapshot, weights: ExplicitWeights): boolean {
    for (let e = 0; e < snapshot.edgeCount; e++) {
        if (weights.isExplicit(e) && !Number.isInteger(weights.value(e))) {
            return false;
        }
    }
    return true;
}

/**
 * How many values of a yfiles column are not serialisable trees.
 * @param column - a json column
 * @returns the count
 */
function countBadTrees(column: Column): number {
    if (column.dtype !== "json") {
        return 0;
    }
    let bad = 0;
    for (let r = 0; r < column.length; r++) {
        if (column.isSet(r) && treeProblem(column.values[r]) !== null) {
            bad++;
        }
    }
    return bad;
}

/**
 * Node ids that read back as another type under the canonical rule: string ids that are
 * canonical integer text, and numbers that are not safe integers.
 * @param snapshot - the snapshot
 * @returns the count
 */
function countIdTypeChanges(snapshot: GraphSnapshot): number {
    const { ids } = snapshot;
    if (ids.kind === "identity" || ids.kind === "dense") {
        return 0;
    }
    let count = 0;
    for (let i = 0; i < ids.size; i++) {
        const id = ids.idOf(i);
        if (typeof id === "string" ? isNmtoken(id) && isCanonicalIntegerText(id) : !Number.isSafeInteger(id)) {
            count++;
        }
    }
    return count;
}

/**
 * Node ids the nmtoken charset cannot hold.
 * @param snapshot - the snapshot
 * @returns the count
 */
function countUnrepresentableNodeIds(snapshot: GraphSnapshot): number {
    const { ids } = snapshot;
    if (ids.kind === "identity" || ids.kind === "dense") {
        return 0;
    }
    let count = 0;
    for (let i = 0; i < ids.size; i++) {
        if (!isNmtoken(String(ids.idOf(i)))) {
            count++;
        }
    }
    return count;
}

/**
 * Edge ids that are not NMTOKENs.
 * @param column - the edge id column
 * @returns the count
 */
function countBadEdgeIds(column: Column): number {
    let bad = 0;
    for (let e = 0; e < column.length; e++) {
        if (column.isSet(e) && !isNmtoken(edgeIdText(column, e))) {
            bad++;
        }
    }
    return bad;
}

/**
 * The text of an edge id cell.
 * @param column - the edge id column
 * @param e - the edge
 * @returns the id as text
 */
function edgeIdText(column: Column, e: number): string {
    const value = column.value(e);
    if (typeof value === "string") {
        return value;
    }
    if (typeof value === "number") {
        return Number.isInteger(value) ? formatInteger(value) : formatF64(value);
    }
    return String(value);
}

/**
 * Whether a logical edge is written as directed: never in an undirected snapshot; never for a
 * half of an expanded pair (an undirected or a mutual source edge, whose halves the resolver
 * flags directed); otherwise per the directed role column, with directed as the default.
 * @param folding - the pair folding
 * @param e - the edge
 * @returns false for a paired edge or an edge flagged undirected
 */
function isDirectedEdge(folding: PairFolding, e: number): boolean {
    return folding.mateOf(e) === INVALID_INDEX && folding.sourceDirected(e);
}

/**
 * The containment order: the children CSR over the parent column (src/children.ts), roots in
 * index order, unreachable nodes (cycles) forced to the top level.
 * @param snapshot - the snapshot
 * @returns the hierarchy
 */
function planHierarchy(snapshot: GraphSnapshot): Hierarchy {
    const parent = snapshot.nodes.byRole("parent");
    const csr = childrenCsr(snapshot, { column: parent !== null && parent.dtype === "u32" ? parent : null });
    const walk = csr.depthFirst();
    // the top-level nodes in written order: the roots, then the cycle members forced out
    const roots = walk.order.filter((u) => walk.depth[u] === 0);
    return {
        parent: csr.column,
        roots,
        childStart: csr.rowPtr,
        childList: csr.children,
        reordered: walk.reordered,
        unreachable: csr.unreachable,
    };
}

// ============================================================ writing

/**
 * The text of one set cell for a `<data>` element or a `<default>`.
 * @param column - the column
 * @param row - the row
 * @param attrType - the declared attr.type
 * @returns the text
 */
function cellText(column: Column, row: number, attrType: string | null): string {
    if (column.meta.components > 1) {
        const value = column.value(row);
        return JSON.stringify(Array.from(value as ArrayLike<number>));
    }
    switch (column.dtype) {
        case "bool":
            return column.value(row) === true ? "true" : "false";
        case "i32":
        case "u32":
        case "u8":
            return String(column.data[row]);
        case "f32":
            return formatF32(column.data[row]);
        case "f64": {
            const value = column.data[row];
            return (attrType === "long" || attrType === "int") && Number.isInteger(value)
                ? formatInteger(value)
                : formatF64(value);
        }
        case "string":
            return column.valueAt(row);
        case "dict":
            return column.dictionary[column.codes[row]];
        case "list":
            return JSON.stringify(Array.from(column.sliceOf(row)));
        case "json":
            return JSON.stringify(column.values[row]) ?? "";
        default: {
            const dtype: never = column;
            throw new GraphFormatError("E_UNSUPPORTED", `unknown dtype ${String(dtype)}`, {});
        }
    }
}

/**
 * The text of a declared default.
 * @param column - the column
 * @param attrType - the declared attr.type
 * @returns the text, or null when the column has no default
 */
function defaultText(column: Column, attrType: string | null): string | null {
    const value = column.meta.default;
    if (value === undefined) {
        return null;
    }
    switch (typeof value) {
        case "boolean":
            return value ? "true" : "false";
        case "number":
            if (column.dtype === "f32") {
                return formatF32(value);
            }
            return (attrType === "long" || attrType === "int") && Number.isInteger(value)
                ? formatInteger(value)
                : formatF64(value);
        case "string":
            return value;
        default:
            return JSON.stringify(value) ?? "";
    }
}

/**
 * Write one `<data>` element.
 * @param plan - the column plan
 * @param row - the row
 * @param indent - the indentation
 * @param out - receives the parts
 */
function writeData(plan: ColumnPlan, row: number, indent: string, out: string[]): void {
    const { column } = plan;
    if (!column.isSet(row)) {
        return;
    }
    if (plan.yfiles && column.dtype === "json") {
        writeTreeData(plan.keyId, column.values[row], indent, out);
        return;
    }
    out.push(`\n${indent}<data key="${plan.keyId}">${escapeXmlText(cellText(column, row, plan.attrType))}</data>`);
}

/**
 * Write a `<data>` holding a yfiles tree.
 * @param keyId - the key id
 * @param value - the tree
 * @param indent - the indentation
 * @param out - receives the parts
 */
function writeTreeData(keyId: string, value: unknown, indent: string, out: string[]): void {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
        const parts: string[] = [];
        writeXmlTree(value, indent, parts);
        out.push(`\n${indent}<data key="${keyId}">${parts.join("")}</data>`);
        return;
    }
    out.push(`\n${indent}<data key="${keyId}">`);
    writeXmlTree(value, `${indent}  `, out);
    out.push(`\n${indent}</data>`);
}

/**
 * The `<key>` element of a plan.
 * @param key - the key plan
 * @param indent - the indentation
 * @returns the element text
 */
function keyElement(key: KeyPlan, indent: string): string {
    const domain = key.domains.length === 3 ? "all" : key.domains[0];
    let text = `${indent}<key id="${escapeXmlAttribute(key.id)}" for="${domain}"`;
    if (key.attrName !== null) {
        text += ` attr.name="${escapeXmlAttribute(key.attrName)}"`;
    }
    if (key.attrType !== null) {
        text += ` attr.type="${escapeXmlAttribute(key.attrType)}"`;
    }
    if (key.yfilesType !== null) {
        text += ` yfiles.type="${escapeXmlAttribute(key.yfilesType)}"`;
    }
    const { column } = key;
    const desc = column?.meta.extra.desc;
    const hasDefault = column !== null && column.meta.default !== undefined;
    if (column === null || (typeof desc !== "string" && !hasDefault)) {
        return `${text}/>\n`;
    }
    const parts: string[] = [`${text}>`];
    const inner = `${indent}  `;
    if (typeof desc === "string") {
        parts.push(`\n${inner}<desc>${escapeXmlText(desc)}</desc>`);
    }
    if (hasDefault) {
        if (key.yfilesType !== null) {
            const value = column.meta.default;
            if (value !== null && typeof value === "object" && !Array.isArray(value)) {
                parts.push(`\n${inner}<default>`);
                writeXmlTree(value, `${inner}  `, parts);
                parts.push(`\n${inner}</default>`);
            } else {
                const tree: string[] = [];
                writeXmlTree(value, inner, tree);
                parts.push(`\n${inner}<default>${tree.join("")}</default>`);
            }
        } else {
            parts.push(`\n${inner}<default>${escapeXmlText(defaultText(column, key.attrType) ?? "")}</default>`);
        }
    }
    parts.push(`\n${indent}</key>\n`);
    return parts.join("");
}

/**
 * Sanitised edge ids: NMTOKENs as they are; the rest refused or mangled per sanitizeIds.
 * @param column - the edge id column, or null
 * @param mode - the resolved sanitizeIds option
 * @param edgeCount - the edge count
 * @returns the id text per edge (null for edges without one)
 */
function sanitizeEdgeIds(column: Column | null, mode: "error" | "mangle", edgeCount: number): (string | null)[] {
    const out: (string | null)[] = new Array<string | null>(edgeCount).fill(null);
    if (column === null) {
        return out;
    }
    const used = new Set<string>();
    const bad: number[] = [];
    for (let e = 0; e < edgeCount; e++) {
        if (!column.isSet(e)) {
            continue;
        }
        const text = edgeIdText(column, e);
        if (isNmtoken(text)) {
            out[e] = text;
            used.add(text);
        } else {
            bad.push(e);
        }
    }
    if (bad.length > 0 && mode === "error") {
        throw new GraphFormatError(
            "E_INVALID_ID",
            `${bad.length} edge id(s) cannot be written as nmtoken (first: ${JSON.stringify(edgeIdText(column, bad[0]))} at edge ${bad[0]}); pass sanitizeIds: "mangle" to rewrite them`,
            { reason: "charset", charset: "nmtoken", count: bad.length, edge: bad[0] },
        );
    }
    for (const e of bad) {
        const base = mangleNmtoken(edgeIdText(column, e));
        let candidate = base;
        for (let k = 2; used.has(candidate); k++) {
            candidate = `${base}_${k}`;
        }
        used.add(candidate);
        out[e] = candidate;
    }
    return out;
}

/**
 * The document as text parts.
 * @param snapshot - the snapshot
 * @param plan - the plan
 * @yields one element (or a group of small ones) at a time
 * @returns nothing
 */
function* writeGraphml(snapshot: GraphSnapshot, plan: Plan): Generator<string, void, undefined> {
    const ids: SanitizedIds = sanitizeIds(snapshot, "nmtoken", plan.options.sanitizeIds);
    const edgeIds = sanitizeEdgeIds(plan.edgeIdColumn, plan.options.sanitizeIds, snapshot.edgeCount);
    const i1 = plan.pretty ? "  " : "";
    const i2 = plan.pretty ? "    " : "";
    const i3 = plan.pretty ? "      " : "";
    const step = plan.pretty ? "  " : "";

    yield '<?xml version="1.0" encoding="UTF-8"?>\n';
    let root = `<graphml xmlns="${GRAPHML_NAMESPACE}"`;
    const namespaces: Record<string, string> = { ...(plan.meta?.namespaces ?? {}) };
    if (
        (plan.nodeColumns.some((c) => c.yfiles) ||
            plan.edgeColumns.some((c) => c.yfiles) ||
            plan.graphColumns.some((c) => c.yfiles)) &&
        !Object.values(namespaces).includes(YFILES_NAMESPACE)
    ) {
        namespaces.y = YFILES_NAMESPACE;
    }
    if (!Object.values(namespaces).includes(XSI_NAMESPACE)) {
        namespaces.xsi = XSI_NAMESPACE;
    }
    for (const [prefix, uri] of Object.entries(namespaces)) {
        root += ` xmlns:${prefix}="${escapeXmlAttribute(uri)}"`;
    }
    const xsiPrefix = Object.entries(namespaces).find(([, uri]) => uri === XSI_NAMESPACE)?.[0] ?? "xsi";
    root += ` ${xsiPrefix}:schemaLocation="${SCHEMA_LOCATION}">\n`;
    yield root;

    for (const key of plan.keys) {
        yield keyElement(key, i1);
    }
    if (plan.originalIdKey !== null) {
        yield `${i1}<key id="${plan.originalIdKey}" for="node" attr.name="${ORIGINAL_ID_ATTRIBUTE}" attr.type="string"/>\n`;
    }

    const graphId = plan.meta?.graphId ?? "G";
    yield `${i1}<graph id="${escapeXmlAttribute(graphId)}" edgedefault="${plan.edgedefault}">`;
    const { description } = snapshot.meta;
    if (description !== null) {
        yield `\n${i2}<desc>${escapeXmlText(description)}</desc>`;
    }
    const graphParts: string[] = [];
    for (const column of plan.graphColumns) {
        writeData(column, 0, i2, graphParts);
    }
    if (graphParts.length > 0) {
        yield graphParts.join("");
    }

    // nodes in containment order, children nested under their parent
    const { hierarchy } = plan;
    const nested = hierarchy.childList.length > 0;
    const stackNode: number[] = [];
    const stackCursor: number[] = [];
    const written = nested ? new Uint8Array(snapshot.nodeCount) : null;
    for (const rootNode of hierarchy.roots) {
        yield* writeNodeSubtree(rootNode, plan, ids, i2, step, stackNode, stackCursor, written);
    }

    // edges: primaries only, in index order
    const list = snapshot.edgeList();
    const { folding } = plan;
    for (let e = 0; e < snapshot.edgeCount; e++) {
        if (folding.folded(e)) {
            continue;
        }
        const parts: string[] = [`\n${i2}<edge`];
        const id = edgeIds[e];
        if (id !== null) {
            parts.push(` id="${escapeXmlAttribute(id)}"`);
        }
        parts.push(
            ` source="${escapeXmlAttribute(String(ids.idAt(list.src[e])))}" target="${escapeXmlAttribute(String(ids.idAt(list.dst[e])))}"`,
        );
        const directed = snapshot.directed && isDirectedEdge(folding, e);
        if (directed !== (plan.edgedefault === "directed")) {
            parts.push(` directed="${directed ? "true" : "false"}"`);
        }
        const sourcePort = portText(plan.sourcePortColumn, e);
        if (sourcePort !== null) {
            parts.push(` sourceport="${escapeXmlAttribute(sourcePort)}"`);
        }
        const targetPort = portText(plan.targetPortColumn, e);
        if (targetPort !== null) {
            parts.push(` targetport="${escapeXmlAttribute(targetPort)}"`);
        }
        const open = parts.length;
        if (plan.weight !== null) {
            const { keyId, attrType } = plan.weight;
            const weight = plan.weight.weights.text(e, attrType === "int" || attrType === "long");
            if (weight !== null) {
                parts.push(`\n${i3}<data key="${keyId}">${escapeXmlText(weight)}</data>`);
            }
        }
        for (const column of plan.edgeColumns) {
            writeData(column, e, i3, parts);
        }
        if (parts.length === open) {
            parts.push("/>");
        } else {
            parts.splice(open, 0, ">");
            parts.push(`\n${i2}</edge>`);
        }
        yield parts.join("");
    }

    yield `\n${i1}</graph>\n</graphml>\n`;
}

/**
 * The text of a port reference cell.
 * @param column - the sourcePort / targetPort column, or null
 * @param e - the edge
 * @returns the port name, or null when unset
 */
function portText(column: Column | null, e: number): string | null {
    if (column === null || !column.isSet(e)) {
        return null;
    }
    const value = column.value(e);
    return typeof value === "string" ? value : String(value);
}

/**
 * Write a top-level node and, nested, every node contained in it (iteratively, so a deep chain
 * cannot overflow the call stack).
 * @param root - the top-level node
 * @param plan - the plan
 * @param ids - the sanitised ids
 * @param indent - the indentation of top-level nodes
 * @param step - one indentation step
 * @param stackNode - a reusable stack of open nodes
 * @param stackCursor - a reusable stack of child cursors
 * @param written - nodes already written (null when nothing is nested)
 * @yields the node elements
 * @returns nothing
 */
function* writeNodeSubtree(
    root: number,
    plan: Plan,
    ids: SanitizedIds,
    indent: string,
    step: string,
    stackNode: number[],
    stackCursor: number[],
    written: Uint8Array | null,
): Generator<string, void, undefined> {
    const { hierarchy } = plan;
    const hasChildren = (u: number): boolean => hierarchy.childStart[u + 1] > hierarchy.childStart[u];
    const openNode = (u: number, depth: number): string => {
        const pad = indent + step.repeat(depth * 2);
        const parts: string[] = [`\n${pad}<node id="${escapeXmlAttribute(String(ids.idAt(u)))}"`];
        const open = parts.length;
        if (plan.originalIdKey !== null && ids.isChanged(u)) {
            parts.push(
                `\n${pad}${step}<data key="${plan.originalIdKey}">${escapeXmlText(String(ids.originalAt(u)))}</data>`,
            );
        }
        for (const column of plan.nodeColumns) {
            writeData(column, u, pad + step, parts);
        }
        if (hasChildren(u)) {
            parts.splice(open, 0, ">");
            parts.push(
                `\n${pad}${step}<graph id="${escapeXmlAttribute(`${String(ids.idAt(u))}:`)}" edgedefault="${plan.edgedefault}">`,
            );
            return parts.join("");
        }
        if (parts.length === open) {
            parts.push("/>");
        } else {
            parts.splice(open, 0, ">");
            parts.push(`\n${pad}</node>`);
        }
        return parts.join("");
    };
    const closeNode = (depth: number): string => {
        const pad = indent + step.repeat(depth * 2);
        return `\n${pad}${step}</graph>\n${pad}</node>`;
    };
    if (written !== null) {
        written[root] = 1;
    }
    yield openNode(root, 0);
    if (!hasChildren(root)) {
        return;
    }
    stackNode.push(root);
    stackCursor.push(hierarchy.childStart[root]);
    while (stackNode.length > 0) {
        const top = stackNode.length - 1;
        const u = stackNode[top];
        const cursor = stackCursor[top];
        if (cursor >= hierarchy.childStart[u + 1]) {
            stackNode.pop();
            stackCursor.pop();
            yield closeNode(top);
            continue;
        }
        stackCursor[top] = cursor + 1;
        const v = hierarchy.childList[cursor];
        if (written !== null && written[v] === 1) {
            continue;
        }
        if (written !== null) {
            written[v] = 1;
        }
        yield openNode(v, top + 1);
        if (hasChildren(v)) {
            stackNode.push(v);
            stackCursor.push(hierarchy.childStart[v]);
        }
    }
}

/**
 * The GraphML exporter (design section 8.5).
 */
export const graphmlExporter: GraphExporter<GraphmlExportOptions> = Object.freeze({
    format: FORMAT,
    capabilities: CAPABILITIES,
    /**
     * Pre-flight: every loss the export would incur.
     * @param snapshot - the snapshot
     * @param options - format-specific and common options
     * @returns the notes
     */
    check(snapshot: GraphSnapshot, options?: GraphmlExportOptions & CommonExportOptions): readonly LossNote[] {
        const plan = planExport(snapshot, resolveExportOptions(options), resolveFormatOptions(options));
        return Object.freeze(plan.notes);
    },
    /**
     * Write the document as UTF-8 chunks.
     * @param snapshot - the snapshot
     * @param options - format-specific and common options
     * @returns the chunks
     */
    export(snapshot: GraphSnapshot, options?: GraphmlExportOptions & CommonExportOptions): AsyncIterable<Uint8Array> {
        const plan = planExport(snapshot, resolveExportOptions(options), resolveFormatOptions(options));
        return encodeChunks(writeGraphml(snapshot, plan));
    },
    /**
     * Write the document as one string.
     * @param snapshot - the snapshot
     * @param options - format-specific and common options
     * @returns the document
     */
    exportToString(snapshot: GraphSnapshot, options?: GraphmlExportOptions & CommonExportOptions): Promise<string> {
        const plan = planExport(snapshot, resolveExportOptions(options), resolveFormatOptions(options));
        return joinText(writeGraphml(snapshot, plan));
    },
});
