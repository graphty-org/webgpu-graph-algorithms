/**
 * What the JSON importer and exporter share (design sections 8.2 and 8.5): the dialect names, the
 * reserved `meta.extra.json` shape record the importer writes and the exporter reads back, the
 * per-dialect capability tables, the fixed key sets of each dialect and the small JSON value
 * helpers both sides use.
 *
 * Dialects: NetworkX node-link (`nodes` + `links` / `edges`, graph-level `directed` / `multigraph`
 * / `graph`), the d3 lineage of the same shape (`links`, `name` ids, integer index endpoints),
 * JSON Graph Format v2 (`graph.nodes` keyed by id, per-edge `directed`, hyperedges), Cytoscape.js
 * elements (`data.id` / `data.source` / `data.target`, `position`, `classes`, `data.parent`),
 * graphology serialisation (`key` / `attributes`, `undirected` edges, `options.type` / `multi`) and
 * vis.js (`from` / `to`).
 */

import { type GraphMeta } from "@graphty/graph-format";

import { capabilities } from "../../common/export.js";
import { type ExportCapabilities } from "../../types.js";

/** The JSON dialects the plugin reads and writes. */
export type JsonDialect = "node-link" | "d3" | "jgf" | "cytoscape" | "graphology" | "vis";

/** Every dialect name, for option checking and messages. */
export const JSON_DIALECTS: readonly JsonDialect[] = Object.freeze([
    "node-link",
    "d3",
    "jgf",
    "cytoscape",
    "graphology",
    "vis",
]);

/** The key under `meta.extra` that holds the shape record (design section 8.5). */
export const META_KEY = "json";

/**
 * The shape information the importer records under `meta.extra.json` so the exporter can write the
 * same file back (design section 8.5). Every field is optional because a snapshot may come from
 * another format or from an older record.
 */
export interface JsonShapeMeta {
    /** The dialect the file was read as. */
    readonly dialect?: JsonDialect | undefined;
    /** node-link / d3: the key the edge array was under ("links" or "edges"). */
    readonly edgesKey?: string | undefined;
    /** node-link / d3: the key holding the node id ("id", "name", ...); null when nodes were positional. */
    readonly nodeIdKey?: string | null | undefined;
    /** node-link / d3: whether edge endpoints were array positions rather than ids. */
    readonly indexLinks?: boolean | undefined;
    /** node-link / d3 / vis: the endpoint keys the file used. */
    readonly sourceKey?: string | undefined;
    /** node-link / d3 / vis: the endpoint keys the file used. */
    readonly targetKey?: string | undefined;
    /** jgf: the graph `id`. */
    readonly id?: string | undefined;
    /** jgf: the graph `type`. */
    readonly type?: string | undefined;
    /** graphology: `options.allowSelfLoops` as declared. */
    readonly allowSelfLoops?: boolean | undefined;
    /** cytoscape: every top-level key besides `elements` and `data` (style, zoom, pan, ...), verbatim. */
    readonly cytoscape?: Readonly<Record<string, unknown>> | undefined;
}

/**
 * Whether a value is a plain JSON object (not null, not an array).
 * @param value - any value
 * @returns true for an object that is not an array
 */
export function isJsonObject(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Whether a record has an own property; JSON.parse output is checked with hasOwnProperty so a key
 * named "constructor" or "__proto__" is never read from the prototype.
 * @param record - the record
 * @param key - the key
 * @returns true when the key is an own property
 */
export function hasKey(record: Readonly<Record<string, unknown>>, key: string): boolean {
    return Object.prototype.hasOwnProperty.call(record, key);
}

/**
 * Whether a text names a dialect.
 * @param value - any value
 * @returns true for one of JSON_DIALECTS
 */
export function isJsonDialect(value: unknown): value is JsonDialect {
    return typeof value === "string" && (JSON_DIALECTS as readonly string[]).includes(value);
}

/**
 * The dialect of a parsed JSON document, by the shape rules of design section 8.2 (Cytoscape:
 * `elements` or a top-level array of `{ data }` elements; JGF: `graph.nodes` / `graph.edges` or
 * `graphs[]`; graphology: `options.type` / `options.multi`, `key` nodes without `id`, edges with
 * `undirected` or an `attributes` record; vis: edges with `from` / `to`; d3: `links` without
 * `directed` / `multigraph` / `graph`; else node-link). Pure: the importer wraps it with its issue
 * codes, the registry's sniff() uses it on a head that parses as a whole document.
 * @param root - the parsed document
 * @returns the dialect, or null when the document is not a graph document in any dialect
 */
export function sniffJsonDialect(root: unknown): JsonDialect | null {
    if (Array.isArray(root)) {
        const first = firstJsonObject(root);
        return root.length === 0 || (first !== null && isJsonObject(first.data)) ? "cytoscape" : null;
    }
    if (!isJsonObject(root)) {
        return null;
    }
    if (hasKey(root, "elements")) {
        return "cytoscape";
    }
    if (isJsonObject(root.graph) && (hasKey(root.graph, "nodes") || hasKey(root.graph, "edges"))) {
        return "jgf";
    }
    if (Array.isArray(root.graphs)) {
        return "jgf";
    }
    if (!hasKey(root, "nodes") && !hasKey(root, "edges") && !hasKey(root, "links")) {
        return null;
    }
    const firstNode = firstJsonObject(root.nodes);
    const firstEdge = firstJsonObject(hasKey(root, "edges") ? root.edges : root.links);
    if (isJsonObject(root.options) && (hasKey(root.options, "type") || hasKey(root.options, "multi"))) {
        return "graphology";
    }
    if (firstNode !== null && hasKey(firstNode, "key") && !hasKey(firstNode, "id")) {
        return "graphology";
    }
    if (firstEdge !== null && !hasKey(firstEdge, "from")) {
        if (hasKey(firstEdge, "undirected") || isJsonObject(firstEdge.attributes)) {
            return "graphology";
        }
    }
    if (firstEdge !== null && hasKey(firstEdge, "from") && hasKey(firstEdge, "to") && !hasKey(firstEdge, "source")) {
        return "vis";
    }
    const bare = !hasKey(root, "directed") && !hasKey(root, "multigraph") && !hasKey(root, "graph");
    return bare && hasKey(root, "links") ? "d3" : "node-link";
}

/**
 * The first object of an array, or null.
 * @param value - maybe an array
 * @returns the first element that is an object
 */
function firstJsonObject(value: unknown): Record<string, unknown> | null {
    if (!Array.isArray(value)) {
        return null;
    }
    const found: unknown = value.find((item) => isJsonObject(item));
    return isJsonObject(found) ? found : null;
}

/**
 * The shape record of a snapshot's metadata, or an empty record when the snapshot did not come
 * from the JSON importer.
 * @param meta - the snapshot's metadata
 * @returns the fields found under `meta.extra.json`, each only when it has the expected type
 */
export function shapeMetaOf(meta: GraphMeta): JsonShapeMeta {
    const raw: unknown = meta.extra[META_KEY];
    if (!isJsonObject(raw)) {
        return {};
    }
    const text = (key: string): string | undefined => (typeof raw[key] === "string" ? raw[key] : undefined);
    const flag = (key: string): boolean | undefined => (typeof raw[key] === "boolean" ? raw[key] : undefined);
    const { nodeIdKey } = raw;
    return {
        dialect: isJsonDialect(raw.dialect) ? raw.dialect : undefined,
        edgesKey: text("edgesKey"),
        nodeIdKey: nodeIdKey === null || typeof nodeIdKey === "string" ? nodeIdKey : undefined,
        indexLinks: flag("indexLinks"),
        sourceKey: text("sourceKey"),
        targetKey: text("targetKey"),
        id: text("id"),
        type: text("type"),
        allowSelfLoops: flag("allowSelfLoops"),
        cytoscape: isJsonObject(raw.cytoscape) ? raw.cytoscape : undefined,
    };
}

/** The direction a dialect assumes when the file declares none (design section 8.4, `defaultDirected`). */
export const DIALECT_DEFAULT_DIRECTED: Readonly<Record<JsonDialect, boolean>> = Object.freeze({
    "node-link": false,
    d3: false,
    jgf: true,
    cytoscape: true,
    graphology: true,
    vis: false,
});

/**
 * The dtypes a JSON dialect keeps as declared: JSON declares no types, so a value reads back as
 * what the importer's inference gives it (design section 5.1: integral numbers i32, other numbers
 * f64, booleans, strings). f32 values re-read as f64 differ from the f32 ones, u32 / u8 come back
 * i32, dict comes back string, a multi-component or list value comes back json.
 */
const JSON_DTYPES = Object.freeze(["f64", "i32", "bool", "string"] as const);

const COMMON = {
    multiEdges: true,
    selfLoops: true,
    idCharset: "any",
    dtypes: JSON_DTYPES,
    components: false,
    lists: false,
    json: true,
    defaults: false,
    options: false,
    temporal: "none",
    positions: false,
    viz: false,
} as const;

const TABLES: Readonly<Record<JsonDialect, ExportCapabilities>> = Object.freeze({
    "node-link": capabilities({
        ...COMMON,
        mixedDirection: false,
        edgeIds: "none",
        hierarchy: false,
        graphAttributes: true,
    }),
    d3: capabilities({
        ...COMMON,
        mixedDirection: false,
        edgeIds: "none",
        hierarchy: false,
        graphAttributes: false,
    }),
    jgf: capabilities({
        ...COMMON,
        mixedDirection: true,
        edgeIds: "optional",
        hierarchy: false,
        graphAttributes: true,
    }),
    cytoscape: capabilities({
        ...COMMON,
        positions: true,
        mixedDirection: false,
        edgeIds: "required",
        hierarchy: true,
        graphAttributes: true,
    }),
    graphology: capabilities({
        ...COMMON,
        mixedDirection: true,
        edgeIds: "optional",
        hierarchy: false,
        graphAttributes: true,
    }),
    vis: capabilities({
        ...COMMON,
        mixedDirection: false,
        edgeIds: "optional",
        hierarchy: false,
        graphAttributes: false,
    }),
});

/**
 * What one dialect can express (design section 8.5): positions and visual columns are written as
 * plain attributes (an array for a multi-component column) and read back without their role,
 * except Cytoscape's `position` object; containment only as the Cytoscape `data.parent`; mixed
 * direction only where the dialect carries a per-edge flag (JGF `directed`, graphology
 * `undirected`); edge ids where the dialect has a slot (not node-link / d3); nothing temporal;
 * graph attributes everywhere but d3 (the bare shape) and vis.
 * @param dialect - the dialect
 * @returns its frozen capability table
 */
export function dialectCapabilities(dialect: JsonDialect): ExportCapabilities {
    return TABLES[dialect];
}

/** The Cytoscape element-level keys (everything else on an element lives under `data`). */
export const CYTOSCAPE_ELEMENT_KEYS: ReadonlySet<string> = new Set([
    "selected",
    "selectable",
    "locked",
    "grabbable",
    "pannable",
    "removed",
    "scratch",
    "renderedPosition",
]);

/** The Cytoscape element keys the importer maps structurally rather than to columns. */
export const CYTOSCAPE_STRUCTURAL_KEYS: ReadonlySet<string> = new Set(["data", "group", "position", "classes"]);

/**
 * The suffix the importer appends to a column name that collides with a structural column or a
 * reserved key of its dialect (design section 5.6 names collisions deterministically); the exporter
 * strips it when the value goes back to the level the suffix names.
 */
export const SUFFIX = Object.freeze({
    /** A Cytoscape `data` key or a JGF / graphology attribute that collides with a structural column. */
    data: "#data",
    /** An element-level key the dialect does not define (Cytoscape, JGF, graphology). */
    element: "#element",
});

/** The default source keys of a node-link edge record, tried in order (as fromRecords does). */
export const NODE_LINK_SOURCE_KEYS: readonly string[] = Object.freeze(["source", "src", "from"]);

/** The default target keys of a node-link edge record, tried in order. */
export const NODE_LINK_TARGET_KEYS: readonly string[] = Object.freeze(["target", "dst", "to"]);

/** The name of the position column every JSON dialect writes / reads (design section 5.2). */
export const POSITION_COLUMN = "position";

/** The name of the Cytoscape classes column (a list of strings with role "classes"). */
export const CLASSES_COLUMN = "classes";

/** The name of the Cytoscape parent column (u32 refersTo node, role "parent"). */
export const PARENT_COLUMN = "parent";
