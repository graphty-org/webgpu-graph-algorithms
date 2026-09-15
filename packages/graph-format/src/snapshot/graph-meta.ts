/**
 * The one resolver of graph-level metadata patches (design section 5.9), shared by the builder's
 * `setMeta` and by `fromCsr`: a GraphMetaPatch merged over the current metadata with the type rules
 * of every field -- strings, enums, a string list for keywords, a boolean or null for
 * declaredMultigraph, a ColumnOrigin for weightOrigin and a JSON object for extra. A field that is
 * `undefined` in the patch keeps the current value; `null` clears it. Every violation is
 * E_COLUMN_TYPE with `details.field`.
 */

import { resolveColumnMeta } from "../columns/column.js";
import { GraphFormatError } from "../errors.js";
import { type ColumnOrigin, type GraphMeta, type GraphMetaPatch } from "../types/index.js";

const ID_TYPES: ReadonlySet<"string" | "integer" | "mixed"> = new Set(["string", "integer", "mixed"] as const);
const TIME_FORMATS: ReadonlySet<"integer" | "double" | "date" | "dateTime"> = new Set([
    "integer",
    "double",
    "date",
    "dateTime",
] as const);
const TIME_REPRESENTATIONS: ReadonlySet<"interval" | "timestamp"> = new Set(["interval", "timestamp"] as const);
const MODES: ReadonlySet<"static" | "dynamic" | "slice"> = new Set(["static", "dynamic", "slice"] as const);

/**
 * A GraphMeta with every field null / empty (design section 5.9), for producers that have no
 * metadata and for tests.
 */
export const EMPTY_GRAPH_META: GraphMeta = Object.freeze({
    name: null,
    description: null,
    creator: null,
    created: null,
    modified: null,
    keywords: Object.freeze([]),
    sourceFormat: null,
    sourceVersion: null,
    idType: null,
    timeFormat: null,
    timeRepresentation: null,
    mode: null,
    declaredMultigraph: null,
    weightOrigin: null,
    extra: Object.freeze({}),
});

/**
 * The E_COLUMN_TYPE error for a metadata field of the wrong type.
 * @param field - the field
 * @param found - the value
 * @returns the error
 */
function metaError(field: string, found: unknown): GraphFormatError {
    return new GraphFormatError("E_COLUMN_TYPE", `meta.${field} has an unsupported value`, {
        field,
        found: typeof found,
    });
}

/**
 * An optional-string metadata field (the value is never undefined here).
 * @param field - the field name
 * @param value - the value
 * @returns the string or null
 */
function metaString(field: string, value: unknown): string | null {
    if (value === null || typeof value === "string") {
        return value;
    }
    throw metaError(field, value);
}

/**
 * An enum-valued metadata field.
 * @param field - the field name
 * @param value - the value
 * @param allowed - the legal strings
 * @returns the value or null
 */
function metaEnum<T extends string>(field: string, value: unknown, allowed: ReadonlySet<T>): T | null {
    if (value === null) {
        return null;
    }
    if (typeof value === "string" && allowed.has(value as T)) {
        return value as T;
    }
    throw metaError(field, value);
}

/**
 * The weightOrigin field as a complete ColumnOrigin.
 * @param value - the value
 * @returns the origin or null
 */
function metaOrigin(value: unknown): ColumnOrigin | null {
    if (value === null) {
        return null;
    }
    if (typeof value !== "object") {
        throw metaError("weightOrigin", value);
    }
    const input = value as Record<string, unknown>;
    const field = (name: string): string | null => {
        const v = input[name];
        if (v === undefined || v === null) {
            return null;
        }
        if (typeof v !== "string") {
            throw metaError(`weightOrigin.${name}`, v);
        }
        return v;
    };
    return Object.freeze({
        format: field("format"),
        id: field("id"),
        title: field("title"),
        type: field("type"),
        namespace: field("namespace"),
    });
}

/**
 * Merge a metadata patch over the current metadata (design section 5.9): undefined fields keep the
 * current value, every other field is type-checked (E_COLUMN_TYPE with details.field).
 * @param current - the metadata before the patch
 * @param patch - the fields to set
 * @returns the merged metadata, frozen
 */
export function resolveGraphMeta(current: GraphMeta, patch: GraphMetaPatch): GraphMeta {
    const pick = <K extends keyof GraphMeta>(key: K, convert: (value: unknown) => GraphMeta[K]): GraphMeta[K] => {
        const value = patch[key];
        return value === undefined ? current[key] : convert(value);
    };
    return Object.freeze({
        name: pick("name", (v) => metaString("name", v)),
        description: pick("description", (v) => metaString("description", v)),
        creator: pick("creator", (v) => metaString("creator", v)),
        created: pick("created", (v) => metaString("created", v)),
        modified: pick("modified", (v) => metaString("modified", v)),
        keywords: pick("keywords", (v) => {
            if (!Array.isArray(v) || v.some((k) => typeof k !== "string")) {
                throw metaError("keywords", v);
            }
            return Object.freeze([...(v as string[])]);
        }),
        sourceFormat: pick("sourceFormat", (v) => metaString("sourceFormat", v)),
        sourceVersion: pick("sourceVersion", (v) => metaString("sourceVersion", v)),
        idType: pick("idType", (v) => metaEnum("idType", v, ID_TYPES)),
        timeFormat: pick("timeFormat", (v) => metaEnum("timeFormat", v, TIME_FORMATS)),
        timeRepresentation: pick("timeRepresentation", (v) => metaEnum("timeRepresentation", v, TIME_REPRESENTATIONS)),
        mode: pick("mode", (v) => metaEnum("mode", v, MODES)),
        declaredMultigraph: pick("declaredMultigraph", (v) => {
            if (v === null || typeof v === "boolean") {
                return v;
            }
            throw metaError("declaredMultigraph", v);
        }),
        weightOrigin: pick("weightOrigin", (v) => metaOrigin(v)),
        extra: pick("extra", (v) => {
            // the column resolver applies the JSON rules of design section 5.9 to `extra`
            const resolved = resolveColumnMeta("meta", "graph", {
                dtype: "json",
                extra: v as Readonly<Record<string, unknown>>,
            });
            return resolved.extra;
        }),
    });
}
