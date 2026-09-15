/**
 * Format detection (design sections 8.2 and 13.1: the registry's `sniff()`, the successor of
 * graphty-element's `format-detection.ts`): which of the registered importers a file is for, from
 * any combination of a filename (its extension), a MIME type and the first bytes of the content.
 *
 * Every importer declares `extensions`, `mimeTypes` and a `sniff(head)` content confidence in
 * 0..1 (design section 8.4); this module combines them into one ranked answer so that content
 * always beats a hint: a `.csv` whose first line is a neo4j-admin header is Neo4j, a `.xml` is
 * GEXF or GraphML by its root element, a `.txt` holding `graph [` is GML. The confidence of a
 * candidate is
 *
 * - `0.5 + 0.35 * content + 0.1 * [extension matches] + 0.05 * [MIME type matches]` when the
 *   importer recognises the content (`content > 0`), so a content match always scores at least
 *   0.5 and at most 1;
 * - `0.3 * [extension matches] + 0.1 * [MIME type matches]` when the content is absent or the
 *   importer rejects it, so a hint alone never reaches 0.5;
 * - nothing (the format is not a candidate) otherwise.
 *
 * Ties are broken by the order the importers were registered in, which is why the default
 * registry lists the common formats first. The JSON dialect is sniffed from the head as a hint
 * (the importer's detection on the parsed document is authoritative, design section 8.2).
 */

import { type JsonDialect, sniffJsonDialect } from "./formats/json/dialect.js";
import { type GraphImporter } from "./types.js";

/** The format names of the eight built-in importers and exporters. */
export type GraphFormatName = "gexf" | "graphml" | "gml" | "dot" | "pajek" | "csv" | "json" | "neo4j";

/**
 * The built-in format names in the default registry's order, which is also the tie-break order of
 * sniffing: the more common format wins an extension two formats claim (`.xml` GraphML before GEXF,
 * `.csv` CSV before Neo4j) when the content does not decide.
 */
export const GRAPH_FORMATS: readonly GraphFormatName[] = Object.freeze([
    "json",
    "graphml",
    "gexf",
    "csv",
    "gml",
    "dot",
    "pajek",
    "neo4j",
]);

/** How many bytes of the input the sniffers look at; the registry reads no more than this before deciding. */
export const SNIFF_HEAD_BYTES = 8192;

/** What is known about an input before it is read. */
export interface SniffHints {
    /** A file name or path; only its extension is used. */
    readonly filename?: string | null | undefined;
    /** A MIME type, with or without parameters (`text/csv; charset=utf-8`). */
    readonly mimeType?: string | null | undefined;
    /** The first bytes (or characters) of the content. */
    readonly head?: Uint8Array | string | null | undefined;
}

/** One ranked candidate of a sniff. */
export interface SniffResult {
    /** The importer's format name. */
    readonly format: string;
    /** The combined confidence in 0..1 (at least 0.5 when the content was recognised). */
    readonly confidence: number;
    /** The importer's own content confidence, 0 when no head was given or it rejected the head. */
    readonly content: number;
    /** Whether the filename's extension is one the importer claims. */
    readonly extension: boolean;
    /** Whether the MIME type is one the importer claims. */
    readonly mimeType: boolean;
    /** For the JSON format: the dialect the head suggests, or null when unknown; always null for other formats. */
    readonly dialect: JsonDialect | null;
}

/**
 * The lower-cased extension of a file name or path, with its dot.
 * @param filename - the name or path
 * @returns the extension (`.gexf`), or null when the name has none
 */
export function extensionOf(filename: string): string | null {
    const base = filename.slice(Math.max(filename.lastIndexOf("/"), filename.lastIndexOf("\\")) + 1);
    const dot = base.lastIndexOf(".");
    if (dot <= 0 || dot === base.length - 1) {
        return null;
    }
    return base.slice(dot).toLowerCase();
}

/**
 * A MIME type without parameters, lower-cased, for comparison with an importer's list.
 * @param mimeType - the type as received (`Text/CSV; charset=utf-8`)
 * @returns the bare type (`text/csv`)
 */
export function normalizeMimeType(mimeType: string): string {
    const semicolon = mimeType.indexOf(";");
    return (semicolon < 0 ? mimeType : mimeType.slice(0, semicolon)).trim().toLowerCase();
}

/**
 * The head as bytes for the importers' sniff functions: at most SNIFF_HEAD_BYTES, a string
 * encoded as UTF-8.
 * @param head - the head as given
 * @returns the bytes
 */
export function headBytes(head: Uint8Array | string): Uint8Array {
    if (typeof head === "string") {
        return new TextEncoder().encode(head.slice(0, SNIFF_HEAD_BYTES)).subarray(0, SNIFF_HEAD_BYTES);
    }
    return head.byteLength > SNIFF_HEAD_BYTES ? head.subarray(0, SNIFF_HEAD_BYTES) : head;
}

/**
 * Rank the registered importers for an input by the rule of the module comment: every importer
 * whose content confidence is positive or whose extension / MIME type matches is a candidate,
 * ordered by confidence (ties in registration order).
 * @param hints - what is known about the input
 * @param importers - the registered importers, in registration order
 * @returns the candidates, best first; empty when nothing matches
 */
export function rankFormats(hints: SniffHints, importers: Iterable<GraphImporter>): readonly SniffResult[] {
    const extension = typeof hints.filename === "string" ? extensionOf(hints.filename) : null;
    const mime = typeof hints.mimeType === "string" ? normalizeMimeType(hints.mimeType) : null;
    const head = hints.head === undefined || hints.head === null ? null : headBytes(hints.head);
    const candidates: { readonly result: SniffResult; readonly rank: number }[] = [];
    let rank = 0;
    for (const importer of importers) {
        const extensionMatch = extension !== null && importer.extensions.some((e) => e.toLowerCase() === extension);
        const mimeMatch = mime !== null && importer.mimeTypes.some((m) => m.toLowerCase() === mime);
        let content = 0;
        if (head !== null && head.byteLength > 0 && typeof importer.sniff === "function") {
            content = clamp(importer.sniff(head));
        }
        let confidence: number;
        if (content > 0) {
            confidence = 0.5 + 0.35 * content + (extensionMatch ? 0.1 : 0) + (mimeMatch ? 0.05 : 0);
        } else if (extensionMatch || mimeMatch) {
            confidence = (extensionMatch ? 0.3 : 0) + (mimeMatch ? 0.1 : 0);
        } else {
            continue;
        }
        const dialect = importer.format === "json" && head !== null ? sniffJsonDialectHead(head) : null;
        const result: SniffResult = Object.freeze({
            format: importer.format,
            confidence: Math.min(1, confidence),
            content,
            extension: extensionMatch,
            mimeType: mimeMatch,
            dialect,
        });
        candidates.push({ result, rank: rank++ });
    }
    candidates.sort((a, b) => b.result.confidence - a.result.confidence || a.rank - b.rank);
    return candidates.map((c) => c.result);
}

/**
 * The best candidate of rankFormats(), or null when no importer matches.
 * @param hints - what is known about the input
 * @param importers - the registered importers, in registration order
 * @returns the best candidate, or null
 */
export function sniffFormat(hints: SniffHints, importers: Iterable<GraphImporter>): SniffResult | null {
    const ranked = rankFormats(hints, importers);
    return ranked.length > 0 ? ranked[0] : null;
}

/**
 * The JSON dialect a head suggests. A head that parses as a whole document is classified exactly
 * (sniffJsonDialect); a truncated head is scanned for its top-level keys, the keys of `graph` and
 * `options`, and the keys of the first object under `nodes` / `edges` / `links`, and the same
 * rule is applied to that skeleton. The result is a hint: the importer decides on the parsed
 * document.
 * @param head - the first bytes or characters of the document
 * @returns the dialect, or null when the head is not a JSON graph document
 */
export function sniffJsonDialectHead(head: Uint8Array | string): JsonDialect | null {
    const text = typeof head === "string" ? head : new TextDecoder("utf-8", { fatal: false }).decode(head);
    const body = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
    const trimmed = body.trimStart();
    if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) {
        return null;
    }
    let root: unknown;
    try {
        root = JSON.parse(trimmed);
    } catch {
        root = skeletonOf(trimmed);
    }
    return sniffJsonDialect(root);
}

/** Where a key was seen while scanning a truncated head. */
type KeyPath = "" | "graph" | "options" | "nodes[0]" | "edges[0]" | "links[0]";

/**
 * A partial document rebuilt from the keys a truncated head reveals: every key gets a placeholder
 * value of the shape the dialect rule tests for (an object for `graph`, `options`, `attributes`
 * and `data`, an array holding the first element's skeleton for `nodes` / `edges` / `links` /
 * `graphs`, `true` otherwise), so sniffJsonDialect() can classify it.
 * @param text - the head, starting with `{` or `[`
 * @returns the skeleton, or null when the scan finds no key at all
 */
function skeletonOf(text: string): unknown {
    const keys = scanKeys(text);
    if (text.startsWith("[")) {
        // a cut array whose first element has not been seen tells nothing (a whole `[]` is Cytoscape)
        const first = keys.get("nodes[0]");
        return first === undefined ? null : [objectOf(first)];
    }
    const rootKeys = keys.get("");
    if (rootKeys === undefined) {
        return null;
    }
    const root: Record<string, unknown> = {};
    for (const key of rootKeys) {
        switch (key) {
            case "graph":
            case "options":
                root[key] = objectOf(keys.get(key));
                break;
            case "graphs":
                root[key] = [];
                break;
            case "nodes":
            case "edges":
            case "links": {
                const first = keys.get(`${key}[0]`);
                root[key] = first === undefined ? [] : [objectOf(first)];
                break;
            }
            default:
                root[key] = true;
                break;
        }
    }
    return root;
}

/**
 * An object holding placeholder values for a key set: `{}` for the keys the rule tests with
 * isJsonObject (`attributes`, `data`), `true` otherwise.
 * @param names - the keys, possibly undefined
 * @returns the object
 */
function objectOf(names: ReadonlySet<string> | undefined): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    if (names !== undefined) {
        for (const name of names) {
            out[name] = name === "attributes" || name === "data" ? {} : true;
        }
    }
    return out;
}

/**
 * Scan a (possibly truncated) JSON text for the keys at the paths the dialect rule reads. The
 * scanner tracks a container stack (the key each object sits under, the index of each array
 * element) and records a key when it is at one of the watched paths; a head cut inside a string
 * or a number simply ends the scan. For a top-level array the first element's keys are recorded
 * under `nodes[0]`.
 * @param text - the head, starting with `{` or `[`
 * @returns key sets by path
 */
function scanKeys(text: string): Map<KeyPath, Set<string>> {
    const found = new Map<KeyPath, Set<string>>();
    // one frame per open container: its kind, the key or index it sits under, and the element index
    const kinds: ("object" | "array")[] = [];
    const labels: string[] = [];
    const counts: number[] = [];
    let pendingKey: string | null = null;
    let expectingKey = false;
    const pathOf = (): KeyPath | null => {
        const depth = kinds.length;
        if (depth === 1) {
            return kinds[0] === "object" ? "" : null;
        }
        if (depth === 2 && kinds[0] === "array" && kinds[1] === "object" && labels[1] === "0") {
            return "nodes[0]";
        }
        if (depth === 2 && kinds[0] === "object" && kinds[1] === "object") {
            return labels[1] === "graph" || labels[1] === "options" ? labels[1] : null;
        }
        if (depth === 3 && kinds[0] === "object" && kinds[1] === "array" && kinds[2] === "object") {
            const section = labels[1];
            if ((section === "nodes" || section === "edges" || section === "links") && labels[2] === "0") {
                return `${section}[0]`;
            }
        }
        return null;
    };
    let i = 0;
    const n = text.length;
    while (i < n) {
        const ch = text[i];
        if (ch === '"') {
            const end = scanString(text, i + 1);
            if (end < 0) {
                break;
            }
            const literal = text.slice(i + 1, end);
            i = end + 1;
            if (expectingKey) {
                pendingKey = decodeKey(literal);
                expectingKey = false;
            }
            continue;
        }
        if (ch === "{" || ch === "[") {
            let label = "";
            if (kinds.length > 0) {
                label = kinds[kinds.length - 1] === "object" ? (pendingKey ?? "") : String(counts[counts.length - 1]);
            }
            kinds.push(ch === "{" ? "object" : "array");
            labels.push(label);
            counts.push(0);
            expectingKey = ch === "{";
            pendingKey = null;
        } else if (ch === "}" || ch === "]") {
            kinds.pop();
            labels.pop();
            counts.pop();
            expectingKey = false;
            pendingKey = null;
        } else if (ch === ":") {
            if (pendingKey !== null) {
                const path = pathOf();
                if (path !== null) {
                    let set = found.get(path);
                    if (set === undefined) {
                        set = new Set();
                        found.set(path, set);
                    }
                    set.add(pendingKey);
                }
            }
        } else if (ch === ",") {
            const top = kinds.length - 1;
            if (top >= 0) {
                counts[top]++;
                expectingKey = kinds[top] === "object";
            }
            pendingKey = null;
        }
        i++;
    }
    return found;
}

/**
 * The index of the closing quote of a JSON string starting after its opening quote.
 * @param text - the text
 * @param from - the index after the opening quote
 * @returns the index of the closing quote, or -1 when the text ends first
 */
function scanString(text: string, from: number): number {
    let i = from;
    while (i < text.length) {
        const ch = text[i];
        if (ch === "\\") {
            i += 2;
            continue;
        }
        if (ch === '"') {
            return i;
        }
        i++;
    }
    return -1;
}

/**
 * A key literal decoded when it holds escapes, kept as written otherwise.
 * @param literal - the text between the quotes
 * @returns the key
 */
function decodeKey(literal: string): string {
    if (!literal.includes("\\")) {
        return literal;
    }
    try {
        return JSON.parse(`"${literal}"`) as string;
    } catch {
        return literal;
    }
}

/**
 * Coerce a sniff confidence into 0..1 (NaN counts as 0).
 * @param value - the importer's answer
 * @returns the clamped value
 */
function clamp(value: number): number {
    if (!(value > 0)) {
        return 0;
    }
    return value > 1 ? 1 : value;
}
