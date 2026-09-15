/**
 * The lexical layer of the Pajek NET format shared by the importer and the exporter (research
 * note 07 section 2.5): the line tokenizer (whitespace-separated tokens, double quotes group
 * spaces and are removed, no escape mechanism), the section headers (`*Vertices N [N1]`,
 * `*Arcs [:k ["name"]]`, `*Edges`, `*Arcslist`, `*Edgeslist`, `*Matrix`, `*Network name` and the
 * project-file sections the importer does not read), the vertex shape keywords, the time interval
 * tokens `[1-5,7-*]` that map to the spells role, and the column names both sides agree on.
 */

/** The node column holding the vertex label (role label). */
export const LABEL_COLUMN = "label";

/** The node column holding the vertex coordinates (role position, f32 x 3, units "file"). */
export const POSITION_COLUMN = "position";

/** The node column holding the vertex shape keyword (dict, no role: Pajek has its own keyword set). */
export const SHAPE_COLUMN = "shape";

/** The edge column holding the relation name of a `*Arcs :k "name"` section (dict). */
export const RELATION_COLUMN = "relation";

/** The node or edge column holding Pajek time intervals (list of f64 pairs, role spells). */
export const SPELLS_COLUMN = "spells";

/** The edge column holding the line value when `weightFrom` names another field or null. */
export const VALUE_COLUMN = "value";

/** The `weightFrom` default: Pajek's third column is the line value (design section 8.4). */
export const VALUE_FIELD = "value";

/** The vertex shape keywords of the Pajek manual. */
export const SHAPES: ReadonlySet<string> = new Set(["ellipse", "box", "diamond", "triangle", "cross", "empty"]);

/** The section keywords the importer reads, lower-cased. */
type SectionKind = "network" | "vertices" | "arcs" | "edges" | "arcslist" | "edgeslist" | "matrix" | "unsupported";

/**
 * The parameter key the exporter writes a node's original id under when `sanitizeIds: "mangle"`
 * renumbers it (design section 8.5: the exporter names the attribute, `restoreMangledIds` reads
 * it back); a user column of that name is reserved under "mangle".
 */
export const ORIGINAL_ID_KEY = "graphty_originalId";

/** A parsed section header line. */
export interface SectionHeader {
    /** The section kind. */
    readonly kind: SectionKind;
    /** The keyword as written, without the asterisk. */
    readonly keyword: string;
    /** The first count (`*Vertices N`), or null. */
    readonly count: number | null;
    /** The second count of a two-mode network (`*Vertices N N1`), or null. */
    readonly secondCount: number | null;
    /** The relation number of `*Arcs :k`, or null. */
    readonly relation: number | null;
    /** The relation or network name, or null. */
    readonly name: string | null;
    /** Tokens the grammar does not account for, for the importer to report. */
    readonly extra: readonly string[];
}

const SECTION_KINDS: ReadonlyMap<string, SectionKind> = new Map([
    ["network", "network"],
    ["vertices", "vertices"],
    ["arcs", "arcs"],
    ["edges", "edges"],
    ["arcslist", "arcslist"],
    ["edgeslist", "edgeslist"],
    ["matrix", "matrix"],
]);

const INTEGER_TEXT = /^[+-]?[0-9]+$/;
const TIME_POINT =
    /^(\*|[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)(?:-(\*|[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?))?$/;

/**
 * Split one line into tokens: runs of non-whitespace, with double quotes grouping whitespace
 * into one token and removed from it (the shlex rule NetworkX applies; Pajek has no escapes, so a
 * quote never appears inside a token). An empty quoted string `""` is one empty token.
 * @param line - the line without its terminator
 * @returns the tokens, or null when a quote is not closed before the end of the line
 */
export function tokenize(line: string): string[] | null {
    const tokens: string[] = [];
    let current = "";
    let started = false;
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
        const c = line.charCodeAt(i);
        if (c === 34) {
            quoted = !quoted;
            started = true;
            continue;
        }
        if (!quoted && (c === 32 || c === 9 || c === 13 || c === 12 || c === 11)) {
            if (started) {
                tokens.push(current);
                current = "";
                started = false;
            }
            continue;
        }
        current += line[i];
        started = true;
    }
    if (quoted) {
        return null;
    }
    if (started) {
        tokens.push(current);
    }
    return tokens;
}

/**
 * Whether a line is a Pajek comment (`%` first) or blank.
 * @param line - the line
 * @returns true when the importer skips it
 */
export function isCommentOrBlank(line: string): boolean {
    for (let i = 0; i < line.length; i++) {
        const c = line.charCodeAt(i);
        if (c === 32 || c === 9 || c === 13 || c === 12 || c === 11) {
            continue;
        }
        return c === 37;
    }
    return true;
}

/**
 * Whether a line starts a section (`*` first, after optional whitespace).
 * @param line - the line
 * @returns true for a header line
 */
export function isSectionLine(line: string): boolean {
    for (let i = 0; i < line.length; i++) {
        const c = line.charCodeAt(i);
        if (c === 32 || c === 9 || c === 13 || c === 12 || c === 11) {
            continue;
        }
        return c === 42;
    }
    return false;
}

/**
 * Parse a section header line: the keyword (case-insensitive), then `N [N1]` for `*Vertices`,
 * `[:k] ["name"]` for the line sections, the name for `*Network`. Anything else after the keyword
 * lands in `extra` so the importer can report it instead of dropping it.
 * @param line - a line isSectionLine() accepted
 * @returns the header, or null when the line has no keyword or a quote is unbalanced
 */
export function parseSectionHeader(line: string): SectionHeader | null {
    const tokens = tokenize(line.trimStart().slice(1));
    if (tokens === null || tokens.length === 0 || tokens[0].length === 0) {
        return null;
    }
    const keyword = tokens[0];
    const kind = SECTION_KINDS.get(keyword.toLowerCase()) ?? "unsupported";
    let count: number | null = null;
    let secondCount: number | null = null;
    let relation: number | null = null;
    let name: string | null = null;
    const extra: string[] = [];
    let i = 1;
    switch (kind) {
        case "vertices":
            if (i < tokens.length && INTEGER_TEXT.test(tokens[i])) {
                count = Number(tokens[i]);
                i++;
                if (i < tokens.length && INTEGER_TEXT.test(tokens[i])) {
                    secondCount = Number(tokens[i]);
                    i++;
                }
            }
            break;
        case "arcs":
        case "edges":
        case "arcslist":
        case "edgeslist":
        case "matrix":
            if (i < tokens.length && tokens[i].startsWith(":") && INTEGER_TEXT.test(tokens[i].slice(1))) {
                relation = Number(tokens[i].slice(1));
                i++;
                if (i < tokens.length) {
                    name = tokens[i];
                    i++;
                }
            }
            break;
        case "network":
            if (i < tokens.length) {
                name = tokens.slice(i).join(" ");
                i = tokens.length;
            }
            break;
        case "unsupported":
            i = tokens.length;
            break;
        default: {
            const unknown: string = kind;
            throw new Error(`unknown section kind ${unknown}`);
        }
    }
    for (; i < tokens.length; i++) {
        extra.push(tokens[i]);
    }
    return { kind, keyword, count, secondCount, relation, name, extra };
}

/**
 * Whether a token is a Pajek integer (a vertex number, a count), without sign.
 * @param token - the token
 * @returns true for one or more ASCII digits
 */
export function isVertexNumber(token: string): boolean {
    if (token.length === 0) {
        return false;
    }
    for (let i = 0; i < token.length; i++) {
        const c = token.charCodeAt(i);
        if (c < 48 || c > 57) {
            return false;
        }
    }
    return true;
}

/**
 * Whether a token is a time interval list `[...]`.
 * @param token - the token
 * @returns true when it starts with `[` and ends with `]`
 */
export function isIntervalToken(token: string): boolean {
    return token.length >= 2 && token.startsWith("[") && token.endsWith("]");
}

/**
 * Parse a Pajek time interval token into spells: `[1-5,7-*]` is `[[1, 5], [7, Infinity]]`, a
 * single time point `[3]` is `[[3, 3]]`, and `*` at either end is the corresponding infinity.
 * @param token - a token isIntervalToken() accepted
 * @returns the spells as [start, end] pairs
 */
export function parseIntervals(token: string): [number, number][] {
    const body = token.slice(1, -1);
    if (body.trim().length === 0) {
        throw new Error(`empty time interval ${token}`);
    }
    const spells: [number, number][] = [];
    for (const part of body.split(",")) {
        const match = TIME_POINT.exec(part.trim());
        if (match === null) {
            throw new Error(`malformed time interval ${token}: "${part}" is not a-b, a-* or a`);
        }
        const start = match[1] === "*" ? -Infinity : Number(match[1]);
        const endText = match[2];
        let end: number;
        if (endText === undefined) {
            end = start;
        } else if (endText === "*") {
            end = Infinity;
        } else {
            end = Number(endText);
        }
        if (start > end) {
            throw new Error(`malformed time interval ${token}: ${match[1]} is after ${endText ?? ""}`);
        }
        spells.push([start, end]);
    }
    return spells;
}

/**
 * Write spells as a Pajek time interval token, the inverse of parseIntervals().
 * @param spells - [start, end] pairs
 * @returns the token, e.g. `[1-5,7-*]`
 */
export function formatIntervals(spells: readonly (readonly [number, number])[]): string {
    const parts: string[] = [];
    for (const [start, end] of spells) {
        const s = start === -Infinity ? "*" : String(start);
        if (start === end) {
            parts.push(s);
        } else {
            parts.push(`${s}-${end === Infinity ? "*" : String(end)}`);
        }
    }
    return `[${parts.join(",")}]`;
}

/**
 * Whether a text can be written as a bare parameter key on a vertex or line row: non-empty, no
 * whitespace or quote, not an interval token, not a number (a number after the label is a
 * coordinate, at the third position of a line row the value) and not a shape keyword.
 * @param text - the candidate key (a column name)
 * @returns true when the importer reads it back as a key
 */
export function isParameterKey(text: string): boolean {
    if (text.length === 0 || /[\s"]/.test(text) || text.startsWith("[") || text.startsWith("*")) {
        return false;
    }
    if (SHAPES.has(text)) {
        return false;
    }
    return !/^[+-]?(\.[0-9]|[0-9])/.test(text);
}
