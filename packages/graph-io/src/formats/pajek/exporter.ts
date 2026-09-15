/**
 * The Pajek NET exporter (design section 8.5; research note 07 sections 2.5 and 9): vertices are
 * numbered 1..N in index order (the `dense-1-based` id charset: an id that is not its 1-based
 * index survives as the label only, which check() reports as W_ID_RENUMBERED; under
 * `sanitizeIds: "mangle"` it is also written as a `graphty_originalId` parameter the importer
 * restores under `restoreMangledIds`), the label role
 * column is the label, the position role column the coordinates, a `shape` column the shape
 * keyword, every other writable column a `key value` parameter and the spells role a time
 * interval token. Logical edges are written in index order in runs of `*Arcs` (directed) and
 * `*Edges` (undirected) sections, folding expanded pairs back through the pair role and reading
 * attributes from the primary half, so a mixed file round-trips in its original edge order. The
 * weight is written for edges whose weight was explicit (the role-weight column's validity).
 */

import { type Column, GraphFormatError, type GraphSnapshot, type NodeId } from "@graphty/graph-format";

import { pairFolding } from "../../common/direction.js";
import { isPajekLabel, quotePajekLabel } from "../../common/escape.js";
import { capabilities, checkCapabilities, LOSS, type SanitizedIds, sanitizeIds } from "../../common/export.js";
import { formatDecimal, formatF32, formatF64, formatInteger } from "../../common/format.js";
import { canonicalId } from "../../common/ids.js";
import { resolveExportOptions } from "../../common/options.js";
import { explicitWeights } from "../../common/weights.js";
import { encodeChunks, joinText } from "../../common/writer.js";
import { type CommonExportOptions, type ExportCapabilities, type GraphExporter, type LossNote } from "../../types.js";
import {
    formatIntervals,
    isParameterKey,
    LABEL_COLUMN,
    ORIGINAL_ID_KEY,
    RELATION_COLUMN,
    SHAPE_COLUMN,
    SHAPES,
} from "./syntax.js";

/** The format-specific options of the Pajek exporter. */
export interface PajekExportOptions {
    /**
     * Write a `*Network <name>` header line (the .paj project-file convention) when the snapshot's
     * meta.name is set; default false, since plain .net readers do not expect it.
     */
    networkHeader?: boolean | undefined;
}

/**
 * The LossNote codes of the Pajek exporter's check(): the Pajek-specific ones and, aliased, the
 * shared ones it records itself (`LOSS` holds the rest of the generic pre-flight's codes). A key
 * is the code without its severity and format prefixes.
 */
export const PAJEK_LOSS = Object.freeze({
    /** A label or text value holds a double quote or a line break; export() will throw E_UNSUPPORTED. */
    TEXT: "E_PAJEK_TEXT",
    /** A column whose name cannot be a parameter key (whitespace, a quote, numeric, a shape keyword); skipped. */
    KEY_DROPPED: "W_PAJEK_KEY_DROPPED",
    /** A label role column that is not text; its values are written as text and re-import as string. */
    LABEL_AS_TEXT: "W_PAJEK_LABEL_AS_TEXT",
    /** A non-finite f64 value; written as Infinity / NaN text, which re-imports as string. */
    NONFINITE_AS_TEXT: "W_PAJEK_NONFINITE_AS_TEXT",
    /** A mutual pair; written as one undirected edge, the mark lost. */
    MUTUAL_AS_UNDIRECTED: LOSS.MUTUAL_AS_UNDIRECTED,
    /** A start / end / timestamp role column; Pajek intervals are written from the spells role only. */
    TEMPORAL_DROPPED: LOSS.TEMPORAL,
    /** A position column with a stride other than 2 or 3. */
    POSITION_STRIDE: "W_PAJEK_POSITION_STRIDE",
    /** meta.extra.pajek.firstMode is not a count within 0..N; the two-mode header is not written. */
    FIRST_MODE_DROPPED: "W_PAJEK_FIRST_MODE_DROPPED",
    /** A role column Pajek has no slot for (kind, ...) written as a plain parameter; the role is lost. */
    ROLE_DROPPED: LOSS.ROLE,
    /** A `shape` column with a value outside the shape keywords is written as a parameter (a string on re-import). */
    SHAPE_AS_PARAMETER: "W_PAJEK_SHAPE_AS_PARAMETER",
    /** A vertex line with coordinates, a shape or parameters needs a label: the id text is written and reads back as a label. */
    LABEL_GAINED: "W_PAJEK_LABEL_GAINED",
    /** A role-less node column named `label` reads back with the label role (its values become the vertex labels). */
    ROLE_ASSUMED: LOSS.ROLE_ASSUMED,
    /** Under sanitizeIds "mangle": an original id whose text reads back as the other type under ids "canonical". */
    ID_TEXT_TYPE: LOSS.ID_TEXT_TYPE,
});

/** The roles Pajek has a slot for beyond the structural, position and temporal ones. */
const SLOT_ROLES: ReadonlySet<string> = new Set(["label"]);

/** The names the importer gives the slot columns, for the name-change notes. */
const ROLE_NAMES: Readonly<Record<string, string>> = Object.freeze({ label: "label" });

/** The roles the exporter handles structurally rather than as parameters. */
const STRUCTURAL_ROLES: ReadonlySet<string> = new Set([
    "directed",
    "pair",
    "mutual",
    "weight",
    "timeText",
    "originalId",
]);

/** The roles the generic checker reports for a format without viz, hierarchy or open-interval support. */
const CHECKED_ROLES: ReadonlySet<string> = new Set([
    "color",
    "size",
    "shape",
    "thickness",
    "parent",
    "parents",
    "open",
]);

/** Roles the generic checker lets through under temporal "spells" that Pajek cannot write. */
const DROPPED_TEMPORAL_ROLES: ReadonlySet<string> = new Set(["start", "end", "timestamp", "timestamps"]);

const CAPABILITIES: ExportCapabilities = capabilities({
    mixedDirection: true,
    multiEdges: true,
    selfLoops: true,
    edgeIds: "none",
    idCharset: "dense-1-based",
    dtypes: ["f64", "i32", "bool", "string"],
    temporal: "spells",
    positions: true,
});

/** Where a column's values go on a row. */
type Slot = "label" | "position" | "shape" | "spells" | "relation" | "param" | "skip";

/** A column with its resolved slot, for the write loops. */
interface PlannedColumn {
    readonly column: Column;
    readonly slot: Slot;
}

/**
 * Decide the slot of every column of a table.
 * @param table - the node or edge table
 * @param domain - node or edge
 * @param notes - the note list check() fills; null when writing
 * @returns the plan, in declaration order
 */
function plan(table: Iterable<Column>, domain: "node" | "edge", notes: LossNote[] | null): PlannedColumn[] {
    const planned: PlannedColumn[] = [];
    const rows = (column: Column): number => column.length - column.nullCount;
    for (const column of table) {
        const { meta } = column;
        const { role, name, dtype } = meta;
        if (role === "label" && domain === "node") {
            if (dtype !== "string" && dtype !== "dict" && notes !== null) {
                notes.push(
                    note(
                        PAJEK_LOSS.LABEL_AS_TEXT,
                        `node column "${name}" (label) is ${dtype}; labels are text and re-import as string`,
                        name,
                        rows(column),
                    ),
                );
            }
            planned.push({ column, slot: "label" });
            continue;
        }
        if (role === "position") {
            if (domain === "node" && (meta.components === 2 || meta.components === 3)) {
                planned.push({ column, slot: "position" });
            } else if (notes !== null && domain === "node") {
                notes.push(
                    note(
                        PAJEK_LOSS.POSITION_STRIDE,
                        `node column "${name}" (position) has ${meta.components} components; Pajek coordinates are x y [z]`,
                        name,
                        rows(column),
                    ),
                );
            } else if (notes !== null) {
                notes.push(
                    note(LOSS.POSITIONS, `edge column "${name}" (position) cannot be written`, name, rows(column)),
                );
            }
            continue;
        }
        if (role === "spells") {
            if (
                dtype === "list" &&
                (meta.itemDtype === "f64" || meta.itemDtype === "i32" || meta.itemDtype === "f32")
            ) {
                planned.push({ column, slot: "spells" });
                continue;
            }
            if (notes !== null) {
                notes.push(
                    note(
                        LOSS.SPELLS,
                        `${domain} column "${name}" (spells) is not a list of number pairs; it cannot be written`,
                        name,
                        rows(column),
                    ),
                );
            }
            continue;
        }
        if (role !== null && DROPPED_TEMPORAL_ROLES.has(role)) {
            if (notes !== null) {
                notes.push(
                    note(
                        PAJEK_LOSS.TEMPORAL_DROPPED,
                        `${domain} column "${name}" (${role}) cannot be written; Pajek intervals come from the spells role`,
                        name,
                        rows(column),
                    ),
                );
            }
            continue;
        }
        if (role !== null && (STRUCTURAL_ROLES.has(role) || CHECKED_ROLES.has(role))) {
            // structural columns are folded into the topology; viz and hierarchy roles are reported by the generic checker
            continue;
        }
        if (domain === "edge" && role === "id") {
            continue;
        }
        if (dtype === "list" || dtype === "json") {
            // reported by the generic checker (lists: false, json: false)
            continue;
        }
        if (domain === "node" && name === SHAPE_COLUMN && (dtype === "dict" || dtype === "string")) {
            if (allShapeKeywords(column)) {
                planned.push({ column, slot: "shape" });
                continue;
            }
            // a value outside the keywords: the whole column is a `shape "<text>"` parameter, which the
            // importer reads back as one string column (a change for a dict column only)
            if (notes !== null && dtype === "dict") {
                notes.push(
                    note(
                        PAJEK_LOSS.SHAPE_AS_PARAMETER,
                        `node column "${name}" holds values outside the Pajek shape keywords; it is written as a parameter and reads back as string`,
                        name,
                        rows(column),
                    ),
                );
            }
            planned.push({ column, slot: "param" });
            continue;
        }
        if (domain === "edge" && name === RELATION_COLUMN && (dtype === "dict" || dtype === "string")) {
            planned.push({ column, slot: "relation" });
            continue;
        }
        if (!isParameterKey(name)) {
            if (notes !== null) {
                notes.push(
                    note(
                        PAJEK_LOSS.KEY_DROPPED,
                        `${domain} column "${name}" cannot be a Pajek parameter key; it is not written`,
                        name,
                        rows(column),
                    ),
                );
            }
            continue;
        }
        planned.push({ column, slot: "param" });
    }
    return planned;
}

/**
 * The notes about vertex labels: Pajek's grammar puts the label second, so a vertex line that
 * carries coordinates, a shape, parameters or an interval needs one, and a node without a label
 * value gets its id text written there (it reads back as a label); a role-less column named
 * `label` reads back with the label role.
 * @param snapshot - the snapshot
 * @param nodePlan - the node column plan
 * @param notes - where to record
 */
function labelNotes(snapshot: GraphSnapshot, nodePlan: readonly PlannedColumn[], notes: LossNote[]): void {
    let labels: Column | null = null;
    let carried = 0;
    for (const { column, slot } of nodePlan) {
        if (slot === "label") {
            labels = column;
        } else if (slot !== "skip" && slot !== "relation") {
            carried++;
        }
    }
    const plain = [...snapshot.nodes].find((c) => c.meta.role === null && c.meta.name === LABEL_COLUMN);
    if (plain !== undefined) {
        notes.push(
            note(
                PAJEK_LOSS.ROLE_ASSUMED,
                `node column "${LABEL_COLUMN}" has no role; written as a parameter, it reads back as the vertex label (role label)`,
                LABEL_COLUMN,
                plain.length - plain.nullCount,
            ),
        );
    }
    if (carried === 0) {
        return;
    }
    let gained = 0;
    for (let i = 0; i < snapshot.nodeCount; i++) {
        if (labels === null || !labels.isSet(i)) {
            gained++;
        }
    }
    if (gained > 0) {
        notes.push(
            note(
                PAJEK_LOSS.LABEL_GAINED,
                `${gained} vertex line(s) carry coordinates, a shape or parameters and need a label; the id text is written there and reads back as a label`,
                labels === null ? LABEL_COLUMN : labels.meta.name,
                gained,
            ),
        );
    }
}

/**
 * Whether every set value of a shape column is a Pajek shape keyword.
 * @param column - the shape column (string or dict)
 * @returns true when the shape slot can hold the column
 */
function allShapeKeywords(column: Column): boolean {
    for (let r = 0; r < column.length; r++) {
        if (column.isSet(r) && !SHAPES.has(column.value(r) as string)) {
            return false;
        }
    }
    return true;
}

/**
 * Build a frozen LossNote.
 * @param code - the code
 * @param message - the message
 * @param column - the column, or null
 * @param count - the count, or null
 * @returns the note
 */
function note(code: string, message: string, column: string | null = null, count: number | null = null): LossNote {
    return Object.freeze({ code, message, column, count });
}

/**
 * The text of one numeric cell: integers as digits, f64 with a decimal point guaranteed so the
 * importer's inference keeps the dtype, f32 as the shortest fround-round-trip decimal.
 * @param value - the value
 * @param dtype - the column dtype
 * @returns the text
 */
function numberText(value: number, dtype: "f32" | "f64" | "i32" | "u32" | "u8"): string {
    switch (dtype) {
        case "i32":
        case "u32":
        case "u8":
            return formatInteger(value);
        case "f32":
            return formatF32(value);
        case "f64":
            return formatDecimal(value, "f64");
        default: {
            const name: string = dtype;
            throw new Error(`unknown numeric dtype ${name}`);
        }
    }
}

/**
 * The text of a set cell as a parameter value or label, before quoting.
 * @param column - the column
 * @param row - the row
 * @returns the text
 */
function cellText(column: Column, row: number): string {
    switch (column.dtype) {
        case "f32":
        case "f64":
        case "i32":
        case "u32":
        case "u8": {
            const value = column.value(row);
            if (typeof value === "number") {
                return numberText(value, column.dtype);
            }
            const parts: string[] = [];
            const vector = value as ArrayLike<number>;
            for (let k = 0; k < vector.length; k++) {
                parts.push(numberText(vector[k], column.dtype));
            }
            return parts.join(" ");
        }
        case "bool":
            return column.value(row) === true ? "true" : "false";
        case "dict":
        case "string":
            return column.value(row) as string;
        case "list":
        case "json":
            return JSON.stringify(column.value(row));
        default: {
            const name: string = (column as Column).dtype;
            throw new Error(`unknown dtype ${name}`);
        }
    }
}

/**
 * Count the cells of a plan that cannot be written (text with a quote or line break, non-finite
 * f64) and add the notes.
 * @param planned - the plan
 * @param domain - node or edge
 * @param notes - the note list
 */
function checkCells(planned: readonly PlannedColumn[], domain: "node" | "edge", notes: LossNote[]): void {
    for (const { column, slot } of planned) {
        if (slot === "position" || slot === "spells") {
            continue;
        }
        const { name } = column.meta;
        let badText = 0;
        let nonFinite = 0;
        for (let r = 0; r < column.length; r++) {
            if (!column.isSet(r)) {
                continue;
            }
            if (column.dtype === "f64" || column.dtype === "f32") {
                const value = column.value(r);
                if (typeof value === "number") {
                    if (!Number.isFinite(value)) {
                        nonFinite++;
                    }
                } else if (value !== undefined && !Array.from(value).every((v) => Number.isFinite(v))) {
                    nonFinite++;
                }
            } else if (column.dtype === "string" || column.dtype === "dict") {
                if (!isPajekLabel(column.value(r) as string)) {
                    badText++;
                }
            }
        }
        if (badText > 0) {
            notes.push(
                note(
                    PAJEK_LOSS.TEXT,
                    `${badText} value(s) of ${domain} column "${name}" hold a double quote or a line break; Pajek cannot write them`,
                    name,
                    badText,
                ),
            );
        }
        if (nonFinite > 0) {
            notes.push(
                note(
                    PAJEK_LOSS.NONFINITE_AS_TEXT,
                    `${nonFinite} non-finite value(s) of ${domain} column "${name}" are written as text`,
                    name,
                    nonFinite,
                ),
            );
        }
    }
}

/**
 * The first-mode count to write in `*Vertices N N1`, from meta.extra.pajek.firstMode.
 * @param snapshot - the snapshot
 * @returns the count, or null when absent or invalid
 */
function firstModeOf(snapshot: GraphSnapshot): number | null {
    const { pajek } = snapshot.meta.extra;
    if (typeof pajek !== "object" || pajek === null) {
        return null;
    }
    const value = (pajek as { firstMode?: unknown }).firstMode;
    if (value === undefined) {
        return null;
    }
    if (typeof value === "number" && Number.isInteger(value) && value >= 0 && value <= snapshot.nodeCount) {
        return value;
    }
    return NaN;
}

/**
 * The label of a vertex: the label column's value when set, the id's text when the id is not
 * the vertex number, otherwise null (no label needed). A label is forced when something follows
 * it on the line, because Pajek's grammar puts the label second.
 * @param labels - the label column, or null
 * @param ids - the sanitised ids
 * @param i - the node index
 * @param needed - whether the line has more after the label
 * @returns the label text, or null
 */
function labelOf(labels: Column | null, ids: SanitizedIds, i: number, needed: boolean): string | null {
    if (labels !== null && labels.isSet(i)) {
        return cellText(labels, i);
    }
    if (needed || ids.isChanged(i)) {
        return idText(ids.originalAt(i));
    }
    return null;
}

/**
 * The text of an id.
 * @param id - the id
 * @returns String(id)
 */
function idText(id: NodeId): string {
    return typeof id === "number" ? formatF64(id) : id;
}

/**
 * The spells text of a row, or null when unset or empty.
 * @param column - the spells list column
 * @param row - the row
 * @returns the interval token or null
 */
function spellsText(column: Column, row: number): string | null {
    if (column.dtype !== "list" || !column.isSet(row)) {
        return null;
    }
    const items = column.sliceOf(row);
    if (items.length === 0) {
        return null;
    }
    const pairs: [number, number][] = [];
    for (const item of items) {
        const pair = item as ArrayLike<number>;
        pairs.push([pair[0], pair[1]]);
    }
    return formatIntervals(pairs);
}

/**
 * The coordinates text of a vertex, or null when unset.
 * @param column - the position column (2 or 3 components)
 * @param i - the node index
 * @param writeZ - whether z is written for a 3-component column
 * @returns "x y" or "x y z", or null
 */
function coordinatesText(column: Column, i: number, writeZ: boolean): string | null {
    if (!column.isSet(i)) {
        return null;
    }
    const dtype = column.dtype as "f32" | "f64" | "i32" | "u32" | "u8";
    const vector = column.value(i) as ArrayLike<number>;
    const x = numberText(vector[0], dtype);
    const y = numberText(vector[1], dtype);
    if (column.meta.components === 3 && writeZ) {
        return `${x} ${y} ${numberText(vector[2], dtype)}`;
    }
    return `${x} ${y}`;
}

/**
 * Whether a 3-component position column needs its z written: the importer recorded sourceDims 3,
 * or any set row has a non-zero z (never drop data because of a 2-D hint).
 * @param column - the position column
 * @returns true when z is written
 */
function needsZ(column: Column): boolean {
    if (column.meta.components !== 3) {
        return false;
    }
    if (column.meta.extra.sourceDims !== 2) {
        return true;
    }
    for (let i = 0; i < column.length; i++) {
        if (column.isSet(i)) {
            const vector = column.value(i) as ArrayLike<number>;
            if (vector[2] !== 0) {
                return true;
            }
        }
    }
    return false;
}

/**
 * The text parts of a Pajek document, one per line.
 * @param snapshot - the snapshot
 * @param options - the resolved format options
 * @param options.networkHeader - whether the `*Network` line is written
 * @param options.mangle - whether the original ids are written as `graphty_originalId` parameters
 * @yields lines with their terminator
 * @returns nothing
 */
function* writeParts(
    snapshot: GraphSnapshot,
    options: { networkHeader: boolean; mangle: boolean },
): Generator<string, void, undefined> {
    const ids = sanitizeIds(snapshot, CAPABILITIES.idCharset, options.mangle ? "mangle" : "error");
    const nodePlan = withoutReservedKey(plan(snapshot.nodes, "node", null), options.mangle);
    const edgePlan = plan(snapshot.edges, "edge", null);
    // fail before writing anything when a text cannot be written (the check() E_ note)
    const preflight: LossNote[] = [];
    checkCells(nodePlan, "node", preflight);
    checkCells(edgePlan, "edge", preflight);
    checkIdTexts(snapshot.nodeCount, ids, options.mangle, preflight);
    for (const n of preflight) {
        if (n.code === PAJEK_LOSS.TEXT) {
            throw new GraphFormatError("E_UNSUPPORTED", n.message, {
                reason: "pajek label",
                column: n.column,
                count: n.count,
            });
        }
    }

    if (options.networkHeader && snapshot.meta.name !== null) {
        yield `*Network ${snapshot.meta.name}\n`;
    }
    const firstMode = firstModeOf(snapshot);
    yield firstMode !== null && !Number.isNaN(firstMode)
        ? `*Vertices ${snapshot.nodeCount} ${firstMode}\n`
        : `*Vertices ${snapshot.nodeCount}\n`;
    yield* writeVertices(snapshot, ids, nodePlan, options.mangle);
    yield* writeLines(snapshot, edgePlan);
}

/**
 * The node plan without a user column named like the exporter's originalId key, which is
 * reserved under "mangle" (check() reports it as not written).
 * @param nodePlan - the node column plan
 * @param mangle - whether the original ids are written as parameters
 * @returns the plan to write
 */
function withoutReservedKey(nodePlan: readonly PlannedColumn[], mangle: boolean): readonly PlannedColumn[] {
    if (!mangle) {
        return nodePlan;
    }
    return nodePlan.filter((p) => !(p.slot === "param" && p.column.meta.name === ORIGINAL_ID_KEY));
}

/**
 * The notes about the id texts a vertex line carries: an id written as a label or, under
 * "mangle", as the `graphty_originalId` parameter must be a Pajek label (E_PAJEK_TEXT otherwise),
 * and under "mangle" an original id whose text reads back as the other type under ids
 * "canonical" is reported (the importer coerces the parameter text like every id cell).
 * @param nodeCount - the node count
 * @param ids - the sanitised ids
 * @param mangle - whether the original ids are written as parameters
 * @param notes - where to record
 */
function checkIdTexts(nodeCount: number, ids: SanitizedIds, mangle: boolean, notes: LossNote[]): void {
    let badText = 0;
    let typeChanged = 0;
    for (let i = 0; i < nodeCount; i++) {
        if (!ids.isChanged(i)) {
            continue;
        }
        const original = ids.originalAt(i);
        const text = idText(original);
        if (!isPajekLabel(text)) {
            badText++;
        } else if (mangle && typeof canonicalId(text) !== typeof original) {
            typeChanged++;
        }
    }
    if (badText > 0) {
        notes.push(
            note(
                PAJEK_LOSS.TEXT,
                `${badText} node id(s) hold a double quote or a line break; Pajek cannot write them as labels or parameters`,
                null,
                badText,
            ),
        );
    }
    if (typeChanged > 0) {
        notes.push(
            note(
                PAJEK_LOSS.ID_TEXT_TYPE,
                `${typeChanged} original id(s) read back from ${ORIGINAL_ID_KEY} as the other type under ids: "canonical" (a string "1" becomes 1, a number 1.5 becomes "1.5")`,
                null,
                typeChanged,
            ),
        );
    }
}

/**
 * The vertex lines: number, label, coordinates, shape, parameters and interval.
 * @param snapshot - the snapshot
 * @param ids - the sanitised ids
 * @param nodePlan - the node column plan
 * @param mangle - whether a renumbered vertex carries its original id as a parameter
 * @yields one line per vertex
 * @returns nothing
 */
function* writeVertices(
    snapshot: GraphSnapshot,
    ids: SanitizedIds,
    nodePlan: readonly PlannedColumn[],
    mangle: boolean,
): Generator<string, void, undefined> {
    let labels: Column | null = null;
    let position: Column | null = null;
    let shape: Column | null = null;
    let nodeSpells: Column | null = null;
    const nodeParams: Column[] = [];
    for (const { column, slot } of nodePlan) {
        switch (slot) {
            case "label":
                labels = column;
                break;
            case "position":
                position = column;
                break;
            case "shape":
                shape = column;
                break;
            case "spells":
                nodeSpells = column;
                break;
            case "param":
                nodeParams.push(column);
                break;
            case "relation":
            case "skip":
                break;
            default: {
                const name: string = slot;
                throw new Error(`unknown slot ${name}`);
            }
        }
    }
    const writeZ = position !== null && needsZ(position);
    for (let i = 0; i < snapshot.nodeCount; i++) {
        const parts: string[] = [];
        if (position !== null) {
            const coordinates = coordinatesText(position, i, writeZ);
            if (coordinates !== null) {
                parts.push(coordinates);
            }
        }
        if (shape !== null && shape.isSet(i)) {
            const keyword = cellText(shape, i);
            if (SHAPES.has(keyword)) {
                parts.push(keyword);
            } else {
                parts.push(`${SHAPE_COLUMN} ${quotePajekLabel(keyword)}`);
            }
        }
        for (const column of nodeParams) {
            if (column.isSet(i)) {
                parts.push(`${column.meta.name} ${quotePajekLabel(cellText(column, i))}`);
            }
        }
        if (mangle && ids.isChanged(i)) {
            parts.push(`${ORIGINAL_ID_KEY} ${quotePajekLabel(idText(ids.originalAt(i)))}`);
        }
        if (nodeSpells !== null) {
            const spells = spellsText(nodeSpells, i);
            if (spells !== null) {
                parts.push(spells);
            }
        }
        const label = labelOf(labels, ids, i, parts.length > 0);
        let line = String(i + 1);
        if (label !== null) {
            line += ` ${quotePajekLabel(label)}`;
        }
        if (parts.length > 0) {
            line += ` ${parts.join(" ")}`;
        }
        yield `${line}\n`;
    }
}

/**
 * The line sections: runs of `*Arcs` / `*Edges` (with a relation number and name when the edge
 * has one), one line per written edge with its weight, parameters and interval.
 * @param snapshot - the snapshot
 * @param edgePlan - the edge column plan
 * @yields one line per section header and edge
 * @returns nothing
 */
function* writeLines(snapshot: GraphSnapshot, edgePlan: readonly PlannedColumn[]): Generator<string, void, undefined> {
    const folding = pairFolding(snapshot, { foldMutual: true });
    const weights = explicitWeights(snapshot);
    const { src, dst } = snapshot.edgeList();
    let relation: Column | null = null;
    let edgeSpells: Column | null = null;
    const edgeParams: Column[] = [];
    for (const { column, slot } of edgePlan) {
        switch (slot) {
            case "relation":
                relation = column;
                break;
            case "spells":
                edgeSpells = column;
                break;
            case "param":
                edgeParams.push(column);
                break;
            case "label":
            case "position":
            case "shape":
            case "skip":
                break;
            default: {
                const name: string = slot;
                throw new Error(`unknown slot ${name}`);
            }
        }
    }
    const relationNumbers = new Map<string, number>();
    let currentKind: "arcs" | "edges" | null = null;
    let currentRelation: string | null = null;
    let sectionOpen = false;
    for (let e = 0; e < snapshot.edgeCount; e++) {
        if (folding.folded(e)) {
            continue;
        }
        // a mutual pair (design section 3.6): both directions, written once as an undirected edge
        const directed = snapshot.directed && folding.sourceDirected(e) && !folding.isMutual(e);
        const kind = directed ? "arcs" : "edges";
        const relationName = relation !== null && relation.isSet(e) ? cellText(relation, e) : null;
        if (!sectionOpen || kind !== currentKind || relationName !== currentRelation) {
            currentKind = kind;
            currentRelation = relationName;
            sectionOpen = true;
            const keyword = kind === "arcs" ? "*Arcs" : "*Edges";
            if (relationName === null) {
                yield `${keyword}\n`;
            } else {
                let number = relationNumbers.get(relationName);
                if (number === undefined) {
                    number = relationNumbers.size + 1;
                    relationNumbers.set(relationName, number);
                }
                yield `${keyword} :${number} ${quotePajekLabel(relationName)}\n`;
            }
        }
        let line = `${src[e] + 1} ${dst[e] + 1}`;
        const weight = weights.text(e);
        if (weight !== null) {
            line += ` ${weight}`;
        }
        for (const column of edgeParams) {
            if (column.isSet(e)) {
                line += ` ${column.meta.name} ${quotePajekLabel(cellText(column, e))}`;
            }
        }
        if (edgeSpells !== null) {
            const spells = spellsText(edgeSpells, e);
            if (spells !== null) {
                line += ` ${spells}`;
            }
        }
        yield `${line}\n`;
    }
    if (!sectionOpen) {
        yield snapshot.directed ? "*Arcs\n" : "*Edges\n";
    }
}

/**
 * Resolve the Pajek export options.
 * @param options - the caller's options
 * @returns the resolved format options; the common options are checked by resolveExportOptions
 */
function resolvePajekOptions(options: (PajekExportOptions & CommonExportOptions) | undefined): {
    networkHeader: boolean;
    mangle: boolean;
} {
    const common = resolveExportOptions(options);
    const value = options?.networkHeader;
    if (value !== undefined && typeof value !== "boolean") {
        throw new GraphFormatError("E_UNSUPPORTED", "option networkHeader: not a boolean", {
            option: "networkHeader",
            found: typeof value,
        });
    }
    return { networkHeader: value ?? false, mangle: common.sanitizeIds === "mangle" };
}

/** The Pajek NET exporter. */
export const pajekExporter: GraphExporter<PajekExportOptions> = Object.freeze({
    format: "pajek",
    capabilities: CAPABILITIES,

    /**
     * Pre-flight: the generic capability gaps plus the Pajek-specific ones.
     * @param snapshot - the snapshot
     * @param options - export options
     * @returns the loss notes, empty when the export is exact
     */
    check(snapshot: GraphSnapshot, options?: PajekExportOptions & CommonExportOptions): readonly LossNote[] {
        const resolved = resolveExportOptions(options);
        const { mangle } = resolvePajekOptions(options);
        const notes: LossNote[] = [];
        const planned = plan(snapshot.nodes, "node", notes);
        const nodePlan = withoutReservedKey(planned, mangle);
        const edgePlan = plan(snapshot.edges, "edge", notes);
        const ids = sanitizeIds(snapshot, CAPABILITIES.idCharset, mangle ? "mangle" : "error");
        checkIdTexts(snapshot.nodeCount, ids, mangle, notes);
        if (nodePlan.length !== planned.length) {
            const reserved = planned.find((p) => p.slot === "param" && p.column.meta.name === ORIGINAL_ID_KEY);
            if (reserved !== undefined) {
                notes.push(
                    note(
                        PAJEK_LOSS.KEY_DROPPED,
                        `node column "${ORIGINAL_ID_KEY}" is the parameter the exporter writes the original ids under (sanitizeIds "mangle"); it is not written`,
                        ORIGINAL_ID_KEY,
                        reserved.column.length - reserved.column.nullCount,
                    ),
                );
            }
        }
        // columns this exporter handles itself although the generic table says otherwise: the spells
        // list (written as an interval token, or reported above with the spells code), the shape
        // keyword and the relation name (dict columns written as keywords)
        const owned = new Set<string>();
        for (const { column, slot } of [...nodePlan, ...edgePlan]) {
            if (slot === "shape" || slot === "relation") {
                owned.add(`${column.meta.domain}:${column.meta.name}`);
            } else if (slot === "param" && column.meta.domain === "node" && column.meta.name === SHAPE_COLUMN) {
                // the shape-as-parameter note already says the column reads back as string
                owned.add(`${column.meta.domain}:${column.meta.name}`);
            }
        }
        for (const table of [snapshot.nodes, snapshot.edges]) {
            for (const column of table) {
                if (column.meta.role === "spells") {
                    owned.add(`${column.meta.domain}:${column.meta.name}`);
                }
            }
        }
        for (const n of checkCapabilities(snapshot, CAPABILITIES, resolved, {
            roles: SLOT_ROLES,
            roleNames: ROLE_NAMES,
        })) {
            if (n.column !== null && (n.code === LOSS.LIST || n.code === LOSS.DTYPE)) {
                const domain = n.message.startsWith("node column") ? "node" : "edge";
                if (owned.has(`${domain}:${n.column}`)) {
                    continue;
                }
            }
            if (n.code === LOSS.COLUMN_NAME_CHANGED && n.message.startsWith("edge column")) {
                // the label slot is a vertex slot; an edge label is a parameter under its own name
                continue;
            }
            notes.push(n);
        }
        labelNotes(snapshot, nodePlan, notes);
        for (const column of snapshot.edges) {
            if (column.meta.role === "label") {
                notes.push(
                    note(
                        LOSS.ROLE,
                        `edge column "${column.meta.name}" (label) is written as a plain parameter; Pajek labels vertices only and the role is lost`,
                        column.meta.name,
                        column.length - column.nullCount,
                    ),
                );
            }
        }
        checkCells(nodePlan, "node", notes);
        checkCells(edgePlan, "edge", notes);
        const folding = pairFolding(snapshot, { foldMutual: true });
        if (folding.mutualCount > 0) {
            notes.push(
                note(
                    PAJEK_LOSS.MUTUAL_AS_UNDIRECTED,
                    `${folding.mutualCount} mutual pair(s) are written as undirected edges; the mutual mark is lost`,
                    snapshot.edges.byRole("mutual")?.meta.name ?? null,
                    folding.mutualCount,
                ),
            );
        }
        if (Number.isNaN(firstModeOf(snapshot))) {
            notes.push(
                note(
                    PAJEK_LOSS.FIRST_MODE_DROPPED,
                    "meta.extra.pajek.firstMode is not a count within 0..nodeCount; the two-mode header is not written",
                ),
            );
        }
        return notes;
    },

    /**
     * Write the snapshot as UTF-8 chunks.
     * @param snapshot - the snapshot
     * @param options - export options
     * @returns the chunks
     */
    export(snapshot: GraphSnapshot, options?: PajekExportOptions & CommonExportOptions): AsyncIterable<Uint8Array> {
        const resolved = resolvePajekOptions(options);
        return encodeChunks(writeParts(snapshot, resolved));
    },

    /**
     * Write the snapshot as one string.
     * @param snapshot - the snapshot
     * @param options - export options
     * @returns the document
     */
    exportToString(snapshot: GraphSnapshot, options?: PajekExportOptions & CommonExportOptions): Promise<string> {
        const resolved = resolvePajekOptions(options);
        return joinText(writeParts(snapshot, resolved));
    },
});
