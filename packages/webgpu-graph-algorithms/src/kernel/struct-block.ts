/**
 * UniformBlock (spec 5.3, D20): one field table generates BOTH the padded WGSL `struct` text spliced into a module
 * and the byte writer / reader the host uses, so the two cannot disagree. The layout is the strict WGSL layout
 * every runtime accepts (Chromium 139 lacks `uniform_buffer_standard_layout`): scalars 4-aligned, vec2 8-aligned,
 * vec4 16-aligned, no vec3, no bool, no arrays, and the total padded to a multiple of 16 (or to `padTo`) through an
 * explicit `@size` on the last member.
 */

import { WebGpuGraphError } from "../errors.js";
import { WGSL_RESERVED_WORDS } from "./prelude.js";

/** Field types a block may hold; no vec3 (spec 5.3), no bool (not host-shareable), no arrays. */
export type UniformFieldType = "u32" | "i32" | "f32" | "vec2f" | "vec2u" | "vec4f" | "vec4u";
/** One field. */
export type UniformField = readonly [name: string, type: UniformFieldType];
/** The values written into or read from a block: scalars as numbers, vectors as number arrays of the vector's width. */
export type UniformValues = Readonly<Record<string, number | readonly number[]>>;

/** The scalar a field is made of. */
type ScalarKind = "u32" | "i32" | "f32";

/** Size, alignment and composition of each field type (WGSL 14.4.1). */
interface TypeInfo {
    readonly size: number;
    readonly align: number;
    readonly width: number;
    readonly scalar: ScalarKind;
}

const TYPE_INFO: Readonly<Record<UniformFieldType, TypeInfo>> = Object.freeze({
    u32: { size: 4, align: 4, width: 1, scalar: "u32" },
    i32: { size: 4, align: 4, width: 1, scalar: "i32" },
    f32: { size: 4, align: 4, width: 1, scalar: "f32" },
    vec2f: { size: 8, align: 8, width: 2, scalar: "f32" },
    vec2u: { size: 8, align: 8, width: 2, scalar: "u32" },
    vec4f: { size: 16, align: 16, width: 4, scalar: "f32" },
    vec4u: { size: 16, align: 16, width: 4, scalar: "u32" },
});

/** One laid-out field. */
interface FieldLayout {
    readonly name: string;
    readonly type: UniformFieldType;
    readonly offset: number;
    readonly info: TypeInfo;
}

const STRUCT_ALIGN = 16;
const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;
const U32_LIMIT = 0xffffffff;
const I32_MIN = -2147483648;
const I32_MAX = 2147483647;

/**
 * Rounds `value` up to the next multiple of `align` (arithmetic, never a bit trick: byte offsets are never operated
 * on bitwise, spec 3.6).
 * @param value - the value to round
 * @param align - the alignment, a positive integer
 * @returns the rounded value
 */
function roundUp(value: number, align: number): number {
    return Math.ceil(value / align) * align;
}

/**
 * Validates a WGSL identifier for a struct or field name.
 * @param argument - the E_INVALID_ARGUMENT `argument` to report
 * @param name - the identifier
 */
function assertIdentifier(argument: string, name: string): void {
    if (!IDENTIFIER.test(name) || WGSL_RESERVED_WORDS.includes(name)) {
        throw new WebGpuGraphError("E_INVALID_ARGUMENT", `UniformBlock: "${name}" is not a usable WGSL identifier`, {
            argument,
            value: name,
            expected: "a WGSL identifier that is not a reserved word",
        });
    }
}

/**
 * Validates one scalar component of a value against its scalar kind.
 * @param field - the field name (for the error)
 * @param kind - the scalar kind
 * @param value - the component
 */
function assertScalar(field: string, kind: ScalarKind, value: unknown): asserts value is number {
    if (typeof value !== "number") {
        throw new WebGpuGraphError("E_INVALID_ARGUMENT", `UniformBlock: field "${field}" expects a number`, {
            argument: field,
            value,
            expected: kind,
        });
    }
    if (kind === "u32" && !(Number.isInteger(value) && value >= 0 && value <= U32_LIMIT)) {
        throw new WebGpuGraphError("E_INVALID_ARGUMENT", `UniformBlock: field "${field}" is not a u32`, {
            argument: field,
            value,
            expected: "an integer in [0, 4294967295]",
        });
    }
    if (kind === "i32" && !(Number.isInteger(value) && value >= I32_MIN && value <= I32_MAX)) {
        throw new WebGpuGraphError("E_INVALID_ARGUMENT", `UniformBlock: field "${field}" is not an i32`, {
            argument: field,
            value,
            expected: "an integer in [-2147483648, 2147483647]",
        });
    }
}

/**
 * Writes one scalar component little-endian.
 * @param view - the target view
 * @param kind - the scalar kind
 * @param offset - the byte offset
 * @param value - the component
 */
function setScalar(view: DataView, kind: ScalarKind, offset: number, value: number): void {
    if (kind === "u32") {
        view.setUint32(offset, value, true);
    } else if (kind === "i32") {
        view.setInt32(offset, value, true);
    } else {
        view.setFloat32(offset, value, true);
    }
}

/**
 * Reads one scalar component little-endian.
 * @param view - the source view
 * @param kind - the scalar kind
 * @param offset - the byte offset
 * @returns the component
 */
function getScalar(view: DataView, kind: ScalarKind, offset: number): number {
    if (kind === "u32") {
        return view.getUint32(offset, true);
    }
    if (kind === "i32") {
        return view.getInt32(offset, true);
    }
    return view.getFloat32(offset, true);
}

/** A generated struct: the padded WGSL text and the byte writer / reader share one field table, so they cannot disagree (D20). */
export class UniformBlock {
    /** The WGSL struct name. */
    readonly name: string;
    /** "uniform" (the default) or "storage"; both lay out identically, the tag documents the block's address space. */
    readonly layout: "uniform" | "storage";
    /** The fields in declaration (= byte) order. */
    readonly fields: readonly UniformField[];
    /** The padded byte length (a multiple of 16). */
    readonly byteLength: number;
    /** The `struct <name> { ... }` text with explicit `@size` / `@align` where padding is needed. */
    readonly wgsl: string;
    /** The laid-out fields by name. */
    private readonly table: ReadonlyMap<string, FieldLayout>;
    /** The laid-out fields in order. */
    private readonly laidOut: readonly FieldLayout[];

    /**
     * Declares a block; fields are laid out in order with 16-byte alignment for vec4 / the struct, 8 for vec2, 4 for
     * scalars; the total is padded to 16 (uniform, storage) or to `padTo` when given.
     * PLAN DECISION: rejects an empty field list and a struct or field name that is not a WGSL identifier or is a
     * reserved word (spec 3.6: every throwing call leaves state unchanged; a reserved name would be invalid WGSL).
     * PLAN DECISION: optional PARAMETERS (`options?` here, `byteOffset?` of write / read / readField) are spelled
     * `?: T` rather than the contract's `?: T | undefined` because the root ESLint rule
     * no-duplicate-type-constituents rejects the explicit undefined on an optional parameter; every call site is
     * identical, and optional PROPERTIES keep `?: T | undefined`.
     * @param name - the WGSL struct name
     * @param fields - the fields in byte order
     * @param options - the layout tag and the optional padding
     * @param options.layout - "uniform" (default) or "storage"; the tag documents the address space, the layout is the same
     * @param options.padTo - the byte length to pad to (a multiple of 16 not below the natural size)
     * @returns the block
     */
    static define(
        name: string,
        fields: readonly UniformField[],
        options?: { readonly layout?: "uniform" | "storage" | undefined; readonly padTo?: number | undefined },
    ): UniformBlock {
        assertIdentifier("name", name);
        if (fields.length === 0) {
            throw new WebGpuGraphError("E_INVALID_ARGUMENT", `UniformBlock ${name}: a block needs at least one field`, {
                argument: "fields",
                value: fields.length,
                expected: ">= 1 field",
            });
        }
        const laidOut: FieldLayout[] = [];
        const table = new Map<string, FieldLayout>();
        let cursor = 0;
        for (const [fieldName, type] of fields) {
            assertIdentifier("fields", fieldName);
            if (table.has(fieldName)) {
                throw new WebGpuGraphError(
                    "E_INVALID_ARGUMENT",
                    `UniformBlock ${name}: duplicate field "${fieldName}"`,
                    {
                        argument: "fields",
                        value: fieldName,
                        expected: "unique field names",
                    },
                );
            }
            const info = TYPE_INFO[type] as TypeInfo | undefined;
            if (info === undefined) {
                throw new WebGpuGraphError("E_INVALID_ARGUMENT", `UniformBlock ${name}: unknown field type "${type}"`, {
                    argument: "fields",
                    value: type,
                    expected: Object.keys(TYPE_INFO).join(" | "),
                });
            }
            const offset = roundUp(cursor, info.align);
            const field: FieldLayout = { name: fieldName, type, offset, info };
            laidOut.push(field);
            table.set(fieldName, field);
            cursor = offset + info.size;
        }
        const natural = roundUp(cursor, STRUCT_ALIGN);
        const padTo = options?.padTo;
        if (padTo !== undefined && (!Number.isInteger(padTo) || padTo % STRUCT_ALIGN !== 0 || padTo < natural)) {
            throw new WebGpuGraphError(
                "E_INVALID_ARGUMENT",
                `UniformBlock ${name}: padTo ${padTo} is not a multiple of 16 >= ${natural}`,
                {
                    argument: "padTo",
                    value: padTo,
                    expected: `a multiple of ${STRUCT_ALIGN} not below ${natural}`,
                },
            );
        }
        const byteLength = padTo ?? natural;
        const lines = laidOut.map((field, index) => {
            const last = index === laidOut.length - 1;
            const end = field.offset + field.info.size;
            const size = last && end < byteLength ? `@size(${byteLength - field.offset}) ` : "";
            return `    ${size}${field.name}: ${field.type},`;
        });
        const wgsl = `struct ${name} {\n${lines.join("\n")}\n}`;
        return new UniformBlock(name, options?.layout ?? "uniform", fields, byteLength, wgsl, laidOut, table);
    }

    /**
     * Built by `define()` only.
     * @param name - the struct name
     * @param layout - the address-space tag
     * @param fields - the declared fields
     * @param byteLength - the padded byte length
     * @param wgsl - the struct text
     * @param laidOut - the laid-out fields in order
     * @param table - the laid-out fields by name
     */
    private constructor(
        name: string,
        layout: "uniform" | "storage",
        fields: readonly UniformField[],
        byteLength: number,
        wgsl: string,
        laidOut: readonly FieldLayout[],
        table: ReadonlyMap<string, FieldLayout>,
    ) {
        this.name = name;
        this.layout = layout;
        this.fields = Object.freeze(fields.map(([fieldName, type]) => Object.freeze([fieldName, type] as const)));
        this.byteLength = byteLength;
        this.wgsl = wgsl;
        this.laidOut = laidOut;
        this.table = table;
    }

    /**
     * Byte offset of a field; E_INVALID_ARGUMENT for an unknown field.
     * @param field - the field name
     * @returns the byte offset inside the block
     */
    offsetOf(field: string): number {
        return this.fieldLayout(field).offset;
    }

    /**
     * Writes `values` at `byteOffset` (default 0); a missing field is written as 0; an unknown key is
     * E_INVALID_ARGUMENT; a vector of the wrong width is E_INVALID_ARGUMENT. Always little-endian. Every byte of the
     * block's region is written (padding as zero), so a written block is byte-exact.
     * PLAN DECISION: every value is validated first (u32 / vec*u components integers in [0, 2^32 - 1], i32 in
     * [-2^31, 2^31 - 1]) and only then is the region zeroed and written, so a rejected call leaves the view unchanged.
     * @param view - the target view (the region must lie inside it)
     * @param values - the field values
     * @param byteOffset - the region's start inside the view
     */
    write(view: DataView, values: UniformValues, byteOffset?: number): void {
        const base = this.regionStart(view, byteOffset);
        for (const key of Object.keys(values)) {
            if (!this.table.has(key)) {
                throw new WebGpuGraphError("E_INVALID_ARGUMENT", `UniformBlock ${this.name}: unknown field "${key}"`, {
                    argument: "values",
                    value: key,
                    expected: this.laidOut.map((f) => f.name).join(", "),
                });
            }
        }
        // validate every value first, so a rejected call leaves the view unchanged (spec 3.6)
        const components: (readonly number[] | null)[] = this.laidOut.map((field) =>
            this.components(field, values[field.name]),
        );
        for (let at = 0; at < this.byteLength; at += 4) {
            view.setUint32(base + at, 0, true);
        }
        this.laidOut.forEach((field, index) => {
            const lanes = components[index];
            if (lanes === null) {
                return;
            }
            lanes.forEach((component, lane) => {
                setScalar(view, field.info.scalar, base + field.offset + 4 * lane, component);
            });
        });
    }

    /**
     * The validated components of one field's value: null for an absent field, else `width` numbers.
     * @param field - the laid-out field
     * @param value - the value given for it (undefined when absent)
     * @returns the components or null
     */
    private components(field: FieldLayout, value: number | readonly number[] | undefined): readonly number[] | null {
        if (value === undefined) {
            return null;
        }
        const { scalar, width } = field.info;
        if (width === 1) {
            assertScalar(field.name, scalar, value);
            return [value];
        }
        if (!Array.isArray(value) || value.length !== width) {
            throw new WebGpuGraphError(
                "E_INVALID_ARGUMENT",
                `UniformBlock ${this.name}: field "${field.name}" expects ${width} components`,
                {
                    argument: field.name,
                    value,
                    expected: `an array of ${width} numbers`,
                },
            );
        }
        const lanes: number[] = [];
        for (let lane = 0; lane < width; lane++) {
            const component: unknown = value[lane];
            assertScalar(field.name, scalar, component);
            lanes.push(component);
        }
        return lanes;
    }

    /**
     * Reads every field at `byteOffset` (storage mode's reader; also used by tests on uniform blocks).
     * @param view - the source view
     * @param byteOffset - the region's start inside the view
     * @returns the field values (vectors as arrays)
     */
    read(view: DataView, byteOffset?: number): UniformValues {
        const base = this.regionStart(view, byteOffset);
        const out: Record<string, number | readonly number[]> = {};
        for (const field of this.laidOut) {
            out[field.name] = this.decode(view, field, base);
        }
        return out;
    }

    /**
     * Reads one field.
     * @param view - the source view
     * @param field - the field name
     * @param byteOffset - the region's start inside the view
     * @returns the value (a number, or an array for a vector)
     */
    readField(view: DataView, field: string, byteOffset?: number): number | readonly number[] {
        const layout = this.fieldLayout(field);
        return this.decode(view, layout, this.regionStart(view, byteOffset));
    }

    /**
     * The laid-out field of a name.
     * @param field - the field name
     * @returns the layout
     */
    private fieldLayout(field: string): FieldLayout {
        const layout = this.table.get(field);
        if (layout === undefined) {
            throw new WebGpuGraphError("E_INVALID_ARGUMENT", `UniformBlock ${this.name}: unknown field "${field}"`, {
                argument: "field",
                value: field,
                expected: this.laidOut.map((f) => f.name).join(", "),
            });
        }
        return layout;
    }

    /**
     * Validates the region `[byteOffset, byteOffset + byteLength)` against the view.
     * @param view - the view
     * @param byteOffset - the requested start (default 0)
     * @returns the start
     */
    private regionStart(view: DataView, byteOffset: number | undefined): number {
        const base = byteOffset ?? 0;
        if (!Number.isInteger(base) || base < 0 || base + this.byteLength > view.byteLength) {
            throw new WebGpuGraphError(
                "E_INVALID_ARGUMENT",
                `UniformBlock ${this.name}: region [${base}, ${base + this.byteLength}) is outside the view`,
                {
                    argument: "byteOffset",
                    value: base,
                    expected: `0 <= byteOffset <= ${view.byteLength - this.byteLength}`,
                },
            );
        }
        return base;
    }

    /**
     * Decodes one field from a view.
     * @param view - the source view
     * @param field - the laid-out field
     * @param base - the region start
     * @returns the value
     */
    private decode(view: DataView, field: FieldLayout, base: number): number | readonly number[] {
        const { scalar, width } = field.info;
        if (width === 1) {
            return getScalar(view, scalar, base + field.offset);
        }
        const lanes: number[] = [];
        for (let lane = 0; lane < width; lane++) {
            lanes.push(getScalar(view, scalar, base + field.offset + 4 * lane));
        }
        return lanes;
    }
}
