/**
 * UniformBlock (spec 5.3, D20; contract 3.9, 3.10.2; 5.5 row struct-block.test.ts): hand-computed strict-layout
 * offsets of a mixed block, 16-byte padding through an explicit @size, the write / read round trip (uniform and
 * storage layouts, little-endian), padTo, every E_INVALID_ARGUMENT case, and the seven generated blocks of 3.10.2
 * at their documented offsets. Pure: no device.
 */

import { STATE_HEADER_BYTES } from "../../src/constants.js";
import { isWebGpuGraphError, type WebGpuGraphError } from "../../src/errors.js";
import { UniformBlock, type UniformField, type UniformFieldType } from "../../src/kernel/struct-block.js";

function catchError(fn: () => unknown): WebGpuGraphError {
    try {
        fn();
    } catch (err) {
        if (isWebGpuGraphError(err)) {
            return err;
        }
        throw err;
    }
    throw new Error("expected a WebGpuGraphError");
}

function offsets(block: UniformBlock): Record<string, number> {
    const out: Record<string, number> = {};
    for (const [name] of block.fields) {
        out[name] = block.offsetOf(name);
    }
    return out;
}

// ---- the 3.10.2 field lists (kernels.ts, P1-T4, exports the same blocks; the offsets are pinned here independently)

const RANGE_PARAMS_FIELDS: readonly UniformField[] = [
    ["start", "u32"],
    ["end", "u32"],
    ["arcBase", "u32"],
    ["arcEnd", "u32"],
    ["accumulate", "u32"],
    ["n", "u32"],
    ["pad0", "u32"],
    ["pad1", "u32"],
];
const REDUCE_PARAMS_FIELDS: readonly UniformField[] = [
    ["count", "u32"],
    ["outOffset", "u32"],
    ["level", "u32"],
    ["pad0", "u32"],
];
const FILL_PARAMS_FIELDS: readonly UniformField[] = [
    ["count", "u32"],
    ["value", "u32"],
    ["mode", "u32"],
    ["pad0", "u32"],
];
const FA2_PARAMS_FIELDS: readonly UniformField[] = [
    ["n", "u32"],
    ["dim", "u32"],
    ["flags", "u32"],
    ["tierStart", "u32"],
    ["tierEnd", "u32"],
    ["iterationIndex", "u32"],
    ["seed", "u32"],
    ["nearMax", "u32"],
    ["scalingRatio", "f32"],
    ["gravity", "f32"],
    ["jitterTolerance", "f32"],
    ["scale", "f32"],
    ["center", "vec4f"],
    ["settleThreshold", "f32"],
    ["extentFactor", "f32"],
    ["gridMax", "u32"],
    ["levels", "u32"],
    ["pad", "vec4f"],
];
const FA2_STATE_FIELDS: readonly UniformField[] = [
    ["speed", "f32"],
    ["speedEfficiency", "f32"],
    ["swing", "f32"],
    ["traction", "f32"],
    ["centroid", "vec4f"],
    ["rmsRadius", "f32"],
    ["radius", "f32"],
    ["meanDisplacement", "f32"],
    ["iteration", "u32"],
    ["min", "vec4f"],
    ["max", "vec4f"],
    ["gridMin", "vec4f"],
    ["eps", "f32"],
    ["settledCount", "u32"],
    ["outsideGrid", "u32"],
    ["maxCellOccupancy", "u32"],
    ["reserved0", "vec4f"],
    ["reserved1", "vec4f"],
    ["reserved2", "vec4f"],
    ["reserved3", "vec4f"],
    ["reserved4", "vec4f"],
    ["reserved5", "vec4f"],
    ["reserved6", "vec4f"],
    ["reserved7", "vec4f"],
    ["reserved8", "vec4f"],
];
const FA2_TRACE_FIELDS: readonly UniformField[] = [
    ["swing", "f32"],
    ["traction", "f32"],
    ["speed", "f32"],
    ["speedEfficiency", "f32"],
    ["meanDisplacement", "f32"],
    ["settledCount", "u32"],
    ["iteration", "u32"],
    ["pad0", "u32"],
];
const FA2_PARTIAL_FIELDS: readonly UniformField[] = [
    ["sum", "vec4f"],
    ["min", "vec4f"],
    ["max", "vec4f"],
    ["swingTraction", "vec2f"],
    ["dispFree", "vec2f"],
];

describe("UniformBlock.define", () => {
    it("lays out every field type at its WGSL size and alignment (scalars 4 / 4, vec2 8 / 8, vec4 16 / 16)", () => {
        const expected: Record<UniformFieldType, { readonly offset: number; readonly end: number }> = {
            u32: { offset: 4, end: 8 },
            i32: { offset: 4, end: 8 },
            f32: { offset: 4, end: 8 },
            vec2f: { offset: 8, end: 16 },
            vec2u: { offset: 8, end: 16 },
            vec4f: { offset: 16, end: 32 },
            vec4u: { offset: 16, end: 32 },
        };
        const types = Object.keys(expected) as UniformFieldType[];
        expect(types).toHaveLength(7);
        for (const type of types) {
            // a leading u32 pushes the field to its alignment; the block then pads to 16
            const block = UniformBlock.define("Probe", [
                ["lead", "u32"],
                ["x", type],
            ]);
            expect(block.offsetOf("x"), type).toBe(expected[type].offset);
            expect(block.byteLength, type).toBe(Math.ceil(expected[type].end / 16) * 16);
            expect(block.wgsl).toContain(`x: ${type},`);
        }
    });

    it("lays a mixed block (u32 x5, f32, vec4f, vec2f) out at the strict WGSL offsets and pads to 16", () => {
        const block = UniformBlock.define("Mixed", [
            ["a", "u32"],
            ["b", "u32"],
            ["c", "u32"],
            ["d", "u32"],
            ["e", "u32"],
            ["f", "f32"],
            ["v", "vec4f"],
            ["w", "vec2f"],
        ]);
        // a 0, b 4, c 8, d 12, e 16, f 20, v roundUp(24, 16) = 32, w roundUp(48, 8) = 48; end 56 -> 64
        expect(offsets(block)).toEqual({ a: 0, b: 4, c: 8, d: 12, e: 16, f: 20, v: 32, w: 48 });
        expect(block.byteLength).toBe(64);
        expect(block.name).toBe("Mixed");
        expect(block.layout).toBe("uniform");
        expect(block.fields).toEqual([
            ["a", "u32"],
            ["b", "u32"],
            ["c", "u32"],
            ["d", "u32"],
            ["e", "u32"],
            ["f", "f32"],
            ["v", "vec4f"],
            ["w", "vec2f"],
        ]);
        expect(block.wgsl).toBe(
            [
                "struct Mixed {",
                "    a: u32,",
                "    b: u32,",
                "    c: u32,",
                "    d: u32,",
                "    e: u32,",
                "    f: f32,",
                "    v: vec4f,",
                "    @size(16) w: vec2f,",
                "}",
            ].join("\n"),
        );
    });

    it("pads a block of scalars to 16 bytes through @size on the last member, and emits no @size when none is needed", () => {
        const one = UniformBlock.define("One", [["n", "u32"]]);
        expect(one.byteLength).toBe(16);
        expect(one.wgsl).toBe("struct One {\n    @size(16) n: u32,\n}");
        const two = UniformBlock.define("Two", [
            ["count", "u32"],
            ["value", "u32"],
        ]);
        expect(two.byteLength).toBe(16);
        expect(two.offsetOf("value")).toBe(4);
        expect(two.wgsl).toBe("struct Two {\n    count: u32,\n    @size(12) value: u32,\n}");
        const four = UniformBlock.define("Four", [
            ["a", "u32"],
            ["b", "u32"],
            ["c", "u32"],
            ["d", "u32"],
        ]);
        expect(four.byteLength).toBe(16);
        expect(four.wgsl).toBe("struct Four {\n    a: u32,\n    b: u32,\n    c: u32,\n    d: u32,\n}");
        const vec = UniformBlock.define("Vec", [
            ["a", "u32"],
            ["v", "vec4f"],
        ]);
        expect(offsets(vec)).toEqual({ a: 0, v: 16 });
        expect(vec.byteLength).toBe(32);
        expect(vec.wgsl).toBe("struct Vec {\n    a: u32,\n    v: vec4f,\n}");
    });

    it("honours padTo (a multiple of 16 not below the natural size) and tags the storage layout", () => {
        const block = UniformBlock.define(
            "Padded",
            [
                ["a", "u32"],
                ["b", "f32"],
            ],
            { layout: "storage", padTo: 64 },
        );
        expect(block.layout).toBe("storage");
        expect(block.byteLength).toBe(64);
        expect(block.wgsl).toBe("struct Padded {\n    a: u32,\n    @size(60) b: f32,\n}");
        expect(UniformBlock.define("Exact", [["a", "u32"]], { padTo: 16 }).byteLength).toBe(16);
        for (const padTo of [20, 8, 0, 15.5]) {
            const err = catchError(() => UniformBlock.define("Bad", [["a", "u32"]], { padTo }));
            expect(err.code).toBe("E_INVALID_ARGUMENT");
            expect(err.details.argument).toBe("padTo");
        }
        const smaller = catchError(() =>
            UniformBlock.define(
                "Bad",
                [
                    ["a", "vec4f"],
                    ["b", "vec4f"],
                ],
                { padTo: 16 },
            ),
        );
        expect(smaller.code).toBe("E_INVALID_ARGUMENT");
        expect(smaller.details.argument).toBe("padTo");
    });

    it("rejects an empty field list, a duplicate field, a reserved or malformed name and an unknown type", () => {
        expect(catchError(() => UniformBlock.define("Empty", [])).details.argument).toBe("fields");
        const dup = catchError(() =>
            UniformBlock.define("Dup", [
                ["a", "u32"],
                ["a", "f32"],
            ]),
        );
        expect(dup.code).toBe("E_INVALID_ARGUMENT");
        expect(dup.details).toMatchObject({ argument: "fields", value: "a" });
        expect(catchError(() => UniformBlock.define("Reserved", [["type", "u32"]])).details.value).toBe("type");
        expect(catchError(() => UniformBlock.define("Space", [["a b", "u32"]])).details.value).toBe("a b");
        expect(catchError(() => UniformBlock.define("target", [["a", "u32"]])).details.argument).toBe("name");
        expect(catchError(() => UniformBlock.define("1st", [["a", "u32"]])).details.argument).toBe("name");
        const unknown = catchError(() => UniformBlock.define("Unknown", [["a", "vec3f" as unknown as "u32"]]));
        expect(unknown.code).toBe("E_INVALID_ARGUMENT");
        expect(unknown.details.value).toBe("vec3f");
    });
});

describe("UniformBlock write / read", () => {
    const block = UniformBlock.define("Mixed", [
        ["a", "u32"],
        ["s", "i32"],
        ["f", "f32"],
        ["v", "vec4f"],
        ["w", "vec2u"],
        ["last", "vec2f"],
    ]);

    it("round-trips every field little-endian and writes a missing field as 0", () => {
        // a 0, s 4, f 8, v 16, w 32, last 40; end 48 -> 48
        expect(offsets(block)).toEqual({ a: 0, s: 4, f: 8, v: 16, w: 32, last: 40 });
        expect(block.byteLength).toBe(48);
        const bytes = new ArrayBuffer(48);
        const view = new DataView(bytes);
        block.write(view, { a: 0xfffffffe, s: -7, f: 1.5, v: [1, 2, 3, 4], w: [5, 6] });
        expect(new Uint8Array(bytes, 0, 4)).toEqual(new Uint8Array([0xfe, 0xff, 0xff, 0xff]));
        expect(view.getInt32(4, true)).toBe(-7);
        expect(view.getFloat32(8, true)).toBe(1.5);
        expect(view.getFloat32(28, true)).toBe(4);
        expect(view.getUint32(36, true)).toBe(6);
        expect(block.read(view)).toEqual({ a: 0xfffffffe, s: -7, f: 1.5, v: [1, 2, 3, 4], w: [5, 6], last: [0, 0] });
        expect(block.readField(view, "v")).toEqual([1, 2, 3, 4]);
        expect(block.readField(view, "s")).toBe(-7);
    });

    it("writes at a byte offset, zeroes the region first and rejects a region outside the view", () => {
        const bytes = new ArrayBuffer(48 * 3);
        const view = new DataView(bytes);
        new Uint8Array(bytes).fill(0xab);
        block.write(view, { a: 9 }, 48);
        expect(block.read(view, 48)).toEqual({ a: 9, s: 0, f: 0, v: [0, 0, 0, 0], w: [0, 0], last: [0, 0] });
        expect(view.getUint8(47)).toBe(0xab);
        expect(view.getUint8(96)).toBe(0xab);
        expect(block.readField(view, "a", 48)).toBe(9);
        for (const offset of [49 * 2 + 1, -48, 1.5]) {
            expect(catchError(() => block.write(view, {}, offset)).details.argument).toBe("byteOffset");
        }
        expect(catchError(() => block.read(new DataView(new ArrayBuffer(16)))).details.argument).toBe("byteOffset");
    });

    it("rejects an unknown key, a wrong vector width, a scalar for a vector, a vector for a scalar and an out-of-range integer", () => {
        const view = new DataView(new ArrayBuffer(48));
        expect(catchError(() => block.write(view, { nope: 1 })).details).toMatchObject({
            argument: "values",
            value: "nope",
        });
        expect(catchError(() => block.write(view, { v: [1, 2, 3] })).details.argument).toBe("v");
        expect(catchError(() => block.write(view, { v: 1 })).details.argument).toBe("v");
        expect(catchError(() => block.write(view, { a: [1] })).details.argument).toBe("a");
        expect(catchError(() => block.write(view, { a: -1 })).details.argument).toBe("a");
        expect(catchError(() => block.write(view, { a: 2 ** 32 })).details.argument).toBe("a");
        expect(catchError(() => block.write(view, { a: 1.5 })).details.argument).toBe("a");
        expect(catchError(() => block.write(view, { s: 2 ** 31 })).details.argument).toBe("s");
        expect(catchError(() => block.write(view, { w: [1, -1] })).details.argument).toBe("w");
        expect(catchError(() => block.offsetOf("nope")).details).toMatchObject({ argument: "field", value: "nope" });
        expect(catchError(() => block.readField(view, "nope")).details.argument).toBe("field");
        // nothing was written by the rejected calls' zeroing of an unrelated field: the view is still all zero
        expect(new Uint8Array(view.buffer).every((b) => b === 0)).toBe(true);
    });

    it("round-trips the storage-mode Fa2State block on the CPU side at the 3.10.2 offsets", () => {
        const state = UniformBlock.define("Fa2State", FA2_STATE_FIELDS, {
            layout: "storage",
            padTo: STATE_HEADER_BYTES,
        });
        const bytes = new ArrayBuffer(STATE_HEADER_BYTES);
        const view = new DataView(bytes);
        const values = {
            speed: 1.5,
            speedEfficiency: 0.25,
            swing: 3,
            traction: 4,
            centroid: [1, 2, 3, 0],
            rmsRadius: 5,
            radius: 6,
            meanDisplacement: 0.125,
            iteration: 7,
            min: [-1, -2, -3, 0],
            max: [1, 2, 3, 0],
            gridMin: [0, 0, 0, 0.5],
            eps: 0.0625,
            settledCount: 8,
            outsideGrid: 9,
            maxCellOccupancy: 10,
            reserved8: [11, 12, 13, 14],
        };
        state.write(view, values);
        expect(view.getFloat32(0, true)).toBe(1.5);
        expect(view.getUint32(44, true)).toBe(7);
        expect(view.getFloat32(240 + 12, true)).toBe(14);
        const back = state.read(view);
        expect(back).toMatchObject(values);
        expect(back.reserved0).toEqual([0, 0, 0, 0]);
        expect(Object.keys(back)).toEqual(FA2_STATE_FIELDS.map(([name]) => name));
    });
});

describe("the generated blocks of 3.10.2", () => {
    it("RangeParams (32 B), ReduceParams (16 B), FillParams (16 B)", () => {
        const range = UniformBlock.define("RangeParams", RANGE_PARAMS_FIELDS);
        expect(offsets(range)).toEqual({
            start: 0,
            end: 4,
            arcBase: 8,
            arcEnd: 12,
            accumulate: 16,
            n: 20,
            pad0: 24,
            pad1: 28,
        });
        expect(range.byteLength).toBe(32);
        const reduce = UniformBlock.define("ReduceParams", REDUCE_PARAMS_FIELDS);
        expect(offsets(reduce)).toEqual({ count: 0, outOffset: 4, level: 8, pad0: 12 });
        expect(reduce.byteLength).toBe(16);
        const fill = UniformBlock.define("FillParams", FILL_PARAMS_FIELDS);
        expect(offsets(fill)).toEqual({ count: 0, value: 4, mode: 8, pad0: 12 });
        expect(fill.byteLength).toBe(16);
    });

    it("Fa2Params (96 B, spec 7.3)", () => {
        const params = UniformBlock.define("Fa2Params", FA2_PARAMS_FIELDS);
        expect(offsets(params)).toEqual({
            n: 0,
            dim: 4,
            flags: 8,
            tierStart: 12,
            tierEnd: 16,
            iterationIndex: 20,
            seed: 24,
            nearMax: 28,
            scalingRatio: 32,
            gravity: 36,
            jitterTolerance: 40,
            scale: 44,
            center: 48,
            settleThreshold: 64,
            extentFactor: 68,
            gridMax: 72,
            levels: 76,
            pad: 80,
        });
        expect(params.byteLength).toBe(96);
        expect(params.wgsl).not.toContain("@size");
    });

    it("Fa2State (storage, padded to STATE_HEADER_BYTES = 256)", () => {
        const state = UniformBlock.define("Fa2State", FA2_STATE_FIELDS, {
            layout: "storage",
            padTo: STATE_HEADER_BYTES,
        });
        expect(offsets(state)).toEqual({
            speed: 0,
            speedEfficiency: 4,
            swing: 8,
            traction: 12,
            centroid: 16,
            rmsRadius: 32,
            radius: 36,
            meanDisplacement: 40,
            iteration: 44,
            min: 48,
            max: 64,
            gridMin: 80,
            eps: 96,
            settledCount: 100,
            outsideGrid: 104,
            maxCellOccupancy: 108,
            reserved0: 112,
            reserved1: 128,
            reserved2: 144,
            reserved3: 160,
            reserved4: 176,
            reserved5: 192,
            reserved6: 208,
            reserved7: 224,
            reserved8: 240,
        });
        expect(state.byteLength).toBe(256);
        expect(state.layout).toBe("storage");
    });

    it("Fa2Trace (32 B record) and Fa2Partial (64 B record)", () => {
        const trace = UniformBlock.define("Fa2Trace", FA2_TRACE_FIELDS, { layout: "storage" });
        expect(offsets(trace)).toEqual({
            swing: 0,
            traction: 4,
            speed: 8,
            speedEfficiency: 12,
            meanDisplacement: 16,
            settledCount: 20,
            iteration: 24,
            pad0: 28,
        });
        expect(trace.byteLength).toBe(32);
        const partial = UniformBlock.define("Fa2Partial", FA2_PARTIAL_FIELDS, { layout: "storage" });
        expect(offsets(partial)).toEqual({ sum: 0, min: 16, max: 32, swingTraction: 48, dispFree: 56 });
        expect(partial.byteLength).toBe(64);
        expect(partial.wgsl).toBe(
            "struct Fa2Partial {\n    sum: vec4f,\n    min: vec4f,\n    max: vec4f,\n    swingTraction: vec2f,\n    dispFree: vec2f,\n}",
        );
    });
});
