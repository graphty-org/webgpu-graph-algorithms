import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { bitmapCount, bitmapToIndices } from "../../src/columns/bitmap.js";
import { GrowableBitmap, GrowableTypedArray } from "../../src/columns/growable.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { HAS_RESIZABLE_ARRAY_BUFFER, STAGING_RESERVE_BYTES } from "../../src/util/typed-array.js";

/** Both growth strategies of design section 6.2, run against every test below. */
const STRATEGIES: readonly { readonly name: string; readonly resizable: boolean }[] = [
    { name: "resizable ArrayBuffer", resizable: true },
    { name: "allocate-and-copy", resizable: false },
];

describe.each(STRATEGIES)("GrowableTypedArray ($name)", ({ resizable }) => {
    it("starts empty and allocates on the first push", () => {
        const g = new GrowableTypedArray(Uint32Array, { resizable });
        expect(g.length).toBe(0);
        expect(g.capacity).toBe(0);
        expect(g.array.length).toBe(0);
        expect(g.array.buffer.resizable).toBe(resizable);
        expect(g.push(7)).toBe(0);
        expect(g.length).toBe(1);
        expect(g.capacity).toBe(16);
        expect(g.array.buffer.byteLength).toBe(64);
        expect(g.get(0)).toBe(7);
    });

    it("honours an initial capacity rounded to the staging granules", () => {
        const g = new GrowableTypedArray(Uint32Array, { capacity: 100, resizable });
        expect(g.length).toBe(0);
        expect(g.capacity).toBe(112); // 100 -> 112 elements (16-granule) = 448 bytes, a multiple of 64
        expect(g.array.buffer.byteLength).toBe(448);
        const bytes = new GrowableTypedArray(Uint8Array, { capacity: 10, resizable });
        expect(bytes.capacity).toBe(64);
        const doubles = new GrowableTypedArray(Float64Array, { capacity: 3, resizable });
        expect(doubles.capacity).toBe(16);
        expect(doubles.array.buffer.byteLength).toBe(128);
    });

    it("grows by doubling and keeps every element", () => {
        const g = new GrowableTypedArray(Uint32Array, { resizable });
        const capacities = new Set<number>();
        for (let i = 0; i < 1000; i++) {
            expect(g.push(i * 3)).toBe(i);
            capacities.add(g.capacity);
        }
        expect(g.length).toBe(1000);
        expect([...capacities]).toEqual([16, 32, 64, 128, 256, 512, 1024]);
        for (let i = 0; i < 1000; i++) {
            expect(g.get(i)).toBe(i * 3);
        }
        expect(g.array.buffer.byteLength % 64).toBe(0);
        expect(g.array.buffer.resizable).toBe(resizable);
    });

    it("pushAll appends a bulk range and returns its first index", () => {
        const g = new GrowableTypedArray(Float32Array, { resizable });
        expect(g.pushAll([1, 2, 3])).toBe(0);
        expect(g.pushAll(new Float32Array(100).fill(0.5))).toBe(3);
        expect(g.pushAll([])).toBe(103);
        expect(g.length).toBe(103);
        expect(g.capacity).toBeGreaterThanOrEqual(103);
        expect(Array.from(g.view().subarray(0, 4))).toEqual([1, 2, 3, 0.5]);
        expect(g.get(102)).toBe(0.5);
    });

    it("set writes in place and get reads it back", () => {
        const g = new GrowableTypedArray(Int32Array, { resizable });
        g.resize(10);
        g.set(9, -5);
        expect(g.get(9)).toBe(-5);
        expect(g.array[9]).toBe(-5);
    });

    it("resize grows with the fill value and shrinks without releasing capacity", () => {
        const g = new GrowableTypedArray(Uint32Array, { resizable });
        g.resize(40, INVALID_INDEX);
        expect(g.length).toBe(40);
        expect(g.capacity).toBe(48);
        expect(Array.from(g.view()).every((v) => v === INVALID_INDEX)).toBe(true);
        g.set(3, 9);
        g.resize(5);
        expect(g.length).toBe(5);
        expect(g.capacity).toBe(48);
        expect(Array.from(g.view())).toEqual([INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, 9, INVALID_INDEX]);
        // regrowing refills only the newly exposed elements
        g.resize(8, 1);
        expect(Array.from(g.view())).toEqual([INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, 9, INVALID_INDEX, 1, 1, 1]);
        g.resize(100);
        expect(g.length).toBe(100);
        expect(g.capacity).toBe(112);
        expect(Array.from(g.view().subarray(8)).every((v) => v === 0)).toBe(true);
        expect(g.get(3)).toBe(9);
        g.resize(0);
        expect(g.length).toBe(0);
        expect(g.view().length).toBe(0);
    });

    it("ensureCapacity reserves without changing length", () => {
        const g = new GrowableTypedArray(Uint8Array, { resizable });
        g.push(1);
        g.ensureCapacity(1000);
        expect(g.length).toBe(1);
        expect(g.capacity).toBeGreaterThanOrEqual(1000);
        expect(g.capacity % 64).toBe(0);
        const { array } = g;
        g.ensureCapacity(500);
        expect(g.array).toBe(array);
        expect(g.get(0)).toBe(1);
    });

    it("view is a zero-copy window of exactly length elements over the staging buffer", () => {
        const g = new GrowableTypedArray(Uint32Array, { resizable });
        g.pushAll([1, 2, 3]);
        const view = g.view();
        expect(view).toBeInstanceOf(Uint32Array);
        expect(view.length).toBe(3);
        expect(view.byteOffset).toBe(0);
        expect(view.buffer).toBe(g.array.buffer);
        view[1] = 20;
        expect(g.get(1)).toBe(20);
        g.set(2, 30);
        expect(view[2]).toBe(30);
    });

    it("trim is an exact-length copy over a fresh fixed-length buffer (invariant I18)", () => {
        const g = new GrowableTypedArray(Float64Array, { resizable });
        g.pushAll([1.25, 2.5, 3.75]);
        const trimmed = g.trim();
        expect(trimmed).toBeInstanceOf(Float64Array);
        expect(trimmed.length).toBe(3);
        expect(trimmed.byteOffset).toBe(0);
        expect(trimmed.buffer.byteLength).toBe(24);
        expect(trimmed.buffer.resizable).toBe(false);
        expect(trimmed.buffer).not.toBe(g.array.buffer);
        expect(Array.from(trimmed)).toEqual([1.25, 2.5, 3.75]);
        g.set(0, 99);
        expect(trimmed[0]).toBe(1.25);
        const empty = new GrowableTypedArray(Uint32Array, { resizable }).trim();
        expect(empty.length).toBe(0);
        expect(empty.buffer.byteLength).toBe(0);
    });

    it("clear keeps the capacity, release drops it, and both leave the array usable", () => {
        const g = new GrowableTypedArray(Uint32Array, { resizable });
        g.pushAll(new Uint32Array(100).fill(5));
        const { capacity } = g;
        g.clear();
        expect(g.length).toBe(0);
        expect(g.capacity).toBe(capacity);
        expect(g.push(1)).toBe(0);
        expect(g.get(0)).toBe(1);
        g.release();
        expect(g.length).toBe(0);
        expect(g.capacity).toBe(0);
        expect(g.array.buffer.byteLength).toBe(0);
        expect(g.push(2)).toBe(0);
        expect(g.get(0)).toBe(2);
        expect(g.capacity).toBe(16);
    });

    it("agrees with a plain array model under random push / set / resize / clear sequences", () => {
        const op = fc.oneof(
            fc.record({ kind: fc.constant("push" as const), value: fc.integer({ min: 0, max: 1000 }) }),
            fc.record({ kind: fc.constant("pushAll" as const), values: fc.array(fc.integer({ min: 0, max: 1000 })) }),
            fc.record({ kind: fc.constant("resize" as const), length: fc.nat(300), fill: fc.nat(9) }),
            fc.record({ kind: fc.constant("set" as const), at: fc.nat(300), value: fc.nat(1000) }),
            fc.record({ kind: fc.constant("clear" as const) }),
        );
        fc.assert(
            fc.property(fc.array(op, { maxLength: 60 }), (ops) => {
                const g = new GrowableTypedArray(Uint32Array, { resizable });
                const model: number[] = [];
                for (const o of ops) {
                    switch (o.kind) {
                        case "push":
                            expect(g.push(o.value)).toBe(model.length);
                            model.push(o.value);
                            break;
                        case "pushAll":
                            expect(g.pushAll(o.values)).toBe(model.length);
                            model.push(...o.values);
                            break;
                        case "resize":
                            g.resize(o.length, o.fill);
                            while (model.length < o.length) {
                                model.push(o.fill);
                            }
                            model.length = o.length;
                            break;
                        case "set":
                            if (o.at < model.length) {
                                g.set(o.at, o.value);
                                model[o.at] = o.value;
                            }
                            break;
                        case "clear":
                            g.clear();
                            model.length = 0;
                            break;
                        default:
                            throw new Error("unreachable");
                    }
                    expect(g.length).toBe(model.length);
                    expect(g.capacity).toBeGreaterThanOrEqual(model.length);
                }
                expect(Array.from(g.view())).toEqual(model);
                expect(Array.from(g.trim())).toEqual(model);
            }),
        );
    });
});

describe("GrowableTypedArray growth mechanics", () => {
    it("keeps the same buffer while growing in place on the resizable path", () => {
        expect(HAS_RESIZABLE_ARRAY_BUFFER).toBe(true);
        const g = new GrowableTypedArray(Uint32Array, { resizable: true });
        g.push(1);
        const { buffer } = g.array;
        expect(buffer.resizable).toBe(true);
        expect(buffer.maxByteLength).toBe(STAGING_RESERVE_BYTES);
        for (let i = 1; i < 5000; i++) {
            g.push(i);
        }
        expect(g.array.buffer).toBe(buffer);
        expect(buffer.byteLength).toBe(g.capacity * 4);
        expect(g.get(4999)).toBe(4999);
    });

    it("replaces the buffer on every growth of the copy path", () => {
        const g = new GrowableTypedArray(Uint32Array, { resizable: false });
        g.push(1);
        const first = g.array.buffer;
        expect(first.resizable).toBe(false);
        for (let i = 1; i < 17; i++) {
            g.push(i);
        }
        expect(g.array.buffer).not.toBe(first);
        expect(g.array.buffer.resizable).toBe(false);
        expect(g.capacity).toBe(32);
        expect(Array.from(g.view())).toEqual(Array.from({ length: 17 }, (_, i) => (i === 0 ? 1 : i)));
    });

    it("defaults to the detected strategy", () => {
        const g = new GrowableTypedArray(Uint32Array);
        g.push(1);
        expect(g.array.buffer.resizable).toBe(HAS_RESIZABLE_ARRAY_BUFFER);
    });
});

describe.each(STRATEGIES)("GrowableBitmap ($name)", ({ resizable }) => {
    it("pushes bits across word boundaries and reads them back", () => {
        const b = new GrowableBitmap({ resizable });
        expect(b.length).toBe(0);
        expect(b.view().length).toBe(0);
        const pattern: boolean[] = [];
        for (let i = 0; i < 100; i++) {
            const value = i % 3 === 0 || i === 31 || i === 32 || i === 99;
            expect(b.push(value)).toBe(i);
            pattern.push(value);
        }
        expect(b.length).toBe(100);
        expect(b.view().length).toBe(4);
        for (let i = 0; i < 100; i++) {
            expect(b.get(i)).toBe(pattern[i]);
        }
        expect(b.count()).toBe(pattern.filter((v) => v).length);
        expect(Array.from(bitmapToIndices(b.view(), 100))).toEqual(pattern.flatMap((v, i) => (v ? [i] : [])));
        // the words beyond the length are clear
        const words = b.view();
        for (let i = 100; i < 128; i++) {
            expect((words[i >>> 5] >>> (i & 31)) & 1).toBe(0);
        }
    });

    it("set writes a bit in place and count stays exact", () => {
        const b = new GrowableBitmap({ resizable });
        b.resize(70);
        expect(b.count()).toBe(0);
        b.set(0, true);
        b.set(31, true);
        b.set(32, true);
        b.set(69, true);
        expect(Array.from(b.view())).toEqual([0x80000001, 1, 0x20]);
        expect(b.count()).toBe(4);
        b.set(31, false);
        expect(b.get(31)).toBe(false);
        expect(b.count()).toBe(3);
    });

    it("resize grows with the fill value across ragged and whole words", () => {
        for (const [from, to] of [
            [0, 1],
            [0, 32],
            [0, 33],
            [5, 10],
            [5, 70],
            [31, 32],
            [31, 33],
            [32, 64],
            [32, 65],
            [33, 100],
            [64, 64],
        ] as const) {
            const filled = new GrowableBitmap({ resizable });
            filled.resize(from);
            filled.resize(to, true);
            expect(filled.length).toBe(to);
            expect(filled.count()).toBe(to - from);
            const words = filled.view();
            expect(words.length).toBe(Math.ceil(to / 32));
            for (let i = 0; i < words.length * 32; i++) {
                const expected = i >= from && i < to;
                expect(((words[i >>> 5] >>> (i & 31)) & 1) === 1).toBe(expected);
            }
            const clear = new GrowableBitmap({ resizable });
            clear.resize(from, true);
            clear.resize(to);
            expect(clear.count()).toBe(from);
            expect(bitmapCount(clear.view(), to)).toBe(from);
        }
    });

    it("resize shrinks and clears the dropped bits so a later regrowth starts clear", () => {
        const b = new GrowableBitmap({ resizable });
        b.resize(100, true);
        b.resize(33);
        expect(b.length).toBe(33);
        expect(Array.from(b.view())).toEqual([0xffffffff, 1]);
        expect(b.count()).toBe(33);
        b.resize(70);
        expect(b.count()).toBe(33);
        expect(Array.from(b.view())).toEqual([0xffffffff, 1, 0]);
        b.resize(0);
        expect(b.length).toBe(0);
        expect(b.view().length).toBe(0);
        b.resize(40);
        expect(b.count()).toBe(0);
        expect(Array.from(b.view())).toEqual([0, 0]);
    });

    it("push after a shrink writes into a clear tail", () => {
        const b = new GrowableBitmap({ resizable });
        b.resize(64, true);
        b.resize(30);
        expect(b.push(false)).toBe(30);
        expect(b.push(true)).toBe(31);
        expect(b.push(false)).toBe(32);
        expect(Array.from(b.view())).toEqual([0xbfffffff, 0]);
        expect(b.count()).toBe(31);
    });

    it("view aliases staging and trim copies into a plain fixed-length buffer", () => {
        const b = new GrowableBitmap({ resizable, capacity: 100 });
        b.resize(40);
        b.set(39, true);
        const view = b.view();
        expect(view.length).toBe(2);
        expect(view.buffer.resizable).toBe(resizable);
        view[0] = 1;
        expect(b.get(0)).toBe(true);
        const trimmed = b.trim();
        expect(trimmed.length).toBe(2);
        expect(trimmed.buffer.resizable).toBe(false);
        expect(trimmed.buffer.byteLength).toBe(8);
        expect(Array.from(trimmed)).toEqual([1, 0x80]);
        b.set(0, false);
        expect(trimmed[0]).toBe(1);
    });

    it("clear and release reset the bitmap and keep it usable", () => {
        const b = new GrowableBitmap({ resizable });
        b.resize(100, true);
        b.clear();
        expect(b.length).toBe(0);
        expect(b.count()).toBe(0);
        expect(b.push(true)).toBe(0);
        expect(b.count()).toBe(1);
        expect(Array.from(b.view())).toEqual([1]);
        b.release();
        expect(b.length).toBe(0);
        expect(b.view().length).toBe(0);
        expect(b.push(false)).toBe(0);
        expect(b.push(true)).toBe(1);
        expect(Array.from(b.view())).toEqual([2]);
    });

    it("agrees with a boolean array model under random push / set / resize sequences", () => {
        const op = fc.oneof(
            fc.record({ kind: fc.constant("push" as const), value: fc.boolean() }),
            fc.record({ kind: fc.constant("resize" as const), length: fc.nat(200), fill: fc.boolean() }),
            fc.record({ kind: fc.constant("set" as const), at: fc.nat(200), value: fc.boolean() }),
        );
        fc.assert(
            fc.property(fc.array(op, { maxLength: 80 }), (ops) => {
                const b = new GrowableBitmap({ resizable });
                const model: boolean[] = [];
                for (const o of ops) {
                    switch (o.kind) {
                        case "push":
                            expect(b.push(o.value)).toBe(model.length);
                            model.push(o.value);
                            break;
                        case "resize":
                            b.resize(o.length, o.fill);
                            while (model.length < o.length) {
                                model.push(o.fill);
                            }
                            model.length = o.length;
                            break;
                        case "set":
                            if (o.at < model.length) {
                                b.set(o.at, o.value);
                                model[o.at] = o.value;
                            }
                            break;
                        default:
                            throw new Error("unreachable");
                    }
                    expect(b.length).toBe(model.length);
                }
                expect(b.count()).toBe(model.filter((v) => v).length);
                const words = b.trim();
                expect(words.length).toBe(Math.ceil(model.length / 32));
                for (let i = 0; i < words.length * 32; i++) {
                    const expected = i < model.length ? model[i] : false;
                    expect(((words[i >>> 5] >>> (i & 31)) & 1) === 1).toBe(expected);
                }
            }),
        );
    });
});
