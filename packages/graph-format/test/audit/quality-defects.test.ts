/**
 * Adversarial quality audit: each test pins one defect found by reading src/ against the design.
 * Every test here asserts the behaviour the design promises; a test that FAILS documents a live
 * defect (listed in the audit report with the file and line of the cause). Nothing in src is
 * changed by this file.
 */

import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import { fromEdgeArrays } from "../../src/populate/from-edge-arrays.js";
import { type GraphSnapshot } from "../../src/snapshot/graph-snapshot.js";
import { fromBytes } from "../../src/wire/bytes.js";

/** Run fn and return the GraphFormatError it throws, asserting the code. */
function expectError(fn: () => unknown, code: string): GraphFormatError {
    let caught: unknown = null;
    try {
        fn();
    } catch (err) {
        caught = err;
    }
    expect(caught).toBeInstanceOf(GraphFormatError);
    const error = caught as GraphFormatError;
    expect(error.code).toBe(code);
    return error;
}

/** A 3-node directed path 0 -> 1 -> 2. */
function pathBuilder(): GraphBuilder {
    const b = new GraphBuilder({ directed: true });
    b.addAnonymousNodes(3);
    b.addEdgeByIndex(0, 1);
    b.addEdgeByIndex(1, 2);
    return b;
}

describe("builder: refersTo values are never checked, so freeze() emits snapshots that fail validate()", () => {
    it("a refersTo node column holding an index >= nodeCount is refused at the write, so the freeze satisfies I12 (src/builder/graph-builder.ts checkReference)", () => {
        const b = pathBuilder();
        const parent = b.declareNodeColumn({ name: "parent", dtype: "u32", refersTo: "node" });
        // design 11.1: malformed input is refused at the call; design 3.2 / 9.5: the builder
        // establishes I1-I13 by construction, so validate() must pass on what it freezes
        expectError(() => b.setNodeValue(parent, 0, 999), "E_INDEX_RANGE");
        expectError(() => b.setNodeValue(parent, 0, -1), "E_INDEX_RANGE");
        expectError(() => b.setNodeColumn("p2", new Uint32Array([0, 3, 1]), { refersTo: "node" }), "E_INDEX_RANGE");
        b.setNodeValue(parent, 0, 2);
        b.setNodeValue(parent, 1, INVALID_INDEX);
        const s = b.freeze();
        expect(() => s.validate({ level: "structure" })).not.toThrow();
        expect(s.nodes.value("parent", 0)).toBe(2);
        expect(s.nodes.isSet("parent", 1)).toBe(false);
        expect(s.nodes.get("p2")).toBeNull();
    });

    it("a non-nullable refersTo column whose target is removed keeps INVALID_INDEX in a SET row (src/builder/compact.ts gather)", () => {
        const b = pathBuilder();
        const parent = b.declareNodeColumn({ name: "parent", dtype: "u32", refersTo: "node", nullable: false });
        b.setNodeValue(parent, 0, 2);
        b.setNodeValue(parent, 1, 2);
        b.setNodeValue(parent, 2, 2);
        b.removeNodeByIndex(2);
        const s = b.freeze();
        const column = s.nodes.requireTyped("parent", "u32");
        // design 5.11: a dangling reference becomes INVALID_INDEX WITH THE ROW UNSET
        expect(column.data[0]).toBe(INVALID_INDEX);
        expect(column.isSet(0)).toBe(false);
        expect(() => s.validate({ level: "structure" })).not.toThrow();
    });
});

describe("AttributeTable.set(): a dict typed array is adopted without checking its codes", () => {
    it("codes beyond the dictionary are accepted, value() returns undefined for a set row and validate() fails I12 (src/columns/column.ts columnFromTypedArray)", () => {
        const s = fromEdgeArrays({
            directed: true,
            nodeCount: 2,
            src: new Uint32Array([0]),
            dst: new Uint32Array([1]),
        });
        // the builder's setNodeColumn rejects the same input with E_COLUMN_TYPE; the table must agree
        const b = new GraphBuilder({ directed: true });
        b.addAnonymousNodes(2);
        expectError(() => b.setNodeColumn("d", new Uint32Array([7, 7]), { dtype: "dict" }), "E_COLUMN_TYPE");
        expectError(() => s.nodes.set("d", new Uint32Array([7, 7]), { dtype: "dict" }), "E_COLUMN_TYPE");
    });
});

describe("wire: the $num tag of design 5.9 collides with user JSON", () => {
    function withJsonCell(value: unknown): GraphSnapshot {
        const b = new GraphBuilder({ directed: true });
        b.addAnonymousNodes(1);
        const h = b.declareNodeColumn({ name: "j", dtype: "json" });
        b.setNodeValue(h, 0, value);
        return b.freeze();
    }

    it("a json cell { $num: 'text' } is a legal JSON value that must round-trip through toBytes / fromBytes (src/wire/from-wire.ts decodeJsonValue)", () => {
        const s = withJsonCell({ $num: "text" });
        const back = fromBytes(s.toBytes());
        expect(back.nodes.require("j").value(0)).toEqual({ $num: "text" });
    });

    it("a json cell { $num: 'NaN' } must come back as the object, not as the number NaN (src/wire/to-wire.ts encodeJsonValue never escapes)", () => {
        const s = withJsonCell({ $num: "NaN" });
        const back = fromBytes(s.toBytes());
        expect(back.nodes.require("j").value(0)).toEqual({ $num: "NaN" });
    });
});

describe("owner count (design 9.1): Column.slice() aliasing is invisible to transferables()", () => {
    it("a zero-copy slice attached to another snapshot lets transferables() list a buffer the source still views (src/columns/column.ts sliceData, src/columns/table.ts set)", () => {
        const source = fromEdgeArrays({
            directed: true,
            nodeCount: 4,
            src: new Uint32Array([0, 1, 2]),
            dst: new Uint32Array([1, 2, 3]),
            nodeColumns: { x: new Uint32Array([10, 20, 30, 40]) },
        });
        const sub = source.inducedSubgraph(new Uint32Array([0, 1])).snapshot;
        const slice = source.nodes.requireTyped("x", "u32").slice(0, 2);
        sub.nodes.set("y", slice);
        const sourceBuffer = source.nodes.requireTyped("x", "u32").data.buffer;
        expect(sub.nodes.requireTyped("y", "u32").data.buffer).toBe(sourceBuffer);
        // 9.1: only EXCLUSIVELY owned buffers are transferable; this one is still viewed by `source`
        expect(sub.transferables()).not.toContain(sourceBuffer);
        const wire = sub.toWire({ transfer: true });
        expect(wire.buffers).not.toContain(sourceBuffer);
    });
});

describe("builder: addGraph() is not atomic", () => {
    it("onDuplicateNode 'error' throws after earlier nodes were already added (src/builder/graph-builder.ts addGraph)", () => {
        const incoming = fromEdgeArrays({
            directed: true,
            ids: ["fresh", "dup"],
            src: new Uint32Array([0]),
            dst: new Uint32Array([1]),
        });
        const b = new GraphBuilder({ directed: true });
        b.addNode("dup");
        const before = b.nodeCount;
        expectError(() => b.addGraph(incoming, { onDuplicateNode: "error" }), "E_DUPLICATE_ID");
        // class contract (design 11.1): "the failing operation is not applied"
        expect(b.nodeCount).toBe(before);
        expect(b.hasNode("fresh")).toBe(false);
    });
});

describe("derived graphs: integer column reducers wrap silently", () => {
    it("contract(..., edgeReducers: { w: 'sum' }) on a u8 column wraps modulo 256 instead of widening or throwing (src/snapshot/derived.ts reduceNumericColumn)", () => {
        const s = fromEdgeArrays({
            directed: true,
            nodeCount: 3,
            src: new Uint32Array([0, 0]),
            dst: new Uint32Array([1, 1]),
            edgeColumns: { w: new Uint8Array([200, 100]) },
        });
        const derived = s.contract(new Uint32Array([0, 1, 2]), { edgeReducers: { w: "sum" } });
        const column = derived.snapshot.edges.require("w");
        // 300 is not representable in u8: every write path rejects that with E_COLUMN_TYPE
        // (src/columns/column.ts writeNumeric); the reducer must not silently store 44
        expect(column.value(0)).toBe(300);
    });
});

describe("bitmap convention: src/columns/bitmap.ts promises every bit >= length is clear", () => {
    it("AttributeTable.set() with a JS array leaves the trailing validity bits SET (src/columns/column.ts columnOfValues fill(0xffffffff))", () => {
        const s = fromEdgeArrays({
            directed: true,
            nodeCount: 3,
            src: new Uint32Array([0]),
            dst: new Uint32Array([1]),
        });
        const viaSet = s.nodes.set("v", [1, undefined, 3], { dtype: "i32" });
        // the builder path (fromEdgeArrays nodeColumns) produces 0b101 for the same logical column
        const viaBuilder = fromEdgeArrays({
            directed: true,
            nodeCount: 3,
            src: new Uint32Array([0]),
            dst: new Uint32Array([1]),
            nodeColumns: { v: { data: [1, undefined, 3], decl: { dtype: "i32" } } },
        }).nodes.require("v");
        expect(viaBuilder.validity?.[0]).toBe(0b101);
        expect(viaSet.validity?.[0]).toBe(0b101);
    });
});

describe("Object.freeze gaps on frozen snapshots", () => {
    it("ColumnMeta.default is a live reference: mutating it changes value() of an immutable column (src/columns/column.ts resolveColumnMeta)", () => {
        const b = new GraphBuilder({ directed: true });
        b.addAnonymousNodes(1);
        b.declareNodeColumn({ name: "d", dtype: "json", default: [1, 2] });
        const s = b.freeze();
        const column = s.nodes.require("d");
        expect(Object.isFrozen(column.meta)).toBe(true);
        (column.meta.default as number[]).push(3);
        expect(column.value(0)).toEqual([1, 2]);
    });

    it("cached view objects are plain mutable objects: a consumer can poison the shared cache (src/snapshot/views.ts cooViewOf)", () => {
        const s = fromEdgeArrays({
            directed: true,
            nodeCount: 2,
            src: new Uint32Array([0]),
            dst: new Uint32Array([1]),
        });
        const coo = s.coo() as { src: Uint32Array };
        const original = coo.src;
        try {
            coo.src = new Uint32Array([7]);
        } catch {
            // a frozen view throws in strict mode: the desired outcome
        }
        expect(s.coo().src).toBe(original);
    });
});
