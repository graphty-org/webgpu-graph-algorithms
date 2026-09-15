/**
 * Audit (API conformance lens): semantics the design states in prose next to the section 12.2
 * signatures, checked against the implementation. Sections covered: 3.9 (queries), 5.5 / 5.7
 * (table set / replaceRole), 6.5 / 6.6 (builder), 7.2 (views), 7.3 (derived graphs), 8.1 (entry
 * points), 9.1 / 9.2 (wire and container), 11.1 / 11.3 (error policy and the situation table).
 *
 * The last describe block holds DELIBERATELY FAILING tests that pin defects found by the audit
 * (see packages/CONFORMANCE.md, findings F1-F5); each names the design clause it enforces. They
 * are expected to fail until src is fixed, and they must not be "fixed" by weakening the assertion.
 */

import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { remapColumn } from "../../src/columns/remap.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError, type GraphFormatErrorCode } from "../../src/errors.js";
import { fromCsr } from "../../src/populate/from-csr.js";
import { fromEdgeArrays } from "../../src/populate/from-edge-arrays.js";
import { fromRecords } from "../../src/populate/from-records.js";
import { type GraphSnapshot } from "../../src/snapshot/graph-snapshot.js";
import { type F32Column, type U32Column, type WireColumn } from "../../src/types/index.js";
import { makeMask, maskSet } from "../../src/util/mask.js";
import { fromBytes } from "../../src/wire/bytes.js";
import { fromWire } from "../../src/wire/from-wire.js";
import { expectError } from "../wire/helpers.js";

function directedFixture(): GraphSnapshot {
    const b = new GraphBuilder({ directed: true });
    // 0: a->b 1, 1: b->a 2 (reciprocal), 2: a->c 3, 3: a->c 4 (parallel), 4: c->c 5 (loop), 5: c->b 6
    b.addEdge("a", "b", 1);
    b.addEdge("b", "a", 2);
    b.addEdge("a", "c", 3);
    b.addEdge("a", "c", 4);
    b.addEdge("c", "c", 5);
    b.addEdge("c", "b", 6);
    return b.freeze();
}

/** A WireColumn whose dtype no reader of this version knows. */
function unknownDtypeColumn(): WireColumn {
    return {
        meta: {
            name: "future",
            domain: "node",
            dtype: "f16" as never,
            components: 1,
            itemDtype: null,
            itemComponents: null,
            nullable: false,
            mutable: false,
            role: null,
            refersTo: null,
            unique: false,
            default: undefined,
            fill: 0,
            options: null,
            origin: null,
            dynamic: false,
            extra: {},
        },
        data: null,
        validity: null,
        nullCount: 0,
        dictionary: null,
        strings: null,
        offsets: null,
        child: null,
        jsonText: null,
    };
}

describe("design section 11.2 / 11.3: every core error code is reachable through a public call", () => {
    // E_IMPORT is reserved for @graphty/graph-io (section 11.2) and has no throw site here.
    const triggers: Record<Exclude<GraphFormatErrorCode, "E_IMPORT">, () => unknown> = {
        E_INVALID_ID: () => new GraphBuilder({ directed: true }).addNode(Number.NaN),
        E_UNKNOWN_NODE: () => new GraphBuilder({ directed: true, addMissingNodes: false }).addEdge("a", "b"),
        E_INDEX_RANGE: () => directedFixture().ids.idOf(99),
        E_TOO_LARGE: () => new GraphBuilder({ directed: true }).reserve(-1),
        E_INVALID_WEIGHT: () => new GraphBuilder({ directed: true }).addEdge("a", "b", Number.NaN),
        E_DIRECTED: () => directedFixture().mate(),
        E_SELF_LOOP: () => {
            const b = new GraphBuilder({ directed: true, selfLoops: "error" });
            b.addEdge("a", "a");
            b.freeze();
        },
        E_DUPLICATE_EDGE: () => {
            const b = new GraphBuilder({ directed: true, duplicateEdges: "error" });
            b.addEdge("a", "b");
            b.addEdge("a", "b");
            b.freeze();
        },
        E_DUPLICATE_EDGE_ID: () => {
            const b = new GraphBuilder({ directed: true });
            b.declareEdgeColumn({ name: "id", dtype: "string", unique: true, role: "id" });
            b.setEdgeValue("id", b.addEdge("a", "b"), "x");
            b.setEdgeValue("id", b.addEdge("b", "c"), "x");
            b.freeze();
        },
        E_DUPLICATE_ID: () => {
            const b = new GraphBuilder({ directed: true });
            b.addNode("a");
            b.addGraph(directedFixture(), { onDuplicateNode: "error" });
        },
        E_DUPLICATE_ROLE: () => {
            const s = directedFixture();
            s.nodes.set("p", new Float32Array(3), { role: "position" });
            s.nodes.set("q", new Float32Array(3), { role: "position" });
        },
        E_UNKNOWN_COLUMN: () => directedFixture().nodes.require("nope"),
        E_COLUMN_TYPE: () => {
            const s = directedFixture();
            s.nodes.set("f", new Float32Array(3));
            s.nodes.requireTyped("f", "u32");
        },
        E_COLUMN_LENGTH: () => directedFixture().nodes.set("x", new Float32Array(2)),
        E_COLUMN_ALIGNMENT: () =>
            directedFixture().nodes.set("u", new Uint8Array(new ArrayBuffer(7), 1, 3), undefined, { adopt: "strict" }),
        E_COLUMN_EXISTS: () => {
            const b = new GraphBuilder({ directed: true });
            b.declareNodeColumn({ name: "c", dtype: "f32" });
            b.declareNodeColumn({ name: "c", dtype: "u32" });
        },
        E_COLUMN_IMMUTABLE: () => {
            const s = directedFixture();
            (s.nodes.set("f", new Float32Array(3)) as F32Column).mutableData();
        },
        E_NO_DEFAULT: () => {
            const s = directedFixture();
            (s.nodes.set("f", new Float32Array(3)) as F32Column).materializeDefault();
        },
        E_PARTITION: () => directedFixture().contract(new Uint32Array(2)),
        E_INVALID_PERMUTATION: () => directedFixture().relabel(new Uint32Array([0, 0, 1])),
        E_MASK_LENGTH: () => directedFixture().filterEdges(new Uint32Array(0)),
        E_GPU_INELIGIBLE: () => {
            const s = directedFixture();
            s.nodes.set("s", ["a", "b", "c"], { dtype: "string" });
            s.nodes.gpuView("s");
        },
        E_INVALID_SNAPSHOT: () => directedFixture().validate({ checksum: true }),
        E_BAD_SERIALIZATION: () => {
            const bytes = directedFixture().toBytes();
            bytes[0] = 0;
            fromBytes(bytes);
        },
        E_UNSUPPORTED_VERSION: () => {
            const wire = directedFixture().toWire();
            fromWire({ manifest: { ...wire.manifest, formatVersion: 2 as never }, buffers: wire.buffers });
        },
        E_DETACHED: () => {
            const s = directedFixture();
            const wire = s.toWire({ transfer: true });
            structuredClone(wire, { transfer: [...wire.buffers] });
            s.outDegree();
        },
        E_BUILDER_DISPOSED: () => {
            const b = new GraphBuilder({ directed: true });
            b.dispose();
            b.addNode("a");
        },
        E_UNSUPPORTED: () => {
            const wire = directedFixture().toWire();
            fromWire({ manifest: { ...wire.manifest, nodeColumns: [unknownDtypeColumn()] }, buffers: wire.buffers });
        },
    };

    for (const [code, trigger] of Object.entries(triggers)) {
        it(`throws ${code} from a public entry point`, () => {
            expectError(trigger, code);
        });
    }

    it("throws only GraphFormatError instances with a frozen details record (11.1, 11.2)", () => {
        const error = expectError(() => directedFixture().ids.idOf(99), "E_INDEX_RANGE");
        expect(error).toBeInstanceOf(Error);
        expect(error.name).toBe("GraphFormatError");
        expect(Object.isFrozen(error.details)).toBe(true);
        expect(error.details).toMatchObject({ index: 99 });
    });
});

describe("design section 11.3: the situation table", () => {
    it("rejects NaN, Infinity, bigint, object, null and undefined ids with E_INVALID_ID", () => {
        for (const id of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 10n, {}, null, undefined]) {
            expectError(() => new GraphBuilder({ directed: true }).addNode(id as never), "E_INVALID_ID");
        }
        const lone = expectError(() => new GraphBuilder({ directed: true }).addNode("\ud800"), "E_INVALID_ID");
        expect(lone.details.reason).toBe("lone surrogate");
    });

    it("stores -0 as 0 and keeps '1' and 1 distinct (SameValueZero)", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNode(-0);
        expect(Object.is(b.idOf(0), 0)).toBe(true);
        expect(b.indexOf(0)).toBe(0);
        expect(b.addNode("1")).toBe(1);
        expect(b.addNode(1)).toBe(2);
    });

    it("creates unknown endpoints by default and throws E_UNKNOWN_NODE with addMissingNodes false", () => {
        expect(new GraphBuilder({ directed: true }).addEdge("x", "y")).toBe(0);
        const error = expectError(
            () => new GraphBuilder({ directed: true, addMissingNodes: false }).addEdge("x", "y"),
            "E_UNKNOWN_NODE",
        );
        expect(error.details.id).toBe("x");
    });

    it("addEdgeByIndex with a dead or out-of-range index is E_UNKNOWN_NODE with details.index", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNode("a");
        b.addNode("b");
        b.removeNodeByIndex(1);
        expect(expectError(() => b.addEdgeByIndex(0, 1), "E_UNKNOWN_NODE").details.index).toBe(1);
        expect(expectError(() => b.addEdgeByIndex(0, 99), "E_UNKNOWN_NODE").details.index).toBe(99);
    });

    it("rejects NaN weights at addEdge, setEdgeWeight and addEdges; accepts Infinity, negative and 0 and reports them in flags", () => {
        const b = new GraphBuilder({ directed: true });
        expectError(() => b.addEdge("a", "b", Number.NaN), "E_INVALID_WEIGHT");
        const e = b.addEdge("a", "b", 2);
        expectError(() => b.setEdgeWeight(e, Number.NaN), "E_INVALID_WEIGHT");
        expectError(
            () => b.addEdges(new Uint32Array([0]), new Uint32Array([1]), new Float32Array([Number.NaN])),
            "E_INVALID_WEIGHT",
        );
        b.addEdge("a", "c", Number.POSITIVE_INFINITY);
        b.addEdge("a", "d", -1);
        b.addEdge("a", "e", 0);
        const omitted = b.addEdge("a", "f");
        expect(b.edgeWeight(omitted)).toBe(1);
        const s = b.freeze();
        expect(s.flags).toMatchObject({
            weighted: true,
            allWeightsOne: false,
            nonNegativeWeights: false,
            finiteWeights: false,
        });
    });

    it("freezes an empty builder to the documented empty snapshot", () => {
        const s = new GraphBuilder({ directed: true }).freeze();
        expect(s.nodeCount).toBe(0);
        expect(Array.from(s.rowPtr)).toEqual([0]);
        expect(s.colIdx.length).toBe(0);
        expect(s.arcToEdge.length).toBe(0);
        expect(s.edgeToArc.length).toBe(0);
        expect(s.weights).toBeNull();
        expect(() => {
            s.validate({ level: "full" });
        }).not.toThrow();
    });

    it("setDirected on an empty unlocked builder takes effect; refusals are E_DIRECTED", () => {
        const b = new GraphBuilder({ directed: true });
        b.setDirected(false);
        expect(b.directed).toBe(false);
        expect(b.options.directed).toBe(false);
        b.addEdge("a", "b", 3);
        expectError(() => b.setDirected(true), "E_DIRECTED");
        b.setDirected(true, { expand: true });
        expect(b.edgeCount).toBe(2);
        expect(b.edgeEndpoints(1)).toEqual([1, 0]);
        expect(b.edgeWeight(1)).toBe(3);
        expectError(() => b.setDirected(false), "E_DIRECTED");
        b.lockDirected();
        expect(b.directedLocked).toBe(true);
        expect(() => {
            b.setDirected(true);
        }).not.toThrow();
        const locked = new GraphBuilder({ directed: true });
        locked.lockDirected();
        expectError(() => locked.setDirected(false), "E_DIRECTED");
        const s = b.freeze();
        expect(s.edges.byRole("directed")?.meta.name).toBe("graphty.directed");
        const pair = s.edges.byRole("pair") as U32Column;
        expect(pair.meta).toMatchObject({ name: "graphty.pair", dtype: "u32", refersTo: "edge" });
        expect(Array.from(pair.data)).toEqual([1, 0]);
    });

    it("table.set: wrong length, u8 alignment under strict, unconditional 4-byte and f64 adoption, replacement", () => {
        const s = directedFixture();
        expectError(() => s.nodes.set("x", new Float32Array(2)), "E_COLUMN_LENGTH");
        const u8 = new Uint8Array(new ArrayBuffer(7), 1, 3);
        expectError(() => s.nodes.set("u", u8, undefined, { adopt: "strict" }), "E_COLUMN_ALIGNMENT");
        expect((s.nodes.set("u", u8) as { data: Uint8Array }).data).not.toBe(u8);
        const f32 = new Float32Array(3);
        expect((s.nodes.set("f", f32) as F32Column).data).toBe(f32);
        const f64 = new Float64Array(3);
        expect((s.nodes.set("d", f64) as { data: Float64Array }).data).toBe(f64);
        const replaced = s.nodes.set("f", new Float32Array(3));
        expect(s.nodes.get("f")).toBe(replaced);
    });

    it("lookups: typed mismatch / get / byRole miss are null; require is E_UNKNOWN_COLUMN; requireTyped is E_COLUMN_TYPE", () => {
        const s = directedFixture();
        s.nodes.set("f", new Float32Array(3));
        expect(s.nodes.typed("f", "u32")).toBeNull();
        expect(s.nodes.get("nope")).toBeNull();
        expect(s.nodes.byRole("weight")).toBeNull();
        expectError(() => s.nodes.require("nope"), "E_UNKNOWN_COLUMN");
        expectError(() => s.nodes.requireTyped("f", "u32"), "E_COLUMN_TYPE");
        expectError(() => s.nodes.value("f", 99), "E_INDEX_RANGE");
        expect(new GraphBuilder({ directed: true }).nodeColumn("zz")).toBe(INVALID_INDEX);
        expect(new GraphBuilder({ directed: true }).edgeColumn("zz")).toBe(INVALID_INDEX);
    });

    it("declareNodeColumn twice: same declaration returns the handle, a different dtype or components is E_COLUMN_EXISTS", () => {
        const b = new GraphBuilder({ directed: true });
        const h = b.declareNodeColumn({ name: "c", dtype: "f32" });
        expect(b.declareNodeColumn({ name: "c", dtype: "f32" })).toBe(h);
        expectError(() => b.declareNodeColumn({ name: "c", dtype: "u32" }), "E_COLUMN_EXISTS");
        expectError(() => b.declareNodeColumn({ name: "c", dtype: "f32", components: 2 }), "E_COLUMN_EXISTS");
    });

    it("non-JSON default / extra values are E_COLUMN_TYPE with details.field; lone surrogates name the reason", () => {
        const s = directedFixture();
        expect(
            expectError(() => s.nodes.set("nd", new Float32Array(3), { default: () => 1 }), "E_COLUMN_TYPE").details
                .field,
        ).toBe("default");
        expect(
            expectError(() => s.nodes.set("ne", new Float32Array(3), { extra: { f: 1n } }), "E_COLUMN_TYPE").details
                .field,
        ).toBe("extra");
        expect(
            expectError(() => s.nodes.set("ls", ["\ud800", "b", "c"], { dtype: "string" }), "E_COLUMN_TYPE").details
                .reason,
        ).toBe("lone surrogate");
    });

    it("fromWire / fromBytes refuse an unknown formatVersion or wire major with details.kind / found / supported", () => {
        const wire = directedFixture().toWire();
        const format = expectError(
            () => fromWire({ manifest: { ...wire.manifest, formatVersion: 2 as never }, buffers: wire.buffers }),
            "E_UNSUPPORTED_VERSION",
        );
        expect(format.details).toMatchObject({ kind: "format", found: 2, supported: 1 });
        const major = expectError(
            () => fromWire({ manifest: { ...wire.manifest, wire: [2, 0] }, buffers: wire.buffers }),
            "E_UNSUPPORTED_VERSION",
        );
        expect(major.details).toMatchObject({ kind: "wire", found: 2, supported: 1 });
    });

    it("fromWire: unknown dtype is E_UNSUPPORTED unless unknownColumns skip (then listed in meta.extra); unknown id kind always throws", () => {
        const wire = directedFixture().toWire();
        const withColumn = {
            manifest: { ...wire.manifest, nodeColumns: [unknownDtypeColumn()] },
            buffers: wire.buffers,
        };
        expect(expectError(() => fromWire(withColumn), "E_UNSUPPORTED").details.dtype).toBe("f16");
        const skipped = fromWire(withColumn, { unknownColumns: "skip" });
        expect(skipped.nodes.has("future")).toBe(false);
        expect(skipped.meta.extra["graphty.skippedColumns"]).toEqual([
            { domain: "node", table: null, name: "future", dtype: "f16" },
        ]);
        const badIds = {
            manifest: { ...wire.manifest, ids: { ...wire.manifest.ids, kind: "weird" as never } },
            buffers: wire.buffers,
        };
        expect(expectError(() => fromWire(badIds), "E_UNSUPPORTED").details.kind).toBe("weird");
        expect(expectError(() => fromWire(badIds, { unknownColumns: "skip" }), "E_UNSUPPORTED").details.kind).toBe(
            "weird",
        );
    });

    it("fromBytes copies a SharedArrayBuffer container and a container at byteOffset 4, adopts at 0 and 8", () => {
        const s = directedFixture();
        const bytes = s.toBytes();
        expect(bytes.length % 256).toBe(0);
        const sab = new SharedArrayBuffer(bytes.length);
        new Uint8Array(sab).set(bytes);
        const fromSab = fromBytes(new Uint8Array(sab));
        expect(fromSab.rowPtr.buffer).toBeInstanceOf(ArrayBuffer);
        const at4 = new Uint8Array(bytes.length + 4);
        at4.set(bytes, 4);
        expect(fromBytes(new Uint8Array(at4.buffer, 4, bytes.length)).rowPtr.buffer).not.toBe(at4.buffer);
        const at8 = new Uint8Array(bytes.length + 8);
        at8.set(bytes, 8);
        const adopted8 = fromBytes(new Uint8Array(at8.buffer, 8, bytes.length));
        expect(adopted8.rowPtr.buffer).toBe(at8.buffer);
        const adopted0 = fromBytes(bytes);
        expect(adopted0.rowPtr.buffer).toBe(bytes.buffer);
        const manifestLength = new DataView(bytes.buffer, bytes.byteOffset).getUint32(12, true);
        const regionStart = 256 * Math.ceil((16 + manifestLength) / 256);
        expect(adopted0.arena?.byteOffset).toBe(regionStart);
        expect(adopted8.arena?.byteOffset).toBe(8 + regionStart);
    });

    it("dispose() makes every further call throw E_BUILDER_DISPOSED", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNode("a");
        b.dispose();
        for (const call of [
            () => b.addNode("b"),
            () => b.freeze(),
            () => b.clear(),
            () => b.byteLength(),
            () => b.indexOf("a"),
            () => b.addEdge("a", "b"),
        ]) {
            expectError(call, "E_BUILDER_DISPOSED");
        }
    });
});

describe("design section 3.9: queries on a directed multigraph with a self-loop", () => {
    const s = (() => {
        const b = new GraphBuilder({ directed: true });
        // 0: a->b, 1: a->c, 2: a->b (parallel), 3: b->b (loop), 4: c->a
        b.addEdge("a", "b", 1);
        b.addEdge("a", "c", 2);
        b.addEdge("a", "b", 3);
        b.addEdge("b", "b", 4);
        b.addEdge("c", "a", 5);
        return b.freeze();
    })();

    it("findArc is the first (lowest edge) arc, arcsBetween a half-open range, multiplicity its width, hasArc the sentinel test", () => {
        expect(s.findArc(0, 1)).toBe(0);
        expect(s.arcToEdge[s.findArc(0, 1)]).toBe(0);
        expect(s.findArc(0, 0)).toBe(INVALID_INDEX);
        expect(s.hasArc(0, 1)).toBe(true);
        expect(s.hasArc(0, 0)).toBe(false);
        expect(s.arcsBetween(0, 1)).toEqual([0, 2]);
        expect(s.arcsBetween(0, 0)).toEqual([0, 0]);
        expect(s.multiplicity(0, 1)).toBe(2);
        expect(s.outArcs(0)).toEqual([0, 3]);
        expect(s.outArcs(0)).not.toBe(s.outArcs(0));
        expect(s.outDegreeOf(0)).toBe(3);
    });

    it("arcSource / edgeSource / edgeTarget / edgeIndexOf", () => {
        expect(s.arcSource(3)).toBe(1);
        expect(s.arcSource(4)).toBe(2);
        expect(s.edgeSource(4)).toBe(2);
        expect(s.edgeTarget(4)).toBe(0);
        expect(s.edgeIndexOf("x")).toBe(INVALID_INDEX);
        expect(s.selfLoopsAt(1)).toBe(1);
        expect(s.selfLoopsAt(0)).toBe(0);
    });

    it("degree conventions of section 3.4 and the weighted views of 7.2", () => {
        expect(Array.from(s.outDegree())).toEqual([3, 1, 1]);
        expect(Array.from(s.inDegree())).toEqual([1, 3, 1]);
        expect(Array.from(s.degree())).toEqual([4, 4, 2]);
        expect(Array.from(s.weightedOutDegree())).toEqual([6, 4, 5]);
        expect(Array.from(s.weightedInDegree())).toEqual([5, 8, 2]);
        expect(Array.from(s.weightedDegree())).toEqual([11, 12, 7]);
        expect(Array.from(s.selfLoopWeight())).toEqual([0, 4, 0]);
        expect(s.totalWeight()).toBe(15);
        expect(Array.from(s.selfLoopArcs())).toEqual([3]);
        expect(Array.from(s.selfLoopsPerNode())).toEqual([0, 1, 0]);
        expect(Array.from(s.degreeOrder().segmentOffsets)).toEqual([0, 0, 0, 3, 3]);
    });
});

describe("design section 7.2: views are shared, cached, prepared and dropped as documented", () => {
    it("undirected: reverse() is the forward arrays, inDegree() is outDegree(), degreeOrder orientations coincide, sum(weightedDegree) is 2 * totalWeight", () => {
        const b = new GraphBuilder({ directed: false });
        b.addEdge("a", "b", 1);
        b.addEdge("a", "b", 2);
        b.addEdge("b", "b", 3);
        b.addEdge("b", "c", 4);
        b.addEdge("c", "a", 5);
        const s = b.freeze();
        const r = s.reverse();
        expect(r.rowPtr).toBe(s.rowPtr);
        expect(r.colIdx).toBe(s.colIdx);
        expect(r.arcToEdge).toBe(s.arcToEdge);
        expect(r.weights).toBe(s.weights);
        expect(Array.from(r.fwdArc)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
        expect(s.inDegree()).toBe(s.outDegree());
        expect(s.weightedInDegree()).toBe(s.weightedOutDegree());
        expect(s.degreeOrder()).toBe(s.degreeOrder({ of: "reverse" }));
        expect(Array.from(s.degree())).toEqual([3, 5, 2]);
        expect(Array.from(s.weightedDegree()).reduce((x, y) => x + y, 0)).toBe(2 * s.totalWeight());
        const mate = s.mate();
        for (let a = 0; a < s.arcCount; a++) {
            expect(mate[mate[a]]).toBe(a);
            expect(s.arcToEdge[mate[a]]).toBe(s.arcToEdge[a]);
        }
        expect(s.isSymmetric()).toBe(true);
        expect(s.toUndirected().snapshot).toBe(s);
        expect(s.transpose().snapshot).toBe(s);
    });

    it("coo aliases everything but src, edgeList.arc aliases edgeToArc, views return the cached object, prepare returns this, dropCaches empties", () => {
        const s = directedFixture();
        const c = s.coo();
        expect(c.dst).toBe(s.colIdx);
        expect(c.arcToEdge).toBe(s.arcToEdge);
        expect(c.weights).toBe(s.weights);
        expect(s.edgeList().arc).toBe(s.edgeToArc);
        expect(s.outDegree()).toBe(s.outDegree());
        expect(s.prepare(["outDegree", "degreeOrder"])).toBe(s);
        expect(s.cachedViews()).toEqual(expect.arrayContaining(["coo", "edgeList", "outDegree", "degreeOrder"]));
        s.dropCaches();
        expect(s.cachedViews()).toEqual([]);
        expect(s.byteLength({ views: true })).toBe(s.byteLength());
    });
});

describe("design section 7.3: derived graphs", () => {
    it("toUndirected collapses reciprocal pairs keep-first, keeps parallels, shares nodes and ids, and satisfies P9b", () => {
        const s = directedFixture();
        const u = s.toUndirected();
        expect(u.snapshot.edgeCount).toBe(5);
        expect(Array.from(u.edgeOrigin ?? [])).toEqual([0, 2, 3, 4, 5]);
        expect(Array.from(u.edgeRemap ?? [])).toEqual([0, 0, 1, 2, 3, 4]);
        expect(u.nodeRemap).toBeNull();
        expect(u.nodeOrigin).toBeNull();
        expect(u.blockSizes).toBeNull();
        expect(u.report).toEqual({ droppedEdges: 0, mergedEdges: 1 });
        expect(Array.from(u.snapshot.edgeList().weights ?? [])).toEqual([1, 3, 4, 5, 6]);
        expect(u.snapshot.nodes).toBe(s.nodes);
        expect(u.snapshot.ids).toBe(s.ids);
        for (const [d, o] of Array.from(u.edgeOrigin ?? []).entries()) {
            expect((u.edgeRemap ?? [])[o]).toBe(d);
        }
        const reciprocal = s.toUndirected({ reciprocal: true, weights: "sum" });
        expect(reciprocal.snapshot.edgeCount).toBe(2);
        expect(Array.from(reciprocal.snapshot.edgeList().weights ?? [])[0]).toBe(3);
    });

    it("transpose swaps the orientation, shares node and edge tables and reports null maps", () => {
        const s = directedFixture();
        const t = s.transpose();
        expect(Array.from(t.snapshot.edgeList().src)).toEqual(Array.from(s.edgeList().dst));
        expect(Array.from(t.snapshot.edgeList().dst)).toEqual(Array.from(s.edgeList().src));
        expect(t.nodeRemap).toBeNull();
        expect(t.edgeRemap).toBeNull();
        expect(t.snapshot.nodes).toBe(s.nodes);
        expect(t.snapshot.edges).toBe(s.edges);
        expect(t.snapshot.serial).not.toBe(s.serial);
    });

    it("simplified: one edge per (u, v), survivor lowest index, multigraph false, reducers per column", () => {
        const s = directedFixture();
        s.edges.set("ec", new Float32Array([0, 100, 200, 300, 400, 500]));
        const sm = s.simplified();
        expect(sm.snapshot.flags.multigraph).toBe(false);
        expect(Array.from(sm.edgeRemap ?? [])).toEqual([0, 1, 2, 2, 3, 4]);
        expect(Array.from((sm.snapshot.edges.get("ec") as F32Column).data)).toEqual([0, 100, 200, 400, 500]);
        const reduced = s.simplified({ weights: "sum", edgeReducers: { ec: "mean" }, selfLoops: "drop" });
        expect(Array.from(reduced.snapshot.edgeList().weights ?? [])).toEqual([1, 2, 7, 6]);
        expect(Array.from((reduced.snapshot.edges.get("ec") as F32Column).data)).toEqual([0, 100, 250, 500]);
        expect(reduced.report).toEqual({ droppedEdges: 1, mergedEdges: 1 });
    });

    it("contract: dense labels are kept as block indices, non-dense labels renumber first-seen, intra-block edges become one self-loop with weight w, unweighted sum materialises multiplicities", () => {
        const s = directedFixture();
        const kept = s.contract(new Uint32Array([1, 1, 0]));
        expect(Array.from(kept.nodeRemap ?? [])).toEqual([1, 1, 0]);
        const renumbered = s.contract(new Uint32Array([5, 5, 9]));
        expect(Array.from(renumbered.nodeRemap ?? [])).toEqual([0, 0, 1]);
        expect(Array.from(renumbered.blockSizes ?? [])).toEqual([2, 1]);
        expect(Array.from(renumbered.nodeOrigin ?? [])).toEqual([0, 2]);
        expect(renumbered.snapshot.ids.kind).toBe("identity");
        expect(Array.from(renumbered.snapshot.edgeList().weights ?? [])).toEqual([3, 7, 5, 6]);
        expect(renumbered.snapshot.nodes.names()).toEqual([]);
        const identity = s.contract(new Uint32Array([0, 1, 2]));
        expect(identity.snapshot.totalWeight()).toBe(s.totalWeight());
        expect(Array.from(identity.snapshot.weightedDegree())).toEqual(Array.from(s.weightedDegree()));
        const unweighted = fromEdgeArrays({
            directed: true,
            nodeCount: 3,
            src: new Uint32Array([0, 0, 1]),
            dst: new Uint32Array([1, 1, 2]),
        });
        expect(Array.from(unweighted.contract(new Uint32Array([0, 0, 1])).snapshot.weights ?? [])).toEqual([2, 1]);
        expect(unweighted.contract(new Uint32Array([0, 0, 1]), { weights: "first" }).snapshot.weights).toBeNull();
    });

    it("inducedSubgraph: list order is the new index order, mask order ascending; relabel: perm[new] = old and the id map follows", () => {
        const s = directedFixture();
        const list = s.inducedSubgraph(new Uint32Array([2, 0]));
        expect(list.snapshot.ids.toArray()).toEqual(["c", "a"]);
        expect(Array.from(list.nodeRemap ?? [])).toEqual([1, INVALID_INDEX, 0]);
        const mask = makeMask(3);
        maskSet(mask, 0, true);
        maskSet(mask, 2, true);
        expect(s.inducedSubgraph({ mask }).snapshot.ids.toArray()).toEqual(["a", "c"]);
        const rl = s.relabel(new Uint32Array([2, 0, 1]));
        expect(rl.snapshot.ids.toArray()).toEqual(["c", "a", "b"]);
        expect(Array.from(rl.nodeRemap ?? [])).toEqual([1, 2, 0]);
        expect(rl.edgeRemap).toBeNull();
    });

    it("withColumns shares core, ids and serial with a cloned column set", () => {
        const s = directedFixture();
        const w = s.withColumns({ extra: new Float32Array(3) });
        expect(w.serial).toBe(s.serial);
        expect(w.rowPtr).toBe(s.rowPtr);
        expect(w.ids).toBe(s.ids);
        expect(w.nodes).not.toBe(s.nodes);
        expect(w.nodes.has("extra")).toBe(true);
        expect(s.nodes.has("extra")).toBe(false);
    });
});

describe("design sections 5.5 / 5.7: roles on set()", () => {
    it("a second column with a taken role is E_DUPLICATE_ROLE; replaceRole removes the previous holder from the table", () => {
        const s = directedFixture();
        const first = s.nodes.set("position", new Float32Array(9), { components: 3, role: "position" });
        expectError(
            () => s.nodes.set("pos2", new Float32Array(9), { components: 3, role: "position" }),
            "E_DUPLICATE_ROLE",
        );
        const second = s.nodes.set(
            "pos2",
            new Float32Array(9),
            { components: 3, role: "position" },
            { replaceRole: true },
        );
        expect(s.nodes.byRole("position")).toBe(second);
        expect(s.nodes.has("position")).toBe(false);
        expect(first.meta.role).toBe("position");
    });
});

describe("design sections 6.5 / 6.6: builder semantics", () => {
    it("a merge policy rewrites the builder, is idempotent, never bumps mutationCount, and edgeRemap names survivors", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b", 1);
        b.addEdge("a", "b", 2);
        b.addEdge("a", "c", 3);
        b.addEdge("a", "b", 4);
        const mutations = b.mutationCount;
        const r = b.freezeWithReport({ duplicateEdges: "sum" });
        expect(r.snapshot.edgeCount).toBe(2);
        expect(Array.from(r.snapshot.edgeList().weights ?? [])).toEqual([7, 3]);
        expect(Array.from(r.report.edgeRemap ?? [])).toEqual([0, 0, 1, 0]);
        expect(r.report.nodeRemap).toBeNull();
        expect(r.report).toMatchObject({ mergedEdges: 2, droppedEdges: 2, compacted: true });
        expect(b.edgeCount).toBe(2);
        expect(b.edgeWeight(0)).toBe(7);
        expect(b.mutationCount).toBe(mutations);
        expect(b.dirty).toBe(false);
        const again = b.freezeWithReport({ duplicateEdges: "sum" });
        expect(again.snapshot.edgeCount).toBe(2);
        expect(again.report.edgeRemap).toBeNull();
    });

    it("re-adding a removed id before the freeze revives the index without its edges; the first freeze reports nodeRemap null with edgeRemap set (I16)", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNode("a");
        b.addNode("b");
        const e0 = b.addEdge("a", "b");
        const removed = b.removeNode("a");
        expect(Array.from(removed)).toEqual([e0]);
        expect(b.addNode("a")).toBe(0);
        expect(b.hasEdge(e0)).toBe(false);
        const r = b.freezeWithReport();
        expect(r.report.nodeRemap).toBeNull();
        expect(Array.from(r.report.edgeRemap ?? [])).toEqual([INVALID_INDEX]);
        expect(r.report.compacted).toBe(true);
        b.removeNode("b");
        const r2 = b.freezeWithReport();
        expect(Array.from(r2.report.nodeRemap ?? [])).toEqual([0, INVALID_INDEX]);
        expect(b.addNode("b")).toBe(1);
        expect(r.snapshot.ids.toArray()).toEqual(["a", "b"]);
        expect(r2.snapshot.ids.toArray()).toEqual(["a"]);
    });

    it("column writes do not bump mutationCount; topology and weight mutations do; freeze does not", () => {
        const b = new GraphBuilder({ directed: true });
        const e = b.addEdge("a", "b", 1);
        const afterAdd = b.mutationCount;
        b.declareNodeColumn({ name: "x", dtype: "f32" });
        b.setNodeValue("x", 0, 1);
        expect(b.mutationCount).toBe(afterAdd);
        b.setEdgeWeight(e, 2);
        expect(b.mutationCount).toBe(afterAdd + 1);
        b.freeze();
        expect(b.mutationCount).toBe(afterAdd + 1);
        expect(b.dirty).toBe(false);
    });

    it("weight bookkeeping of 3.7: explicit / omitted mix keeps a role-weight column whose validity marks explicit rows; weightDtype f64 keeps a shadow only when not f32-exact", () => {
        const mixed = new GraphBuilder({ directed: true });
        mixed.addEdge("a", "b");
        mixed.addEdge("b", "c", 2);
        const ms = mixed.freeze();
        const column = ms.edges.byRole("weight");
        expect(column?.meta.dtype).toBe("f32");
        expect(column?.nullCount).toBe(1);
        expect(column?.isSet(0)).toBe(false);
        expect(column?.isSet(1)).toBe(true);
        const declared = new GraphBuilder({ directed: true, weighted: true });
        declared.addEdge("a", "b");
        const ds = declared.freeze();
        expect(Array.from(ds.weights ?? [])).toEqual([1]);
        expect(ds.edges.byRole("weight")?.nullCount).toBe(1);
        const inexact = new GraphBuilder({ directed: true, weightDtype: "f64" });
        inexact.addEdge("a", "b", 0.1);
        expect(inexact.freeze().edges.byRole("weight")?.meta.dtype).toBe("f64");
        const exact = new GraphBuilder({ directed: true, weightDtype: "f64" });
        exact.addEdge("a", "b", 1.5);
        expect(exact.freeze().edges.byRole("weight")).toBeNull();
        expect(new GraphBuilder({ directed: true }).freeze().edges.byRole("weight")).toBeNull();
    });
});

describe("design section 8.1: entry point defaults", () => {
    it("fromEdgeArrays: identity ids with no storage, isolates preserved, F64 weights downcast with a shadow only when inexact", () => {
        const s = fromEdgeArrays({
            directed: true,
            nodeCount: 4,
            src: new Uint32Array([0, 2]),
            dst: new Uint32Array([1, 3]),
        });
        expect(s.ids.kind).toBe("identity");
        expect(s.ids.byteLength()).toBe(0);
        expect(s.nodeCount).toBe(4);
        const inexact = fromEdgeArrays({
            directed: true,
            nodeCount: 2,
            src: new Uint32Array([0]),
            dst: new Uint32Array([1]),
            weights: new Float64Array([0.1]),
        });
        expect(inexact.edges.byRole("weight")?.meta.dtype).toBe("f64");
        const exact = fromEdgeArrays({
            directed: true,
            nodeCount: 2,
            src: new Uint32Array([0]),
            dst: new Uint32Array([1]),
            weights: new Float64Array([0.5]),
        });
        expect(exact.edges.byRole("weight")).toBeNull();
    });

    it("fromCsr: adopts by reference with arena null, validates full by default, rebuilds unsorted rows without touching the input, aliases edgeList().weights", () => {
        const rowPtr = new Uint32Array([0, 2, 3, 3]);
        const colIdx = new Uint32Array([1, 2, 2]);
        const weights = new Float32Array([1, 2, 3]);
        const s = fromCsr({ directed: true, nodeCount: 3, rowPtr, colIdx, weights });
        expect(s.rowPtr).toBe(rowPtr);
        expect(s.arena).toBeNull();
        expect(s.flags.arcToEdgeIsIdentity).toBe(true);
        expect(s.edgeList().weights).toBe(weights);
        const bad = expectError(
            () => fromCsr({ directed: true, nodeCount: 3, rowPtr, colIdx: new Uint32Array([1, 2, 9]) }),
            "E_INVALID_SNAPSHOT",
        );
        expect(bad.details.invariant).toBe("I2");
        const unsortedColIdx = new Uint32Array([2, 1, 2]);
        const rebuilt = fromCsr({ directed: true, nodeCount: 3, rowPtr, colIdx: unsortedColIdx });
        expect(Array.from(unsortedColIdx)).toEqual([2, 1, 2]);
        expect(Array.from(rebuilt.colIdx)).toEqual([1, 2, 2]);
        expect(Array.from(rebuilt.arcToEdge)).toEqual([1, 0, 2]);
        expect(rebuilt.arena).not.toBeNull();
        const asserted = expectError(
            () => fromCsr({ directed: true, nodeCount: 3, rowPtr, colIdx: unsortedColIdx }, { sortRows: false }),
            "E_INVALID_SNAPSHOT",
        );
        expect(asserted.details.invariant).toBe("I4");
    });

    it("fromRecords: endpoint fallbacks, nodeId null, unweighted with edgeWeight null, id coercion modes", () => {
        expect(fromRecords({ directed: true, edges: [{ src: "a", dst: "b" }] }).snapshot.edgeCount).toBe(1);
        expect(fromRecords({ directed: true, edges: [{ from: "a", to: "b" }] }).snapshot.edgeCount).toBe(1);
        expectError(() => fromRecords({ directed: true, edges: [{ target: "b" }] }), "E_INVALID_ID");
        const d3 = fromRecords({
            directed: true,
            nodes: [{ x: 1 }, { x: 2 }],
            edges: [{ source: 0, target: 1 }],
            nodeId: null,
        }).snapshot;
        expect(d3.ids.kind).toBe("identity");
        expect(d3.edgeCount).toBe(1);
        expect(
            fromRecords({ directed: true, edges: [{ source: "a", target: "b", weight: 5 }], edgeWeight: null }).snapshot
                .weights,
        ).toBeNull();
        expect(
            fromRecords({
                directed: true,
                edges: [{ source: "1", target: "01" }],
                ids: "canonical",
            }).snapshot.ids.toArray(),
        ).toEqual([1, "01"]);
        expect(
            fromRecords({ directed: true, edges: [{ source: 1, target: "01" }], ids: "string" }).snapshot.ids.toArray(),
        ).toEqual(["1", "01"]);
        expect(
            fromRecords({
                directed: true,
                edges: [{ source: "1", target: "2" }],
                ids: "number",
            }).snapshot.ids.toArray(),
        ).toEqual([1, 2]);
        expect(fromRecords({ directed: true, edges: [{ source: "1", target: 1 }] }).snapshot.ids.toArray()).toEqual([
            "1",
            1,
        ]);
        const inferred = fromRecords({
            directed: true,
            nodes: [
                { id: 1, v: true },
                { id: 2, v: 3 },
                { id: 3, v: 1.5 },
                { id: 4, v: "s" },
            ],
            edges: [],
        }).snapshot;
        expect(inferred.nodes.get("v")?.meta.dtype).toBe("string");
    });
});

describe("design sections 9.1 / 9.2: wire and container", () => {
    it("manifest shape: identity permutations are null, views null unless requested, scalar views ignored, only exclusive buffers transfer", () => {
        const s = directedFixture();
        const w = s.toWire();
        expect(w.manifest.format).toBe("graphty-snapshot");
        expect(w.manifest.wire).toEqual([1, 0]);
        expect(w.manifest.formatVersion).toBe(1);
        expect(w.manifest.views).toBeNull();
        expect(w.manifest.copied).toEqual([]);
        expect(w.buffers[0]).toBe(s.arena?.buffer);
        const sorted = fromEdgeArrays({
            directed: true,
            nodeCount: 3,
            src: new Uint32Array([0, 0, 1]),
            dst: new Uint32Array([1, 2, 2]),
        });
        expect(sorted.arcToEdge.length).toBe(3);
        expect(sorted.toWire().manifest.core.arcToEdge).toBeNull();
        expect(sorted.toWire().manifest.core.edgeToArc).toBeNull();
        expect(sorted.transferables()).toHaveLength(1);
        s.reverse();
        const { views } = s.toWire({ includeViews: ["reverse", "totalWeight", "symmetric"] }).manifest;
        expect(Object.keys(views ?? {})).toEqual(["reverse"]);
        const sibling = s.withColumns();
        expect(sibling.toWire({ transfer: true }).manifest.copied).toContain(0);
        expect(() => structuredClone(s)).toThrow();
        expect(structuredClone(w).buffers).toHaveLength(w.buffers.length);
    });

    it("container header: magic GSNP, u16 major 1 / minor 0, host-order probe, 256-byte multiple; chunks concatenate to the same bytes and adopt with arena null", () => {
        const s = directedFixture();
        const bytes = s.toBytes();
        expect(Array.from(bytes.subarray(0, 4))).toEqual([0x47, 0x53, 0x4e, 0x50]);
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        expect(view.getUint16(4, true)).toBe(1);
        expect(view.getUint16(6, true)).toBe(0);
        expect(Array.from(bytes.subarray(8, 12))).toEqual([4, 3, 2, 1]);
        expect(bytes.length % 256).toBe(0);
        const chunks = [...s.toByteChunks()];
        const joined = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
        let offset = 0;
        for (const chunk of chunks) {
            joined.set(chunk, offset);
            offset += chunk.length;
        }
        expect(Array.from(joined)).toEqual(Array.from(bytes));
        expect(chunks.length).toBeGreaterThan(1);
        for (const chunk of chunks.slice(1)) {
            expect(chunk.length % 256).toBe(0);
        }
        expect(s.toBytes()).toEqual(bytes);
        expect(s.contentHash()).toMatch(/^[0-9a-f]{16}$/);
    });

    it("a real transfer detaches the source: detached is derived, core accessors and wire methods throw E_DETACHED, view caches are dropped", () => {
        const s = directedFixture();
        s.outDegree();
        const wire = s.toWire({ transfer: true });
        expect(s.detached).toBe(false);
        const moved = structuredClone(wire, { transfer: [...wire.buffers] });
        expect(s.detached).toBe(true);
        expect(s.rowPtr.length).toBe(0);
        for (const call of [
            () => s.outDegree(),
            () => s.findArc(0, 1),
            () => s.arcToEdge,
            () => s.edgeToArc,
            () => s.coo(),
            () => s.toWire(),
            () => s.toBytes(),
            () => s.validate(),
            () => s.contentHash(),
            () => s.outArcs(0),
            () => s.transpose(),
            () => s.withColumns(),
        ]) {
            expectError(call, "E_DETACHED");
        }
        expect(fromWire(moved, { validate: "none" }).edgeCount).toBe(6);
    });
});

describe("AUDIT FINDINGS: deliberately failing tests that pin defects (see packages/CONFORMANCE.md)", () => {
    it("F1 (11.1, 6.5): freeze({ duplicateEdges }) with a value outside DuplicatePolicy must throw E_UNSUPPORTED and leave the builder untouched, not merge as 'first'", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b", 1);
        b.addEdge("a", "b", 5);
        b.addEdge("a", "c", 3);
        const { edgeCount } = b;
        expectError(() => b.freeze({ duplicateEdges: "bogus" as never }), "E_UNSUPPORTED");
        expect(b.edgeCount).toBe(edgeCount);
        expect(b.edgeWeight(1)).toBe(5);
    });

    it("F2 (11.1): addEdgeRecord / addNodeRecord that throw on an attribute value must not be partially applied", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNode("a");
        b.addNode("b");
        const before = { edgeCount: b.edgeCount, mutationCount: b.mutationCount, edgeBound: b.edgeBound };
        expectError(() => b.addEdgeRecord("a", "b", { x: 1, y: "\ud800" }), "E_COLUMN_TYPE");
        expect({ edgeCount: b.edgeCount, mutationCount: b.mutationCount, edgeBound: b.edgeBound }).toEqual(before);
        const nodes = new GraphBuilder({ directed: true });
        expectError(() => nodes.addNodeRecord("n", { x: 1, y: 1n }), "E_COLUMN_TYPE");
        expect(nodes.hasNode("n")).toBe(false);
        expect(nodes.nodeCount).toBe(0);
    });

    it("F3 (3.6, 7.2, P8): isSymmetric() is 'the arc set is closed under reversal with equal weights' (a set comparison), also on a multigraph", () => {
        const b = new GraphBuilder({ directed: true });
        b.addEdge("a", "b", 1);
        b.addEdge("a", "b", 2);
        b.addEdge("b", "a", 2);
        b.addEdge("b", "a", 1);
        const s = b.freeze();
        // every arc (u, v, w) has a mirror (v, u, w): {a->b: 1, 2} vs {b->a: 2, 1}
        expect(s.isSymmetric()).toBe(true);
    });

    it("F4 (5.8, 12.2): a snapshot's own enumerable keys are the public fields plus the documented clone guard; lazy state is private", () => {
        const s = directedFixture();
        const publicFields = [
            "serial",
            "label",
            "formatVersion",
            "directed",
            "nodeCount",
            "edgeCount",
            "arcCount",
            "selfLoopCount",
            "rowPtr",
            "colIdx",
            "weights",
            "flags",
            "ids",
            "nodes",
            "edges",
            "graph",
            "extensions",
            "meta",
            "arena",
            "__graphtyNoStructuredClone",
        ];
        expect(Object.keys(s).filter((k) => !publicFields.includes(k))).toEqual([]);
    });

    it("F5 (5.11, 12.2): remapColumn must not rewrite refersTo values through a remap of the OTHER index space", () => {
        const s = directedFixture();
        const column = s.nodes.set("firstEdge", new Uint32Array([0, 1, 2]), { refersTo: "edge" });
        const nodeRemap = new Uint32Array([1, INVALID_INDEX, 0]);
        let remapped: U32Column | null = null;
        try {
            remapped = remapColumn(column, nodeRemap, 2) as U32Column;
        } catch (error) {
            expect(error).toBeInstanceOf(GraphFormatError);
        }
        if (remapped !== null) {
            // rows moved (new 0 <- old 2, new 1 <- old 0); the EDGE references they hold are unchanged
            expect(Array.from(remapped.data)).toEqual([2, 0]);
        }
    });
});
