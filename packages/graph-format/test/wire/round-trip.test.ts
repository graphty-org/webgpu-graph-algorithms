/**
 * Round trips of design section 16.3: toWire -> structuredClone -> fromWire, toWire({ transfer })
 * -> transfer -> fromWire, toBytes -> fromBytes at every validation level and byte offset,
 * toByteChunks -> fromByteChunks, for every id-map kind, every dtype, extension tables, undirected
 * and directed graphs and the empty graph.
 */

import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { remapColumn } from "../../src/columns/remap.js";
import { ALIGNMENT, FORMAT_VERSION, INVALID_INDEX, WIRE_MAJOR, WIRE_MINOR } from "../../src/constants.js";
import { equalsTopology } from "../../src/snapshot/graph-snapshot.js";
import { type NodeId, type ValidationLevel, type ViewName, type WireSnapshot } from "../../src/types/index.js";
import { isShared, noteShared } from "../../src/util/shared-buffers.js";
import { fromByteChunks, fromBytes } from "../../src/wire/bytes.js";
import { fromWire } from "../../src/wire/from-wire.js";
import { WIRE_PRODUCER } from "../../src/wire/to-wire.js";
import { assertInvariants } from "../helpers/invariants.js";
import { expectError, expectSnapshotsEqual, idsSnapshot, richSnapshot, splitContainer } from "./helpers.js";

const LEVELS: readonly ValidationLevel[] = ["none", "structure", "full"];

describe("toWire / fromWire", () => {
    it("produces a manifest with the documented header fields and one buffer per distinct ArrayBuffer", () => {
        const s = richSnapshot();
        const wire = s.toWire();
        const { manifest } = wire;
        expect(manifest.format).toBe("graphty-snapshot");
        expect(manifest.wire).toEqual([WIRE_MAJOR, WIRE_MINOR]);
        expect(manifest.producer).toBe(WIRE_PRODUCER);
        expect(manifest.formatVersion).toBe(FORMAT_VERSION);
        expect(manifest.directed).toBe(false);
        expect(manifest.counts).toEqual({ nodes: 4, edges: 5, arcs: 9, selfLoops: 1 });
        expect(manifest.flags).toEqual(s.flags);
        expect(manifest.copied).toEqual([]);
        expect(manifest.label).toBe("rich");
        expect(manifest.views).toBeNull();
        expect(manifest.arena).not.toBeNull();
        expect(manifest.core.rowPtr.buffer).toBe(manifest.arena?.buffer);
        expect(manifest.core.colIdx?.buffer).toBe(manifest.arena?.buffer);
        expect(wire.buffers[manifest.core.rowPtr.buffer]).toBe(s.arena?.buffer);
        expect(manifest.core.rowPtr.byteOffset).toBe(s.rowPtr.byteOffset);
        expect(new Set(wire.buffers).size).toBe(wire.buffers.length);
        expect(manifest.ids.kind).toBe("string");
        expect(manifest.nodeColumns.map((c) => c.meta.name)).toEqual(s.nodes.names());
        expect(manifest.extensions.map((e) => e.name)).toEqual(["temporal:node:price"]);
        expect(JSON.parse(JSON.stringify(manifest))).toEqual(manifest);
    });

    it("tags non-finite numbers and -0 in metadata so JSON.stringify is lossless", () => {
        const s = richSnapshot();
        const { manifest } = s.toWire();
        const end = manifest.nodeColumns.find((c) => c.meta.name === "end");
        expect(end?.meta.default).toEqual({ $num: "Infinity" });
        expect(end?.meta.fill).toEqual({ $num: "Infinity" });
        const nanfill = manifest.nodeColumns.find((c) => c.meta.name === "nanfill");
        expect(nanfill?.meta.fill).toEqual({ $num: "NaN" });
        const blob = manifest.nodeColumns.find((c) => c.meta.name === "blob");
        expect(blob?.meta.extra).toEqual({ note: "nested", limit: { $num: "-0" } });
        expect(manifest.meta.extra).toEqual({
            nested: {
                inf: { $num: "Infinity" },
                ninf: { $num: "-Infinity" },
                nan: { $num: "NaN" },
                nz: { $num: "-0" },
                list: [1, "two", null],
            },
        });
        const wire = s.toWire();
        const back = fromWire({ manifest: JSON.parse(JSON.stringify(wire.manifest)) as never, buffers: wire.buffers });
        expect(back.nodes.require("end").meta.default).toBe(Infinity);
        expect(Number.isNaN(back.nodes.require("nanfill").meta.fill)).toBe(true);
        expect(Object.is((back.nodes.require("blob").meta.extra as { limit: number }).limit, -0)).toBe(true);
        expectSnapshotsEqual(s, back);
    });

    it("round-trips the rich snapshot through structuredClone at every level", () => {
        const s = richSnapshot();
        for (const level of LEVELS) {
            const wire = structuredClone(s.toWire());
            const back = fromWire(wire, { validate: level });
            assertInvariants(back);
            expectSnapshotsEqual(s, back);
            expect(back.arena).not.toBeNull();
            expect(back.arena?.buffer).toBe(wire.buffers[wire.manifest.arena?.buffer ?? -1]);
            expect(back.serial).not.toBe(s.serial);
        }
    });

    it("adopts buffers by reference and copies them with copy: true", () => {
        const s = richSnapshot();
        const wire = s.toWire();
        const adopted = fromWire(wire);
        expect(adopted.rowPtr.buffer).toBe(s.rowPtr.buffer);
        expect(adopted.nodes.requireTyped("pos", "f32").data.buffer).toBe(
            s.nodes.requireTyped("pos", "f32").data.buffer,
        );
        const copied = fromWire(wire, { copy: true });
        expect(copied.rowPtr.buffer).not.toBe(s.rowPtr.buffer);
        expect(copied.arena).not.toBeNull();
        expectSnapshotsEqual(s, copied);
    });

    it("copies a SharedArrayBuffer instead of adopting it", () => {
        const s = idsSnapshot([1, 2, 3]);
        const wire = s.toWire();
        const shared = wire.buffers.map((b) => {
            const sab = new SharedArrayBuffer(b.byteLength);
            new Uint8Array(sab).set(new Uint8Array(b));
            return sab as unknown as ArrayBuffer;
        });
        const back = fromWire({ manifest: wire.manifest, buffers: shared });
        expect(back.rowPtr.buffer).toBeInstanceOf(ArrayBuffer);
        expectSnapshotsEqual(s, back);
        expectError(() => fromWire({ manifest: wire.manifest, buffers: [5 as never] }), "E_BAD_SERIALIZATION");
    });

    it("round-trips every id-map kind", () => {
        const cases: { readonly ids: readonly NodeId[]; readonly kind: string }[] = [
            { ids: [0, 1, 2, 3], kind: "identity" },
            { ids: [1, 2, 3, 4], kind: "identity" },
            { ids: [0, 5, 2, 3], kind: "dense" },
            { ids: [0.5, 1e9, -3, 7], kind: "numeric" },
            { ids: ["a", "b", "\u00e9\u00e8", ""], kind: "string" },
            { ids: [1, "1", 2.5, "x"], kind: "mixed" },
        ];
        for (const { ids, kind } of cases) {
            const s = idsSnapshot(ids);
            expect(s.ids.kind).toBe(kind);
            const back = fromWire(structuredClone(s.toWire()), { validate: "full" });
            expect(back.ids.kind).toBe(kind);
            expect(back.ids.toArray()).toEqual([...ids]);
            expectSnapshotsEqual(s, back);
            const bytes = fromBytes(s.toBytes());
            expect(bytes.ids.kind).toBe(kind);
            expectSnapshotsEqual(s, bytes);
            expect(bytes.ids.indexOf(ids[1])).toBe(1);
        }
    });

    it("round-trips the empty graph, a node-only graph and a directed graph with an identity permutation", () => {
        const empty = new GraphBuilder({ directed: true }).freeze();
        const emptyBack = fromWire(structuredClone(empty.toWire()), { validate: "full" });
        expectSnapshotsEqual(empty, emptyBack);
        expect(emptyBack.nodeCount).toBe(0);
        expect(emptyBack.rowPtr.length).toBe(1);
        expect(emptyBack.colIdx.length).toBe(0);
        expect(emptyBack.weights).toBeNull();
        expect(emptyBack.edgeToArc.length).toBe(0);
        const emptyWire = empty.toWire().manifest;
        expect(emptyWire.core.colIdx).toBeNull();
        expect(emptyWire.core.arcToEdge).toBeNull();
        expect(emptyWire.core.edgeToArc).toBeNull();
        expectSnapshotsEqual(empty, fromBytes(empty.toBytes()));

        const nodesOnly = new GraphBuilder({ directed: false, weighted: true });
        nodesOnly.addNodes(["x", "y"]);
        const nodesOnlySnapshot = nodesOnly.freeze();
        expect(nodesOnlySnapshot.weights?.length).toBe(0);
        const nodesOnlyBack = fromBytes(nodesOnlySnapshot.toBytes());
        expect(nodesOnlyBack.weights?.length).toBe(0);
        expect(nodesOnlyBack.flags.weighted).toBe(true);
        expectSnapshotsEqual(nodesOnlySnapshot, nodesOnlyBack);

        const directed = idsSnapshot([0, 1, 2, 3]);
        expect(directed.flags.arcToEdgeIsIdentity).toBe(true);
        expect(directed.arcToEdge.length).toBe(3);
        const wire = directed.toWire();
        expect(wire.manifest.core.arcToEdge).toBeNull();
        expect(wire.manifest.core.edgeToArc).toBeNull();
        const back = fromWire(wire, { validate: "full" });
        expect(back.flags.arcToEdgeIsIdentity).toBe(true);
        expect(Array.from(back.arcToEdge)).toEqual([0, 1, 2]);
        expectSnapshotsEqual(directed, back);
    });

    it("round-trips a directed rich snapshot with a non-identity permutation", () => {
        const s = richSnapshot({ directed: true });
        expect(s.flags.arcToEdgeIsIdentity).toBe(false);
        expectSnapshotsEqual(s, fromWire(structuredClone(s.toWire()), { validate: "full" }));
        expectSnapshotsEqual(s, fromBytes(s.toBytes()));
    });

    it("omits attribute tables with includeColumns: false", () => {
        const s = richSnapshot();
        const wire = s.toWire({ includeColumns: false });
        expect(wire.manifest.nodeColumns).toEqual([]);
        expect(wire.manifest.edgeColumns).toEqual([]);
        expect(wire.manifest.graphColumns).toEqual([]);
        expect(wire.manifest.extensions).toEqual([]);
        const back = fromWire(wire, { validate: "full" });
        expect(equalsTopology(s, back)).toBe(true);
        expect(back.nodes.names()).toEqual([]);
        expect(back.extensions.size).toBe(0);
    });

    it("carries cached views named by includeViews; the receiver installs them below full and recomputes under full", () => {
        const s = richSnapshot({ directed: true });
        expect(s.toWire({ includeViews: ["reverse", "outDegree"] }).manifest.views).toBeNull();
        s.prepare(["reverse", "outDegree", "degreeOrder", "totalWeight", "edgeList"]);
        const wire = s.toWire({
            includeViews: ["reverse", "outDegree", "totalWeight", "symmetric", "coo", "edgeList"],
        });
        const { views } = wire.manifest;
        expect(views).not.toBeNull();
        expect(Object.keys(views ?? {}).sort()).toEqual(["edgeList", "outDegree", "reverse"]);
        expect(views?.outDegree.data.length).toBe(s.nodeCount);
        expect(Object.keys(views?.reverse ?? {})).toEqual(["rowPtr", "colIdx", "weights", "fwdArc"]);
        const back = fromWire(wire, { validate: "full" });
        expect(back.cachedViews()).toEqual([]);
        expect(Array.from(back.outDegree())).toEqual(Array.from(s.outDegree()));
        expectSnapshotsEqual(s, back);
        // below "full" the carried views are installed, aliasing the wire's buffers
        const seeded = fromWire(wire, { validate: "structure" });
        expect(seeded.cachedViews()).toEqual(["reverse", "edgeList", "outDegree"]);
        expect(seeded.outDegree().buffer).toBe(wire.buffers[views?.outDegree.data.buffer ?? -1]);
        expect(Array.from(seeded.outDegree())).toEqual(Array.from(s.outDegree()));
        expect(Array.from(seeded.reverse().rowPtr)).toEqual(Array.from(s.reverse().rowPtr));
        expect(Array.from(seeded.reverse().colIdx)).toEqual(Array.from(s.reverse().colIdx));
        expect(Array.from(seeded.reverse().fwdArc)).toEqual(Array.from(s.reverse().fwdArc));
        expect(Array.from(seeded.reverse().arcToEdge)).toEqual(Array.from(s.reverse().arcToEdge));
        expect(Array.from(seeded.reverse().weights ?? [])).toEqual(Array.from(s.reverse().weights ?? []));
        expect(Array.from(seeded.edgeList().src)).toEqual(Array.from(s.edgeList().src));
        expect(Array.from(seeded.edgeList().dst)).toEqual(Array.from(s.edgeList().dst));
        expect(Array.from(seeded.edgeList().weights ?? [])).toEqual(Array.from(s.edgeList().weights ?? []));
        expect(seeded.edgeList().arc).toBe(seeded.edgeToArc);
        expect(Array.from(seeded.inDegree())).toEqual(Array.from(s.inDegree()));
        expectSnapshotsEqual(s, seeded);
        const bytes = s.toBytes({ includeViews: ["outDegree", "reverse"] });
        const parts = splitContainer(bytes);
        expect(Object.keys(parts.manifest.views as object).sort()).toEqual(["outDegree", "reverse"]);
        expectSnapshotsEqual(s, fromBytes(bytes));
        expect(fromBytes(bytes, { validate: "none" }).cachedViews()).toEqual(["reverse", "outDegree"]);
        const undirected = richSnapshot();
        undirected.prepare(["reverse"]);
        const uWire = undirected.toWire({ includeViews: ["reverse"] });
        expect(uWire.manifest.views?.reverse.rowPtr).toEqual(uWire.manifest.core.rowPtr);
        const uBytes = splitContainer(undirected.toBytes({ includeViews: ["reverse"] }));
        const uViews = uBytes.manifest.views as Record<string, Record<string, unknown>>;
        expect(uViews.reverse.rowPtr).toEqual((uBytes.manifest.core as Record<string, unknown>).rowPtr);
        expect(uViews.reverse.colIdx).toEqual((uBytes.manifest.core as Record<string, unknown>).colIdx);
        // an undirected reverse is an alias of the forward arrays, so the receiver keeps its own
        const uBack = fromWire(uWire, { validate: "structure" });
        expect(uBack.cachedViews()).toEqual([]);
        expect(uBack.reverse().rowPtr).toBe(uBack.rowPtr);
    });

    it("installs every carried view kind and rejects one of the wrong shape", () => {
        const s = richSnapshot({ directed: true });
        const all: ViewName[] = [
            "reverse",
            "coo",
            "edgeList",
            "outDegree",
            "inDegree",
            "degree",
            "weightedOutDegree",
            "weightedInDegree",
            "weightedDegree",
            "selfLoopWeight",
            "selfLoopArcs",
            "selfLoopsPerNode",
            "degreeOrder",
            "reverseDegreeOrder",
        ];
        s.prepare(all);
        const wire = s.toWire({ includeViews: all });
        const back = fromWire(wire, { validate: "structure" });
        expect(back.cachedViews()).toEqual(s.cachedViews());
        for (const name of ["outDegree", "inDegree", "degree", "selfLoopArcs", "selfLoopsPerNode"] as const) {
            expect(Array.from(back[name]())).toEqual(Array.from(s[name]()));
        }
        for (const name of ["weightedOutDegree", "weightedInDegree", "weightedDegree", "selfLoopWeight"] as const) {
            expect(Array.from(back[name]())).toEqual(Array.from(s[name]()));
        }
        expect(Array.from(back.coo().src)).toEqual(Array.from(s.coo().src));
        expect(back.coo().dst).toBe(back.colIdx);
        expect(Array.from(back.degreeOrder().perm)).toEqual(Array.from(s.degreeOrder().perm));
        expect(Array.from(back.degreeOrder({ of: "reverse" }).segmentOffsets)).toEqual(
            Array.from(s.degreeOrder({ of: "reverse" }).segmentOffsets),
        );
        const u = richSnapshot();
        u.prepare(["mate", "outDegree", "inDegree", "reverseDegreeOrder"]);
        const uBack = fromWire(u.toWire({ includeViews: ["mate", "outDegree", "inDegree", "reverseDegreeOrder"] }), {
            validate: "none",
        });
        expect(uBack.cachedViews()).toEqual(["outDegree", "mate"]);
        expect(Array.from(uBack.mate())).toEqual(Array.from(u.mate()));
        expect(uBack.inDegree()).toBe(uBack.outDegree());
        // wrong shape: the outDegree carried has arcCount entries instead of nodeCount
        const bad = structuredClone(wire);
        const views = bad.manifest.views as unknown as Record<string, Record<string, Record<string, unknown>>>;
        views.outDegree.data = { ...views.coo.src };
        const error = expectError(() => fromWire(bad, { validate: "none" }), "E_BAD_SERIALIZATION");
        expect(error.details.ref).toBe("views.outDegree.data");
        // a missing required member
        const missing = structuredClone(wire);
        delete (missing.manifest.views as Record<string, Record<string, unknown>>).reverse.fwdArc;
        expect(expectError(() => fromWire(missing, { validate: "none" }), "E_BAD_SERIALIZATION").details.ref).toBe(
            "views.reverse.fwdArc",
        );
        // weights carried for an unweighted snapshot
        const ub = new GraphBuilder({ directed: true });
        ub.addEdge(0, 1);
        ub.addEdge(1, 2);
        const unweighted = ub.freeze();
        unweighted.prepare(["reverse"]);
        const uw = unweighted.toWire({ includeViews: ["reverse"] });
        const withWeights = structuredClone(uw);
        const rv = (withWeights.manifest.views as Record<string, Record<string, unknown>>).reverse;
        rv.weights = { ...(rv.rowPtr as object), dtype: "f32", length: 0, byteLength: 0 };
        expectError(() => fromWire(withWeights, { validate: "none" }), "E_BAD_SERIALIZATION");
    });
});

describe("transfer", () => {
    it("transferables() lists the distinct exclusive buffers and toWire({ transfer }) detaches the source", () => {
        const s = richSnapshot();
        const list = s.transferables();
        expect(new Set(list).size).toBe(list.length);
        expect(list).toContain(s.arena?.buffer);
        expect(list).toContain(s.nodes.requireTyped("pos", "f32").data.buffer);
        const wire = s.toWire({ transfer: true });
        expect(wire.manifest.copied).toEqual([]);
        expect(wire.buffers).toEqual(list);
        const clone = structuredClone(wire, { transfer: s.transferables() });
        expect(s.detached).toBe(true);
        expectError(() => s.outDegreeOf(0), "E_DETACHED");
        expectError(() => s.toWire(), "E_DETACHED");
        const back = fromWire(clone, { validate: "full" });
        expect(back.nodeCount).toBe(4);
        expect(back.detached).toBe(false);
        expect(back.nodes.value("label", 0)).toBe("node 0 \u00e9");
    });

    it("copies buffers recorded as shared and lists them in manifest.copied", () => {
        const s = richSnapshot();
        const arena = s.arena?.buffer as ArrayBuffer;
        const pos = s.nodes.requireTyped("pos", "f32").data.buffer;
        expect(isShared(arena)).toBe(false);
        noteShared(arena);
        noteShared(pos);
        expect(isShared(arena)).toBe(true);
        const list = s.transferables();
        expect(list).not.toContain(arena);
        expect(list).not.toContain(pos);
        const wire = s.toWire({ transfer: true });
        const arenaIndex = wire.manifest.core.rowPtr.buffer;
        expect(wire.manifest.copied).toContain(arenaIndex);
        expect(wire.buffers[arenaIndex]).not.toBe(arena);
        expect(wire.buffers[arenaIndex].byteLength).toBe(arena.byteLength);
        expect(wire.manifest.arena?.buffer).toBe(arenaIndex);
        expect(s.toWire().manifest.copied).toEqual([]);
        const clone = structuredClone(wire, { transfer: s.transferables() });
        expect(s.detached).toBe(false);
        expect(s.nodes.requireTyped("pos", "f32").data.length).toBe(12);
        expect(s.nodes.requireTyped("end", "f64").data.length).toBe(0);
        const back = fromWire(clone, { validate: "full" });
        expectSnapshotsEqual(richSnapshot(), back);
    });

    it("copies the core of a withColumns() sibling and transfers the sibling's own columns (design 16.3)", () => {
        const s = richSnapshot({ directed: true });
        const sibling = s.withColumns({ score: new Float32Array(s.nodeCount) });
        const own = sibling.nodes.requireTyped("score", "f32").data.buffer;
        expect(s.arena).not.toBeNull();
        const arena = s.arena?.buffer as ArrayBuffer;
        expect(isShared(arena)).toBe(true);
        expect(sibling.transferables()).toEqual([own]);
        expect(s.transferables()).toEqual([]);
        const wire = sibling.toWire({ transfer: true });
        const arenaIndex = wire.manifest.core.rowPtr.buffer;
        expect(wire.manifest.copied).toContain(arenaIndex);
        expect(wire.buffers[arenaIndex]).not.toBe(arena);
        const ownIndex = wire.buffers.indexOf(own);
        expect(ownIndex).toBeGreaterThanOrEqual(0);
        expect(wire.manifest.copied).not.toContain(ownIndex);
        const clone = structuredClone(wire, { transfer: sibling.transferables() });
        expect(s.detached).toBe(false);
        expect(sibling.detached).toBe(false);
        expect(s.nodes.requireTyped("pos", "f32").data.length).toBe(s.nodeCount * 3);
        expect(sibling.nodes.requireTyped("score", "f32").data.length).toBe(0);
        const back = fromWire(clone, { validate: "full" });
        expect(equalsTopology(back, s)).toBe(true);
        expect(back.nodes.has("score")).toBe(true);
        // a string column materialised AFTER the sibling was created is still recognised as shared
        expect(s.toWire({ transfer: true }).manifest.copied.length).toBe(s.toWire().buffers.length);
    });

    it("copies the node table shared with a derived graph and the reverse arrays adopted by transpose()", () => {
        const s = richSnapshot({ directed: true });
        const filtered = s.withoutSelfLoops().snapshot;
        expect(filtered.nodes).toBe(s.nodes);
        const pos = s.nodes.requireTyped("pos", "f32").data.buffer;
        expect(filtered.transferables()).not.toContain(pos);
        expect(s.transferables()).not.toContain(pos);
        expect(filtered.transferables()).toContain(filtered.arena?.buffer);
        const wire = filtered.toWire({ transfer: true });
        structuredClone(wire, { transfer: filtered.transferables() });
        expect(filtered.detached).toBe(true);
        expect(s.detached).toBe(false);
        expect(s.nodes.requireTyped("pos", "f32").data.length).toBe(s.nodeCount * 3);
        const t = richSnapshot({ directed: true });
        const transposed = t.transpose().snapshot;
        expect(transposed.transferables()).not.toContain(transposed.rowPtr.buffer);
        structuredClone(transposed.toWire({ transfer: true }), { transfer: transposed.transferables() });
        expect(transposed.detached).toBe(false);
        expect(t.reverse().rowPtr.length).toBe(t.nodeCount + 1);
    });

    it("copies a Column object moved between tables and the buffers behind a renamed shared column", () => {
        const a = idsSnapshot([0, 1, 2]);
        const b = idsSnapshot([0, 1, 2]);
        const column = a.nodes.set("x", new Float32Array([1, 2, 3]));
        const { buffer } = a.nodes.requireTyped("x", "f32").data;
        expect(a.transferables()).toContain(buffer);
        b.nodes.set("x", column);
        expect(b.nodes.get("x")).toBe(column);
        expect(isShared(column)).toBe(true);
        expect(a.transferables()).not.toContain(buffer);
        expect(b.transferables()).not.toContain(buffer);
        // moved under a new name: a re-wrapped column sharing every buffer with one another table holds
        const c = idsSnapshot([0, 1, 2]);
        const moved = c.nodes.set("y", column);
        expect(moved).not.toBe(column);
        expect(isShared(moved)).toBe(true);
        expect(c.transferables()).not.toContain(buffer);
        b.nodes.rename("x", "renamed");
        expect(b.transferables()).not.toContain(buffer);
        // a fresh column adopted by reference is exclusive, before and after a rename
        const d = idsSnapshot([0, 1]);
        d.nodes.set("w", new Uint32Array([4, 5]), undefined, { adopt: "strict" });
        const fresh = d.nodes.requireTyped("w", "u32").data.buffer;
        expect(d.transferables()).toContain(fresh);
        d.nodes.rename("w", "v");
        expect(d.transferables()).toContain(fresh);
        // a free-standing column (never held by a table) moved in is exclusive
        const e = idsSnapshot([0, 1]);
        const free = remapColumn(a.nodes.require("x"), new Uint32Array([0, 1, INVALID_INDEX]), 2);
        e.nodes.set("x", free);
        expect(e.transferables()).toContain(e.nodes.requireTyped("x", "f32").data.buffer);
    });

    it("hands a snapshot to a MessagePort with postMessage(wire, transferables()) in O(1) per buffer", async () => {
        const s = richSnapshot();
        const wire = s.toWire({ transfer: true });
        const list = s.transferables();
        const { port1, port2 } = new MessageChannel();
        const received = new Promise<WireSnapshot>((resolve) => {
            port2.once("message", (message: WireSnapshot) => {
                resolve(message);
            });
        });
        port1.postMessage(wire, list);
        expect(s.detached).toBe(true);
        expect(list.every((buffer) => buffer.byteLength === 0)).toBe(true);
        const back = fromWire(await received, { validate: "none" });
        port1.close();
        port2.close();
        expect(back.detached).toBe(false);
        expect(back.arena).not.toBeNull();
        expectSnapshotsEqual(richSnapshot(), back);
        assertInvariants(back);
    });

    it("structuredClone of the wire works while structuredClone of the snapshot throws", () => {
        const s = idsSnapshot(["a", "b"]);
        expect(() => structuredClone(s)).toThrow();
        const wire = structuredClone(s.toWire());
        expect(wire.buffers[0]).not.toBe(s.toWire().buffers[0]);
        expectSnapshotsEqual(s, fromWire(wire));
    });
});

describe("toBytes / fromBytes", () => {
    it("writes the documented header and a 256-aligned region whose prefix is the arena", () => {
        const s = richSnapshot();
        const bytes = s.toBytes();
        expect(bytes.byteOffset).toBe(0);
        expect(Array.from(bytes.subarray(0, 4))).toEqual([0x47, 0x53, 0x4e, 0x50]);
        expect(Array.from(bytes.subarray(8, 12))).toEqual([4, 3, 2, 1]);
        const parts = splitContainer(bytes);
        expect(parts.major).toBe(WIRE_MAJOR);
        expect(parts.minor).toBe(WIRE_MINOR);
        expect(parts.manifest.wire).toEqual([WIRE_MAJOR, WIRE_MINOR]);
        expect(parts.region.byteLength % ALIGNMENT).toBe(0);
        const core = parts.manifest.core as Record<string, { buffer: number; byteOffset: number; byteLength: number }>;
        expect(core.rowPtr).toMatchObject({ buffer: 0, byteOffset: 0, byteLength: 20 });
        expect(core.colIdx.byteOffset).toBe(256);
        expect(core.weights.byteOffset).toBe(512);
        expect(core.arcToEdge.byteOffset).toBe(768);
        expect(core.edgeToArc.byteOffset).toBe(1024);
        expect(parts.manifest.arena).toEqual({
            buffer: 0,
            byteOffset: 0,
            byteLength: 1024 + 20,
            hotByteLength: 512 + 36,
        });
        const refs: number[] = [];
        JSON.stringify(parts.manifest, (key, value: unknown) => {
            if (key === "byteOffset" && typeof value === "number") {
                refs.push(value);
            }
            return value;
        });
        expect(refs.every((offset) => offset % ALIGNMENT === 0)).toBe(true);
        expect(Array.from(new Uint32Array(bytes.buffer, parts.region.byteOffset, 5))).toEqual(Array.from(s.rowPtr));
    });

    it("is deterministic: the same snapshot serialises to the same bytes twice, whatever getters were touched", () => {
        const a = richSnapshot({ directed: true });
        const b = richSnapshot({ directed: true });
        expect(b.arcToEdge.length).toBe(b.arcCount);
        b.prepare(["reverse", "coo"]);
        b.nodes.require("label").value(0);
        expect(Array.from(a.toBytes())).toEqual(Array.from(b.toBytes()));
        expect(Array.from(a.toBytes())).toEqual(Array.from(a.toBytes()));
    });

    it("round-trips at every level, adopting at offsets 0 and 8 and copying at 4 and from a SharedArrayBuffer", () => {
        const s = richSnapshot();
        const bytes = s.toBytes();
        for (const level of LEVELS) {
            for (const offset of [0, 4, 8]) {
                const padded = new Uint8Array(bytes.byteLength + offset);
                padded.set(bytes, offset);
                const view = padded.subarray(offset);
                const back = fromBytes(view, { validate: level });
                assertInvariants(back);
                expectSnapshotsEqual(s, back);
                expect(back.arena).not.toBeNull();
                if (offset % 8 === 0) {
                    expect(back.rowPtr.buffer).toBe(padded.buffer);
                    expect(back.arena?.byteOffset).toBe(
                        offset + (bytes.byteLength - splitContainer(bytes).region.byteLength),
                    );
                    expect(back.arena?.buffer).toBe(padded.buffer);
                } else {
                    expect(back.rowPtr.buffer).not.toBe(padded.buffer);
                    expect(back.arena?.byteOffset).toBe(0);
                }
                assertInvariants(back);
            }
        }
        const sab = new SharedArrayBuffer(bytes.byteLength);
        new Uint8Array(sab).set(bytes);
        const fromShared = fromBytes(sab);
        expect(fromShared.rowPtr.buffer).toBeInstanceOf(ArrayBuffer);
        expectSnapshotsEqual(s, fromShared);
        const fromBuffer = fromBytes(bytes.buffer);
        expect(fromBuffer.rowPtr.buffer).toBe(bytes.buffer);
        expectSnapshotsEqual(s, fromBuffer);
        const copied = fromBytes(bytes, { copy: true });
        expect(copied.rowPtr.buffer).not.toBe(bytes.buffer);
        expectSnapshotsEqual(s, copied);
    });

    it("keeps a default of Infinity and a fill of NaN across the container", () => {
        const s = richSnapshot();
        const back = fromBytes(s.toBytes());
        const end = back.nodes.requireTyped("end", "f64");
        expect(end.meta.default).toBe(Infinity);
        expect(end.value(1)).toBe(Infinity);
        expect(end.data[1]).toBe(Infinity);
        const nanfill = back.nodes.requireTyped("nanfill", "f64");
        expect(Number.isNaN(nanfill.meta.fill)).toBe(true);
        expect(Number.isNaN(nanfill.data[0])).toBe(true);
        expect(nanfill.value(1)).toBeCloseTo(1 / 3);
        const ext = back.extensions.get("temporal:node:price");
        expect(ext?.value("end", 1)).toBe(Infinity);
        expect(ext?.isSet("end", 1)).toBe(false);
        expect(back.meta.extra).toEqual(s.meta.extra);
    });

    it("toByteChunks yields the same bytes as toBytes and fromByteChunks adopts per chunk with arena === null", () => {
        const s = richSnapshot();
        const chunks = [...s.toByteChunks()];
        expect(chunks.length).toBeGreaterThan(2);
        expect(chunks.every((c) => c.byteLength % ALIGNMENT === 0)).toBe(true);
        const total = chunks.reduce((n, c) => n + c.byteLength, 0);
        const joined = new Uint8Array(total);
        let cursor = 0;
        for (const chunk of chunks) {
            joined.set(chunk, cursor);
            cursor += chunk.byteLength;
        }
        expect(Array.from(joined)).toEqual(Array.from(s.toBytes()));
        const back = fromByteChunks(chunks);
        assertInvariants(back);
        expect(back.arena).toBeNull();
        expect(back.rowPtr.buffer).toBe(chunks[1].buffer);
        expectSnapshotsEqual(s, back);
        const copied = fromByteChunks(chunks, { copy: true });
        expect(copied.rowPtr.buffer).not.toBe(chunks[1].buffer);
        expectSnapshotsEqual(s, copied);
        const single = fromByteChunks([s.toBytes()]);
        expect(single.arena).toBeNull();
        expectSnapshotsEqual(s, single);
        const bytes = s.toBytes();
        const odd: Uint8Array[] = [];
        for (let i = 0; i < bytes.byteLength; i += 100) {
            odd.push(bytes.subarray(i, Math.min(i + 100, bytes.byteLength)));
        }
        expectSnapshotsEqual(s, fromByteChunks(odd));
        expectError(() => fromByteChunks([bytes.subarray(0, 10)]), "E_BAD_SERIALIZATION");
        expectError(() => fromByteChunks([5 as never]), "E_BAD_SERIALIZATION");
    });

    it("chunks the empty graph into a header chunk and one rowPtr chunk", () => {
        const empty = new GraphBuilder({ directed: false }).freeze();
        const chunks = [...empty.toByteChunks()];
        expect(chunks.length).toBe(2);
        expect(chunks[1].byteLength).toBe(ALIGNMENT);
        expect(Array.from(new Uint32Array(chunks[1].buffer, 0, 1))).toEqual([0]);
        expectSnapshotsEqual(empty, fromByteChunks(chunks));
        const bytes = empty.toBytes();
        expect(bytes.byteLength).toBe(chunks[0].byteLength + ALIGNMENT);
        expectSnapshotsEqual(empty, fromBytes(bytes));
        const withColumn = new GraphBuilder({ directed: true });
        withColumn.declareNodeColumn({ name: "x", dtype: "f32" });
        withColumn.declareNodeColumn({ name: "s", dtype: "string" });
        const columnSnapshot = withColumn.freeze();
        const columnChunks = [...columnSnapshot.toByteChunks()];
        expect(columnChunks.length).toBe(3);
        expectSnapshotsEqual(columnSnapshot, fromByteChunks(columnChunks));
        const columnBytes = columnSnapshot.toBytes();
        expect(columnBytes.byteLength).toBe(columnChunks[0].byteLength + 2 * ALIGNMENT);
        const back = fromBytes(columnBytes);
        expectSnapshotsEqual(columnSnapshot, back);
        expect(back.nodes.requireTyped("x", "f32").data.length).toBe(0);
    });

    it("round-trips a snapshot frozen without an arena", () => {
        const b = new GraphBuilder({ directed: false });
        b.addEdge("p", "q", 2);
        const s = b.freeze({ arena: false });
        expect(s.arena).toBeNull();
        const wire = s.toWire();
        expect(wire.manifest.arena).toBeNull();
        expect(wire.buffers.length).toBeGreaterThan(1);
        const back = fromWire(wire, { validate: "full" });
        expect(back.arena).toBeNull();
        expectSnapshotsEqual(s, back);
        const fromContainer = fromBytes(s.toBytes());
        expect(fromContainer.arena).not.toBeNull();
        expectSnapshotsEqual(s, fromContainer);
    });

    it("refuses unknown option values", () => {
        const s = idsSnapshot([0, 1]);
        const wire = s.toWire();
        expectError(() => fromWire(wire, { validate: "loose" as never }), "E_UNSUPPORTED");
        expectError(() => fromWire(wire, { unknownColumns: "drop" as never }), "E_UNSUPPORTED");
        expectError(() => fromWire(wire, { copy: "yes" as never }), "E_UNSUPPORTED");
        expectError(() => fromWire(5 as never), "E_BAD_SERIALIZATION");
        expectError(() => fromWire({ manifest: wire.manifest, buffers: 5 as never }), "E_BAD_SERIALIZATION");
    });

    it("transferables() and toWire() reuse the lazily materialised representations", () => {
        const s = richSnapshot({ ids: [1, "1", 2.5, "x"] });
        const first = s.toWire();
        const second = s.toWire();
        expect(second.buffers).toEqual(first.buffers);
        expect(s.transferables()).toEqual(first.buffers.filter((b) => !isShared(b)));
    });
});
