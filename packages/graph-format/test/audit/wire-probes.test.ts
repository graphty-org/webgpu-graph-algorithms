/**
 * Targeted audit probes of the wire module (design sections 4.5, 9.1 - 9.5, 11.1 - 11.4): the
 * semantics the fuzzer cannot judge on its own. The first block documents behaviour that holds
 * (structuredClone round trips, transfer detaching exactly the sender, shared node tables of derived
 * graphs, checksum records, unknownColumns "skip", contentHash stability). The second block
 * ("open defects") holds DELIBERATELY FAILING tests, one per finding of the audit, each asserting
 * what the design promises; the wire-fuzz.test.ts oracle recognises their signatures so it keeps
 * reporting new anomalies while these stay open.
 */

import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import { equalsTopology, type GraphSnapshot } from "../../src/snapshot/graph-snapshot.js";
import { type NodeId, type ValidationLevel, type WireBufferRef, type WireSnapshot } from "../../src/types/index.js";
import { fromByteChunks, fromBytes } from "../../src/wire/bytes.js";
import { fromWire, SKIPPED_COLUMNS_KEY } from "../../src/wire/from-wire.js";
import {
    at,
    buildContainer,
    expectError,
    expectSnapshotsEqual,
    idsSnapshot,
    rebuildContainer,
    richSnapshot,
    splitContainer,
} from "../wire/helpers.js";

const LEVELS: readonly ValidationLevel[] = ["full", "structure", "none"];

interface IdCase {
    readonly kind: string;
    readonly ids: readonly NodeId[];
}

const ID_CASES: readonly IdCase[] = [
    { kind: "identity", ids: [0, 1, 2, 3] },
    { kind: "identity", ids: [1, 2, 3, 4] },
    { kind: "dense", ids: [0, 2, 4, 6] },
    { kind: "numeric", ids: [0.5, 100, -3, 1e10] },
    { kind: "string", ids: ["a", "b", "\u00e9\u4e2d", ""] },
    { kind: "mixed", ids: [1, "a", 2, "b"] },
];

/** A typed view over the wire buffer a reference names. */
function viewOf(wire: WireSnapshot, ref: WireBufferRef): Uint32Array {
    return new Uint32Array(wire.buffers[ref.buffer], ref.byteOffset, ref.byteLength / 4);
}

/** Every view name, for prepare(). */
const ALL_VIEWS = [
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
] as const;

describe("wire probes: documented semantics that hold", () => {
    it("structuredClone(toWire(s)) round-trips every id-map kind and every dtype at every level", () => {
        const snapshots: GraphSnapshot[] = [
            richSnapshot(),
            richSnapshot({ directed: true, checksum: true }),
            ...ID_CASES.map((c) => idsSnapshot(c.ids)),
            ...ID_CASES.map((c) => idsSnapshot(c.ids, false)),
        ];
        for (const s of snapshots) {
            const wire = s.toWire();
            expect(JSON.parse(JSON.stringify(wire.manifest))).toEqual(wire.manifest);
            const clone = structuredClone(wire);
            expect(clone.buffers.every((b, i) => b !== wire.buffers[i])).toBe(true);
            for (const level of LEVELS) {
                const back = fromWire(clone, { validate: level });
                expectSnapshotsEqual(s, back);
                back.validate({ level: "full" });
                expect(back.ids.kind).toBe(s.ids.kind);
                expect(back.contentHash()).toBe(s.contentHash());
                // the clone's buffers are adopted by reference, the sender's are untouched
                expect(back.rowPtr.buffer).toBe(clone.buffers[clone.manifest.core.rowPtr.buffer]);
                expect(s.detached).toBe(false);
            }
        }
        for (const c of ID_CASES) {
            expect(idsSnapshot(c.ids).ids.kind).toBe(c.kind);
        }
    });

    it("toWire({ transfer: true }) plus transferables() detaches exactly the sender and leaves the receiver usable", () => {
        const s = richSnapshot({ directed: true });
        s.prepare(["outDegree", "reverse"]);
        const wire = s.toWire({ transfer: true });
        const list = s.transferables();
        // the transfer list is exactly the wire's buffers minus the copied ones (design 9.1)
        expect(wire.manifest.copied).toEqual([]);
        expect(new Set(list)).toEqual(new Set(wire.buffers));
        expect(list).toContain(s.arena?.buffer);
        expect(s.detached).toBe(false);
        // fan-out before the transfer: a structuredClone per receiver copies every buffer
        const fanOut = structuredClone(wire);
        expect(fanOut.buffers.every((b, i) => b !== wire.buffers[i])).toBe(true);
        const moved = structuredClone(wire, { transfer: list });
        expect(list.every((b) => b.byteLength === 0)).toBe(true);
        expect(s.detached).toBe(true);
        for (const call of [
            (): unknown => s.arcToEdge,
            (): unknown => s.edgeToArc,
            (): unknown => s.outDegree(),
            (): unknown => s.outDegreeOf(0),
            (): unknown => s.reverse(),
            (): unknown => s.edgeSource(0),
            (): unknown => s.contentHash(),
            (): unknown => s.toWire(),
            (): unknown => s.toBytes(),
            (): unknown => s.transferables(),
            (): unknown => s.validate(),
            (): unknown => s.withColumns({}),
            (): unknown => s.transpose(),
            (): unknown => s.byteLength(),
        ]) {
            expectError(call, "E_DETACHED");
        }
        expect(s.rowPtr.length).toBe(0);
        // the receiver is complete, the fan-out copy too
        const back = fromWire(moved, { validate: "full" });
        expectSnapshotsEqual(richSnapshot({ directed: true }), back);
        expect(back.detached).toBe(false);
        expectSnapshotsEqual(richSnapshot({ directed: true }), fromWire(fanOut, { validate: "full" }));
        // a receiver can be transferred on in turn (forwarding, design 9.1)
        const forwarded = structuredClone(back.toWire({ transfer: true }), { transfer: back.transferables() });
        expect(back.detached).toBe(true);
        expectSnapshotsEqual(richSnapshot({ directed: true }), fromWire(forwarded, { validate: "full" }));
        // without transfer nothing is transferable and the sender survives a clone
        const t = richSnapshot();
        const plain = t.toWire();
        structuredClone(plain);
        expect(t.detached).toBe(false);
        expect(plain.buffers).toContain(t.arena?.buffer);
    });

    it("toWire of a derived graph sharing the node table shares the buffers and copies them only under transfer", () => {
        const s = richSnapshot({ directed: true });
        const derived = s.withoutSelfLoops().snapshot;
        expect(derived.nodes).toBe(s.nodes);
        const posBuffer = s.nodes.requireTyped("pos", "f32").data.buffer;
        const wire = derived.toWire();
        // shared by reference: the same ArrayBuffer objects appear in the derived wire
        expect(wire.buffers).toContain(posBuffer);
        expect(wire.manifest.copied).toEqual([]);
        const back = fromWire(structuredClone(wire), { validate: "full" });
        expect(back.nodes.names()).toEqual(s.nodes.names());
        expect(back.nodes.value("label", 2)).toBe(s.nodes.value("label", 2));
        expect(back.edgeCount).toBe(s.edgeCount - s.selfLoopCount);
        expect(equalsTopology(back, derived)).toBe(true);
        // under transfer the shared node table is copied and listed, the derived core is transferred
        const transferWire = derived.toWire({ transfer: true });
        const posIndex = wire.buffers.indexOf(posBuffer);
        expect(transferWire.manifest.copied).toContain(posIndex);
        expect(transferWire.buffers[posIndex]).not.toBe(posBuffer);
        expect(derived.transferables()).not.toContain(posBuffer);
        expect(derived.transferables()).toContain(derived.arena?.buffer);
        structuredClone(transferWire, { transfer: derived.transferables() });
        expect(derived.detached).toBe(true);
        expect(s.detached).toBe(false);
        expect(s.nodes.value("label", 2)).toBe("node 2 \u00e9");
        expect(s.nodes.requireTyped("pos", "f32").data.length).toBe(s.nodeCount * 3);
        // a derived graph with a gathered node table owns its node columns exclusively (the graph table and
        // the extension tables are still shared with the source and therefore copied)
        const induced = s.inducedSubgraph(new Uint32Array([0, 1])).snapshot;
        expect(induced.nodes).not.toBe(s.nodes);
        expect(induced.graph).toBe(s.graph);
        const inducedWire = induced.toWire({ transfer: true });
        const inducedPos = induced.nodes.requireTyped("pos", "f32").data.buffer;
        expect(inducedPos).not.toBe(posBuffer);
        expect(inducedWire.manifest.copied).not.toContain(inducedWire.buffers.indexOf(inducedPos));
        expect(induced.transferables()).toContain(inducedPos);
        const graphCount = s.graph.requireTyped("count", "i32").data.buffer;
        const graphIndex = induced.toWire().buffers.indexOf(graphCount);
        expect(graphIndex).toBeGreaterThanOrEqual(0);
        expect(inducedWire.manifest.copied).toContain(graphIndex);
        expect(inducedWire.buffers[graphIndex]).not.toBe(graphCount);
        expect(fromWire(structuredClone(inducedWire), { validate: "full" }).nodes.value("label", 0)).toBe(
            "node 0 \u00e9",
        );
    });

    it("checksum records stay with the freezing realm: the sender verifies, the receiver reports no-checksum", () => {
        const s = richSnapshot({ checksum: true });
        s.validate({ checksum: true });
        const bytes = s.toBytes();
        const wire = s.toWire();
        // encoding does not disturb the records
        s.validate({ checksum: true, level: "full" });
        for (const back of [fromBytes(bytes), fromWire(structuredClone(wire)), fromByteChunks([bytes])]) {
            const error = expectError(() => back.validate({ checksum: true }), "E_INVALID_SNAPSHOT");
            expect(error.details.reason).toBe("no-checksum");
            back.validate({ level: "full" });
        }
        // a snapshot frozen without checksums reports the same on both sides
        const plain = richSnapshot();
        expect(expectError(() => plain.validate({ checksum: true }), "E_INVALID_SNAPSHOT").details.reason).toBe(
            "no-checksum",
        );
        // a write into the sender's frozen core after the encode is caught by the sender's checksums
        const colIdx = new Uint32Array(s.colIdx.buffer, s.colIdx.byteOffset, s.colIdx.length);
        const saved = colIdx[0];
        colIdx[0] = saved === 0 ? 1 : 0;
        const mismatch = expectError(() => s.validate({ checksum: true }), "E_INVALID_SNAPSHOT");
        expect(mismatch.details.reason).toBe("checksum");
        colIdx[0] = saved;
        s.validate({ checksum: true });
        // and the bytes written before the corruption still decode to the original graph
        expectSnapshotsEqual(richSnapshot(), fromBytes(bytes));
    });

    it("unknownColumns: skip drops exactly the unknown columns and reports domain, table, name and dtype", () => {
        const s = richSnapshot();
        const bytes = s.toBytes();
        const unknown = rebuildContainer(bytes, (m) => {
            at(m, "nodeColumns.5.meta").dtype = "f16";
            at(m, "edgeColumns.1.meta").dtype = "decimal";
            at(m, "graphColumns.1.meta").dtype = "i64";
            at(m, "extensions.0.columns.3.meta").dtype = "u16";
            // a list whose item dtype is unknown is skipped as a whole
            at(m, "nodeColumns.9.meta").itemDtype = "f16";
        });
        const refused = expectError(() => fromBytes(unknown), "E_UNSUPPORTED");
        expect(refused.details.dtype).toBe("f16");
        expect(refused.details.column).toBe("byte");
        for (const level of LEVELS) {
            const skipped = fromBytes(unknown, { validate: level, unknownColumns: "skip" });
            expect(skipped.nodes.has("byte")).toBe(false);
            expect(skipped.nodes.has("tags")).toBe(false);
            expect(skipped.edges.has("kind")).toBe(false);
            expect(skipped.graph.has("count")).toBe(false);
            expect(skipped.extensions.get("temporal:node:price")?.has("value")).toBe(false);
            // every other column survives untouched
            expect(skipped.nodes.names()).toEqual(s.nodes.names().filter((n) => n !== "byte" && n !== "tags"));
            expect(skipped.edges.names()).toEqual(s.edges.names().filter((n) => n !== "kind"));
            expect(skipped.nodes.value("label", 0)).toBe("node 0 \u00e9");
            expect(skipped.edgeIndexOf("e3")).toBe(3);
            expect(skipped.meta.extra[SKIPPED_COLUMNS_KEY]).toEqual([
                { domain: "node", table: null, name: "byte", dtype: "f16" },
                { domain: "node", table: null, name: "tags", dtype: "f16" },
                { domain: "edge", table: null, name: "kind", dtype: "decimal" },
                { domain: "graph", table: null, name: "count", dtype: "i64" },
                { domain: "extension", table: "temporal:node:price", name: "value", dtype: "u16" },
            ]);
            expect(Object.isFrozen(skipped.meta.extra[SKIPPED_COLUMNS_KEY])).toBe(true);
            if (level !== "none") {
                skipped.validate({ level: "full" });
            }
            // the report survives a further round trip
            const again = fromBytes(skipped.toBytes());
            expect(again.meta.extra[SKIPPED_COLUMNS_KEY]).toEqual(skipped.meta.extra[SKIPPED_COLUMNS_KEY]);
        }
        // the same policy through fromWire
        const wire = structuredClone(s.toWire());
        (wire.manifest.nodeColumns[5].meta as { dtype: string }).dtype = "f16";
        expectError(() => fromWire(wire), "E_UNSUPPORTED");
        expect(fromWire(wire, { unknownColumns: "skip" }).nodes.has("byte")).toBe(false);
        // an unknown id-map kind is never skippable (ids are not optional, design 9.1)
        const badIds = rebuildContainer(bytes, (m) => {
            at(m, "ids").kind = "sparse";
        });
        expect(expectError(() => fromBytes(badIds, { unknownColumns: "skip" }), "E_UNSUPPORTED").details.kind).toBe(
            "sparse",
        );
    });

    it("contentHash is stable across toBytes / fromBytes / fromWire / chunks / copy and independent of touched getters", () => {
        const cases: GraphSnapshot[] = [
            richSnapshot(),
            richSnapshot({ directed: true }),
            idsSnapshot([0, 1, 2, 3]),
            idsSnapshot(["a", "b", "c"], false),
            new GraphBuilder({ directed: true }).freeze(),
        ];
        for (const s of cases) {
            const hash = s.contentHash();
            expect(hash).toMatch(/^[0-9a-f]{16}$/);
            const bytes = s.toBytes();
            for (const level of LEVELS) {
                expect(fromBytes(bytes, { validate: level }).contentHash()).toBe(hash);
                expect(fromBytes(bytes, { validate: level, copy: true }).contentHash()).toBe(hash);
                expect(fromWire(structuredClone(s.toWire()), { validate: level }).contentHash()).toBe(hash);
                expect(fromByteChunks(s.toByteChunks(), { validate: level }).contentHash()).toBe(hash);
            }
            const offset8 = new Uint8Array(bytes.byteLength + 8);
            offset8.set(bytes, 8);
            expect(fromBytes(offset8.subarray(8)).contentHash()).toBe(hash);
            expect(
                fromBytes(
                    new Uint8Array(new SharedArrayBuffer(bytes.byteLength)).fill(0).map((_, i) => bytes[i]),
                ).contentHash(),
            ).toBe(hash);
        }
        // an identity permutation hashes the same whether or not arcToEdge was materialised (P11)
        const a = idsSnapshot([0, 1, 2, 3]);
        const b = idsSnapshot([0, 1, 2, 3]);
        expect(b.arcToEdge.length).toBe(b.arcCount);
        expect(b.edgeToArc.length).toBe(b.edgeCount);
        expect(a.contentHash()).toBe(b.contentHash());
        expect(fromBytes(b.toBytes()).contentHash()).toBe(a.contentHash());
        // and a different topology hashes differently
        expect(idsSnapshot([0, 1, 2, 3], false).contentHash()).not.toBe(a.contentHash());
        expect(idsSnapshot([0, 1, 2]).contentHash()).not.toBe(a.contentHash());
    });

    it("carried views survive the container at structure level and are recomputed under full", () => {
        const s = richSnapshot({ directed: true });
        s.prepare(ALL_VIEWS);
        const bytes = s.toBytes({ includeViews: ALL_VIEWS });
        const full = fromBytes(bytes);
        expect(full.cachedViews()).toEqual([]);
        const structure = fromBytes(bytes, { validate: "structure" });
        expect(new Set(structure.cachedViews())).toEqual(new Set(s.cachedViews()));
        for (const name of ["outDegree", "inDegree", "degree", "selfLoopArcs", "selfLoopsPerNode"] as const) {
            expect(Array.from(structure[name]())).toEqual(Array.from(s[name]()));
        }
        expect(Array.from(structure.degreeOrder().perm)).toEqual(Array.from(s.degreeOrder().perm));
        expectSnapshotsEqual(s, structure);
        expect(structure.contentHash()).toBe(s.contentHash());
    });
});

// ============================================================ open defects (deliberately failing pins)

describe("wire probes: open defects", () => {
    it("toWire({ transfer: true, includeColumns: false }) and transferables() agree, so no column is lost", () => {
        // design 9.1: transferables() returns the same exclusive set toWire({ transfer: true }) put into buffers
        const s = richSnapshot();
        const wire = s.toWire({ transfer: true, includeColumns: false });
        const list = s.transferables();
        const posBuffer = s.nodes.requireTyped("pos", "f32").data.buffer;
        expect(wire.buffers).not.toContain(posBuffer);
        expect
            .soft(
                list.filter((b) => !wire.buffers.includes(b)).length,
                "transferables() lists buffers the wire does not carry",
            )
            .toBe(0);
        const moved = structuredClone(wire, { transfer: list });
        // the receiver never had the columns...
        expect(fromWire(moved, { validate: "full" }).nodes.names()).toEqual([]);
        // ...and the sender must still have them
        expect.soft(s.nodes.requireTyped("pos", "f32").data.length, "sender lost its pos column").toBe(s.nodeCount * 3);
        expect(s.nodes.value("label", 0), "sender lost its label column").toBe("node 0 \u00e9");
    });

    it("column and id accessors after a consuming transfer throw E_DETACHED, never a TypeError or undefined", () => {
        // design 11.1: only GraphFormatError is thrown; 11.3: any accessor after a consuming transfer is E_DETACHED
        const s = richSnapshot();
        s.prepare(["outDegree"]);
        structuredClone(s.toWire({ transfer: true }), { transfer: s.transferables() });
        expect(s.detached).toBe(true);
        // row 0 of "end" and "pos" is set in the fixture; the detached validity bitmap now reads as unset
        const calls: readonly [string, () => unknown][] = [
            ["nodes.value(pos, 0) (3 components: a raw TypeError)", () => s.nodes.value("pos", 0)],
            ["nodes.value(label, 0) (a set row reads as undefined)", () => s.nodes.value("label", 0)],
            ["nodes.value(solid, 0) (a non-nullable u32 reads as undefined)", () => s.nodes.value("solid", 0)],
            ["nodes.isSet(end, 0) (a set row reports false)", () => s.nodes.isSet("end", 0)],
            [
                "validate({ checksum: true }) with a shared core (a raw TypeError)",
                () => richSnapshotDetached().validate({ checksum: true }),
            ],
        ];
        for (const [label, call] of calls) {
            let caught: unknown = null;
            let result: unknown = null;
            try {
                result = call();
            } catch (err) {
                caught = err;
            }
            expect.soft(caught, `${label}: returned ${JSON.stringify(result)}`).toBeInstanceOf(GraphFormatError);
            if (caught instanceof GraphFormatError) {
                expect.soft(caught.code, label).toBe("E_DETACHED");
            }
        }
        // design 9.1: view caches of a detached snapshot are dropped
        expect.soft(s.cachedViews(), "cachedViews() of a detached snapshot").toEqual([]);
        // a wire-decoded snapshot that is transferred on: its id map reads detached bytes
        const back = fromWire(structuredClone(richSnapshot().toWire()), { validate: "full" });
        structuredClone(back.toWire({ transfer: true }), { transfer: back.transferables() });
        expect(back.detached).toBe(true);
        expectError(() => back.ids.toArray(), "E_DETACHED");
        // VERIFIER (round 1): at "full" the reverse Map is built eagerly, so the lookups above never
        // read the store; at the default level the first indexOf() decodes the Utf8 store lazily and
        // read detached bytes (a raw TypeError) -- every lookup of every typed id-map kind is E_DETACHED
        const idSets: readonly (readonly NodeId[])[] = [
            ["a", "b", "c", "d"],
            [1, "x", 3, "y"],
            [10, 20, 30, 40],
        ];
        for (const ids of idSets) {
            const lazy = fromWire(structuredClone(idsSnapshot(ids).toWire()));
            structuredClone(lazy.toWire({ transfer: true }), { transfer: lazy.transferables() });
            expect(lazy.detached).toBe(true);
            const lookups: readonly [string, () => unknown][] = [
                ["indexOf", () => lazy.ids.indexOf(ids[0])],
                ["has", () => lazy.ids.has(ids[0])],
                ["requireIndex", () => lazy.ids.requireIndex(ids[0])],
                ["indicesOf", () => lazy.ids.indicesOf(ids)],
                ["stringIndex", () => lazy.ids.stringIndex()],
                ["idOf", () => lazy.ids.idOf(0)],
                ["idsSlice", () => lazy.ids.idsSlice(0, 2)],
                ["toArray", () => lazy.ids.toArray()],
                ["toMap", () => lazy.ids.toMap(new Uint32Array(ids.length))],
                ["entries", () => [...lazy.ids.entries(new Uint32Array(ids.length))]],
            ];
            for (const [label, call] of lookups) {
                let caught: unknown = null;
                let result: unknown = null;
                try {
                    result = call();
                } catch (err) {
                    caught = err;
                }
                const where = `${lazy.ids.kind} ids: ${label} returned ${JSON.stringify(result)}`;
                expect.soft(caught, where).toBeInstanceOf(GraphFormatError);
                if (caught instanceof GraphFormatError) {
                    expect.soft(caught.code, where).toBe("E_DETACHED");
                }
            }
        }
    });

    it("a same-realm fromWire receiver counts as a holder, so a later transfer copies rather than empties it", () => {
        // design 9.1: "no sibling snapshot is ever silently emptied"; the owner count records every second holder
        const s = richSnapshot();
        const receiver = fromWire(s.toWire(), { validate: "full" });
        expect(receiver.nodes.value("label", 0)).toBe("node 0 \u00e9");
        const wire = s.toWire({ transfer: true });
        const list = s.transferables();
        // the core is recognised as shared (copied), the columns and the id store are not
        expect(wire.manifest.copied).toContain(wire.manifest.core.rowPtr.buffer);
        structuredClone(wire, { transfer: list });
        expect(receiver.detached).toBe(false);
        expect.soft(receiver.nodes.value("label", 0), "receiver's string column was emptied").toBe("node 0 \u00e9");
        expect
            .soft(receiver.nodes.requireTyped("pos", "f32").data.length, "receiver's f32 column was emptied")
            .toBe(receiver.nodeCount * 3);
        expect(receiver.ids.toArray(), "receiver's id store was emptied").toEqual(["a", "b", "c", "d"]);
    });

    it("the mutable-column overlap rule holds by buffer identity, not by buffer index", () => {
        // design 9.5 (structure): no byte overlap between a mutable column and any core or immutable segment
        const s = richSnapshot();
        const wire = structuredClone(s.toWire());
        const core = wire.manifest.core.rowPtr;
        const buffers = [...wire.buffers, wire.buffers[core.buffer]];
        const pos = wire.manifest.nodeColumns.find((c) => c.meta.name === "pos") as { data: WireBufferRef | null };
        pos.data = {
            buffer: buffers.length - 1,
            byteOffset: core.byteOffset,
            byteLength: 48,
            dtype: "f32",
            length: 12,
        };
        const aliased: WireSnapshot = { manifest: wire.manifest, buffers };
        let decoded: GraphSnapshot | null = null;
        try {
            decoded = fromWire(aliased, { validate: "structure" });
        } catch (err) {
            expect(err).toBeInstanceOf(GraphFormatError);
            expect((err as GraphFormatError).code).toBe("E_BAD_SERIALIZATION");
        }
        if (decoded !== null) {
            const before = Array.from(decoded.rowPtr);
            decoded.nodes.requireTyped("pos", "f32").mutableData().fill(0);
            expect(Array.from(decoded.rowPtr), "writing the mutable column rewrote rowPtr").toEqual(before);
        }
    });

    it("carried views are content-checked below full validation (I2 applies to every view array)", () => {
        // design 3.2 I2: INVALID_INDEX never appears in any view array; 9.5: structure checks I2
        const s = idsSnapshot([0, 1, 2, 3]);
        s.prepare(["degreeOrder", "outDegree", "coo", "selfLoopArcs"]);
        const wire = structuredClone(s.toWire({ includeViews: ["degreeOrder", "outDegree", "coo", "selfLoopArcs"] }));
        const views = wire.manifest.views as Record<string, Record<string, WireBufferRef>>;
        viewOf(wire, views.degreeOrder.perm).fill(INVALID_INDEX);
        viewOf(wire, views.coo.src).fill(INVALID_INDEX);
        viewOf(wire, views.outDegree.data).fill(0xdeadbeef);
        let decoded: GraphSnapshot | null = null;
        try {
            decoded = fromWire(wire, { validate: "structure" });
        } catch (err) {
            expect(err).toBeInstanceOf(GraphFormatError);
            expect((err as GraphFormatError).code).toBe("E_INVALID_SNAPSHOT");
        }
        if (decoded !== null) {
            // validate() does not look at resident views either
            decoded.validate({ level: "full" });
            expect(decoded.cachedViews()).toContain("degreeOrder");
            expect
                .soft(Array.from(decoded.degreeOrder().perm), "installed perm holds INVALID_INDEX")
                .toEqual([0, 1, 2, 3]);
            expect.soft(Array.from(decoded.coo().src), "installed coo.src holds INVALID_INDEX").toEqual([0, 1, 2]);
            expect(Array.from(decoded.outDegree()), "installed outDegree holds garbage").toEqual([1, 1, 1, 0]);
        }
    });

    it("validate() reports a violated unique column as E_INVALID_SNAPSHOT", () => {
        // design 11.4: validate() throws E_INVALID_SNAPSHOT with details.invariant; 9.5: any failure of an
        // untrusted read is E_INVALID_SNAPSHOT or E_BAD_SERIALIZATION (E_DUPLICATE_* are freeze-time codes)
        const s = richSnapshot();
        const bytes = s.toBytes();
        const parts = splitContainer(bytes);
        const idColumn = (parts.manifest.edgeColumns as { strings: { utf8: WireBufferRef } }[])[0];
        const { utf8 } = idColumn.strings;
        const corrupted = bytes.slice();
        const regionStart = bytes.byteLength - parts.region.byteLength;
        // "e1" -> "e0": the id column is declared unique
        corrupted[regionStart + utf8.byteOffset + 3] = 0x30;
        const fromContainer = expectError(() => fromBytes(corrupted), "E_INVALID_SNAPSHOT");
        expect(fromContainer.details.invariant).toBe("I12");
        const decoded = fromBytes(corrupted, { validate: "structure" });
        expectError(() => decoded.validate({ level: "full" }), "E_INVALID_SNAPSHOT");
    });

    it("a bigint or null-prototype manifest value is refused with E_BAD_SERIALIZATION", () => {
        // design 11.1: only GraphFormatError instances are thrown; structuredClone carries both value kinds
        const s = idsSnapshot([0, 1, 2]);
        const cases: readonly [string, (wire: WireSnapshot) => void][] = [
            ["core.rowPtr.dtype = 5n", (w) => ((w.manifest.core.rowPtr as { dtype: unknown }).dtype = 5n)],
            ["ids.kind = 5n", (w) => ((w.manifest.ids as { kind: unknown }).kind = 5n)],
            ["format = 5n", (w) => ((w.manifest as { format: unknown }).format = 5n)],
            [
                "formatVersion = Object.create(null)",
                (w) => ((w.manifest as { formatVersion: unknown }).formatVersion = Object.create(null)),
            ],
            ["meta.idType = 5n", (w) => ((w.manifest.meta as { idType: unknown }).idType = 5n)],
        ];
        for (const [label, poison] of cases) {
            const wire = structuredClone(s.toWire());
            poison(wire);
            let caught: unknown = null;
            try {
                fromWire(wire, { validate: "none" });
            } catch (err) {
                caught = err;
            }
            expect.soft(caught, label).toBeInstanceOf(GraphFormatError);
        }
    });

    it("negative zero is not accepted as a count, an identity offset or a numeric id", () => {
        // design 4.1: -0 equals 0 and is stored as 0; a count of -0 is not "an integer in [0, MAX_COUNT]"
        const s = idsSnapshot([0, 1, 2, 3]);
        const counts = structuredClone(s.toWire());
        (counts.manifest.counts as { selfLoops: number }).selfLoops = -0;
        const back = fromWire(counts, { validate: "full" });
        expect.soft(Object.is(back.selfLoopCount, 0), "selfLoopCount is -0").toBe(true);
        const offset = structuredClone(s.toWire());
        (offset.manifest.ids as { offset: number }).offset = -0;
        expect.soft(Object.is(fromWire(offset, { validate: "full" }).ids.offset, 0), "ids.offset is -0").toBe(true);
        const numeric = idsSnapshot([0.5, 7, -3]);
        const wire = structuredClone(numeric.toWire());
        const values = wire.manifest.ids.values as WireBufferRef;
        new Float64Array(wire.buffers[values.buffer], values.byteOffset, values.byteLength / 8)[1] = -0;
        const decoded = fromWire(wire, { validate: "full" });
        expect(decoded.ids.indexOf(0)).toBe(1);
        expect(Object.is(decoded.ids.idOf(1), 0), "a numeric id of -0 is returned as -0").toBe(true);
    });

    it("an identity offset whose ids leave the safe-integer range is refused at the structure level", () => {
        // design 4.2: identity means id === index + offset for every index; the check is O(1)
        const s = idsSnapshot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        const wire = structuredClone(s.toWire());
        (wire.manifest.ids as { offset: number }).offset = Number.MAX_SAFE_INTEGER - 2;
        expectError(() => fromWire(wire, { validate: "full" }), "E_INVALID_SNAPSHOT");
        let decoded: GraphSnapshot | null = null;
        try {
            decoded = fromWire(wire, { validate: "structure" });
        } catch (err) {
            expect(err).toBeInstanceOf(GraphFormatError);
            expect((err as GraphFormatError).code).toBe("E_BAD_SERIALIZATION");
        }
        if (decoded !== null) {
            const ids = decoded.ids.toArray();
            expect(new Set(ids).size, "structure-level fromWire produced colliding identity ids").toBe(ids.length);
        }
    });

    it("a dense id map checks its bound before building the inverse at every level", () => {
        // design 4.2: dense requires maxId + 1 <= 2 * nodeCount; one corrupted value must not allocate 16 GB
        const s = idsSnapshot([0, 2, 4, 6]);
        const wire = structuredClone(s.toWire());
        const values = wire.manifest.ids.values as WireBufferRef;
        viewOf(wire, values)[3] = 0xfffffff0;
        expectError(() => fromWire(wire, { validate: "structure" }), "E_BAD_SERIALIZATION");
        let decoded: GraphSnapshot | null = null;
        try {
            decoded = fromWire(wire, { validate: "none" });
        } catch (err) {
            expect(err).toBeInstanceOf(GraphFormatError);
        }
        if (decoded !== null) {
            expect(
                decoded.ids.byteLength(),
                "the inverse array of a 4-node map was allocated for id 0xfffffff0",
            ).toBeLessThan(1024);
        }
    });

    it("deeply nested metadata in an in-memory manifest is refused with E_BAD_SERIALIZATION, not a RangeError", () => {
        // design 11.1: only GraphFormatError instances are thrown; the container path already maps this case
        const s = idsSnapshot([0, 1, 2]);
        const deepText = `${"[".repeat(100000)}${"]".repeat(100000)}`;
        const deep = JSON.parse(`{"x":${deepText}}`) as Record<string, unknown>;
        // the container path parses the manifest with a reviver, whose own recursion is caught and mapped
        const parts = splitContainer(s.toBytes());
        expect(parts.manifestText.split('"extra":{}').length).toBe(2);
        const container = buildContainer(
            parts.manifestText.replace('"extra":{}', `"extra":{"x":${deepText}}`),
            parts.region,
        );
        expectError(() => fromBytes(container), "E_BAD_SERIALIZATION");
        const cases: readonly [string, (wire: WireSnapshot) => void][] = [
            ["meta.extra", (w) => ((w.manifest.meta as { extra: unknown }).extra = deep)],
            ["column meta.extra", (w) => ((w.manifest.nodeColumns[0].meta as { extra: unknown }).extra = deep)],
            ["column meta.default", (w) => ((w.manifest.nodeColumns[0].meta as { default: unknown }).default = deep.x)],
        ];
        for (const [label, poison] of cases) {
            const t = idsSnapshot([0, 1, 2]);
            t.nodes.set("j", [null, { a: 1 }, [1]]);
            const wire = structuredClone(t.toWire());
            poison(wire);
            let caught: unknown = null;
            try {
                fromWire(wire, { validate: "none" });
            } catch (err) {
                caught = err;
            }
            expect.soft(caught, label).toBeInstanceOf(GraphFormatError);
        }
    });
});

/** A checksum-recording snapshot whose exclusive column buffers were transferred while its core stayed shared. */
function richSnapshotDetached(): GraphSnapshot {
    const s = richSnapshot({ checksum: true });
    // a consumer that kept its own view over a column buffer is outside the owner count (design 9.1)
    // and can transfer it away underneath the snapshot; the core stays attached
    const pos = s.nodes.requireTyped("pos", "f32").data.buffer;
    structuredClone(pos, { transfer: [pos] });
    expect(s.detached).toBe(false);
    return s;
}
