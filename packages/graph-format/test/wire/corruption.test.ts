/**
 * Corruption tests of design section 16.3: every validate branch of the container and manifest
 * readers, each asserting the documented error code and, where applicable, the invariant number or
 * the offending reference.
 */

import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { FORMAT_VERSION, WIRE_MAJOR } from "../../src/constants.js";
import { type FromWireOptions, type WireBufferRef, type WireSnapshot } from "../../src/types/index.js";
import { fromByteChunks, fromBytes } from "../../src/wire/bytes.js";
import { decodeJsonValue, fromWire, parseGuardedJson, SKIPPED_COLUMNS_KEY } from "../../src/wire/from-wire.js";
import { assertLittleEndianHost, encodeJsonValue } from "../../src/wire/to-wire.js";
import {
    at,
    buildContainer,
    expectError,
    idsSnapshot,
    rebuildContainer,
    richSnapshot,
    splitContainer,
} from "./helpers.js";

/** The container offset of the buffer region. */
function regionStart(bytes: Uint8Array): number {
    return bytes.byteLength - splitContainer(bytes).region.byteLength;
}

/** The container offset of a core array's segment. */
function segmentStart(bytes: Uint8Array, name: string): number {
    const ref = at(splitContainer(bytes).manifest, `core.${name}`) as unknown as WireBufferRef;
    return regionStart(bytes) + ref.byteOffset;
}

/** A copy of the bytes with one byte replaced. */
function patched(bytes: Uint8Array, offset: number, value: number): Uint8Array {
    const out = bytes.slice();
    out[offset] = value;
    return out;
}

const LEVELS: readonly FromWireOptions[] = [{ validate: "none" }, { validate: "structure" }, { validate: "full" }];

describe("container header corruption", () => {
    const s = richSnapshot();
    const bytes = s.toBytes();

    it("refuses a changed magic", () => {
        const error = expectError(() => fromBytes(patched(bytes, 0, 0x58)), "E_BAD_SERIALIZATION");
        expect(error.details.ref).toBe("magic");
    });

    it("refuses a truncated header", () => {
        const error = expectError(() => fromBytes(bytes.subarray(0, 12)), "E_BAD_SERIALIZATION");
        expect(error.details.ref).toBe("header");
        expectError(() => fromBytes(new ArrayBuffer(0)), "E_BAD_SERIALIZATION");
    });

    it("refuses an unknown wire major with E_UNSUPPORTED_VERSION kind wire", () => {
        const error = expectError(() => fromBytes(patched(bytes, 4, WIRE_MAJOR + 1)), "E_UNSUPPORTED_VERSION");
        expect(error.details).toMatchObject({ kind: "wire", found: WIRE_MAJOR + 1, supported: WIRE_MAJOR });
        const manifestOnly = rebuildContainer(bytes, (m) => {
            m.wire = [WIRE_MAJOR + 1, 0];
        });
        expect(expectError(() => fromBytes(manifestOnly), "E_UNSUPPORTED_VERSION").details.kind).toBe("wire");
        expect(
            expectError(
                () => fromWire({ ...s.toWire(), manifest: { ...s.toWire().manifest, wire: [9, 0] } }),
                "E_UNSUPPORTED_VERSION",
            ).details.kind,
        ).toBe("wire");
    });

    it("refuses a header minor that disagrees with the manifest and accepts a newer minor in both", () => {
        const error = expectError(() => fromBytes(patched(bytes, 6, 3)), "E_BAD_SERIALIZATION");
        expect(error.details.ref).toBe("wire");
        const newer = rebuildContainer(
            bytes,
            (m) => {
                m.wire = [WIRE_MAJOR, 7];
                m.futureField = { anything: true };
            },
            { minor: 7 },
        );
        const back = fromBytes(newer);
        expect(back.nodeCount).toBe(4);
        const wire = s.toWire();
        const withField = fromWire({
            manifest: { ...wire.manifest, wire: [WIRE_MAJOR, 9], unknownField: 1 } as never,
            buffers: wire.buffers,
        });
        expect(withField.nodeCount).toBe(4);
    });

    it("refuses a formatVersion mismatch with an unchanged layout, kind format", () => {
        const wrong = rebuildContainer(bytes, (m) => {
            m.formatVersion = FORMAT_VERSION + 1;
        });
        const error = expectError(() => fromBytes(wrong), "E_UNSUPPORTED_VERSION");
        expect(error.details).toMatchObject({ kind: "format", found: FORMAT_VERSION + 1, supported: FORMAT_VERSION });
    });

    it("refuses the probe bytes 01 02 03 04", () => {
        const swapped = bytes.slice();
        swapped.set([1, 2, 3, 4], 8);
        const error = expectError(() => fromBytes(swapped), "E_BAD_SERIALIZATION");
        expect(error.details.ref).toBe("endianness");
    });

    it("refuses a manifest length beyond the file and a truncated manifest", () => {
        const parts = splitContainer(bytes);
        const lying = buildContainer(parts.manifestText, parts.region, { declaredLength: bytes.byteLength + 10 });
        expect(expectError(() => fromBytes(lying), "E_BAD_SERIALIZATION").details.ref).toBe("manifest");
        const truncated = bytes.subarray(0, 40);
        expect(expectError(() => fromBytes(truncated), "E_BAD_SERIALIZATION").details.ref).toBe("manifest");
    });

    it("refuses a manifest that is not JSON, not UTF-8 or not an object", () => {
        const parts = splitContainer(bytes);
        expectError(() => fromBytes(buildContainer("{ not json", parts.region)), "E_BAD_SERIALIZATION");
        expectError(() => fromBytes(buildContainer("[1, 2]", parts.region)), "E_BAD_SERIALIZATION");
        const bad = buildContainer("{}", parts.region);
        bad[16] = 0xff;
        bad[17] = 0xfe;
        expect(expectError(() => fromBytes(bad), "E_BAD_SERIALIZATION").details.reason).toBe("not valid UTF-8");
        expectError(() => fromBytes(buildContainer('{"format": "other"}', parts.region)), "E_BAD_SERIALIZATION");
    });

    it("refuses the prototype-pollution keys in the manifest and in json column text", () => {
        const parts = splitContainer(bytes);
        const polluted = parts.manifestText.replace('"label":"rich"', '"label":"rich","__proto__":{"x":1}');
        expect(polluted).not.toBe(parts.manifestText);
        const error = expectError(() => fromBytes(buildContainer(polluted, parts.region)), "E_BAD_SERIALIZATION");
        expect(error.details.key).toBe("__proto__");
        const inExtra = rebuildContainer(bytes, (m) => {
            (m.meta as Record<string, unknown>).extra = { constructor: 1 };
        });
        expect(expectError(() => fromBytes(inExtra), "E_BAD_SERIALIZATION").details.key).toBe("constructor");
        expect(expectError(() => parseGuardedJson('{"prototype": 1}', "x"), "E_BAD_SERIALIZATION").details.key).toBe(
            "prototype",
        );
        expectError(() => parseGuardedJson("nope", "x"), "E_BAD_SERIALIZATION");
        const wire = s.toWire();
        const extra = { ...wire.manifest.meta.extra, __proto__: { polluted: true } } as Record<string, unknown>;
        expect(Object.keys(extra)).not.toContain("__proto__");
        const manifest = {
            ...wire.manifest,
            meta: { ...wire.manifest.meta, extra: JSON.parse('{"__proto__": 1}') as never },
        };
        expectError(() => fromWire({ manifest, buffers: wire.buffers }), "E_BAD_SERIALIZATION");
    });

    it("refuses truncated regions at every level", () => {
        for (const options of LEVELS) {
            const cut = bytes.subarray(0, regionStart(bytes) + 300);
            const error = expectError(() => fromBytes(cut, options), "E_BAD_SERIALIZATION");
            expect(typeof error.details.ref).toBe("string");
            expectError(() => fromByteChunks([cut], options), "E_BAD_SERIALIZATION");
        }
        const noRegion = bytes.subarray(0, regionStart(bytes));
        expect(expectError(() => fromBytes(noRegion), "E_BAD_SERIALIZATION").details.ref).toBe("core.rowPtr");
    });
});

describe("buffer reference corruption", () => {
    const s = richSnapshot();
    const bytes = s.toBytes();

    it("refuses an odd byteOffset, a lying byteLength, a range past the buffer and a bad buffer index", () => {
        const odd = rebuildContainer(bytes, (m) => {
            at(m, "core.colIdx").byteOffset = 258;
        });
        expect(expectError(() => fromBytes(odd), "E_BAD_SERIALIZATION").details.ref).toBe("core.colIdx");
        const lying = rebuildContainer(bytes, (m) => {
            at(m, "core.colIdx").byteLength = 8;
        });
        expect(expectError(() => fromBytes(lying), "E_BAD_SERIALIZATION").details.ref).toBe("core.colIdx");
        const past = rebuildContainer(bytes, (m) => {
            const ref = at(m, "core.colIdx");
            ref.length = 1_000_000;
            ref.byteLength = 4_000_000;
        });
        expect(expectError(() => fromBytes(past, { validate: "none" }), "E_BAD_SERIALIZATION").details.ref).toBe(
            "core.colIdx",
        );
        const badIndex = rebuildContainer(bytes, (m) => {
            at(m, "core.rowPtr").buffer = 3;
        });
        expect(expectError(() => fromBytes(badIndex), "E_BAD_SERIALIZATION").details.ref).toBe("core.rowPtr");
        const negative = rebuildContainer(bytes, (m) => {
            at(m, "core.rowPtr").byteOffset = -4;
        });
        expectError(() => fromBytes(negative), "E_BAD_SERIALIZATION");
    });

    it("refuses an unknown wire dtype and a dtype that does not match the slot", () => {
        const unknown = rebuildContainer(bytes, (m) => {
            at(m, "core.rowPtr").dtype = "f16";
        });
        expect(expectError(() => fromBytes(unknown), "E_BAD_SERIALIZATION").details.ref).toBe("core.rowPtr");
        const mismatch = rebuildContainer(bytes, (m) => {
            at(m, "core.rowPtr").dtype = "f32";
        });
        expect(expectError(() => fromBytes(mismatch), "E_BAD_SERIALIZATION").details).toMatchObject({
            ref: "core.rowPtr",
            expected: "u32",
            found: "f32",
        });
        const notAnObject = rebuildContainer(bytes, (m) => {
            (m.core as Record<string, unknown>).rowPtr = 5;
        });
        expectError(() => fromBytes(notAnObject), "E_BAD_SERIALIZATION");
    });

    it("refuses a mutable column overlapping a core or immutable segment at the structure level", () => {
        const posIndex = s.nodes.names().indexOf("pos");
        const overlap = rebuildContainer(bytes, (m) => {
            const data = at(m, `nodeColumns.${posIndex}.data`);
            data.byteOffset = 256;
        });
        const error = expectError(() => fromBytes(overlap, { validate: "structure" }), "E_BAD_SERIALIZATION");
        expect(error.details.ref).toBe(`nodeColumns[${posIndex}].data`);
        expect(error.details.overlaps).toBe("core.colIdx");
        const lenient = fromBytes(overlap, { validate: "none" });
        expect(lenient.nodes.requireTyped("pos", "f32").data.byteOffset).toBe(lenient.colIdx.byteOffset);
        const scoreIndex = s.nodes.names().indexOf("score");
        const immutableOverlap = rebuildContainer(bytes, (m) => {
            at(m, `nodeColumns.${scoreIndex}.data`).byteOffset = 256;
        });
        const aliased = fromBytes(immutableOverlap, { validate: "full" });
        expect(Array.from(aliased.nodes.requireTyped("score", "i32").data)).toEqual(
            Array.from(s.colIdx.subarray(0, 4)),
        );
    });

    it("refuses arena descriptors that lie", () => {
        const hot = rebuildContainer(bytes, (m) => {
            at(m, "arena").hotByteLength = 4;
        });
        expect(expectError(() => fromBytes(hot), "E_BAD_SERIALIZATION").details.ref).toBe("arena.hotByteLength");
        expect(fromBytes(hot, { validate: "none" }).arena?.hotByteLength).toBe(s.arena?.hotByteLength);
        const past = rebuildContainer(bytes, (m) => {
            at(m, "arena").byteLength = 1 << 30;
        });
        expect(expectError(() => fromBytes(past), "E_BAD_SERIALIZATION").details.ref).toBe("arena");
        const badIndex = rebuildContainer(bytes, (m) => {
            at(m, "arena").buffer = 4;
        });
        expect(expectError(() => fromBytes(badIndex), "E_BAD_SERIALIZATION").details.ref).toBe("arena.buffer");
        const misaligned = rebuildContainer(bytes, (m) => {
            at(m, "arena").byteOffset = 4;
            at(m, "arena").byteLength = (at(m, "arena").byteLength as number) - 4;
        });
        expect(expectError(() => fromBytes(misaligned), "E_BAD_SERIALIZATION").details.ref).toBe("core.colIdx");
        const noArena = rebuildContainer(bytes, (m) => {
            m.arena = null;
        });
        expect(fromBytes(noArena).arena).toBeNull();
    });

    it("refuses core references that disagree with the flags and counts", () => {
        const cases: readonly [string, (m: Record<string, unknown>) => void][] = [
            [
                "colIdx null with arcs",
                (m) => {
                    (m.core as Record<string, unknown>).colIdx = null;
                },
            ],
            [
                "weights without flag",
                (m) => {
                    (m.flags as Record<string, unknown>).weighted = false;
                },
            ],
            [
                "weights null with flag",
                (m) => {
                    (m.core as Record<string, unknown>).weights = null;
                },
            ],
            [
                "arcToEdge null without identity",
                (m) => {
                    (m.core as Record<string, unknown>).arcToEdge = null;
                },
            ],
            [
                "edgeToArc null without identity",
                (m) => {
                    (m.core as Record<string, unknown>).edgeToArc = null;
                },
            ],
            [
                "identity flag on an undirected graph",
                (m) => {
                    (m.flags as Record<string, unknown>).arcToEdgeIsIdentity = true;
                },
            ],
            [
                "flag not a boolean",
                (m) => {
                    (m.flags as Record<string, unknown>).multigraph = "yes";
                },
            ],
            [
                "count not an integer",
                (m) => {
                    (m.counts as Record<string, unknown>).nodes = 1.5;
                },
            ],
            [
                "directed not a boolean",
                (m) => {
                    m.directed = 1;
                },
            ],
        ];
        for (const [name, edit] of cases) {
            const corrupt = rebuildContainer(bytes, edit);
            expect(() => fromBytes(corrupt, { validate: "none" }), name).toThrow();
            expectError(() => fromBytes(corrupt, { validate: "none" }), "E_BAD_SERIALIZATION");
        }
        const directed = idsSnapshot([0, 1, 2]).toBytes();
        const present = rebuildContainer(directed, (m) => {
            (m.core as Record<string, unknown>).arcToEdge = at(m, "core.colIdx");
        });
        expect(expectError(() => fromBytes(present), "E_BAD_SERIALIZATION").details.ref).toBe("core.arcToEdge");
    });

    it("reports count lies as column-length or invariant violations", () => {
        const nodes = rebuildContainer(bytes, (m) => {
            (m.counts as Record<string, unknown>).nodes = 5;
        });
        expect(expectError(() => fromBytes(nodes, { validate: "none" }), "E_BAD_SERIALIZATION").details.cause).toBe(
            "E_COLUMN_LENGTH",
        );
        const ids = rebuildContainer(bytes, (m) => {
            (m.ids as Record<string, unknown>).size = 3;
        });
        expect(expectError(() => fromBytes(ids, { validate: "none" }), "E_INVALID_SNAPSHOT").details.invariant).toBe(
            "I11",
        );
        expect(expectError(() => fromBytes(ids, { validate: "structure" }), "E_BAD_SERIALIZATION").details.ref).toBe(
            "ids.offsets",
        );
        const bare = idsSnapshot([0, 1, 2], false).toBytes();
        const bareNodes = rebuildContainer(bare, (m) => {
            (m.counts as Record<string, unknown>).nodes = 4;
        });
        expect(
            expectError(() => fromBytes(bareNodes, { validate: "none" }), "E_INVALID_SNAPSHOT").details.invariant,
        ).toBe("I1");
        const bareEdges = rebuildContainer(bare, (m) => {
            (m.counts as Record<string, unknown>).edges = 3;
        });
        expect(
            expectError(() => fromBytes(bareEdges, { validate: "none" }), "E_INVALID_SNAPSHOT").details.invariant,
        ).toBe("I5");
        const bareIds = rebuildContainer(bare, (m) => {
            (m.ids as Record<string, unknown>).size = 2;
        });
        expect(
            expectError(() => fromBytes(bareIds, { validate: "none" }), "E_INVALID_SNAPSHOT").details.invariant,
        ).toBe("I11");
    });
});

describe("core data corruption", () => {
    const s = richSnapshot({ directed: true });
    const bytes = s.toBytes();

    it("flipping a colIdx byte fails full validation with the invariant number", () => {
        const outOfRange = patched(bytes, segmentStart(bytes, "colIdx"), 200);
        expect(expectError(() => fromBytes(outOfRange), "E_INVALID_SNAPSHOT").details.invariant).toBe("I2");
        expect(
            expectError(() => fromBytes(outOfRange, { validate: "structure" }), "E_INVALID_SNAPSHOT").details.invariant,
        ).toBe("I2");
        expect(fromBytes(outOfRange, { validate: "none" }).colIdx[0]).toBe(200);
    });

    it("unsorting a row fails full validation with I4 and passes structure", () => {
        const start = segmentStart(bytes, "colIdx");
        const row = Array.from(s.colIdx.subarray(s.rowPtr[0], s.rowPtr[1]));
        expect(row.length).toBeGreaterThan(1);
        const unsorted = bytes.slice();
        const view = new Uint32Array(unsorted.buffer, start, s.arcCount);
        view[s.rowPtr[0]] = row[row.length - 1];
        view[s.rowPtr[1] - 1] = row[0];
        if (row[0] !== row[row.length - 1]) {
            expect(expectError(() => fromBytes(unsorted), "E_INVALID_SNAPSHOT").details.invariant).toBe("I4");
        }
        expect(fromBytes(unsorted, { validate: "structure" }).nodeCount).toBe(4);
    });

    it("breaking a permutation fails full validation with I5 or I6", () => {
        const start = segmentStart(bytes, "arcToEdge");
        const broken = bytes.slice();
        const view = new Uint32Array(broken.buffer, start, s.arcCount);
        view[0] = view[1];
        const error = expectError(() => fromBytes(broken), "E_INVALID_SNAPSHOT");
        expect(["I4", "I5", "I6"]).toContain(error.details.invariant);
        const orientation = bytes.slice();
        new Uint32Array(orientation.buffer, segmentStart(bytes, "edgeToArc"), s.edgeCount)[0] = s.edgeToArc[1];
        const error2 = expectError(() => fromBytes(orientation), "E_INVALID_SNAPSHOT");
        expect(["I5", "I6"]).toContain(error2.details.invariant);
    });

    it("a NaN weight fails full validation with I8", () => {
        const start = segmentStart(bytes, "weights");
        const nan = bytes.slice();
        new Float32Array(nan.buffer, start, 1)[0] = Number.NaN;
        expect(expectError(() => fromBytes(nan), "E_INVALID_SNAPSHOT").details.invariant).toBe("I8");
    });

    it("a flag that lies fails full validation with I9", () => {
        const lie = rebuildContainer(bytes, (m) => {
            (m.flags as Record<string, unknown>).multigraph = !s.flags.multigraph;
        });
        expect(expectError(() => fromBytes(lie), "E_INVALID_SNAPSHOT").details.invariant).toBe("I9");
        expect(fromBytes(lie, { validate: "structure" }).flags.multigraph).toBe(!s.flags.multigraph);
    });
});

describe("id map corruption", () => {
    it("refuses an unknown id-map kind at every level", () => {
        const s = idsSnapshot(["a", "b"]);
        const bytes = s.toBytes();
        for (const options of LEVELS) {
            const unknown = rebuildContainer(bytes, (m) => {
                (m.ids as Record<string, unknown>).kind = "interned";
            });
            const error = expectError(() => fromBytes(unknown, options), "E_UNSUPPORTED");
            expect(error.details.kind).toBe("interned");
        }
    });

    it("refuses invalid UTF-8 in string ids under full validation and duplicate ids", () => {
        const s = idsSnapshot(["ab", "cd"]);
        const bytes = s.toBytes();
        const utf8 = at(splitContainer(bytes).manifest, "ids.utf8") as unknown as WireBufferRef;
        const bad = patched(bytes, regionStart(bytes) + utf8.byteOffset, 0xff);
        const error = expectError(() => fromBytes(bad), "E_INVALID_SNAPSHOT");
        expect(error.details.invariant).toBe("I11");
        expect(fromBytes(bad, { validate: "structure" }).ids.idOf(0)).toBe("\ufffdb");
        const dup = bytes.slice();
        dup.set([0x61, 0x62], regionStart(bytes) + utf8.byteOffset + 2);
        expect(expectError(() => fromBytes(dup), "E_INVALID_SNAPSHOT").details.invariant).toBe("I11");
        const dense = idsSnapshot([0, 5, 2]);
        const denseBytes = dense.toBytes();
        const values = at(splitContainer(denseBytes).manifest, "ids.values") as unknown as WireBufferRef;
        const dupDense = patched(denseBytes, regionStart(denseBytes) + values.byteOffset + 4, 0);
        expect(
            expectError(() => fromBytes(dupDense, { validate: "none" }), "E_INVALID_SNAPSHOT").details.invariant,
        ).toBe("I11");
    });

    it("refuses id maps whose slots are missing or of the wrong dtype", () => {
        const s = idsSnapshot([1, "1", 2.5]);
        const bytes = s.toBytes();
        const missing = rebuildContainer(bytes, (m) => {
            (m.ids as Record<string, unknown>).tags = null;
        });
        expect(expectError(() => fromBytes(missing), "E_BAD_SERIALIZATION").details.ref).toBe("ids.tags");
        const wrong = rebuildContainer(bytes, (m) => {
            at(m, "ids.numbers").dtype = "f32";
        });
        expect(expectError(() => fromBytes(wrong), "E_BAD_SERIALIZATION").details.ref).toBe("ids.numbers");
        const badTag = bytes.slice();
        const tags = at(splitContainer(bytes).manifest, "ids.tags") as unknown as WireBufferRef;
        badTag[regionStart(bytes) + tags.byteOffset] = 7;
        expect(expectError(() => fromBytes(badTag), "E_BAD_SERIALIZATION").details.ref).toBe("ids.tags");
        const badSize = rebuildContainer(bytes, (m) => {
            (m.ids as Record<string, unknown>).size = -1;
        });
        expectError(() => fromBytes(badSize), "E_BAD_SERIALIZATION");
        const badOffset = rebuildContainer(bytes, (m) => {
            (m.ids as Record<string, unknown>).offset = "one";
        });
        expectError(() => fromBytes(badOffset), "E_BAD_SERIALIZATION");
    });
});

describe("column corruption", () => {
    const s = richSnapshot();
    const bytes = s.toBytes();
    const index = (name: string): number => s.nodes.names().indexOf(name);

    it("refuses an unknown column dtype unless unknownColumns is skip, which records the column", () => {
        const unknown = rebuildContainer(bytes, (m) => {
            at(m, `nodeColumns.${index("score")}.meta`).dtype = "f16";
            at(m, `edgeColumns.1.meta`).dtype = "decimal";
            at(m, `nodeColumns.${index("tags")}.meta`).itemDtype = "f16";
            at(m, `extensions.0.columns.3.meta`).dtype = "i64";
        });
        for (const options of LEVELS) {
            const error = expectError(() => fromBytes(unknown, options), "E_UNSUPPORTED");
            expect(error.details).toMatchObject({ dtype: "f16", column: "score" });
        }
        const skipped = fromBytes(unknown, { unknownColumns: "skip" });
        expect(skipped.nodes.has("score")).toBe(false);
        expect(skipped.nodes.has("tags")).toBe(false);
        expect(skipped.nodes.has("pos")).toBe(true);
        expect(skipped.edges.has("kind")).toBe(false);
        expect(skipped.extensions.get("temporal:node:price")?.has("value")).toBe(false);
        expect(skipped.meta.extra[SKIPPED_COLUMNS_KEY]).toEqual([
            { domain: "node", table: null, name: "score", dtype: "f16" },
            { domain: "node", table: null, name: "tags", dtype: "f16" },
            { domain: "edge", table: null, name: "kind", dtype: "decimal" },
            { domain: "extension", table: "temporal:node:price", name: "value", dtype: "i64" },
        ]);
        expect(skipped.meta.name).toBe("rich graph");
        expect(Object.isFrozen(skipped.meta.extra)).toBe(true);
        const wire = s.toWire();
        const wireUnknown = structuredClone(wire);
        (wireUnknown.manifest.nodeColumns[index("byte")].meta as { dtype: string }).dtype = "u16";
        expect(expectError(() => fromWire(wireUnknown), "E_UNSUPPORTED").details.dtype).toBe("u16");
        expect(fromWire(wireUnknown, { unknownColumns: "skip" }).nodes.has("byte")).toBe(false);
    });

    it("refuses a nullCount that disagrees with the bitmap at the structure level", () => {
        const lie = rebuildContainer(bytes, (m) => {
            at(m, `nodeColumns.${index("score")}`).nullCount = 0;
        });
        const error = expectError(() => fromBytes(lie), "E_BAD_SERIALIZATION");
        expect(error.details.ref).toBe(`nodeColumns[${index("score")}].nullCount`);
        expect(fromBytes(lie, { validate: "none" }).nodes.require("score").nullCount).toBe(2);
    });

    it("refuses malformed metadata and length rules as E_BAD_SERIALIZATION", () => {
        const badComponents = rebuildContainer(bytes, (m) => {
            at(m, `nodeColumns.${index("pos")}.meta`).components = 99;
        });
        const error = expectError(() => fromBytes(badComponents), "E_BAD_SERIALIZATION");
        expect(error.details.cause).toBe("E_COLUMN_TYPE");
        const stringComponents = rebuildContainer(bytes, (m) => {
            at(m, `nodeColumns.${index("pos")}.meta`).components = "3";
        });
        expect(expectError(() => fromBytes(stringComponents), "E_BAD_SERIALIZATION").details.ref).toBe(
            `nodeColumns[${index("pos")}].meta.components`,
        );
        const badFill = rebuildContainer(bytes, (m) => {
            at(m, `nodeColumns.${index("score")}.meta`).fill = "zero";
        });
        expectError(() => fromBytes(badFill), "E_BAD_SERIALIZATION");
        const badTag = rebuildContainer(bytes, (m) => {
            at(m, `nodeColumns.${index("end")}.meta`).default = { $num: "huge" };
        });
        expectError(() => fromBytes(badTag), "E_BAD_SERIALIZATION");
        const shortValidity = rebuildContainer(bytes, (m) => {
            const ref = at(m, `nodeColumns.${index("score")}.validity`);
            ref.length = 0;
            ref.byteLength = 0;
        });
        expect(expectError(() => fromBytes(shortValidity), "E_BAD_SERIALIZATION").details.cause).toBe(
            "E_COLUMN_LENGTH",
        );
        const validityOnSolid = rebuildContainer(bytes, (m) => {
            at(m, `nodeColumns.${index("solid")}`).validity = at(m, `nodeColumns.${index("score")}.validity`);
        });
        expectError(() => fromBytes(validityOnSolid), "E_BAD_SERIALIZATION");
        const noName = rebuildContainer(bytes, (m) => {
            delete at(m, `nodeColumns.${index("score")}.meta`).name;
        });
        expectError(() => fromBytes(noName), "E_BAD_SERIALIZATION");
        const notAColumn = rebuildContainer(bytes, (m) => {
            (m.nodeColumns as unknown[])[0] = 5;
        });
        expectError(() => fromBytes(notAColumn), "E_BAD_SERIALIZATION");
        const notAList = rebuildContainer(bytes, (m) => {
            m.nodeColumns = {};
        });
        expectError(() => fromBytes(notAList), "E_BAD_SERIALIZATION");
    });

    it("refuses duplicate names, duplicate roles and duplicate extension tables", () => {
        const dupName = rebuildContainer(bytes, (m) => {
            at(m, `nodeColumns.${index("score")}.meta`).name = "pos";
        });
        expectError(() => fromBytes(dupName), "E_BAD_SERIALIZATION");
        const dupRole = rebuildContainer(bytes, (m) => {
            at(m, `nodeColumns.${index("score")}.meta`).role = "position";
        });
        expect(expectError(() => fromBytes(dupRole), "E_BAD_SERIALIZATION").details.cause).toBe("E_DUPLICATE_ROLE");
        const dupExtension = rebuildContainer(bytes, (m) => {
            (m.extensions as unknown[]).push((m.extensions as unknown[])[0]);
        });
        expectError(() => fromBytes(dupExtension), "E_BAD_SERIALIZATION");
        const badExtension = rebuildContainer(bytes, (m) => {
            (m.extensions as unknown[])[0] = { name: 5 };
        });
        expectError(() => fromBytes(badExtension), "E_BAD_SERIALIZATION");
    });

    it("refuses broken dictionaries, string stores, lists and json text", () => {
        const dictIndex = index("cat");
        const dictOffsets = at(
            splitContainer(bytes).manifest,
            `nodeColumns.${dictIndex}.dictionary.offsets`,
        ) as unknown as WireBufferRef;
        const decreasing = bytes.slice();
        new Uint32Array(decreasing.buffer, regionStart(bytes) + dictOffsets.byteOffset, 2)[1] = 1000;
        for (const options of LEVELS) {
            expectError(() => fromBytes(decreasing, options), "E_BAD_SERIALIZATION");
        }
        const labelIndex = index("label");
        const strings = at(
            splitContainer(bytes).manifest,
            `nodeColumns.${labelIndex}.strings.utf8`,
        ) as unknown as WireBufferRef;
        const badUtf8 = patched(bytes, regionStart(bytes) + strings.byteOffset, 0xc3);
        expect(expectError(() => fromBytes(badUtf8), "E_BAD_SERIALIZATION").details.ref).toBe(
            `nodeColumns[${labelIndex}].strings`,
        );
        expect(fromBytes(badUtf8, { validate: "structure" }).nodes.value("label", 0)).toContain("\ufffd");
        const stringOffsets = at(
            splitContainer(bytes).manifest,
            `nodeColumns.${labelIndex}.strings.offsets`,
        ) as unknown as WireBufferRef;
        const overrun = bytes.slice();
        new Uint32Array(overrun.buffer, regionStart(bytes) + stringOffsets.byteOffset, 5)[4] = 9999;
        expect(
            expectError(() => fromBytes(overrun, { validate: "structure" }), "E_BAD_SERIALIZATION").details.ref,
        ).toBe(`nodeColumns[${labelIndex}].strings`);
        expect(expectError(() => fromBytes(overrun, { validate: "none" }), "E_BAD_SERIALIZATION").details.cause).toBe(
            "E_COLUMN_LENGTH",
        );
        const emptyOffsets = rebuildContainer(bytes, (m) => {
            const ref = at(m, `nodeColumns.${labelIndex}.strings.offsets`);
            ref.length = 0;
            ref.byteLength = 0;
        });
        expectError(() => fromBytes(emptyOffsets, { validate: "none" }), "E_BAD_SERIALIZATION");
        const tagsIndex = index("tags");
        const listOffsets = rebuildContainer(bytes, (m) => {
            const ref = at(m, `nodeColumns.${tagsIndex}.offsets`);
            ref.length = 3;
            ref.byteLength = 12;
        });
        expect(expectError(() => fromBytes(listOffsets), "E_BAD_SERIALIZATION").details.ref).toBe(
            `nodeColumns[${tagsIndex}].offsets`,
        );
        const noChild = rebuildContainer(bytes, (m) => {
            at(m, `nodeColumns.${tagsIndex}`).child = null;
        });
        expectError(() => fromBytes(noChild), "E_BAD_SERIALIZATION");
        const listChild = rebuildContainer(bytes, (m) => {
            const column = at(m, `nodeColumns.${tagsIndex}`);
            column.child = {
                meta: { name: "items", dtype: "list", itemDtype: "u32", nullable: false },
                data: null,
                validity: null,
                nullCount: 0,
                dictionary: null,
                strings: null,
                offsets: at(m, "core.rowPtr"),
                child: {
                    meta: { name: "leaf", dtype: "u32", nullable: false },
                    data: at(m, "core.colIdx"),
                    validity: null,
                    nullCount: 0,
                    dictionary: null,
                    strings: null,
                    offsets: null,
                    child: null,
                    jsonText: null,
                },
                jsonText: null,
            };
        });
        expect(expectError(() => fromBytes(listChild, { validate: "none" }), "E_BAD_SERIALIZATION").details.ref).toBe(
            `nodeColumns[${tagsIndex}].child`,
        );
        const childMismatch = rebuildContainer(bytes, (m) => {
            at(m, `nodeColumns.${tagsIndex}.child.meta`).dtype = "u32";
            at(m, `nodeColumns.${tagsIndex}.child`).data = at(m, "core.colIdx");
            at(m, `nodeColumns.${tagsIndex}.child`).strings = null;
        });
        expectError(() => fromBytes(childMismatch), "E_BAD_SERIALIZATION");
        const blobIndex = index("blob");
        const jsonText = at(
            splitContainer(bytes).manifest,
            `nodeColumns.${blobIndex}.jsonText.utf8`,
        ) as unknown as WireBufferRef;
        const badJson = patched(bytes, regionStart(bytes) + jsonText.byteOffset, 0x5d);
        expect(
            expectError(() => fromBytes(badJson, { validate: "none" }), "E_BAD_SERIALIZATION").details.ref,
        ).toContain(`nodeColumns[${blobIndex}].jsonText[`);
    });

    it("copies a u8 column whose bytes end exactly at the buffer end so the padded view is constructible", () => {
        const b = new GraphBuilder({ directed: true });
        b.addNodes([0, 1, 2]);
        b.declareNodeColumn({ name: "byte", dtype: "u8", nullable: false });
        b.setNodeValue("byte", 0, 1);
        b.setNodeValue("byte", 1, 2);
        b.setNodeValue("byte", 2, 3);
        const snap = b.freeze();
        const wire = snap.toWire();
        const column = wire.manifest.nodeColumns[0];
        const data = column.data as WireBufferRef;
        const tight = new Uint8Array(3);
        tight.set(new Uint8Array(wire.buffers[data.buffer], data.byteOffset, 3));
        const buffers = [...wire.buffers, tight.buffer];
        const manifest = {
            ...wire.manifest,
            nodeColumns: [{ ...column, data: { ...data, buffer: buffers.length - 1, byteOffset: 0 } }],
        };
        const back = fromWire({ manifest, buffers }, { validate: "full" });
        const col = back.nodes.requireTyped("byte", "u8");
        expect(Array.from(col.data)).toEqual([1, 2, 3]);
        expect(col.data.buffer).not.toBe(tight.buffer);
        expect(col.paddedU32View().length).toBe(1);
    });
});

describe("views and metadata", () => {
    const s = richSnapshot({ directed: true });
    s.prepare(["outDegree", "reverse"]);
    const bytes = s.toBytes({ includeViews: ["outDegree", "reverse"] });

    it("ignores unknown view names and checks the references of carried views at every level", () => {
        const unknown = rebuildContainer(bytes, (m) => {
            (m.views as Record<string, unknown>).futureView = (m.views as Record<string, unknown>).outDegree;
        });
        expect(fromBytes(unknown).cachedViews()).toEqual([]);
        expect(fromBytes(unknown, { validate: "structure" }).cachedViews()).toContain("outDegree");
        const bad = rebuildContainer(bytes, (m) => {
            at(m, "views.outDegree.data").byteOffset = 3;
        });
        expect(expectError(() => fromBytes(bad), "E_BAD_SERIALIZATION").details.ref).toBe("views.outDegree.data");
        expect(expectError(() => fromBytes(bad, { validate: "none" }), "E_BAD_SERIALIZATION").details.ref).toBe(
            "views.outDegree.data",
        );
        const notObject = rebuildContainer(bytes, (m) => {
            m.views = 5;
        });
        expectError(() => fromBytes(notObject), "E_BAD_SERIALIZATION");
    });

    it("refuses metadata fields of the wrong type and accepts missing ones", () => {
        const badName = rebuildContainer(bytes, (m) => {
            (m.meta as Record<string, unknown>).name = 5;
        });
        expect(expectError(() => fromBytes(badName), "E_BAD_SERIALIZATION").details.ref).toBe("meta.name");
        const badEnum = rebuildContainer(bytes, (m) => {
            (m.meta as Record<string, unknown>).mode = "loud";
        });
        expect(expectError(() => fromBytes(badEnum), "E_BAD_SERIALIZATION").details.ref).toBe("meta.mode");
        const badKeywords = rebuildContainer(bytes, (m) => {
            (m.meta as Record<string, unknown>).keywords = [1];
        });
        expectError(() => fromBytes(badKeywords), "E_BAD_SERIALIZATION");
        const badOrigin = rebuildContainer(bytes, (m) => {
            (m.meta as Record<string, unknown>).weightOrigin = { format: 3 };
        });
        expectError(() => fromBytes(badOrigin), "E_BAD_SERIALIZATION");
        const badMultigraph = rebuildContainer(bytes, (m) => {
            (m.meta as Record<string, unknown>).declaredMultigraph = "yes";
        });
        expectError(() => fromBytes(badMultigraph), "E_BAD_SERIALIZATION");
        const badExtra = rebuildContainer(bytes, (m) => {
            (m.meta as Record<string, unknown>).extra = [1];
        });
        expectError(() => fromBytes(badExtra), "E_BAD_SERIALIZATION");
        const badLabel = rebuildContainer(bytes, (m) => {
            m.label = 5;
        });
        expectError(() => fromBytes(badLabel), "E_BAD_SERIALIZATION");
        const missing = rebuildContainer(bytes, (m) => {
            m.meta = {};
            delete m.views;
            delete m.extensions;
            delete m.label;
        });
        const back = fromBytes(missing);
        expect(back.meta.name).toBeNull();
        expect(back.meta.keywords).toEqual([]);
        expect(back.meta.extra).toEqual({});
        expect(back.label).toBeNull();
        expect(back.extensions.size).toBe(0);
    });
});

describe("JSON tagging helpers and host checks", () => {
    it("encode and decode are inverse on every tagged value and reject unknown tags", () => {
        const value = { a: [Infinity, -Infinity, Number.NaN, -0, 0, 1.5, "s", null, true], b: { c: -0 } };
        const encoded = encodeJsonValue(value);
        expect(JSON.parse(JSON.stringify(encoded))).toEqual(encoded);
        const decoded = decodeJsonValue(JSON.parse(JSON.stringify(encoded)), "x") as typeof value;
        expect(decoded.a[0]).toBe(Infinity);
        expect(decoded.a[1]).toBe(-Infinity);
        expect(Number.isNaN(decoded.a[2])).toBe(true);
        expect(Object.is(decoded.a[3], -0)).toBe(true);
        expect(Object.is(decoded.a[4], 0)).toBe(true);
        expect(decoded.a.slice(5)).toEqual([1.5, "s", null, true]);
        expect(Object.is(decoded.b.c, -0)).toBe(true);
        expect(encodeJsonValue(undefined)).toBeUndefined();
        expect(decodeJsonValue(undefined, "x")).toBeUndefined();
        expect(decodeJsonValue({ $num: "1", extra: 2 }, "x")).toEqual({ $num: "1", extra: 2 });
        expectError(() => decodeJsonValue({ $num: "huge" }, "x"), "E_BAD_SERIALIZATION");
        expectError(
            () => decodeJsonValue({ nested: JSON.parse('{"__proto__": 1}') as unknown }, "x"),
            "E_BAD_SERIALIZATION",
        );
        const ownProto = JSON.parse('{"__proto__": {"polluted": true}}') as Record<string, unknown>;
        const encodedProto = encodeJsonValue(ownProto) as Record<string, unknown>;
        expect(Object.getPrototypeOf(encodedProto)).toBe(Object.prototype);
        expect(Object.keys(encodedProto)).toEqual(["__proto__"]);
        const date = new Date(0);
        expect(encodeJsonValue(date)).toBe(date);
        expect(decodeJsonValue(date, "x")).toBe(date);
    });

    it("refuses to write on a big-endian host", () => {
        const error = expectError(() => assertLittleEndianHost(false), "E_UNSUPPORTED");
        expect(error.details.reason).toBe("big-endian host");
        expect(() => assertLittleEndianHost(true)).not.toThrow();
    });

    it("fromWire refuses a manifest of the wrong shape", () => {
        const s = idsSnapshot([0, 1]);
        const wire = s.toWire();
        const bad = (edit: (m: Record<string, unknown>) => void): WireSnapshot => {
            const manifest = structuredClone(wire.manifest) as unknown as Record<string, unknown>;
            edit(manifest);
            return { manifest: manifest as never, buffers: wire.buffers };
        };
        expectError(() => fromWire(bad((m) => (m.format = "x"))), "E_BAD_SERIALIZATION");
        expectError(() => fromWire(bad((m) => (m.wire = "1.0"))), "E_BAD_SERIALIZATION");
        expectError(() => fromWire(bad((m) => (m.wire = [1]))), "E_BAD_SERIALIZATION");
        expectError(() => fromWire(bad((m) => (m.counts = null))), "E_BAD_SERIALIZATION");
        expectError(() => fromWire(bad((m) => (m.core = []))), "E_BAD_SERIALIZATION");
        expectError(() => fromWire(bad((m) => (m.ids = 1))), "E_BAD_SERIALIZATION");
        expectError(() => fromWire(bad((m) => (m.meta = "meta"))), "E_BAD_SERIALIZATION");
        expectError(() => fromWire(bad((m) => (m.extensions = [1]))), "E_BAD_SERIALIZATION");
        expectError(() => fromWire(bad((m) => (m.nodeColumns = [{ meta: 1 }]))), "E_BAD_SERIALIZATION");
        expectError(() => fromWire({ manifest: null as never, buffers: [] }), "E_BAD_SERIALIZATION");
    });
});
