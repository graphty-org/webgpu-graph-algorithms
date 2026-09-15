/**
 * Golden container test (design section 16.3): test/fixtures/rich-v1.gsnp was written by
 * tmp/make-golden.ts from the fixture graph of fixture-graph.ts. Re-reading it guards against
 * accidental wire changes; the comparison masks `producer`.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { WIRE_MAJOR, WIRE_MINOR } from "../../src/constants.js";
import { fromByteChunks, fromBytes } from "../../src/wire/bytes.js";
import { assertInvariants } from "../helpers/invariants.js";
import { expectSnapshotsEqual, maskedContainer, richSnapshot, splitContainer } from "./helpers.js";

const FIXTURE = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "rich-v1.gsnp");

describe("golden container", () => {
    const golden = new Uint8Array(readFileSync(FIXTURE));

    it("has the documented header", () => {
        expect(Array.from(golden.subarray(0, 4))).toEqual([0x47, 0x53, 0x4e, 0x50]);
        const parts = splitContainer(golden);
        expect(parts.major).toBe(WIRE_MAJOR);
        expect(parts.minor).toBe(WIRE_MINOR);
        expect(parts.manifest.format).toBe("graphty-snapshot");
        expect(parts.manifest.formatVersion).toBe(1);
        expect(golden.byteLength % 256).toBe(0);
    });

    it("decodes to the fixture graph at full validation", () => {
        const back = fromBytes(golden);
        assertInvariants(back);
        expectSnapshotsEqual(richSnapshot(), back);
        expect(back.nodes.value("label", 2)).toBe("node 2 \u00e9");
        expect(back.edgeIndexOf("e3")).toBe(3);
        expect(back.nodes.requireTyped("cat", "dict").dictionary).toEqual(["x", "y", "z"]);
        expect(back.nodes.value("blob", 0)).toEqual({ deep: [1, "x", null, { k: 0 }], inf: Infinity, negZero: -0 });
        expectSnapshotsEqual(richSnapshot(), fromByteChunks([golden]));
    });

    it("re-serialises to the same bytes modulo producer", () => {
        const fresh = richSnapshot().toBytes();
        expect(maskedContainer(fresh)).toEqual(maskedContainer(golden));
        const decoded = fromBytes(golden).toBytes();
        expect(maskedContainer(decoded)).toEqual(maskedContainer(golden));
    });
});
