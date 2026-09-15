import { describe, expect, it } from "vitest";

import * as graphFormat from "../src/index.js";

/** The value exports of design section 12.2 (classes, functions and constants); types are checked by test/types. */
const VALUE_EXPORTS = [
    "AttributeTable",
    "FORMAT_VERSION",
    "GraphBuilder",
    "GraphFormatError",
    "GraphSnapshot",
    "INVALID_INDEX",
    "MAX_COUNT",
    "NodeIdMap",
    "SNAPSHOT_BRAND",
    "equalsTopology",
    "expandEdges",
    "foldArcs",
    "fromByteChunks",
    "fromBytes",
    "fromCsr",
    "fromEdgeArrays",
    "fromRecords",
    "fromWire",
    "gatherArray",
    "gatherColumn",
    "gpuEligibility",
    "isGraphSnapshot",
    "makeMask",
    "maskCount",
    "maskSet",
    "maskTest",
    "maskToIndices",
    "paddedU32View",
    "remapArray",
    "remapColumn",
    "renumberPartition",
    "scatterArray",
    "withComponents",
];

describe("public barrel (design section 12.2)", () => {
    it("exports exactly the value surface of section 12.2 and no default export", () => {
        expect(Object.keys(graphFormat).sort()).toEqual([...VALUE_EXPORTS].sort());
        expect((graphFormat as Record<string, unknown>).default).toBeUndefined();
    });

    it("exposes the classes as constructors and the constants with their design values", () => {
        expect(typeof graphFormat.GraphBuilder).toBe("function");
        expect(typeof graphFormat.GraphSnapshot).toBe("function");
        expect(typeof graphFormat.AttributeTable).toBe("function");
        expect(typeof graphFormat.NodeIdMap).toBe("function");
        expect(graphFormat.INVALID_INDEX).toBe(0xffffffff);
        expect(graphFormat.MAX_COUNT).toBe(0xfffffffe);
        expect(graphFormat.FORMAT_VERSION).toBe(1);
        expect(graphFormat.SNAPSHOT_BRAND).toBe(Symbol.for("@graphty/graph-format/snapshot"));
    });

    it("round-trips a graph through the barrel's builder, snapshot and wire entry points", () => {
        const builder = new graphFormat.GraphBuilder({ directed: false });
        builder.addEdge("a", "b", 2);
        builder.addEdge("b", "c", 3);
        const snapshot = builder.freeze();
        expect(snapshot).toBeInstanceOf(graphFormat.GraphSnapshot);
        expect(snapshot.ids).toBeInstanceOf(graphFormat.NodeIdMap);
        expect(snapshot.nodes).toBeInstanceOf(graphFormat.AttributeTable);
        expect(graphFormat.isGraphSnapshot(snapshot)).toBe(true);
        const back = graphFormat.fromBytes(snapshot.toBytes());
        expect(graphFormat.equalsTopology(snapshot, back)).toBe(true);
        expect(graphFormat.equalsTopology(snapshot, graphFormat.fromWire(snapshot.toWire()))).toBe(true);
        const derived = snapshot.transpose().snapshot;
        expect(derived).toBeInstanceOf(graphFormat.GraphSnapshot);
    });
});
