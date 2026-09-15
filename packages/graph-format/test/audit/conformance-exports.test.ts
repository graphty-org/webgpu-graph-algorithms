/**
 * Audit (API conformance lens): the export list of src/index.ts must equal the 126 declarations of
 * design section 12.2, name for name and kind for kind. The design says the listing "is the shape
 * of dist/graph-format.d.ts: everything below is exported from src/index.ts and nothing else is".
 *
 * Two independent checks:
 * - runtime: `Object.keys(barrel)` equals the value exports (classes, functions, constants);
 * - compile-time: the TypeScript checker enumerates every export of src/index.ts (type-only
 *   exports included) and classifies it as class / interface / type alias / function / const, and
 *   the result must equal the section 12.2 listing transcribed below.
 *
 * The listing is transcribed verbatim from the design; keep it in sync with section 12.2, never
 * with src/index.ts (the point is to catch the barrel drifting from the design).
 */

import { resolve } from "node:path";

import ts from "typescript";
import { describe, expect, it } from "vitest";

import * as barrel from "../../src/index.js";

type Kind = "class" | "interface" | "type" | "function" | "const";

/** Design section 12.2, grouped by declaration kind. */
const DESIGN_12_2: Readonly<Record<Kind, readonly string[]>> = {
    class: ["AttributeTable", "GraphBuilder", "GraphFormatError", "GraphSnapshot", "NodeIdMap"],
    const: ["FORMAT_VERSION", "INVALID_INDEX", "MAX_COUNT", "SNAPSHOT_BRAND"],
    function: [
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
    ],
    interface: [
        "AdjacencyView",
        "ArenaLayout",
        "ArenaSegment",
        "BoolColumn",
        "ByteLengthOptions",
        "ColumnBase",
        "ColumnDecl",
        "ColumnInput",
        "ColumnMeta",
        "ColumnOrigin",
        "ContractOptions",
        "CooView",
        "CsrInput",
        "DegreeOrderOptions",
        "DegreeOrderView",
        "DerivedGraph",
        "DictColumn",
        "EdgeArraysInput",
        "EdgeListView",
        "F32Column",
        "F64Column",
        "FreezeOptions",
        "FreezeReport",
        "FromCsrOptions",
        "FromWireOptions",
        "GraphBuilderOptions",
        "GraphMeta",
        "GraphSink",
        "I32Column",
        "JsonColumn",
        "ListColumn",
        "RecordsInput",
        "ResolvedBuilderOptions",
        "ReverseView",
        "SetDirectedOptions",
        "SetOptions",
        "SimplifyOptions",
        "SnapshotFlags",
        "StringColumn",
        "ToBytesOptions",
        "ToUndirectedOptions",
        "ToWireOptions",
        "U32Column",
        "U8Column",
        "ValidateOptions",
        "WireArena",
        "WireBufferRef",
        "WireColumn",
        "WireIdMap",
        "WireManifest",
        "WireSnapshot",
        "WireUtf8",
    ],
    type: [
        "ArcIndex",
        "BuilderOptionsPatch",
        "Column",
        "ColumnDeclPatch",
        "ColumnDomain",
        "ColumnHandle",
        "ColumnOf",
        "ColumnOriginInput",
        "ColumnReducer",
        "ColumnRole",
        "CoreArrayName",
        "Dtype",
        "DtypeValue",
        "DuplicatePolicy",
        "EdgeId",
        "EdgeIndex",
        "EdgeMask",
        "ExtensionHandle",
        "F32",
        "F64",
        "FlagClaims",
        "GpuEligibility",
        "GraphFormatErrorCode",
        "GraphMetaPatch",
        "I32",
        "IdCoercion",
        "KnownColumnRole",
        "Loose",
        "NodeId",
        "NodeIdMapKind",
        "NodeIndex",
        "NodeMask",
        "NumericVector",
        "ScalarDtype",
        "TypedArrayData",
        "U32",
        "U8",
        "ValidationLevel",
        "ViewName",
        "WeightReducer",
        "WireDtype",
    ],
};

const KINDS: readonly Kind[] = ["class", "const", "function", "interface", "type"];

function kindOf(symbol: ts.Symbol): Kind | "other" {
    if (symbol.flags & ts.SymbolFlags.Class) {
        return "class";
    }
    if (symbol.flags & ts.SymbolFlags.Interface) {
        return "interface";
    }
    if (symbol.flags & ts.SymbolFlags.TypeAlias) {
        return "type";
    }
    if (symbol.flags & ts.SymbolFlags.Function) {
        return "function";
    }
    if (symbol.flags & ts.SymbolFlags.Variable) {
        return "const";
    }
    return "other";
}

/**
 * Enumerate the exports of src/index.ts through the TypeScript checker, resolving re-export
 * aliases to the declarations behind them.
 * @returns every export name mapped to its declaration kind
 */
function barrelExports(): Map<string, Kind | "other"> {
    const indexPath = resolve("src/index.ts");
    const program = ts.createProgram([indexPath, resolve("src/lib-resizable-array-buffer.d.ts")], {
        target: ts.ScriptTarget.ES2020,
        module: ts.ModuleKind.ES2020,
        moduleResolution: ts.ModuleResolutionKind.Bundler,
        lib: ["lib.es2020.d.ts"],
        types: [],
        strict: true,
        skipLibCheck: true,
        noEmit: true,
    });
    const checker = program.getTypeChecker();
    const sourceFile = program.getSourceFile(indexPath);
    expect(sourceFile).toBeDefined();
    if (sourceFile === undefined) {
        throw new Error("src/index.ts not in the program");
    }
    const moduleSymbol = checker.getSymbolAtLocation(sourceFile);
    expect(moduleSymbol).toBeDefined();
    if (moduleSymbol === undefined) {
        throw new Error("src/index.ts has no module symbol");
    }
    const out = new Map<string, Kind | "other">();
    for (const exported of checker.getExportsOfModule(moduleSymbol)) {
        const resolved = exported.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(exported) : exported;
        out.set(exported.name, kindOf(resolved));
    }
    return out;
}

describe("design section 12.2: the export list of src/index.ts", () => {
    const expectedAll = KINDS.flatMap((kind) => DESIGN_12_2[kind]).sort();

    it("transcribes 126 distinct names (4 constants, 5 classes, 24 functions, 52 interfaces, 41 type aliases)", () => {
        expect(expectedAll).toHaveLength(126);
        expect(new Set(expectedAll).size).toBe(126);
        expect(DESIGN_12_2.const).toHaveLength(4);
        expect(DESIGN_12_2.class).toHaveLength(5);
        expect(DESIGN_12_2.function).toHaveLength(24);
        expect(DESIGN_12_2.interface).toHaveLength(52);
        expect(DESIGN_12_2.type).toHaveLength(41);
    });

    it("exports at runtime exactly the value declarations (classes, functions, constants) and no default", () => {
        const runtimeValues = [...DESIGN_12_2.class, ...DESIGN_12_2.function, ...DESIGN_12_2.const].sort();
        expect(Object.keys(barrel).sort()).toEqual(runtimeValues);
        expect((barrel as Record<string, unknown>).default).toBeUndefined();
        for (const name of DESIGN_12_2.class) {
            expect(typeof (barrel as Record<string, unknown>)[name], name).toBe("function");
        }
        for (const name of DESIGN_12_2.function) {
            expect(typeof (barrel as Record<string, unknown>)[name], name).toBe("function");
        }
    });

    it("exports through the type checker exactly the 126 names of the listing, with the same kinds", () => {
        const actual = barrelExports();
        expect([...actual.keys()].sort()).toEqual(expectedAll);
        for (const kind of KINDS) {
            const actualOfKind = [...actual.entries()].filter(([, k]) => k === kind).map(([name]) => name);
            expect(actualOfKind.sort(), `exports of kind ${kind}`).toEqual([...DESIGN_12_2[kind]].sort());
        }
        expect([...actual.values()].filter((k) => k === "other")).toEqual([]);
    });

    it("gives the four constants the values the design assigns", () => {
        expect(barrel.INVALID_INDEX).toBe(0xffffffff);
        expect(barrel.MAX_COUNT).toBe(0xfffffffe);
        expect(barrel.FORMAT_VERSION).toBe(1);
        expect(barrel.SNAPSHOT_BRAND).toBe(Symbol.for("@graphty/graph-format/snapshot"));
    });
});
