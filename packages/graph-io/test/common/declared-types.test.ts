import { GraphFormatError } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import {
    type DeclaringFormat,
    mapDeclaredType,
    parseBooleanText,
    parseDecimalText,
    parsePointText,
    parseScalarText,
    stringSpec,
} from "../../src/common/declared-types.js";

function codeOf(fn: () => unknown): string | null {
    try {
        fn();
    } catch (err) {
        return err instanceof GraphFormatError ? err.code : "other";
    }
    return null;
}

describe("declared type table (design 5.1)", () => {
    it.each<[DeclaringFormat, string, string, string | null, string]>([
        // GEXF 1.2 / 1.3
        ["gexf", "integer", "i32", null, "integer"],
        ["gexf", "long", "f64", null, "long"],
        ["gexf", "double", "f64", null, "double"],
        ["gexf", "float", "f32", null, "float"],
        ["gexf", "boolean", "bool", null, "boolean"],
        ["gexf", "string", "string", null, "string"],
        ["gexf", "anyURI", "string", null, "string"],
        ["gexf", "char", "string", null, "string"],
        ["gexf", "byte", "i32", null, "integer"],
        ["gexf", "short", "i32", null, "integer"],
        ["gexf", "bigdecimal", "string", null, "string"],
        ["gexf", "biginteger", "string", null, "string"],
        ["gexf", "date", "f64", null, "temporal"],
        ["gexf", "dateTime", "f64", null, "temporal"],
        ["gexf", "liststring", "list", "string", "string"],
        ["gexf", "listboolean", "list", "bool", "boolean"],
        ["gexf", "listinteger", "list", "i32", "integer"],
        ["gexf", "listlong", "list", "f64", "long"],
        ["gexf", "listfloat", "list", "f32", "float"],
        ["gexf", "listdouble", "list", "f64", "double"],
        ["gexf", "listbyte", "list", "i32", "integer"],
        ["gexf", "listshort", "list", "i32", "integer"],
        ["gexf", "listbigdecimal", "list", "string", "string"],
        ["gexf", "listbiginteger", "list", "string", "string"],
        ["gexf", "listchar", "list", "string", "string"],
        // GraphML
        ["graphml", "boolean", "bool", null, "boolean"],
        ["graphml", "int", "i32", null, "integer"],
        ["graphml", "long", "f64", null, "long"],
        ["graphml", "float", "f32", null, "float"],
        ["graphml", "double", "f64", null, "double"],
        ["graphml", "string", "string", null, "string"],
        // GML
        ["gml", "int", "i32", null, "integer"],
        ["gml", "real", "f64", null, "double"],
        ["gml", "string", "string", null, "string"],
        ["gml", "record", "json", null, "json"],
        ["gml", "list", "list", "json", "json"],
        // Neo4j
        ["neo4j", "int", "i32", null, "integer"],
        ["neo4j", "integer", "f64", null, "long"],
        ["neo4j", "long", "f64", null, "long"],
        ["neo4j", "float", "f32", null, "float"],
        ["neo4j", "double", "f64", null, "double"],
        ["neo4j", "boolean", "bool", null, "boolean"],
        ["neo4j", "byte", "i32", null, "integer"],
        ["neo4j", "short", "i32", null, "integer"],
        ["neo4j", "char", "string", null, "string"],
        ["neo4j", "string", "string", null, "string"],
        ["neo4j", "point", "json", null, "point"],
        ["neo4j", "date", "f64", null, "temporal"],
        ["neo4j", "localtime", "f64", null, "temporal"],
        ["neo4j", "time", "f64", null, "temporal"],
        ["neo4j", "localdatetime", "f64", null, "temporal"],
        ["neo4j", "datetime", "f64", null, "temporal"],
        ["neo4j", "duration", "string", null, "duration"],
        ["neo4j", "string[]", "list", "string", "string"],
        ["neo4j", "long[]", "list", "f64", "long"],
        ["neo4j", "boolean[]", "list", "bool", "boolean"],
        ["neo4j", "INT", "i32", null, "integer"],
        ["neo4j", "DateTime", "f64", null, "temporal"],
    ])("%s %j -> %s (item %s, kind %s)", (format, type, dtype, itemDtype, kind) => {
        const spec = mapDeclaredType(format, type, "f64");
        expect(spec).not.toBeNull();
        expect(spec?.dtype).toBe(dtype);
        expect(spec?.itemDtype).toBe(itemDtype);
        expect(spec?.kind).toBe(kind);
        expect(spec?.list).toBe(dtype === "list");
        expect(spec?.declared).toBe(type);
        expect(spec?.precision).toBe(kind === "long");
        expect(spec?.temporal !== null).toBe(kind === "temporal");
    });

    it("maps temporal kinds", () => {
        expect(mapDeclaredType("gexf", "date", "f64")?.temporal).toBe("date");
        expect(mapDeclaredType("gexf", "dateTime", "f64")?.temporal).toBe("dateTime");
        expect(mapDeclaredType("neo4j", "localdatetime", "f64")?.temporal).toBe("localDateTime");
        expect(mapDeclaredType("neo4j", "time", "f64")?.temporal).toBe("time");
        expect(mapDeclaredType("neo4j", "localtime", "f64")?.temporal).toBe("localTime");
        expect(mapDeclaredType("neo4j", "date", "f64")?.temporal).toBe("date");
    });

    it('stores long as string under long: "string" (whole column, lists included)', () => {
        const scalar = mapDeclaredType("gexf", "long", "string");
        expect(scalar).toMatchObject({ dtype: "string", kind: "string", precision: false, declared: "long" });
        const list = mapDeclaredType("gexf", "listlong", "string");
        expect(list).toMatchObject({ dtype: "list", itemDtype: "string", kind: "string", precision: false });
        expect(mapDeclaredType("graphml", "long", "string")?.dtype).toBe("string");
        expect(mapDeclaredType("neo4j", "long[]", "string")?.itemDtype).toBe("string");
    });

    it("returns null for a type the format does not define", () => {
        expect(mapDeclaredType("gexf", "yfiles.type", "f64")).toBeNull();
        expect(mapDeclaredType("graphml", "integer", "f64")).toBeNull();
        expect(mapDeclaredType("graphml", "liststring", "f64")).toBeNull();
        expect(mapDeclaredType("gml", "double", "f64")).toBeNull();
        expect(mapDeclaredType("neo4j", "map", "f64")).toBeNull();
        expect(mapDeclaredType("gexf", "list", "f64")).toBeNull();
        expect(mapDeclaredType("gexf", "", "f64")).toBeNull();
    });

    it("stringSpec is the untyped fallback", () => {
        expect(stringSpec(null)).toMatchObject({ dtype: "string", kind: "string", declared: "", list: false });
        expect(stringSpec("weird").declared).toBe("weird");
    });
});

describe("parseScalarText", () => {
    it("parses booleans as true/false in any case and 0/1", () => {
        expect(parseBooleanText("true")).toBe(true);
        expect(parseBooleanText("True")).toBe(true);
        expect(parseBooleanText("FALSE")).toBe(false);
        expect(parseBooleanText("1")).toBe(true);
        expect(parseBooleanText("0")).toBe(false);
        expect(parseBooleanText(" true ")).toBe(true);
        expect(parseBooleanText("yes")).toBeNull();
        expect(parseBooleanText("")).toBeNull();
        expect(parseScalarText("true", "boolean")).toBe(true);
        expect(codeOf(() => parseScalarText("yes", "boolean"))).toBe("E_COLUMN_TYPE");
    });

    it("parses integers within i32 and rejects the rest", () => {
        expect(parseScalarText("42", "integer")).toBe(42);
        expect(parseScalarText("-42", "integer")).toBe(-42);
        expect(parseScalarText("+7", "integer")).toBe(7);
        expect(parseScalarText(" 12 \n", "integer")).toBe(12);
        expect(parseScalarText("007", "integer")).toBe(7);
        expect(codeOf(() => parseScalarText("2147483648", "integer"))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parseScalarText("1.5", "integer"))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parseScalarText("", "integer"))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parseScalarText("abc", "integer"))).toBe("E_COLUMN_TYPE");
    });

    it("parses longs as numbers beyond i32 (precision is the importer's check)", () => {
        expect(parseScalarText("2147483648", "long")).toBe(2147483648);
        expect(parseScalarText("9007199254740993", "long")).toBe(9007199254740992);
        expect(codeOf(() => parseScalarText("1.0", "long"))).toBe("E_COLUMN_TYPE");
    });

    it("parses floats and doubles including the XSD and JSON non-finite spellings", () => {
        expect(parseScalarText("1.5", "double")).toBe(1.5);
        expect(parseScalarText("1e-7", "float")).toBe(1e-7);
        expect(parseScalarText(".5", "double")).toBe(0.5);
        expect(parseScalarText("5.", "double")).toBe(5);
        expect(parseScalarText("-3", "double")).toBe(-3);
        expect(parseScalarText("  2.25 ", "double")).toBe(2.25);
        expect(parseDecimalText("INF")).toBe(Infinity);
        expect(parseDecimalText("-INF")).toBe(-Infinity);
        expect(parseDecimalText("Infinity")).toBe(Infinity);
        expect(parseDecimalText("-Infinity")).toBe(-Infinity);
        expect(Number.isNaN(parseDecimalText("NaN"))).toBe(true);
        expect(codeOf(() => parseDecimalText("1,5"))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parseDecimalText(""))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parseDecimalText("0x10"))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parseDecimalText("nan"))).toBe("E_COLUMN_TYPE");
    });

    it("keeps strings and durations exactly", () => {
        expect(parseScalarText("  keep me ", "string")).toBe("  keep me ");
        expect(parseScalarText("P1Y2M", "duration")).toBe("P1Y2M");
    });

    it("parses temporal values through the temporal kind", () => {
        expect(parseScalarText("2009-03-01", "temporal", "date")).toBe(Date.UTC(2009, 2, 1));
        expect(parseScalarText(" 2009-03-01T12:00:00Z ", "temporal", "dateTime")).toBe(Date.UTC(2009, 2, 1, 12));
        expect(codeOf(() => parseScalarText("2009-03-01", "temporal", null))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parseScalarText("yesterday", "temporal", "date"))).toBe("E_COLUMN_TYPE");
    });

    it("parses Neo4j points and JSON", () => {
        expect(parsePointText("{x:1.0, y:2.5, crs:'cartesian'}")).toEqual({ x: 1, y: 2.5, crs: "cartesian" });
        expect(parsePointText('{ "latitude": 55.5, "longitude": 12.3, "crs": "wgs-84" }')).toEqual({
            latitude: 55.5,
            longitude: 12.3,
            crs: "wgs-84",
        });
        expect(parsePointText("{x:1, y:2, z:3, srid:9157}")).toEqual({ x: 1, y: 2, z: 3, srid: 9157 });
        expect(parsePointText("{}")).toEqual({});
        expect(codeOf(() => parsePointText("x:1"))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parsePointText("{x}"))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parsePointText("{x:abc}"))).toBe("E_COLUMN_TYPE");
        expect(codeOf(() => parsePointText("{:1}"))).toBe("E_COLUMN_TYPE");
        expect(parseScalarText("{x:1}", "point")).toEqual({ x: 1 });
        expect(parseScalarText('{"a":[1,2]}', "json")).toEqual({ a: [1, 2] });
        expect(codeOf(() => parseScalarText("{a:1}", "json"))).toBe("E_COLUMN_TYPE");
    });

    it("truncates long texts in error messages", () => {
        try {
            parseScalarText("x".repeat(100), "integer");
        } catch (err) {
            expect((err as Error).message.length).toBeLessThan(80);
            expect((err as GraphFormatError).details.value).toBe("x".repeat(100));
        }
    });
});
