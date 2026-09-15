import { GraphFormatError } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { formatHeaderField, isHeaderRecord, parseHeaderField } from "../../../src/formats/neo4j/header.js";

describe("parseHeaderField", () => {
    it("parses the structural fields", () => {
        expect(parseHeaderField(":ID")).toMatchObject({ name: "", kind: "ID", type: null, space: null });
        expect(parseHeaderField("personId:ID(Person)")).toMatchObject({
            name: "personId",
            kind: "ID",
            space: "Person",
        });
        expect(parseHeaderField(":LABEL")).toMatchObject({ name: "", kind: "LABEL" });
        expect(parseHeaderField(":START_ID(Person)")).toMatchObject({ kind: "START_ID", space: "Person" });
        expect(parseHeaderField(":END_ID")).toMatchObject({ kind: "END_ID", space: null });
        expect(parseHeaderField(":TYPE")).toMatchObject({ kind: "TYPE" });
        expect(parseHeaderField("junk:IGNORE")).toMatchObject({ name: "junk", kind: "IGNORE" });
        expect(parseHeaderField(":ignore")).toMatchObject({ kind: "IGNORE" });
    });

    it("parses property fields with and without a type", () => {
        expect(parseHeaderField("name")).toMatchObject({ name: "name", kind: "PROPERTY", type: null });
        expect(parseHeaderField("age:int")).toMatchObject({ name: "age", kind: "PROPERTY", type: "int" });
        expect(parseHeaderField("tags:string[]")).toMatchObject({ name: "tags", type: "string[]" });
        expect(parseHeaderField(" spaced : LONG ")).toMatchObject({ name: "spaced", type: "LONG" });
    });

    it("parses brace options with lower-cased keys", () => {
        const field = parseHeaderField(":ID(Space){label:Person, Id-Type: string}");
        expect(field.space).toBe("Space");
        expect([...field.options]).toEqual([
            ["label", "Person"],
            ["id-type", "string"],
        ]);
        expect(parseHeaderField(":ID{}").options.size).toBe(0);
    });

    it("keeps the cell text", () => {
        expect(parseHeaderField(" name:int ").text).toBe(" name:int ");
    });

    it.each([
        "",
        "   ",
        ":",
        "name:",
        ":string",
        "name:string(Person)",
        "name:LABEL",
        "x:START_ID",
        "x:END_ID",
        "x:TYPE",
        "name(Person)",
        "name{label:X}",
        "name:LABEL(Space)",
        ":ID{label}",
        ":ID{:x}",
        "a:b:c",
        "a(b",
    ])("rejects %j", (cell) => {
        expect(() => parseHeaderField(cell)).toThrow(GraphFormatError);
        try {
            parseHeaderField(cell);
        } catch (err) {
            expect((err as GraphFormatError).code).toBe("E_UNSUPPORTED");
            expect((err as GraphFormatError).details.reason).toBe("header");
        }
    });
});

describe("isHeaderRecord", () => {
    it("recognises node and relationship headers", () => {
        expect(isHeaderRecord([":ID", "name"], 2)).toBe(true);
        expect(isHeaderRecord(["personId:ID(Person)", ":LABEL"], 2)).toBe(true);
        expect(isHeaderRecord([":START_ID", ":END_ID", ":TYPE"], 3)).toBe(true);
        expect(isHeaderRecord(["since:int", " :end_id "], 2)).toBe(true);
        expect(isHeaderRecord([":ID{label:X}"], 1)).toBe(true);
    });

    it("does not mistake data rows or other headers", () => {
        expect(isHeaderRecord(["1", "Alice"], 2)).toBe(false);
        expect(isHeaderRecord(["source", "target"], 2)).toBe(false);
        expect(isHeaderRecord([":LABEL", "name"], 2)).toBe(false);
        expect(isHeaderRecord(["urn:IDENT", "x"], 2)).toBe(false);
        expect(isHeaderRecord([":ID", "x"], 1)).toBe(true);
        expect(isHeaderRecord(["x", ":ID"], 1)).toBe(false);
    });
});

describe("formatHeaderField", () => {
    it("writes the inverse of parseHeaderField", () => {
        expect(formatHeaderField("", "ID", null)).toBe(":ID");
        expect(formatHeaderField("personId", "ID", "Person")).toBe("personId:ID(Person)");
        expect(formatHeaderField("age", "int", null)).toBe("age:int");
        expect(parseHeaderField(formatHeaderField("tags", "string[]", null))).toMatchObject({
            name: "tags",
            type: "string[]",
        });
    });
});
