import { GraphFormatError } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { ImportReportBuilder } from "../../../src/common/report.js";
import {
    BAD_QUOTE_CODE,
    checkRecordSyntax,
    RecordReader,
    UNCLOSED_QUOTE_CODE,
} from "../../../src/formats/csv/records.js";
import { ImportError, type ImportInput } from "../../../src/types.js";
import { byteChunks, textChunksOf } from "../../helpers/corpus.js";

interface Record {
    cells: string[];
    quoted: boolean[];
    line: number;
}

async function read(
    input: ImportInput,
    syntax = { delimiter: ",", quote: '"' },
    report = new ImportReportBuilder("neo4j", 100),
): Promise<Record[]> {
    const reader = new RecordReader(input, report, syntax, {});
    const out: Record[] = [];
    for await (const count of reader) {
        out.push({ cells: reader.cells.slice(0, count), quoted: reader.quoted.slice(0, count), line: reader.line });
    }
    return out;
}

function cellsOf(records: readonly Record[]): string[][] {
    return records.map((r) => r.cells);
}

describe("RecordReader", () => {
    it("splits plain records and keeps line numbers", async () => {
        const records = await read("a,b,c\n1,2,3\n4,5,6\n");
        expect(cellsOf(records)).toEqual([
            ["a", "b", "c"],
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]);
        expect(records.map((r) => r.line)).toEqual([1, 2, 3]);
    });

    it("yields a final record without a line terminator", async () => {
        expect(cellsOf(await read("a,b\n1,2"))).toEqual([
            ["a", "b"],
            ["1", "2"],
        ]);
    });

    it("handles CRLF and lone CR terminators", async () => {
        const records = await read("a,b\r\n1,2\r3,4\r\n");
        expect(cellsOf(records)).toEqual([
            ["a", "b"],
            ["1", "2"],
            ["3", "4"],
        ]);
        expect(records.map((r) => r.line)).toEqual([1, 2, 3]);
    });

    it("skips blank lines but keeps a record of empty cells", async () => {
        const records = await read("a,b\n\n\n1,\n,\n");
        expect(cellsOf(records)).toEqual([
            ["a", "b"],
            ["1", ""],
            ["", ""],
        ]);
        expect(records.map((r) => r.line)).toEqual([1, 4, 5]);
    });

    it("distinguishes a quoted empty field from an unquoted empty field", async () => {
        const [record] = await read('"",,x');
        expect(record.cells).toEqual(["", "", "x"]);
        expect(record.quoted).toEqual([true, false, false]);
    });

    it("keeps a quoted empty field on its own line as a record", async () => {
        const records = await read('a\n""\nb\n');
        expect(cellsOf(records)).toEqual([["a"], [""], ["b"]]);
        expect(records[1].quoted).toEqual([true]);
    });

    it("decodes quoted fields with delimiters, doubled quotes and line breaks", async () => {
        const records = await read('"a,b","say ""hi""","line 1\nline 2\r\nline 3",plain\nnext,1\n');
        expect(cellsOf(records)).toEqual([
            ["a,b", 'say "hi"', "line 1\nline 2\r\nline 3", "plain"],
            ["next", "1"],
        ]);
        // the second record starts on line 4: the quoted field spanned lines 1..3
        expect(records[1].line).toBe(4);
    });

    it("text after a closing quote is a fatal parse error with the record's line (one policy for CSV and Neo4j CSV)", async () => {
        // RFC 4180 has no such form and neo4j-admin import refuses it; the CSV importer already
        // aborted on it, so the shared reader keeps the strict policy for both formats
        const report = new ImportReportBuilder("neo4j", 100);
        await expect(read('"ab"cd,x\n', undefined, report)).rejects.toThrow(ImportError);
        expect(report.issues.map((i) => i.code)).toEqual([BAD_QUOTE_CODE]);
        expect(report.issues[0].severity).toBe("error");
        expect(report.issues[0].line).toBe(1);
        const eof = new ImportReportBuilder("neo4j", 100);
        await expect(read('a,b\n1,"ab"cd', undefined, eof)).rejects.toThrow(ImportError);
        expect(eof.issues.map((i) => i.code)).toEqual([BAD_QUOTE_CODE]);
        expect(eof.issues[0].line).toBe(2);
        const chunked = new ImportReportBuilder("neo4j", 100);
        await expect(read(textChunksOf('"ab"cd,x\n', 5), undefined, chunked)).rejects.toThrow(ImportError);
        expect(chunked.issues).toHaveLength(1);
    });

    it("aborts on a quoted field that is never closed", async () => {
        const report = new ImportReportBuilder("neo4j", 100);
        await expect(read('a,b\n1,"open\n2,x\n', undefined, report)).rejects.toThrow(ImportError);
        try {
            await read('a,b\n1,"open\n2,x\n');
        } catch (err) {
            expect(err).toBeInstanceOf(ImportError);
            const { report: partial } = err as ImportError;
            expect(partial.issues[0].code).toBe(UNCLOSED_QUOTE_CODE);
            expect(partial.issues[0].line).toBe(2);
        }
    });

    it("produces the same records whatever the chunk boundaries", async () => {
        const text = `id,name\n1,"Zo${String.fromCharCode(0xeb)}"\r\n2,"a,b\r\nc"\n3,""\n4,\n`;
        const expected = await read(text);
        const bytes = new TextEncoder().encode(text);
        for (const size of [1, 2, 3, 5, 7, 64]) {
            expect(await read(byteChunks(bytes, size))).toEqual(expected);
            expect(await read(textChunksOf(text, size))).toEqual(expected);
        }
    });

    it("supports a tab delimiter and another quote character", async () => {
        const records = await read("a\tb\n'x\ty'\t2\n", { delimiter: "\t", quote: "'" });
        expect(cellsOf(records)).toEqual([
            ["a", "b"],
            ["x\ty", "2"],
        ]);
    });

    it("strips a BOM", async () => {
        const records = await read(`${String.fromCharCode(0xfeff)}a,b\n`);
        expect(cellsOf(records)).toEqual([["a", "b"]]);
    });
});

describe("checkRecordSyntax", () => {
    it("accepts one-character delimiter and quote", () => {
        expect(checkRecordSyntax({ delimiter: "\t", quote: "'" })).toEqual({ delimiter: "\t", quote: "'" });
    });

    it.each([
        [{ delimiter: ",,", quote: '"' }, "delimiter"],
        [{ delimiter: "", quote: '"' }, "delimiter"],
        [{ delimiter: "\n", quote: '"' }, "delimiter"],
        [{ delimiter: ",", quote: '""' }, "quote"],
        [{ delimiter: ",", quote: "," }, "delimiter"],
    ])("rejects %j", (syntax, option) => {
        expect(() => checkRecordSyntax(syntax)).toThrow(GraphFormatError);
        try {
            checkRecordSyntax(syntax);
        } catch (err) {
            expect((err as GraphFormatError).code).toBe("E_UNSUPPORTED");
            expect((err as GraphFormatError).details.option).toBe(option);
        }
    });
});
