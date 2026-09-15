import { GraphBuilder, GraphFormatError, type GraphSnapshot, INVALID_INDEX } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { RENAMED_CODE } from "../../../src/common/attributes.js";
import { DIRECTION_FORCED_CODE, DIRECTION_REFUSED_CODE, MIXED_DIRECTION_CODE } from "../../../src/common/direction.js";
import { INVALID_UTF8_CODE } from "../../../src/common/input.js";
import { SINK_OPTION_CODE } from "../../../src/common/options.js";
import {
    BAD_TYPE_CODE,
    COLUMN_MISSING_CODE,
    csvImporter,
    DUPLICATE_NODE_CODE,
    EMPTY_INPUT_CODE,
    FIELD_COUNT_CODE,
    ID_MERGED_CODE,
    MISSING_ENDPOINT_CODE,
    MISSING_ID_CODE,
    NO_DATA_ROWS_CODE,
    NO_ENDPOINT_COLUMNS_CODE,
    NO_ID_COLUMN_CODE,
    ROLE_TAKEN_CODE,
} from "../../../src/formats/csv/importer.js";
import { BAD_QUOTE_CODE, UNCLOSED_QUOTE_CODE } from "../../../src/formats/csv/records.js";
import { type CommonImportOptions, ImportError, type ImportReport } from "../../../src/types.js";
import {
    byteChunks,
    corpusEntry,
    corpusFiles,
    inputShapes,
    malformedFiles,
    readCorpusBytes,
    readCorpusText,
    readMalformedBytes,
    textChunksOf,
} from "../../helpers/corpus.js";

type Options = Parameters<typeof csvImporter.import>[2];

async function load(
    input: Parameters<typeof csvImporter.import>[0],
    options?: Options,
    builderOptions: { directed?: boolean; locked?: boolean } = {},
): Promise<{ snapshot: GraphSnapshot; report: ImportReport; builder: GraphBuilder }> {
    const builder = new GraphBuilder({ directed: builderOptions.directed ?? true, weightDtype: "f64" });
    if (builderOptions.locked) {
        builder.lockDirected();
    }
    const report = await csvImporter.import(input, builder, options);
    return { snapshot: builder.freeze(), report, builder };
}

async function failure(input: string, options?: Options): Promise<ImportError> {
    const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
    try {
        await csvImporter.import(input, builder, options);
    } catch (err) {
        if (err instanceof ImportError) {
            return err;
        }
        throw err;
    }
    throw new Error("expected an ImportError");
}

function codes(report: ImportReport): string[] {
    return report.issues.map((i) => i.code);
}

function edgesOf(s: GraphSnapshot): string[] {
    const list = s.edgeList();
    const out: string[] = [];
    for (let e = 0; e < s.edgeCount; e++) {
        out.push(`${String(s.ids.idOf(list.src[e]))}->${String(s.ids.idOf(list.dst[e]))}`);
    }
    return out;
}

function weightsOf(s: GraphSnapshot): (number | undefined)[] {
    const shadow = s.edges.byRole("weight");
    const list = s.edgeList();
    const out: (number | undefined)[] = [];
    for (let e = 0; e < s.edgeCount; e++) {
        if (shadow !== null) {
            out.push(shadow.isSet(e) ? (shadow.value(e) as number) : undefined);
        } else {
            out.push(list.weights === null ? undefined : list.weights[e]);
        }
    }
    return out;
}

function column(s: GraphSnapshot, table: "nodes" | "edges", name: string): unknown[] {
    const c = s[table].get(name);
    if (c === null) {
        throw new Error(`no ${table} column ${name}`);
    }
    return Array.from({ length: c.length }, (_, r) => (c.isSet(r) ? c.value(r) : undefined));
}

describe("csvImporter: the corpus", () => {
    for (const entry of corpusFiles("csv")) {
        it(`imports ${entry.path} with the manifest's counts`, async () => {
            const { snapshot, report } = await load(readCorpusBytes("csv", entry.path));
            expect(snapshot.nodeCount).toBe(entry.expectedNodes);
            // the manifest counts source rows; an expanded undirected row of a mixed file is two logical edges
            expect(snapshot.edgeCount - report.counts.expandedMixed).toBe(entry.expectedEdges);
            expect(report.counts.nodes).toBe(entry.expectedNodes);
            expect(report.counts.edges).toBe(snapshot.edgeCount);
            expect(report.errorCount).toBe(0);
            expect(report.format).toBe("csv");
            expect(snapshot.directed).toBe(!entry.features.includes("undirected"));
        });

        it(`imports ${entry.path} identically from every input shape`, async () => {
            const bytes = readCorpusBytes("csv", entry.path);
            const reference = await load(bytes);
            for (const shape of inputShapes(bytes)) {
                const { snapshot } = await load(shape.make());
                expect(snapshot.ids.toArray(), shape.name).toEqual(reference.snapshot.ids.toArray());
                expect(edgesOf(snapshot), shape.name).toEqual(edgesOf(reference.snapshot));
                expect(weightsOf(snapshot), shape.name).toEqual(weightsOf(reference.snapshot));
                expect(snapshot.directed, shape.name).toBe(reference.snapshot.directed);
            }
        });
    }

    it("simple-edges.csv: string ids, f64 weights, directed by default", async () => {
        const { snapshot } = await load(readCorpusText("csv", "simple-edges.csv"));
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c", "d", "e"]);
        expect(edgesOf(snapshot)).toEqual(["a->b", "b->c", "c->d", "d->e", "a->c"]);
        expect(weightsOf(snapshot)).toEqual([1, 2, 1.5, 1, 0.5]);
        expect(snapshot.flags.weighted).toBe(true);
        expect(snapshot.edges.names()).toEqual([]);
    });

    it("gephi-format.csv: per-row Type expands the undirected rows into pairs and keeps the Label", async () => {
        const { snapshot, report } = await load(readCorpusText("csv", "gephi-format.csv"));
        expect(snapshot.directed).toBe(true);
        expect(snapshot.edgeCount).toBe(7);
        expect(report.counts.expandedMixed).toBe(2);
        expect(edgesOf(snapshot)).toEqual([
            "Alice->Bob",
            "Bob->Charlie",
            "Charlie->David",
            "David->Charlie",
            "David->Alice",
            "Alice->Charlie",
            "Charlie->Alice",
        ]);
        expect(column(snapshot, "edges", "graphty.directed")).toEqual([true, true, false, false, true, false, false]);
        expect(column(snapshot, "edges", "graphty.pair")).toEqual([undefined, undefined, 3, 2, undefined, 6, 5]);
        const label = snapshot.edges.get("Label");
        expect(label?.meta.role).toBe("label");
        expect(label?.dtype).toBe("string");
        expect(column(snapshot, "edges", "Label")).toEqual([
            "friendship",
            "colleague",
            "acquaintance",
            undefined,
            "neighbor",
            "colleague",
            undefined,
        ]);
        expect(weightsOf(snapshot)).toEqual([5, 3, 1, 1, 2, 4, 4]);
    });

    it("dolphins-medium.csv: all-Undirected rows import as an undirected graph with integer ids", async () => {
        const { snapshot } = await load(readCorpusText("csv", "dolphins-medium.csv"));
        expect(snapshot.directed).toBe(false);
        expect(snapshot.edges.byRole("pair")).toBeNull();
        expect(snapshot.ids.idOf(0)).toBe(8);
        expect(snapshot.ids.idOf(1)).toBe(3);
        expect(typeof snapshot.ids.idOf(30)).toBe("number");
        expect(snapshot.arcCount).toBe(100);
    });

    it("got-edges.csv (CRLF, no terminator on the last line) reads every row", async () => {
        const { snapshot } = await load(readCorpusText("csv", "got-edges.csv"));
        expect(edgesOf(snapshot)[0]).toBe("Aemon->Grenn");
        expect(edgesOf(snapshot)[351]).toBe("Ygritte->Rattleshirt");
        expect(weightsOf(snapshot)[351]).toBe(9);
        expect(snapshot.ids.toArray()).not.toContain("Grenn\r");
    });

    it("got-nodes.csv + got-edges.csv: the node table comes first and gives the Label", async () => {
        const entry = corpusEntry("csv", "got-edges.csv");
        const { snapshot, report } = await load(readCorpusText("csv", "got-edges.csv"), {
            nodes: readCorpusText("csv", "got-nodes.csv"),
        });
        expect(snapshot.nodeCount).toBe(entry.expectedNodes);
        expect(snapshot.edgeCount).toBe(entry.expectedEdges);
        expect(report.counts.nodes).toBe(107);
        expect(report.errorCount).toBe(0);
        expect(snapshot.ids.idOf(0)).toBe("Aemon");
        expect(snapshot.ids.idOf(106)).toBe("Walton");
        const label = snapshot.nodes.get("Label");
        expect(label?.meta.role).toBe("label");
        expect(label?.value(106)).toBe("Walton");
    });

    it("got-nodes.csv alone imports as a node table under table auto", async () => {
        const { snapshot, report } = await load(readCorpusText("csv", "got-nodes.csv"));
        expect(snapshot.nodeCount).toBe(107);
        expect(snapshot.edgeCount).toBe(0);
        expect(report.counts.nodes).toBe(107);
        expect(snapshot.nodes.get("Label")?.value(3)).toBe("Aerys");
    });
});

describe("csvImporter: the malformed corpus", () => {
    // fatal: the import aborts regardless of the error limit
    const fatal: Record<string, string> = {
        "binary-content.csv": INVALID_UTF8_CODE,
        "empty-file.csv": EMPTY_INPUT_CODE,
        "missing-source-column.csv": NO_ENDPOINT_COLUMNS_CODE,
        "missing-target-column.csv": NO_ENDPOINT_COLUMNS_CODE,
        "unclosed-quote.csv": UNCLOSED_QUOTE_CODE,
    };
    // recoverable: per-row errors under the default limit, ImportError under errorLimit 0
    const recoverable: Record<string, { errors: number; code: string; nodes: number; edges: number }> = {
        "empty-values.csv": { errors: 3, code: MISSING_ENDPOINT_CODE, nodes: 2, edges: 1 },
        "garbage-content.csv": { errors: 3, code: "E_INVALID_WEIGHT", nodes: 0, edges: 0 },
        "inconsistent-columns.csv": { errors: 3, code: FIELD_COUNT_CODE, nodes: 2, edges: 1 },
    };
    // a header without rows is an empty graph (the exporter writes exactly that for an empty snapshot)
    const headerOnly = ["header-only.csv"];

    it("covers every malformed file", () => {
        const covered = new Set([
            ...Object.keys(fatal),
            ...Object.keys(recoverable),
            ...headerOnly,
            "wrong-delimiter.csv",
        ]);
        expect([...malformedFiles("csv")].sort()).toEqual([...covered].sort());
    });

    for (const [name, code] of Object.entries(fatal)) {
        it(`${name} aborts with ${code} and a report`, async () => {
            const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
            const err = await csvImporter.import(readMalformedBytes("csv", name), builder).then(
                () => null,
                (e: unknown) => e,
            );
            expect(err).toBeInstanceOf(ImportError);
            const { report } = err as ImportError;
            expect((err as ImportError).code).toBe("E_IMPORT");
            expect(report.format).toBe("csv");
            expect(report.issues.at(-1)?.code).toBe(code);
            expect(report.errorCount).toBeGreaterThan(0);
        });
    }

    for (const [name, expected] of Object.entries(recoverable)) {
        it(`${name} recovers with ${expected.errors} error(s) under the default limit`, async () => {
            const { snapshot, report } = await load(readMalformedBytes("csv", name));
            expect(report.errorCount).toBe(expected.errors);
            expect(codes(report)).toContain(expected.code);
            expect(report.truncated).toBe(false);
            expect(snapshot.nodeCount).toBe(expected.nodes);
            expect(snapshot.edgeCount).toBe(expected.edges);
            expect(report.counts.skippedEdges).toBe(expected.errors);
            for (const issue of report.issues) {
                expect(issue.line).not.toBeNull();
            }
        });

        it(`${name} aborts with ImportError under errorLimit 0`, async () => {
            const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
            const err = await csvImporter.import(readMalformedBytes("csv", name), builder, { errorLimit: 0 }).then(
                () => null,
                (e: unknown) => e,
            );
            expect(err).toBeInstanceOf(ImportError);
            const { report } = err as ImportError;
            expect(report.truncated).toBe(true);
            expect(report.errorCount).toBe(1);
            expect(report.issues[0].code).toBe(expected.code);
        });
    }

    for (const name of headerOnly) {
        it(`${name} is an empty graph with a no-rows warning`, async () => {
            const { snapshot, report } = await load(readMalformedBytes("csv", name));
            expect(snapshot.nodeCount).toBe(0);
            expect(snapshot.edgeCount).toBe(0);
            expect(codes(report)).toEqual([NO_DATA_ROWS_CODE]);
            expect(report.errorCount).toBe(0);
        });
    }

    it("wrong-delimiter.csv reads through delimiter sniffing and fails when the delimiter is forced", async () => {
        const { snapshot, report } = await load(readMalformedBytes("csv", "wrong-delimiter.csv"));
        expect(report.issues).toEqual([]);
        expect(edgesOf(snapshot)).toEqual(["A->B", "C->D", "E->F"]);
        expect(weightsOf(snapshot)).toEqual([1, 2, 3]);
        const err = await failure(new TextDecoder().decode(readMalformedBytes("csv", "wrong-delimiter.csv")), {
            delimiter: ",",
            table: "edges",
        });
        expect(err.report.issues[0].code).toBe(NO_ENDPOINT_COLUMNS_CODE);
        // under table auto a single-column headerless file is a node list
        const asNodes = await load(readMalformedBytes("csv", "wrong-delimiter.csv"), { delimiter: "," });
        expect(asNodes.snapshot.ids.toArray()).toEqual(["Source;Target;Weight", "A;B;1", "C;D;2", "E;F;3"]);
    });

    it("the unclosed quote report names the line", async () => {
        const err = await failure('Source,Target\n"a,b\nc,d\n');
        expect(err.report.issues[0]).toMatchObject({ code: UNCLOSED_QUOTE_CODE, line: 2, category: "parse-error" });
    });

    it("a malformed closing quote aborts with its line", async () => {
        const err = await failure('source,target\n"a"x,"b"\nc,d\n');
        expect(err.report.issues[0]).toMatchObject({ code: BAD_QUOTE_CODE, line: 2, category: "parse-error" });
        expect(err.report.counts.edges).toBe(0);
    });

    it("invalid UTF-8 is a fatal parse error", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const bytes = new Uint8Array([...new TextEncoder().encode("source,target\na,"), 0xff, 0xfe, 10]);
        const err = await csvImporter.import(bytes, builder).then(
            () => null,
            (e: unknown) => e,
        );
        expect(err).toBeInstanceOf(ImportError);
        expect((err as ImportError).report.issues[0].code).toBe(INVALID_UTF8_CODE);
    });
});

describe("csvImporter: header and delimiter handling", () => {
    it("skips the leading # / % comment lines of SNAP and KONECT files and reads their direction", async () => {
        const snap = await load("# Undirected graph\n# Nodes: 3 Edges: 2\n# FromNodeId\tToNodeId\n1\t2\n2\t3\n");
        expect(snap.report.issues).toEqual([]);
        expect(snap.snapshot.directed).toBe(false);
        expect(snap.snapshot.ids.toArray()).toEqual([1, 2, 3]);
        expect(snap.snapshot.edgeCount).toBe(2);
        const konect = await load("% asym positive\n% 2 3 3\n1 2 16777217\n2 3 0.5\n");
        expect(konect.report.issues).toEqual([]);
        expect(konect.snapshot.directed).toBe(true);
        expect(konect.snapshot.edges.byRole("weight")?.value(0)).toBe(16777217);
        expect(konect.snapshot.edges.byRole("weight")?.value(1)).toBe(0.5);
        // the caller's defaultDirected applies when the comments say nothing about direction
        const silent = await load("# a comment\n1,2\n", { defaultDirected: false });
        expect(silent.snapshot.directed).toBe(false);
        // a later line starting with # is a record, not a comment (the field-count error says so)
        const body = await load("source,target\na,b\n# not a comment\nb,c\n");
        expect(body.report.issues.map((i) => i.code)).toEqual(["E_CSV_FIELD_COUNT"]);
        expect(body.snapshot.edgeCount).toBe(2);
    });

    it("sniffs tab, semicolon, pipe and space delimiters", async () => {
        for (const [delimiter, name] of [
            ["\t", "tab"],
            [";", "semicolon"],
            ["|", "pipe"],
            [" ", "space"],
        ] as const) {
            const header = ["source", "target", "weight"].join(delimiter);
            const text = `${header}\na${delimiter}b${delimiter}2\nb${delimiter}c${delimiter}3\n`;
            const { snapshot } = await load(text);
            expect(edgesOf(snapshot), name).toEqual(["a->b", "b->c"]);
            expect(weightsOf(snapshot), name).toEqual([2, 3]);
        }
    });

    it("prefers the delimiter with the most consistent field count", async () => {
        const { snapshot } = await load("source,target,note\na,b,hello world\nb,c,one two three\n");
        expect(edgesOf(snapshot)).toEqual(["a->b", "b->c"]);
        expect(column(snapshot, "edges", "note")).toEqual(["hello world", "one two three"]);
    });

    it("accepts an explicit delimiter", async () => {
        const { snapshot } = await load("source;target\na,b;c,d\n", { delimiter: ";" });
        expect(edgesOf(snapshot)).toEqual(["a,b->c,d"]);
    });

    it("rejects an unusable delimiter option", async () => {
        const builder = new GraphBuilder({ directed: true });
        await expect(csvImporter.import("a,b", builder, { delimiter: '"' })).rejects.toMatchObject({
            code: "E_UNSUPPORTED",
        });
        await expect(csvImporter.import("a,b", builder, { delimiter: "" })).rejects.toMatchObject({
            code: "E_UNSUPPORTED",
        });
    });

    it("detects a header by a known column name, case-insensitively", async () => {
        for (const header of ["source,target", "SOURCE,TARGET", "From,To", "src,dst", "Source,Target"]) {
            const { snapshot } = await load(`${header}\nx,y\n`);
            expect(edgesOf(snapshot), header).toEqual(["x->y"]);
        }
    });

    it("detects a header when a text row precedes numeric rows and asks for its columns", async () => {
        const err = await failure("u,v\n1,2\n2,3\n");
        expect(err.report.issues[0].code).toBe(NO_ENDPOINT_COLUMNS_CODE);
        expect(err.report.issues[0].message).toContain('"u", "v"');
        const { snapshot } = await load("u,v\n1,2\n2,3\n", { sourceColumn: "u", targetColumn: "v" });
        expect(snapshot.ids.toArray()).toEqual([1, 2, 3]);
        expect(edgesOf(snapshot)).toEqual(["1->2", "2->3"]);
    });

    it("treats a text-only first row as data when nothing below is numeric", async () => {
        const { snapshot } = await load("alice,bob\nbob,carol\n");
        expect(snapshot.ids.toArray()).toEqual(["alice", "bob", "carol"]);
        expect(edgesOf(snapshot)).toEqual(["alice->bob", "bob->carol"]);
    });

    it("reads a headerless numeric edge list positionally with the third column as the weight", async () => {
        const { snapshot } = await load("1 2 0.5\n2 3 1.5\n3 1\n".replace("3 1\n", "3 1 2\n"));
        expect(snapshot.ids.toArray()).toEqual([1, 2, 3]);
        expect(weightsOf(snapshot)).toEqual([0.5, 1.5, 2]);
        expect(snapshot.edges.names()).toEqual([]);
    });

    it("names the extra columns of a headerless file by position and honours weightFrom null", async () => {
        const { snapshot } = await load("a,b,x,7\nb,c,y,8\n", { weightFrom: null });
        expect(snapshot.flags.weighted).toBe(false);
        // declaration order follows the dict decision: a numeric column is created at its first cell,
        // a text column when its sample decides
        expect([...snapshot.edges.names()].sort()).toEqual(["column3", "column4"]);
        expect(column(snapshot, "edges", "column3")).toEqual(["x", "y"]);
        expect(column(snapshot, "edges", "column4")).toEqual([7, 8]);
    });

    it("honours header: false and header: true", async () => {
        const forced = await load("source,target\na,b\n", { header: false });
        expect(edgesOf(forced.snapshot)).toEqual(["source->target", "a->b"]);
        const declared = await load("p,q\na,b\n", { header: true, sourceColumn: "p", targetColumn: "q" });
        expect(edgesOf(declared.snapshot)).toEqual(["a->b"]);
    });

    it("resolves explicit endpoint columns by name and by position", async () => {
        const byName = await load("weight,to,from\n2,b,a\n", { sourceColumn: "from", targetColumn: "to" });
        expect(edgesOf(byName.snapshot)).toEqual(["a->b"]);
        expect(weightsOf(byName.snapshot)).toEqual([2]);
        const byPosition = await load("x,y,z\na,b,c\n", { header: true, sourceColumn: 2, targetColumn: 0 });
        expect(edgesOf(byPosition.snapshot)).toEqual(["c->a"]);
        expect(column(byPosition.snapshot, "edges", "y")).toEqual(["b"]);
        const headerless = await load("a,b,c\nd,e,f\n", { sourceColumn: 2, targetColumn: 0 });
        expect(edgesOf(headerless.snapshot)).toEqual(["c->a", "f->d"]);
        expect(headerless.snapshot.flags.weighted).toBe(false);
        expect(column(headerless.snapshot, "edges", "column2")).toEqual(["b", "e"]);
    });

    it("refuses an explicit column the file does not have", async () => {
        const builder = new GraphBuilder({ directed: true });
        await expect(csvImporter.import("a,b\n", builder, { sourceColumn: "nope" })).rejects.toMatchObject({
            code: "E_UNSUPPORTED",
        });
        await expect(csvImporter.import("a,b\n", builder, { sourceColumn: 5 })).rejects.toMatchObject({
            code: "E_UNSUPPORTED",
        });
        await expect(
            csvImporter.import("source,target\na,b\n", builder, { sourceColumn: "source", targetColumn: "source" }),
        ).rejects.toMatchObject({ code: "E_UNSUPPORTED" });
        await expect(csvImporter.import("a,b\n", builder, { sourceColumn: -1 })).rejects.toMatchObject({
            code: "E_UNSUPPORTED",
        });
    });

    it("fails on a header without endpoint columns when the table must be edges", async () => {
        const err = await failure("id,name\n1,a\n", { table: "edges" });
        expect(err.report.issues[0].code).toBe(NO_ENDPOINT_COLUMNS_CODE);
        expect(err.report.issues[0].line).toBe(1);
        const neither = await failure("foo,bar\n1,2\n2,3\n");
        expect(neither.report.issues[0].code).toBe(NO_ENDPOINT_COLUMNS_CODE);
        expect(neither.report.issues[0].message).toContain("neither");
    });

    it("renames repeated header names and reports them", async () => {
        const { snapshot, report } = await load("source,target,x,x\na,b,1,2\n");
        expect(codes(report)).toEqual([RENAMED_CODE]);
        expect(report.issues[0].element).toBe("x");
        expect(snapshot.edges.names()).toEqual(["x", "x#4"]);
        expect(column(snapshot, "edges", "x#4")).toEqual([2]);
    });

    it("names an empty header cell by its position and trims header cells", async () => {
        const { snapshot } = await load(" source , target ,,\na,b,1,2\n");
        expect(edgesOf(snapshot)).toEqual(["a->b"]);
        expect(snapshot.edges.names()).toEqual(["column3", "column4"]);
    });

    it("reads quoted cells with embedded delimiters, quotes and line breaks and keeps line numbers", async () => {
        const text = 'source,target,note\n"a,1","b""q","multi\nline"\nc,,x\n';
        const { snapshot, report } = await load(text);
        expect(edgesOf(snapshot)).toEqual(['a,1->b"q']);
        expect(column(snapshot, "edges", "note")).toEqual(["multi\nline"]);
        expect(report.issues[0]).toMatchObject({ code: MISSING_ENDPOINT_CODE, line: 4 });
    });

    it("reads lone-CR line endings", async () => {
        const { snapshot } = await load("source,target\ra,b\rb,c\r");
        expect(edgesOf(snapshot)).toEqual(["a->b", "b->c"]);
    });

    it("reads mixed LF and CRLF line endings", async () => {
        const { snapshot } = await load("source,target\r\na,b\nb,c\r\n");
        expect(edgesOf(snapshot)).toEqual(["a->b", "b->c"]);
    });

    it("skips blank lines and a BOM", async () => {
        const { snapshot, report } = await load(`${String.fromCharCode(0xfeff)}source,target\n\na,b\n   \n\nb,c\n`);
        expect(edgesOf(snapshot)).toEqual(["a->b", "b->c"]);
        expect(report.issues).toEqual([]);
    });

    it("streams a file longer than the preview in small chunks with exact line numbers", async () => {
        const rows = ["source,target,weight"];
        for (let i = 0; i < 300; i++) {
            rows.push(`n${i},n${i + 1},${i % 7 === 6 ? "bad" : i}`);
        }
        const text = `${rows.join("\n")}\n`;
        const bytes = new TextEncoder().encode(text);
        const { snapshot, report } = await load(byteChunks(bytes, 13));
        expect(snapshot.edgeCount).toBe(300 - report.errorCount);
        expect(report.errorCount).toBe(42);
        expect(report.issues[0]).toMatchObject({ code: "E_INVALID_WEIGHT", line: 8 });
        expect(report.issues[1].line).toBe(15);
    });
});

describe("csvImporter: ids, weights and direction", () => {
    it("applies the canonical rule: 1 becomes a number while 01, 1.0 and +1 stay strings", async () => {
        const { snapshot } = await load("source,target\n1,01\n1.0,+1\n-0,0\n");
        expect(snapshot.ids.toArray()).toEqual([1, "01", "1.0", "+1", "-0", 0]);
    });

    it("keeps every id a string under ids string and merges under ids number with a coercion warning", async () => {
        const strings = await load("source,target\n1,2\n", { ids: "string" });
        expect(strings.snapshot.ids.toArray()).toEqual(["1", "2"]);
        const numbers = await load("source,target\n1,01\n1.0,2\n", { ids: "number" });
        expect(numbers.snapshot.ids.toArray()).toEqual([1, 2]);
        expect(numbers.snapshot.edgeCount).toBe(2);
        expect(codes(numbers.report)).toEqual([ID_MERGED_CODE, ID_MERGED_CODE]);
        expect(numbers.report.issues[0].line).toBe(2);
        expect(numbers.report.issues[1].message).toContain("2 id cell(s)");
    });

    it("rejects a non-numeric id under ids number per row and continues", async () => {
        const { snapshot, report } = await load("source,target\n1,2\nx,3\n", { ids: "number" });
        expect(snapshot.ids.toArray()).toEqual([1, 2]);
        expect(report.issues[0]).toMatchObject({ code: "E_INVALID_ID", line: 3, element: "x->3" });
        expect(report.counts.skippedEdges).toBe(1);
    });

    it("uses one coercion rule across the node table and the edge table", async () => {
        const { snapshot } = await load("source,target\n1,2\n", { nodes: "id,name\n1,one\n2,two\n" });
        expect(snapshot.ids.toArray()).toEqual([1, 2]);
        expect(column(snapshot, "nodes", "name")).toEqual(["one", "two"]);
        expect(edgesOf(snapshot)).toEqual(["1->2"]);
    });

    it("matches the weight column case-insensitively and omits blank weights", async () => {
        const { snapshot } = await load("source,target,Weight\na,b,2.5\nb,c,\nc,d, 3 \n");
        expect(weightsOf(snapshot)).toEqual([2.5, undefined, 3]);
        expect(snapshot.flags.weighted).toBe(true);
    });

    it("takes a custom weight column and warns when an explicit one is missing", async () => {
        const custom = await load("source,target,cost,weight\na,b,7,1\n", { weightFrom: "cost" });
        expect(weightsOf(custom.snapshot)).toEqual([7]);
        expect(column(custom.snapshot, "edges", "weight")).toEqual([1]);
        const missing = await load("source,target\na,b\n", { weightFrom: "cost" });
        expect(codes(missing.report)).toEqual([COLUMN_MISSING_CODE]);
        expect(missing.snapshot.flags.weighted).toBe(false);
    });

    it("keeps a column named weight as data under weightFrom null", async () => {
        const { snapshot } = await load("source,target,weight\na,b,2\n", { weightFrom: null });
        expect(snapshot.flags.weighted).toBe(false);
        expect(column(snapshot, "edges", "weight")).toEqual([2]);
    });

    it("reports an invalid weight per row before the sink is touched", async () => {
        const { snapshot, report } = await load("source,target,weight\na,b,x\nb,c,NaN\nc,d,Infinity\n");
        expect(report.issues.map((i) => [i.code, i.line])).toEqual([
            ["E_INVALID_WEIGHT", 2],
            ["E_INVALID_WEIGHT", 3],
        ]);
        expect(snapshot.ids.toArray()).toEqual(["c", "d"]);
        expect(weightsOf(snapshot)).toEqual([Infinity]);
    });

    it("keeps 0.1 and 16777217 exact through the f64 default", async () => {
        const { snapshot } = await load("source,target,weight\na,b,0.1\nb,c,16777217\n");
        expect(weightsOf(snapshot)).toEqual([0.1, 16777217]);
    });

    it("imports as undirected under defaultDirected false", async () => {
        const { snapshot } = await load("source,target\na,b\n", { defaultDirected: false });
        expect(snapshot.directed).toBe(false);
    });

    it("reads a Gephi Type column per row with blank cells taking the default", async () => {
        const { snapshot, report } = await load("Source,Target,Type\na,b,\nb,c,undirected\nc,d,MUTUAL\n", {
            defaultDirected: true,
        });
        expect(snapshot.directed).toBe(true);
        expect(edgesOf(snapshot)).toEqual(["a->b", "b->c", "c->b", "c->d", "d->c"]);
        expect(column(snapshot, "edges", "graphty.directed")).toEqual([true, false, false, true, true]);
        expect(column(snapshot, "edges", "graphty.mutual")).toEqual([undefined, undefined, undefined, true, undefined]);
        expect(report.counts.expandedMixed).toBe(2);
        expect(snapshot.edges.get("Type")).toBeNull();
    });

    it("a first Undirected row makes the sink undirected and a later Directed row expands it once", async () => {
        const { snapshot, report } = await load("Source,Target,Type\na,b,Undirected\nb,c,Directed\n");
        expect(snapshot.directed).toBe(true);
        expect(edgesOf(snapshot)).toEqual(["a->b", "b->a", "b->c"]);
        expect(report.counts.expandedMixed).toBe(1);
        expect(report.counts.edges).toBe(3);
    });

    it("rejects an unknown Type word per row", async () => {
        const { snapshot, report } = await load("Source,Target,Type\na,b,Sideways\nb,c,Directed\n");
        expect(report.issues[0]).toMatchObject({ code: BAD_TYPE_CODE, line: 2 });
        expect(edgesOf(snapshot)).toEqual(["b->c"]);
    });

    it("treats Type as an attribute outside the Gephi dialect and honours typeColumn", async () => {
        const generic = await load("source,target,Type\na,b,Undirected\n");
        expect(generic.snapshot.directed).toBe(true);
        expect(column(generic.snapshot, "edges", "Type")).toEqual(["Undirected"]);
        const explicit = await load("source,target,kind\na,b,Undirected\n", { typeColumn: "kind" });
        expect(explicit.snapshot.directed).toBe(false);
        const disabled = await load("Source,Target,Type\na,b,Undirected\n", { typeColumn: null });
        expect(disabled.snapshot.directed).toBe(true);
        expect(column(disabled.snapshot, "edges", "Type")).toEqual(["Undirected"]);
    });

    it("forces or refuses mixed direction per onMixedDirection", async () => {
        const text = "Source,Target,Type\na,b,Directed\nb,c,Undirected\n";
        const forced = await load(text, { onMixedDirection: "directed" });
        expect(forced.snapshot.edgeCount).toBe(2);
        expect(codes(forced.report)).toEqual([DIRECTION_FORCED_CODE]);
        const undirected = await load(text, { onMixedDirection: "undirected" });
        expect(undirected.snapshot.directed).toBe(false);
        expect(undirected.snapshot.edgeCount).toBe(2);
        const err = await failure(text, { onMixedDirection: "error" });
        expect(err.report.issues[0]).toMatchObject({ code: MIXED_DIRECTION_CODE, line: 3 });
        expect(err.report.counts.edges).toBe(1);
    });

    it("follows a locked sink and reports the refused direction", async () => {
        const { snapshot, report } = await load("source,target\na,b\n", { defaultDirected: false }, { locked: true });
        expect(snapshot.directed).toBe(true);
        expect(codes(report)).toEqual([DIRECTION_REFUSED_CODE]);
        expect(snapshot.edgeCount).toBe(2);
        expect(report.counts.expandedMixed).toBe(1);
    });

    it("leaves the sink's direction alone when the file has no edge rows", async () => {
        const builder = new GraphBuilder({ directed: false, weightDtype: "f64" });
        builder.addEdge("x", "y");
        await csvImporter.import("source,target\n", builder);
        expect(builder.directed).toBe(false);
        expect(builder.edgeCount).toBe(1);
    });

    it("reports builder-policy options the sink does not honour", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        const report = await csvImporter.import("source,target\na,b\n", builder, {
            addMissingNodes: false,
            duplicateEdges: "sum",
            selfLoops: "keep",
            weightDtype: "f64",
        });
        expect(report.issues.map((i) => [i.code, i.element])).toEqual([
            [SINK_OPTION_CODE, "addMissingNodes"],
            [SINK_OPTION_CODE, "duplicateEdges"],
        ]);
    });

    it("records an unknown endpoint when the sink refuses to create nodes", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64", addMissingNodes: false });
        const report = await csvImporter.import("source,target\na,b\nb,c\n", builder, { nodes: "id\na\nb\n" });
        expect(report.issues.map((i) => [i.code, i.category, i.line])).toEqual([
            ["E_UNKNOWN_NODE", "missing-value", 3],
        ]);
        expect(builder.edgeCount).toBe(1);
        expect(report.counts.skippedEdges).toBe(1);
    });

    it("rejects an option outside its set before reading", async () => {
        const builder = new GraphBuilder({ directed: true });
        const bad = [
            { ids: "weird" },
            { header: "yes" },
            { table: "graph" },
            { typeColumn: 1.5 },
            { onMixedDirection: "flip" },
        ] as unknown as CommonImportOptions[];
        for (const options of bad) {
            await expect(csvImporter.import("source,target\na,b\n", builder, options)).rejects.toBeInstanceOf(
                GraphFormatError,
            );
        }
    });
});

describe("csvImporter: attribute columns", () => {
    it("infers bool, i32, f64 and string columns and widens per column", async () => {
        const text = "source,target,flag,count,ratio,name,mixed\na,b,true,1,0.5,x,1\nb,c,false,2,2,y,01\n";
        const { snapshot } = await load(text);
        const dtypes = Object.fromEntries(
            ["flag", "count", "ratio", "name", "mixed"].map((n) => [n, snapshot.edges.get(n)?.dtype]),
        );
        expect(dtypes).toEqual({ flag: "bool", count: "i32", ratio: "f64", name: "string", mixed: "string" });
        expect(column(snapshot, "edges", "flag")).toEqual([true, false]);
        expect(column(snapshot, "edges", "mixed")).toEqual(["1", "01"]);
    });

    it("leaves empty cells unset and keeps an all-empty column as a nullable string column", async () => {
        const { snapshot } = await load("source,target,a,b\nx,y,,\ny,z,5,\n");
        expect(column(snapshot, "edges", "a")).toEqual([undefined, 5]);
        const b = snapshot.edges.get("b");
        expect(b?.dtype).toBe("string");
        expect(b?.nullCount).toBe(2);
    });

    it("makes a low-cardinality text column a dict and a high-cardinality one a string", async () => {
        const rows = ["source,target,kind,name"];
        for (let i = 0; i < 40; i++) {
            rows.push(`n${i},n${i + 1},${i % 3 === 0 ? "friend" : "foe"},name${i}`);
        }
        const { snapshot } = await load(`${rows.join("\n")}\n`);
        expect(snapshot.edges.get("kind")?.dtype).toBe("dict");
        expect(snapshot.edges.get("name")?.dtype).toBe("string");
        expect(column(snapshot, "edges", "kind").slice(0, 3)).toEqual(["friend", "foe", "foe"]);
    });

    it("never makes a dict of a column with numeric text", async () => {
        const rows = ["source,target,year"];
        for (let i = 0; i < 40; i++) {
            rows.push(`n${i},n${i + 1},${2000 + (i % 2)}`);
        }
        const { snapshot } = await load(`${rows.join("\n")}\n`);
        expect(snapshot.edges.get("year")?.dtype).toBe("i32");
    });

    it("decides the dict after the sample and keeps later numeric text as dict members", async () => {
        const rows = ["source,target,kind"];
        for (let i = 0; i < 1030; i++) {
            let kind = "7";
            if (i < 1024) {
                kind = i % 2 === 0 ? "a" : "b";
            }
            rows.push(`n${i},n${i + 1},${kind}`);
        }
        const { snapshot } = await load(`${rows.join("\n")}\n`);
        const kind = snapshot.edges.get("kind");
        expect(kind?.dtype).toBe("dict");
        expect(kind?.value(1029)).toBe("7");
    });

    it("declares an edge id column (role id, unique) and label columns (role label) as strings", async () => {
        const { snapshot } = await load("Source,Target,Id,Label\na,b,7,seven\nb,c,8,\n", {
            nodes: "Id,Label\na,A\nb,2\nc,\n",
        });
        const id = snapshot.edges.byRole("id");
        expect(id?.meta.name).toBe("Id");
        expect(id?.dtype).toBe("string");
        expect(id?.meta.unique).toBe(true);
        expect(column(snapshot, "edges", "Id")).toEqual(["7", "8"]);
        expect(column(snapshot, "edges", "Label")).toEqual(["seven", undefined]);
        expect(snapshot.nodes.byRole("label")?.dtype).toBe("string");
        expect(column(snapshot, "nodes", "Label")).toEqual(["A", "2", undefined]);
        expect(snapshot.edgeIndexOf("8")).toBe(1);
    });

    it("keeps values without the role when the sink already holds the role elsewhere", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.declareEdgeColumn({ name: "name", dtype: "string", role: "label" });
        const report = await csvImporter.import("source,target,label\na,b,x\n", builder);
        expect(codes(report)).toEqual([ROLE_TAKEN_CODE]);
        const s = builder.freeze();
        expect(s.edges.get("label")?.meta.role).toBeNull();
        expect(s.edges.get("label")?.value(0)).toBe("x");
    });

    it("renames a role column the sink already declared differently", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.declareEdgeColumn({ name: "label", dtype: "i32" });
        const report = await csvImporter.import("source,target,label\na,b,x\n", builder);
        expect(codes(report)).toEqual([RENAMED_CODE]);
        const s = builder.freeze();
        expect(s.edges.get("label#3")?.meta.role).toBe("label");
    });

    it("writes cell text into a caller's string or dict column and parsed values into a numeric one", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.declareEdgeColumn({ name: "s", dtype: "string" });
        builder.declareEdgeColumn({ name: "d", dtype: "dict" });
        builder.declareEdgeColumn({ name: "n", dtype: "f64" });
        await csvImporter.import("source,target,s,d,n\na,b,1.50,007,1.50\n", builder);
        const s = builder.freeze();
        expect(s.edges.get("s")?.value(0)).toBe("1.50");
        expect(s.edges.get("d")?.value(0)).toBe("007");
        expect(s.edges.get("n")?.value(0)).toBe(1.5);
    });

    it("records a value a caller's typed column refuses and keeps the row", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        builder.declareEdgeColumn({ name: "n", dtype: "i32" });
        const report = await csvImporter.import("source,target,n\na,b,x\nb,c,4\n", builder);
        expect(report.issues.map((i) => [i.code, i.line, i.element])).toEqual([["E_COLUMN_TYPE", null, "n"]]);
        expect(builder.edgeCount).toBe(2);
        const s = builder.freeze();
        expect(s.edges.get("n")?.value(1)).toBe(4);
    });
});

describe("csvImporter: node tables", () => {
    it("imports a node table first, merges duplicates with a warning and reports blank ids", async () => {
        const { snapshot, report } = await load("source,target\nb,a\n", {
            nodes: "id,age,city\na,30,Paris\nb,,Rome\na,31,\n,5,x\n",
        });
        expect(snapshot.ids.toArray()).toEqual(["a", "b"]);
        expect(column(snapshot, "nodes", "age")).toEqual([31, undefined]);
        expect(column(snapshot, "nodes", "city")).toEqual(["Paris", "Rome"]);
        expect(report.issues.map((i) => [i.code, i.line, i.element])).toEqual([
            [DUPLICATE_NODE_CODE, 4, "a"],
            [MISSING_ID_CODE, 5, null],
        ]);
        expect(report.counts.nodes).toBe(2);
        expect(report.counts.skippedNodes).toBe(1);
    });

    it("resolves the id column by candidates or by option, and fails without one", async () => {
        const named = await load("source,target\n1,2\n", { nodes: "name,x\n1,a\n2,b\n" });
        expect(column(named.snapshot, "nodes", "x")).toEqual(["a", "b"]);
        const explicit = await load("source,target\n1,2\n", { nodes: "key1,x\n1,a\n2,b\n", idColumn: "key1" });
        expect(column(explicit.snapshot, "nodes", "x")).toEqual(["a", "b"]);
        const err = await failure("source,target\n1,2\n", { nodes: "foo,bar\n1,a\n" });
        expect(err.report.issues[0].code).toBe(NO_ID_COLUMN_CODE);
        const headerless = await load("source,target\n1,2\n", { nodes: "1,a\n2,b\n" });
        expect(column(headerless.snapshot, "nodes", "column2")).toEqual(["a", "b"]);
    });

    it("keeps node-table row order and lets the edges add missing nodes after it", async () => {
        const { snapshot, report } = await load("source,target\nz,a\n", { nodes: "id\nb\na\n" });
        expect(snapshot.ids.toArray()).toEqual(["b", "a", "z"]);
        expect(report.counts.nodes).toBe(3);
    });

    it("honours nodeIdFrom label and index", async () => {
        const byLabel = await load("source,target\nA,B\n", {
            nodes: "id,label\n1,A\n2,B\n",
            nodeIdFrom: "label",
        });
        expect(byLabel.snapshot.ids.toArray()).toEqual(["A", "B"]);
        expect(column(byLabel.snapshot, "nodes", "id")).toEqual([1, 2]);
        expect(column(byLabel.snapshot, "nodes", "label")).toEqual(["A", "B"]);
        const byIndex = await load("source,target\n0,1\n", { nodes: "id\nx\ny\n", nodeIdFrom: "index" });
        expect(byIndex.snapshot.ids.toArray()).toEqual([0, 1]);
        expect(column(byIndex.snapshot, "nodes", "id")).toEqual(["x", "y"]);
        const err = await failure("source,target\n0,1\n", { nodes: "id\nx\n", nodeIdFrom: "label" });
        expect(err.report.issues[0].code).toBe(NO_ID_COLUMN_CODE);
    });

    it("imports the main input as a node table under table nodes", async () => {
        const { snapshot } = await load("source,target,id\na,b,c\n", { table: "nodes" });
        expect(snapshot.ids.toArray()).toEqual(["c"]);
        expect(column(snapshot, "nodes", "source")).toEqual(["a"]);
    });

    it("reads a headerless single-column input as a node list", async () => {
        const { snapshot } = await load("alice\nbob\n");
        expect(snapshot.ids.toArray()).toEqual(["alice", "bob"]);
    });
});

describe("csvImporter: cancellation, progress and sniffing", () => {
    it("reports byte progress with the total for in-memory input", async () => {
        const calls: [number, number | undefined][] = [];
        const text = "source,target\na,b\n";
        await load(text, { onProgress: (done, total) => calls.push([done, total]) });
        expect(calls.at(-1)).toEqual([text.length, text.length]);
    });

    it("stops on an aborted signal with the abort reason", async () => {
        const controller = new AbortController();
        controller.abort();
        const builder = new GraphBuilder({ directed: true });
        await expect(
            csvImporter.import("source,target\na,b\n", builder, { signal: controller.signal }),
        ).rejects.toMatchObject({
            name: "AbortError",
        });
    });

    it("cancels a stream when the import aborts midway", async () => {
        let cancelled = false;
        let pulls = 0;
        const stream = new ReadableStream<Uint8Array>({
            pull(controller): void {
                pulls++;
                if (pulls === 1) {
                    controller.enqueue(new TextEncoder().encode("source,target,weight\na,b,bad\n"));
                } else {
                    controller.enqueue(new TextEncoder().encode(`n${pulls},n${pulls + 1},1\n`));
                }
            },
            cancel(): void {
                cancelled = true;
            },
        });
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        await expect(csvImporter.import(stream, builder, { errorLimit: 0 })).rejects.toBeInstanceOf(ImportError);
        expect(cancelled).toBe(true);
        expect(pulls).toBeLessThan(10000);
    });

    it("completes a quoted multi-line cell split across chunks and keeps the line count", async () => {
        const text = 'source,target,note\na,b,"multi\nline\ncell"\nc,,x\n';
        const { snapshot, report } = await load(textChunksOf(text, 3));
        expect(column(snapshot, "edges", "note")).toEqual(["multi\nline\ncell"]);
        expect(report.issues[0]).toMatchObject({ code: MISSING_ENDPOINT_CODE, line: 5 });
        expect(report.lossy).toEqual([]);
        expect(report.durationMs).toBeGreaterThanOrEqual(0);
    });

    it("sniffs CSV heads by their headers and refuses other formats", () => {
        const { sniff } = csvImporter;
        if (sniff === undefined) {
            throw new Error("no sniff");
        }
        const enc = (t: string): Uint8Array => new TextEncoder().encode(t);
        expect(sniff(enc("Source,Target,Weight\na,b,1\n"))).toBe(0.9);
        expect(sniff(enc("Id,Label\na,A\n"))).toBe(0.6);
        expect(sniff(enc("1\t2\n2\t3\n"))).toBe(0.3);
        expect(sniff(enc("hello\nworld\n"))).toBe(0);
        expect(sniff(enc(""))).toBe(0);
        expect(sniff(enc('<?xml version="1.0"?><gexf/>'))).toBe(0);
        expect(sniff(enc('{"nodes": []}'))).toBe(0);
        expect(sniff(enc("graph [\n node [ id 1 ]\n]"))).toBe(0);
        expect(sniff(enc("digraph G { a -> b }"))).toBe(0);
        expect(sniff(enc('*Vertices 3\n1 "a"\n'))).toBe(0);
        expect(sniff(enc('Creator "x"\ngraph [\n]'))).toBe(0);
        for (const entry of corpusFiles("csv")) {
            expect(sniff(readCorpusBytes("csv", entry.path)), entry.path).toBe(0.9);
        }
    });

    it("declares the format, extensions and mime types", () => {
        expect(csvImporter.format).toBe("csv");
        expect(csvImporter.extensions).toContain(".csv");
        expect(csvImporter.extensions).toContain(".tsv");
        expect(csvImporter.mimeTypes).toContain("text/csv");
    });

    it("exposes INVALID_INDEX for unknown ids after an import", async () => {
        const { snapshot } = await load("source,target\na,b\n");
        expect(snapshot.ids.indexOf("zzz")).toBe(INVALID_INDEX);
    });
});
