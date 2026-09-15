/**
 * Semantic audit of the Neo4j (neo4j-admin import CSV) importer against the corpus fixtures
 * themselves (facts derived here by manual line parsing of the raw text) and against the quirks
 * graphty-element's CSVDataSource neo4j branch and research note 07 sections 2.6 / 2.10 describe:
 * header typing, id spaces, labels, arrays, empty vs quoted-empty cells, :IGNORE, temporal types,
 * multi-section files. Tests named "DEFECT:" pin a defect and fail until the importer is fixed.
 */

import { type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { importGraph, type ImportGraphOptions, type ImportGraphResult } from "../../src/registry.js";
import { readCorpusBytes, readCorpusText } from "../helpers/corpus.js";

/** The byte-order mark, built from its code so this file stays ASCII. */
const BOM = String.fromCharCode(0xfeff);

async function load(name: string, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
    return importGraph(readCorpusBytes("neo4j", name), { format: "neo4j", ...options });
}

async function parse(text: string, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
    return importGraph(text, { format: "neo4j", ...options });
}

function endpoints(snapshot: GraphSnapshot, e: number): [string | number, string | number] {
    const list = snapshot.edgeList();
    return [snapshot.ids.idOf(list.src[e]), snapshot.ids.idOf(list.dst[e])];
}

function edgeStrings(snapshot: GraphSnapshot): string[] {
    const out: string[] = [];
    for (let e = 0; e < snapshot.edgeCount; e++) {
        out.push(endpoints(snapshot, e).join("->"));
    }
    return out;
}

/** Values of a column over rows 0..n-1, "<unset>" for unset rows. */
function column(snapshot: GraphSnapshot, table: "nodes" | "edges", name: string): unknown[] {
    const t = snapshot[table];
    const out: unknown[] = [];
    for (let i = 0; i < t.rowCount; i++) {
        out.push(t.isSet(name, i) ? t.value(name, i) : "<unset>");
    }
    return out;
}

describe("Neo4j corpus facts: social-bundle.csv (node and relationship sections in one file)", () => {
    const text = readCorpusText("neo4j", "social-bundle.csv");

    it("the raw file: a typed node header, six users (one quoted multi-line), a relationship header, eight rows", () => {
        const lines = text.split("\n");
        expect(lines[0]).toBe("userId:ID,:LABEL,name,age:int,score:double,active:boolean,tags:string[],joined:date");
        expect(lines[8]).toBe(":START_ID,:END_ID,:TYPE,since:int,weight:double");
        expect(lines.slice(1, 8).join("\n")).toContain('"Zo');
        expect((text.match(/^u\d,(?!u\d,)/gm) ?? []).length).toBe(6);
        expect((text.match(/^u\d,u\d,/gm) ?? []).length).toBe(8);
        expect(text).toContain("u6,u6,LIKES,2021,7.5");
        expect(text).toContain('"Eve ""the"" Great"');
        expect(text).toContain('u4,,Dan,41,3,true,"",2021-06-30');
    });

    it("declares every property column from the header with its neo4j type in origin", async () => {
        const { snapshot, report } = await load("social-bundle.csv");
        expect(report.errorCount).toBe(0);
        const typeOf = (name: string): [string, string | null | undefined] => {
            const c = snapshot.nodes.require(name);
            return [c.dtype + (c.meta.itemDtype === null ? "" : `<${c.meta.itemDtype}>`), c.meta.origin?.type];
        };
        expect(typeOf("labels")).toEqual(["list<dict>", "LABEL"]);
        expect(snapshot.nodes.get("labels")?.meta.role).toBe("labels");
        expect(typeOf("name")).toEqual(["string", null]);
        expect(typeOf("age")).toEqual(["i32", "int"]);
        expect(typeOf("score")).toEqual(["f64", "double"]);
        expect(typeOf("active")).toEqual(["bool", "boolean"]);
        expect(typeOf("tags")).toEqual(["list<string>", "string[]"]);
        expect(typeOf("joined")).toEqual(["f64", "date"]);
        expect(snapshot.nodes.get("userId")?.meta.origin?.type).toBe("ID");
        expect(snapshot.edges.get("type")?.meta.role).toBe("kind");
        expect(snapshot.edges.get("since")?.dtype).toBe("i32");
    });

    it('reads every node cell: labels (multiple, none), quoted commas / quotes / line breaks, blanks unset, "" as empty', async () => {
        const { snapshot } = await load("social-bundle.csv");
        expect(snapshot.ids.toArray()).toEqual(["u1", "u2", "u3", "u4", "u5", "u6"]);
        expect(column(snapshot, "nodes", "labels")).toEqual([
            ["Person"],
            ["Person", "Admin"],
            ["Person"],
            "<unset>",
            ["Person"],
            ["Person"],
        ]);
        expect(column(snapshot, "nodes", "name")).toEqual([
            "Alice",
            "Bob",
            "Carol, Jr.",
            "Dan",
            'Eve "the" Great',
            `Zo${String.fromCharCode(0xeb)}\nLine two`,
        ]);
        expect(column(snapshot, "nodes", "age")).toEqual([34, 28, "<unset>", 41, 19, 55]);
        expect(column(snapshot, "nodes", "score")).toEqual([0.5, 1.25, "<unset>", 3, -2.5, 1000]);
        expect(column(snapshot, "nodes", "active")).toEqual([true, false, true, true, false, true]);
        expect(column(snapshot, "nodes", "tags")).toEqual([
            ["admin", "editor"],
            "<unset>",
            ["reader"],
            [],
            ["x", "y", "z"],
            ["solo"],
        ]);
        expect(column(snapshot, "nodes", "joined")).toEqual([
            Date.UTC(2020, 0, 15),
            Date.UTC(2019, 10, 2),
            "<unset>",
            Date.UTC(2021, 5, 30),
            Date.UTC(2018, 1, 28),
            Date.UTC(2022, 11, 31),
        ]);
    });

    it("reads every relationship: directed, typed, since with a blank, weights with blanks, one self-loop, one parallel pair", async () => {
        const { snapshot } = await load("social-bundle.csv");
        expect(snapshot.directed).toBe(true);
        expect(edgeStrings(snapshot)).toEqual([
            "u1->u2",
            "u2->u3",
            "u3->u1",
            "u1->u4",
            "u4->u5",
            "u5->u6",
            "u6->u6",
            "u1->u2",
        ]);
        expect(column(snapshot, "edges", "type")).toEqual([
            "KNOWS",
            "KNOWS",
            "FOLLOWS",
            "MANAGES",
            "KNOWS",
            "BLOCKS",
            "LIKES",
            "LIKES",
        ]);
        expect(column(snapshot, "edges", "since")).toEqual([2015, 2016, 2017, 2018, "<unset>", 2020, 2021, 2019]);
        const weight = snapshot.edges.byRole("weight");
        expect([0, 1, 2, 3, 4, 5, 6, 7].map((e) => (weight?.isSet(e) === true ? weight.value(e) : "<unset>"))).toEqual([
            0.5,
            "<unset>",
            2,
            1,
            0.25,
            "<unset>",
            7.5,
            "<unset>",
        ]);
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([0.5, 1, 2, 1, 0.25, 1, 7.5, 1]);
        expect(snapshot.selfLoopCount).toBe(1);
        expect(snapshot.flags.multigraph).toBe(true);
        expect(snapshot.multiplicity(0, 1)).toBe(2);
        expect(snapshot.edges.has("weight")).toBe(false);
    });
});

describe("Neo4j corpus facts: typed-properties.csv (every header type, :IGNORE, temporal companions)", () => {
    const text = readCorpusText("neo4j", "typed-properties.csv");

    it("the raw header declares 23 node columns including junk:IGNORE, and three LINK relationships", () => {
        const header = text.split("\n")[0].split(",");
        expect(header).toHaveLength(23);
        expect(header).toContain("junk:IGNORE");
        expect(header).toContain("l:long");
        expect(header).toContain("zdt:datetime");
        expect(text).toContain("9007199254740993");
        expect(text).toContain("2024-02-29T13:45:00+02:00");
        expect((text.match(/,LINK,/g) ?? []).length).toBe(3);
    });

    it("maps every declared type: bool, int, long (precision issue), float f32, double, string, char, byte, short, point json, lists", async () => {
        const { snapshot, report } = await load("typed-properties.csv");
        expect(snapshot.ids.toArray()).toEqual([1, 2, 3]);
        expect(column(snapshot, "nodes", "b")).toEqual([true, false, true]);
        expect(column(snapshot, "nodes", "i")).toEqual([42, -7, 0]);
        expect(column(snapshot, "nodes", "l")).toEqual([9007199254740992, -12, 0]);
        expect(report.issues.filter((i) => i.code === "W_PRECISION" && i.category === "precision")).toHaveLength(1);
        expect(snapshot.nodes.get("f")?.dtype).toBe("f32");
        expect(column(snapshot, "nodes", "f")).toEqual([0.5, Math.fround(0.001), "<unset>"]);
        expect(column(snapshot, "nodes", "d")).toEqual([0.1, 2.5e10, "<unset>"]);
        expect(column(snapshot, "nodes", "s")).toEqual(["hello", "multi\nline", ""]);
        expect(column(snapshot, "nodes", "c")).toEqual(["x", "y", "<unset>"]);
        expect(column(snapshot, "nodes", "by")).toEqual([127, "<unset>", 0]);
        expect(column(snapshot, "nodes", "sh")).toEqual([-32768, "<unset>", 0]);
        expect(snapshot.nodes.get("p")?.dtype).toBe("json");
        expect(column(snapshot, "nodes", "p")).toEqual([{ x: 1.5, y: 2, crs: "cartesian" }, "<unset>", "<unset>"]);
        expect(column(snapshot, "nodes", "li")).toEqual([[1, 2, 3], "<unset>", "<unset>"]);
        expect(column(snapshot, "nodes", "ld")).toEqual([[0.5, 1.5], "<unset>", "<unset>"]);
        expect(column(snapshot, "nodes", "lb")).toEqual([[true, false], "<unset>", "<unset>"]);
        expect(column(snapshot, "nodes", "ls")).toEqual([["a", "b", "c"], "<unset>", []]);
        expect(column(snapshot, "nodes", "untyped")).toEqual(["plain", "<unset>", ""]);
        expect(column(snapshot, "nodes", "du")).toEqual(["P1Y2M3D", "PT5M", "<unset>"]);
        expect(snapshot.nodes.get("du")?.meta.origin?.type).toBe("duration");
    });

    it("converts the temporal types to milliseconds and keeps text companions for non-canonical forms", async () => {
        const { snapshot } = await load("typed-properties.csv");
        expect(column(snapshot, "nodes", "dt")).toEqual([Date.UTC(2024, 1, 29), Date.UTC(1999, 11, 31), "<unset>"]);
        expect(column(snapshot, "nodes", "lt")).toEqual([(13 * 3600 + 45 * 60) * 1000, 0, "<unset>"]);
        expect(column(snapshot, "nodes", "t")).toEqual([
            (13 * 3600 + 45 * 60) * 1000,
            (23 * 3600 + 59 * 60 + 59.5) * 1000,
            "<unset>",
        ]);
        expect(column(snapshot, "nodes", "ldt")).toEqual([
            Date.UTC(2024, 1, 29, 13, 45),
            Date.UTC(1999, 11, 31, 23, 59, 59, 123),
            "<unset>",
        ]);
        expect(column(snapshot, "nodes", "zdt")).toEqual([
            Date.UTC(2024, 1, 29, 11, 45),
            Date.UTC(1999, 11, 31, 23, 59, 59),
            "<unset>",
        ]);
        const zdtText = snapshot.nodes.require("zdt.text");
        expect(zdtText.meta.role).toBe("timeText");
        expect(zdtText.meta.extra.for).toBe("zdt");
        expect(column(snapshot, "nodes", "zdt.text")).toEqual(["2024-02-29T13:45:00+02:00", "<unset>", "<unset>"]);
        const tText = snapshot.nodes.require("t.text");
        expect(tText.meta.extra.for).toBe("t");
        expect(column(snapshot, "nodes", "t.text")).toEqual(["<unset>", "23:59:59.5Z", "<unset>"]);
    });

    it("skips the :IGNORE column with a loss note and reads the typed relationship properties and weights", async () => {
        const { snapshot, report } = await load("typed-properties.csv");
        expect(snapshot.nodes.has("junk")).toBe(false);
        expect(report.lossy).toEqual([expect.objectContaining({ code: "W_NEO4J_IGNORED_COLUMNS", count: 1 })]);
        expect(edgeStrings(snapshot)).toEqual(["1->2", "2->3", "3->1"]);
        expect(column(snapshot, "edges", "type")).toEqual(["LINK", "LINK", "LINK"]);
        expect(column(snapshot, "edges", "d")).toEqual([0.25, "<unset>", 1e21]);
        expect(column(snapshot, "edges", "ls")).toEqual([["q", "r"], "<unset>", ["solo"]]);
        expect(column(snapshot, "edges", "dt")).toEqual([Date.UTC(2020, 4, 5), "<unset>", Date.UTC(2020, 4, 6)]);
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([1.5, 1, 0]);
        expect(snapshot.edges.byRole("weight")?.isSet(1)).toBe(false);
    });
});

describe("Neo4j corpus facts: movies-nodes.csv + movies-rels.csv (id spaces, stored ids, two sections each)", () => {
    it("reads two node sections with different id spaces and stored-id properties", async () => {
        const text = readCorpusText("neo4j", "movies-nodes.csv");
        expect(text.split("\n")[0]).toBe("movieId:ID(Movie),title,released:int,:LABEL");
        expect(text.split("\n")[4]).toBe("personId:ID(Person),name,born:int,:LABEL");
        const { snapshot, report } = await load("movies-nodes.csv");
        expect(report.errorCount).toBe(0);
        expect(snapshot.ids.toArray()).toEqual(["m1", "m2", "m3", "p1", "p2", "p3", "p4"]);
        expect(column(snapshot, "nodes", "idSpace")).toEqual([
            "Movie",
            "Movie",
            "Movie",
            "Person",
            "Person",
            "Person",
            "Person",
        ]);
        expect(snapshot.nodes.get("idSpace")?.meta.role).toBe("idSpace");
        expect(column(snapshot, "nodes", "labels")).toEqual([
            ["Movie"],
            ["Movie", "Sequel"],
            ["Movie"],
            ["Person", "Actor"],
            ["Person", "Actor"],
            ["Person", "Director"],
            ["Person", "Actor"],
        ]);
        expect(column(snapshot, "nodes", "title")).toEqual([
            "The Matrix",
            "Matrix, Reloaded",
            "Cloud Atlas",
            "<unset>",
            "<unset>",
            "<unset>",
            "<unset>",
        ]);
        expect(column(snapshot, "nodes", "released")).toEqual([
            1999,
            2003,
            2012,
            "<unset>",
            "<unset>",
            "<unset>",
            "<unset>",
        ]);
        expect(column(snapshot, "nodes", "born")).toEqual(["<unset>", "<unset>", "<unset>", 1964, 1967, 1965, 1956]);
        expect(snapshot.nodes.get("movieId")?.meta.origin?.namespace).toBe("Movie");
        expect(column(snapshot, "nodes", "movieId")).toEqual([
            "m1",
            "m2",
            "m3",
            "<unset>",
            "<unset>",
            "<unset>",
            "<unset>",
        ]);
        expect(column(snapshot, "nodes", "personId").slice(3)).toEqual(["p1", "p2", "p3", "p4"]);
    });

    it("joins the relationship file through the relationships option: six typed edges with role lists", async () => {
        const rels = readCorpusText("neo4j", "movies-rels.csv");
        expect(rels.split("\n")[0]).toBe(":START_ID(Person),:END_ID(Movie),:TYPE,roles:string[]");
        expect(rels).toContain("p4,m3,ACTED_IN,Zachry;Dr. Henry Goose");
        const { snapshot } = await load("movies-nodes.csv", {
            relationships: [readCorpusBytes("neo4j", "movies-rels.csv")],
        });
        expect(snapshot.nodeCount).toBe(7);
        expect(edgeStrings(snapshot)).toEqual(["p1->m1", "p1->m2", "p2->m1", "p4->m3", "p3->p1", "p2->p1"]);
        expect(column(snapshot, "edges", "type")).toEqual([
            "ACTED_IN",
            "ACTED_IN",
            "ACTED_IN",
            "ACTED_IN",
            "DIRECTED_WITH",
            "KNOWS",
        ]);
        expect(column(snapshot, "edges", "roles")).toEqual([
            ["Neo"],
            ["Neo"],
            ["Trinity"],
            ["Zachry", "Dr. Henry Goose"],
            "<unset>",
            "<unset>",
        ]);
        expect(snapshot.directed).toBe(true);
        const alone = await load("movies-rels.csv");
        expect(alone.snapshot.nodeCount).toBe(7);
        expect(alone.snapshot.ids.toArray()).toEqual(["p1", "m1", "m2", "p2", "p4", "m3", "p3"]);
        expect(alone.report.counts.nodes).toBe(0);
    });
});

describe("Neo4j corpus facts: karate-neo4j.csv and crlf-tabs.tsv", () => {
    it("karate: 34 numeric ids, one label, a club string, 78 TIES relationships", async () => {
        const text = readCorpusText("neo4j", "karate-neo4j.csv");
        const nodeRows = text
            .split(":START_ID")[0]
            .split("\n")
            .slice(1)
            .filter((l) => l.length > 0);
        const relRows = text
            .split(":START_ID,:END_ID,:TYPE\n")[1]
            .split("\n")
            .filter((l) => l.length > 0);
        expect(nodeRows).toHaveLength(34);
        expect(relRows).toHaveLength(78);
        expect(nodeRows.filter((r) => r.endsWith("Mr. Hi"))).toHaveLength(17);
        const { snapshot } = await load("karate-neo4j.csv");
        expect(snapshot.ids.kind).toBe("identity");
        expect(snapshot.ids.toArray()).toEqual(nodeRows.map((r) => Number(r.split(",")[0])));
        expect(snapshot.edgeCount).toBe(78);
        expect(snapshot.nodes.value("club", 0)).toBe("Mr. Hi");
        expect(snapshot.nodes.value("club", 33)).toBe("Officer");
        expect(snapshot.nodes.value("labels", 33)).toEqual(["Member"]);
        for (let e = 0; e < relRows.length; e++) {
            const [s, t, type] = relRows[e].split(",");
            expect(endpoints(snapshot, e)).toEqual([Number(s), Number(t)]);
            expect(snapshot.edges.value("type", e)).toBe(type);
        }
        expect(snapshot.degree()[snapshot.ids.requireIndex(34)]).toBe(17);
    });

    it("crlf-tabs.tsv with the tab delimiter: BOM, CRLF, blank line, {label:Thing} header option, quoted tab", async () => {
        const raw = readCorpusText("neo4j", "crlf-tabs.tsv");
        expect(raw.startsWith(BOM)).toBe(true);
        expect(raw).toContain("\r\n");
        expect(raw).toContain('c\t"Gam\tma"\t3');
        const { snapshot } = await load("crlf-tabs.tsv", { delimiter: "\t" });
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c"]);
        expect(column(snapshot, "nodes", "labels")).toEqual([["Thing"], ["Thing"], ["Thing"]]);
        expect(column(snapshot, "nodes", "name")).toEqual(["Alpha", "Beta", "Gam\tma"]);
        expect(column(snapshot, "nodes", "n")).toEqual([1, 2, 3]);
        expect(edgeStrings(snapshot)).toEqual(["a->b", "b->c"]);
        expect(column(snapshot, "edges", "type")).toEqual(["NEXT", "NEXT"]);
    });

    it("DEFECT: a .tsv file (an advertised extension) must import without a delimiter option", async () => {
        // The importer lists ".tsv" and "text/tab-separated-values" but always splits on "," unless
        // the caller passes delimiter, so importGraph(bytes, { filename: "x.tsv" }) sniffs neo4j and
        // then fails with E_HEADER on the tab-joined header cell.
        const { snapshot } = await importGraph(readCorpusBytes("neo4j", "crlf-tabs.tsv"), {
            filename: "crlf-tabs.tsv",
        });
        expect(snapshot.nodeCount).toBe(3);
        expect(snapshot.edgeCount).toBe(2);
    });
});

describe("Neo4j quirks from the CSVDataSource neo4j branch and research note 07", () => {
    it("reads relationship sections before node sections, blank lines between sections, CRLF and a BOM", async () => {
        const { snapshot } = await parse(`${BOM}:START_ID,:END_ID,:TYPE\r\n1,2,R\r\n\r\nid:ID,name\r\n1,a\r\n2,b\r\n`);
        expect(snapshot.ids.toArray()).toEqual([1, 2]);
        expect(column(snapshot, "nodes", "name")).toEqual(["a", "b"]);
        expect(edgeStrings(snapshot)).toEqual(["1->2"]);
    });

    it("applies one id rule to every section: canonical keeps 01 and 1.0 as strings; number merges with a warning; string keeps text", async () => {
        const canonical = await parse("id:ID,name\n1,a\n01,b\n1.0,c\n");
        expect(canonical.snapshot.ids.toArray()).toEqual([1, "01", "1.0"]);
        const numbers = await parse("id:ID,name\n1,a\n01,b\n", { ids: "number" });
        expect(numbers.snapshot.nodeCount).toBe(1);
        expect(numbers.report.issues.map((i) => i.code)).toEqual(
            expect.arrayContaining(["W_ID_MERGED", "W_DUPLICATE_NODE"]),
        );
        const strings = await parse("id:ID,name\n1,a\n01,b\n", { ids: "string" });
        expect(strings.snapshot.ids.toArray()).toEqual(["1", "01"]);
        const big = await parse("id:ID,name\n9223372036854775807,a\n");
        expect(big.snapshot.ids.idOf(0)).toBe("9223372036854775807");
    });

    it("refuses the same id in two id spaces with an explicit error rather than merging silently", async () => {
        const { snapshot, report } = await parse(
            "id:ID(A),name\n1,a1\nid:ID(B),name\n1,b1\n:START_ID(A),:END_ID(B),:TYPE\n1,1,R\n",
        );
        expect(snapshot.nodeCount).toBe(1);
        expect(report.issues.map((i) => i.code)).toContain("E_NEO4J_ID_SPACE_COLLISION");
        expect(snapshot.nodes.value("name", 0)).toBe("a1");
    });

    it('splits :LABEL and array cells on ; (or the arrayDelimiter option), quoted delimiters included, and "" as an empty list', async () => {
        const { snapshot } = await parse('id:ID,:LABEL,tags:string[]\n1,A;B,"a;b;c"\n2,,"x, y;z"\n3,A,""\n4,B,\n');
        expect(column(snapshot, "nodes", "labels")).toEqual([["A", "B"], "<unset>", ["A"], ["B"]]);
        expect(column(snapshot, "nodes", "tags")).toEqual([["a", "b", "c"], ["x, y", "z"], [], "<unset>"]);
        const pipes = await parse("id:ID,tags:string[]\n1,a|b\n", { arrayDelimiter: "|" });
        expect(pipes.snapshot.nodes.value("tags", 0)).toEqual(["a", "b"]);
    });

    it("skips :IGNORE columns with a loss note, warns on unknown types and unhandled header options, reads {label:} options", async () => {
        const ignore = await parse("id:ID,x:IGNORE,y\n1,skip,keep\n");
        expect(ignore.snapshot.nodes.has("x")).toBe(false);
        expect(ignore.snapshot.nodes.value("y", 0)).toBe("keep");
        expect(ignore.report.lossy.map((n) => n.code)).toContain("W_NEO4J_IGNORED_COLUMNS");
        const unknownType = await parse(":START_ID,:END_ID,:TYPE,since:foo\n1,2,R,x\n");
        expect(unknownType.report.issues.map((i) => i.code)).toContain("W_UNKNOWN_ATTR_TYPE");
        expect(unknownType.snapshot.edges.value("since", 0)).toBe("x");
        const options = await parse("id:ID{label:Thing},name\n1,a\n");
        expect(options.snapshot.nodes.value("labels", 0)).toEqual(["Thing"]);
        const idType = await parse("id:ID{id-type:string},name\n1,a\n");
        expect(idType.report.issues.map((i) => i.code)).toContain("W_NEO4J_HEADER_OPTION_IGNORED");
    });

    it("reads booleans as true / false / 1 / 0 (case-insensitive) and reports other texts", async () => {
        const { snapshot, report } = await parse("id:ID,b:boolean\n1,true\n2,TRUE\n3,1\n4,false\n5,x\n");
        expect(column(snapshot, "nodes", "b")).toEqual([true, true, true, false]);
        expect(snapshot.nodeCount).toBe(4);
        expect(report.issues.filter((i) => i.code === "E_COLUMN_TYPE")).toHaveLength(1);
    });

    it("reports malformed rows explicitly: wrong cell count, blank :ID, blank endpoint, duplicate id, two :ID columns, unnamed property", async () => {
        const counts = await parse("id:ID,name,age:int\n1,a\n2,b,3,extra\n3,c,4\n");
        expect(counts.snapshot.ids.toArray()).toEqual([3]);
        expect(counts.report.issues.filter((i) => i.code === "E_NEO4J_COLUMN_COUNT")).toHaveLength(2);
        const blankId = await parse("id:ID,name\n,a\n2,b\n");
        expect(blankId.report.issues.map((i) => i.code)).toContain("E_MISSING_ID");
        const blankEnd = await parse(":START_ID,:END_ID,:TYPE\n1,,R\n");
        expect(blankEnd.report.issues.map((i) => i.code)).toContain("E_MISSING_ENDPOINT");
        const dup = await parse("id:ID,name\n1,a\n1,b\n");
        expect(dup.report.issues.map((i) => i.code)).toContain("W_DUPLICATE_NODE");
        expect(dup.snapshot.nodes.value("name", 0)).toBe("b");
        await expect(parse("id:ID,other:ID\n1,2\n")).rejects.toMatchObject({
            report: { issues: [{ code: "E_NEO4J_HEADER" }] },
        });
        await expect(parse("id:ID,:string\n1,x\n")).rejects.toMatchObject({
            report: { issues: [{ code: "E_NEO4J_HEADER" }] },
        });
        await expect(parse("name,age\n1,x\n")).rejects.toMatchObject({
            report: { issues: [{ code: "E_NEO4J_HEADER" }] },
        });
    });

    it("keeps relationships directed (Neo4j has no undirected relationship) unless onMixedDirection says undirected", async () => {
        const directed = await parse(":START_ID,:END_ID,:TYPE\n1,2,R\n", { defaultDirected: false });
        expect(directed.snapshot.directed).toBe(true);
        const undirected = await parse(":START_ID,:END_ID,:TYPE\n1,2,R\n", { onMixedDirection: "undirected" });
        expect(undirected.snapshot.directed).toBe(false);
        expect(undirected.report.issues.map((i) => i.code)).toContain("W_DIRECTION_FORCED");
    });

    it("renames a property whose name collides with labels / type / idSpace and warns", async () => {
        const { snapshot, report } = await parse("id:ID,labels,:LABEL\n1,x,A\n:START_ID,:END_ID,:TYPE,type\n1,1,R,y\n");
        expect(snapshot.nodes.value("labels", 0)).toEqual(["A"]);
        expect(snapshot.nodes.value("labels#labels", 0)).toBe("x");
        expect(snapshot.edges.value("type", 0)).toBe("R");
        expect(snapshot.edges.value("type#type", 0)).toBe("y");
        expect(report.issues.filter((i) => i.code === "W_COLUMN_RENAMED")).toHaveLength(2);
    });

    it("reads a weight:double property as the weight (blank = unset) or another property under weightFrom", async () => {
        const { snapshot } = await parse(":START_ID,:END_ID,:TYPE,weight:double\n1,2,R,0.5\n2,1,R,\n");
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([0.5, 1]);
        expect(snapshot.edges.byRole("weight")?.isSet(1)).toBe(false);
        const cost = await parse(":START_ID,:END_ID,:TYPE,cost:double\n1,2,R,0.5\n", { weightFrom: "cost" });
        expect(Array.from(cost.snapshot.edgeList().weights ?? [])).toEqual([0.5]);
        const none = await parse(":START_ID,:END_ID,:TYPE,weight:double\n1,2,R,0.5\n", { weightFrom: null });
        expect(none.snapshot.flags.weighted).toBe(false);
        expect(none.snapshot.edges.value("weight", 0)).toBe(0.5);
    });

    it("reports a date in a non-ISO form and a point in an unknown syntax instead of guessing", async () => {
        const { snapshot, report } = await parse("id:ID,d:date,p:point\n1,20240229,\n2,,POINT(1 2)\n3,2024-02-29,\n");
        expect(report.issues.filter((i) => i.code === "E_COLUMN_TYPE")).toHaveLength(2);
        expect(snapshot.ids.toArray()).toEqual([3]);
    });
});
