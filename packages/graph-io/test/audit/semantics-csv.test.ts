/**
 * Semantic audit of the CSV importer against the corpus files themselves (facts derived here by
 * manual line splitting of the raw text) and against the quirks graphty-element's CSVDataSource /
 * csv-variant-detection and research note 07 section 2.6 describe (Gephi tables, SNAP and KONECT
 * comment headers, adjacency lists, Cytoscape interaction tables, paired node files). Tests named
 * "DEFECT:" pin a defect and fail until the importer is fixed.
 */

import { type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { importGraph, type ImportGraphOptions, type ImportGraphResult } from "../../src/registry.js";
import { readCorpusBytes, readCorpusText } from "../helpers/corpus.js";

/** The byte-order mark, built from its code so this file stays ASCII. */
const BOM = String.fromCharCode(0xfeff);

async function load(name: string, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
    return importGraph(readCorpusBytes("csv", name), { format: "csv", ...options });
}

async function parse(text: string, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
    return importGraph(text, { format: "csv", ...options });
}

/** A plain-comma table without quoting: header names and rows as cell arrays. */
function table(text: string): { header: string[]; rows: string[][] } {
    const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
    return { header: lines[0].split(","), rows: lines.slice(1).map((l) => l.split(",")) };
}

/** Node ids in first-appearance order over the source / target cells of every row. */
function firstAppearance(rows: readonly string[][], s: number, t: number): string[] {
    const seen: string[] = [];
    for (const row of rows) {
        for (const cell of [row[s], row[t]]) {
            if (!seen.includes(cell)) {
                seen.push(cell);
            }
        }
    }
    return seen;
}

function endpoints(snapshot: GraphSnapshot, e: number): [string, string] {
    const list = snapshot.edgeList();
    return [String(snapshot.ids.idOf(list.src[e])), String(snapshot.ids.idOf(list.dst[e]))];
}

function edgeStrings(snapshot: GraphSnapshot): string[] {
    const out: string[] = [];
    for (let e = 0; e < snapshot.edgeCount; e++) {
        out.push(endpoints(snapshot, e).join("->"));
    }
    return out;
}

describe("CSV corpus facts: gephi-format.csv (Source,Target,Type,Weight,Label; mixed direction)", () => {
    const text = readCorpusText("csv", "gephi-format.csv");
    const { header, rows } = table(text);

    it("the raw file has the Gephi header, 5 rows, 3 Directed and 2 Undirected, integral weights", () => {
        expect(header).toEqual(["Source", "Target", "Type", "Weight", "Label"]);
        expect(rows).toHaveLength(5);
        expect(rows.map((r) => r[2])).toEqual(["Directed", "Directed", "Undirected", "Directed", "Undirected"]);
        expect(rows.map((r) => r[3])).toEqual(["5.0", "3.0", "1.0", "2.0", "4.0"]);
        expect(rows.map((r) => r[4])).toEqual(["friendship", "colleague", "acquaintance", "neighbor", "colleague"]);
        expect(firstAppearance(rows, 0, 1)).toEqual(["Alice", "Bob", "Charlie", "David"]);
    });

    it("imports a directed graph whose two undirected rows are expanded into pairs (7 logical edges)", async () => {
        const { snapshot, report } = await load("gephi-format.csv");
        expect(report.errorCount).toBe(0);
        expect(snapshot.directed).toBe(true);
        expect(snapshot.nodeCount).toBe(4);
        expect(snapshot.ids.toArray()).toEqual(["Alice", "Bob", "Charlie", "David"]);
        expect(snapshot.edgeCount).toBe(7);
        expect(report.counts.expandedMixed).toBe(2);
        expect(edgeStrings(snapshot)).toEqual([
            "Alice->Bob",
            "Bob->Charlie",
            "Charlie->David",
            "David->Charlie",
            "David->Alice",
            "Alice->Charlie",
            "Charlie->Alice",
        ]);
        const directed = snapshot.edges.byRole("directed");
        const pair = snapshot.edges.byRole("pair");
        expect([0, 1, 2, 3, 4, 5, 6].map((e) => directed?.value(e))).toEqual([
            true,
            true,
            false,
            false,
            true,
            false,
            false,
        ]);
        expect(pair?.value(2)).toBe(3);
        expect(pair?.value(6)).toBe(5);
        expect(pair?.isSet(0)).toBe(false);
    });

    it("reads Weight into the weights and Label into the label role, mirrored rows unset", async () => {
        const { snapshot } = await load("gephi-format.csv");
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([5, 3, 1, 1, 2, 4, 4]);
        const label = snapshot.edges.byRole("label");
        expect(label?.meta.name).toBe("Label");
        expect([0, 1, 2, 4, 5].map((e) => label?.value(e))).toEqual([
            "friendship",
            "colleague",
            "acquaintance",
            "neighbor",
            "colleague",
        ]);
        expect(label?.isSet(3)).toBe(false);
        expect(label?.isSet(6)).toBe(false);
        expect(snapshot.edges.has("Type")).toBe(false);
        expect(snapshot.edges.has("Weight")).toBe(false);
    });
});

describe("CSV corpus facts: got-edges.csv with got-nodes.csv (paired Gephi tables)", () => {
    const edgesText = readCorpusText("csv", "got-edges.csv");
    const nodesText = readCorpusText("csv", "got-nodes.csv");
    const edges = table(edgesText);
    const nodes = table(nodesText);

    it("the raw files: 352 weighted rows without a Type column; 107 Id,Label rows", () => {
        expect(edges.header).toEqual(["Source", "Target", "Weight"]);
        expect(edges.rows).toHaveLength(352);
        expect(edges.rows[0]).toEqual(["Aemon", "Grenn", "5"]);
        expect(edges.rows[351]).toEqual(["Ygritte", "Rattleshirt", "9"]);
        expect(edges.rows.reduce((sum, r) => sum + Number(r[2]), 0)).toBe(4324);
        expect(nodes.header).toEqual(["Id", "Label"]);
        expect(nodes.rows).toHaveLength(107);
        expect(nodes.rows.every((r) => r[0] === r[1])).toBe(true);
        expect(firstAppearance(edges.rows, 0, 1)).toHaveLength(107);
    });

    it("imports the edge table alone as directed (Gephi's default when Type is absent) in first-appearance order", async () => {
        const { snapshot, report } = await load("got-edges.csv");
        expect(report.errorCount).toBe(0);
        expect(snapshot.directed).toBe(true);
        expect(snapshot.nodeCount).toBe(107);
        expect(snapshot.edgeCount).toBe(352);
        expect(snapshot.ids.toArray()).toEqual(firstAppearance(edges.rows, 0, 1));
        const list = snapshot.edgeList();
        let sum = 0;
        for (let e = 0; e < edges.rows.length; e++) {
            expect(endpoints(snapshot, e)).toEqual([edges.rows[e][0], edges.rows[e][1]]);
            expect(list.weights?.[e]).toBe(Number(edges.rows[e][2]));
            sum += list.weights?.[e] ?? 0;
        }
        expect(sum).toBe(4324);
        expect(snapshot.edges.byRole("weight")).toBeNull();
        expect(snapshot.selfLoopCount).toBe(0);
        expect(snapshot.flags.multigraph).toBe(false);
    });

    it("imports the node table through the nodes option with the same ids and a Label column", async () => {
        const { snapshot } = await load("got-edges.csv", { nodes: readCorpusBytes("csv", "got-nodes.csv") });
        expect(snapshot.nodeCount).toBe(107);
        expect(snapshot.edgeCount).toBe(352);
        expect(snapshot.ids.toArray()).toEqual(nodes.rows.map((r) => r[0]));
        const label = snapshot.nodes.byRole("label");
        expect(label?.meta.name).toBe("Label");
        expect(label?.nullCount).toBe(0);
        for (let i = 0; i < nodes.rows.length; i++) {
            expect(label?.value(i)).toBe(nodes.rows[i][1]);
        }
        expect(snapshot.degree()[snapshot.ids.requireIndex("Tyrion")]).toBe(36);
    });

    it("reads the node table on its own as a node list", async () => {
        const { snapshot } = await load("got-nodes.csv");
        expect(snapshot.nodeCount).toBe(107);
        expect(snapshot.edgeCount).toBe(0);
        expect(snapshot.ids.idOf(106)).toBe("Walton");
        expect(snapshot.nodes.value("Label", 106)).toBe("Walton");
    });
});

describe("CSV corpus facts: dolphins-medium.csv and simple-edges.csv", () => {
    it("dolphins-medium: 50 Undirected rows with numeric ids become an undirected graph of 31 canonical numbers", async () => {
        const { header, rows } = table(readCorpusText("csv", "dolphins-medium.csv"));
        expect(header).toEqual(["Source", "Target", "Type", "Weight"]);
        expect(rows).toHaveLength(50);
        expect(rows.every((r) => r[2] === "Undirected" && r[3] === "1")).toBe(true);
        const order = firstAppearance(rows, 0, 1);
        expect(order).toHaveLength(31);
        expect(order.slice(0, 5)).toEqual(["8", "3", "9", "5", "6"]);
        const { snapshot, report } = await load("dolphins-medium.csv");
        expect(report.counts.expandedMixed).toBe(0);
        expect(snapshot.directed).toBe(false);
        expect(snapshot.nodeCount).toBe(31);
        expect(snapshot.edgeCount).toBe(50);
        expect(snapshot.ids.toArray()).toEqual(order.map(Number));
        expect(snapshot.ids.kind).toBe("dense");
        expect(snapshot.flags.weighted).toBe(true);
        expect(snapshot.flags.allWeightsOne).toBe(true);
        for (let e = 0; e < rows.length; e++) {
            expect(endpoints(snapshot, e)).toEqual([rows[e][0], rows[e][1]]);
        }
        expect(snapshot.edges.byRole("directed")).toBeNull();
    });

    it("simple-edges: a lower-case generic edge list with fractional weights, directed by default", async () => {
        const { header, rows } = table(readCorpusText("csv", "simple-edges.csv"));
        expect(header).toEqual(["source", "target", "weight"]);
        expect(rows.map((r) => r[2])).toEqual(["1.0", "2.0", "1.5", "1.0", "0.5"]);
        const { snapshot } = await load("simple-edges.csv");
        expect(snapshot.directed).toBe(true);
        expect(snapshot.ids.toArray()).toEqual(["a", "b", "c", "d", "e"]);
        expect(edgeStrings(snapshot)).toEqual(["a->b", "b->c", "c->d", "d->e", "a->c"]);
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([1, 2, 1.5, 1, 0.5]);
        const undirected = await load("simple-edges.csv", { defaultDirected: false });
        expect(undirected.snapshot.directed).toBe(false);
    });
});

describe("CSV quirks from CSVDataSource, csv-variant-detection and research note 07", () => {
    it("reads the full Gephi edge table: Id as the edge id, Label, Type per row (any case, Mutual), Weight", async () => {
        const { snapshot } = await parse(
            "Source,Target,Type,Id,Label,Weight\na,b,directed,e1,first,1\nb,c,undirected,e2,second,2\nc,a,Mutual,e3,third,3\n",
        );
        expect(snapshot.edges.byRole("id")?.meta.name).toBe("Id");
        expect(snapshot.edges.byRole("id")?.value(0)).toBe("e1");
        expect(snapshot.edges.byRole("label")?.value(0)).toBe("first");
        expect(edgeStrings(snapshot)).toEqual(["a->b", "b->c", "c->b", "c->a", "a->c"]);
        expect(snapshot.edges.byRole("mutual")?.value(3)).toBe(true);
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([1, 2, 2, 3, 3]);
    });

    it("keeps Weight 0.1 and 16777217 exact in the f64 shadow (design 16.5) and a blank weight unset", async () => {
        const { snapshot } = await parse("Source,Target,Weight\na,b,0.1\nb,c,16777217\nc,d,\n");
        const weight = snapshot.edges.byRole("weight");
        expect(weight?.dtype).toBe("f64");
        expect([0, 1].map((e) => weight?.value(e))).toEqual([0.1, 16777217]);
        expect(weight?.isSet(2)).toBe(false);
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([Math.fround(0.1), 16777216, 1]);
    });

    it("keeps 01, 1 and 1.0 as distinct ids under the canonical rule, and the same rule across paired tables", async () => {
        const { snapshot } = await parse("source,target\n01,1\n1.0,1\n+1,-0\n");
        expect(snapshot.ids.toArray()).toEqual(["01", 1, "1.0", "+1", "-0"]);
        const paired = await parse("source,target\n01,2\n", { nodes: "id,name\n01,A\n2,B\n" });
        expect(paired.snapshot.ids.toArray()).toEqual(["01", 2]);
        expect(paired.snapshot.nodes.value("name", 0)).toBe("A");
        expect(paired.snapshot.edgeCount).toBe(1);
    });

    it("resolves source / target header variants: SOURCE, from/to, src/dst, and a WEIGHT column", async () => {
        for (const head of ["SOURCE,TARGET,WEIGHT", "from,to,weight", "src,dst,Weight"]) {
            const { snapshot } = await parse(`${head}\na,b,2\n`);
            expect(edgeStrings(snapshot)).toEqual(["a->b"]);
            expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([2]);
        }
        const value = await parse("source,target,value\na,b,2\n");
        expect(value.snapshot.flags.weighted).toBe(false);
        expect(value.snapshot.edges.value("value", 0)).toBe(2);
    });

    it("trims header cells, strips a BOM, reads CRLF, quoted embedded line breaks and sniffed ; / tab delimiters", async () => {
        const bom = await parse(`${BOM} source , target \r\na,b\r\n`);
        expect(edgeStrings(bom.snapshot)).toEqual(["a->b"]);
        const quoted = await parse('source,target,note\na,b,"line1\nline2"\n');
        expect(quoted.snapshot.edges.value("note", 0)).toBe("line1\nline2");
        const semicolon = await parse("source;target;weight\na;b;1\nb;c;2\n");
        expect(edgeStrings(semicolon.snapshot)).toEqual(["a->b", "b->c"]);
        const tab = await parse("source\ttarget\na\tb\n");
        expect(edgeStrings(tab.snapshot)).toEqual(["a->b"]);
        const spaces = await parse('"node a" "node b"\n"node b" "node c"\n');
        expect(spaces.snapshot.ids.toArray()).toEqual(["node a", "node b", "node c"]);
    });

    it("reads a headerless numeric edge list positionally (source, target, weight) as design 8.4 requires", async () => {
        const { snapshot } = await parse("1 2 0.5\n2 3 1.5\n");
        expect(snapshot.ids.toArray()).toEqual([1, 2, 3]);
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([0.5, 1.5]);
        const bare = await parse("1,2\n2,3\n");
        expect(bare.snapshot.edgeCount).toBe(2);
        expect(bare.snapshot.flags.weighted).toBe(false);
    });

    it("DEFECT: SNAP edge lists start with # comment lines (design 8.4, note 07 2.6) and must import", async () => {
        // snap.stanford.edu files: "# Directed graph ...", "# Nodes: N Edges: M", "# FromNodeId\tToNodeId",
        // then tab-separated integer pairs. The importer has no comment-line rule: the first # line
        // becomes an 11-column header and every data row is a field-count error, so nothing loads.
        const snap =
            "# Directed graph (each unordered pair of nodes is saved once)\n# Foo\n# Nodes: 3 Edges: 3\n# FromNodeId\tToNodeId\n1\t2\n2\t3\n3\t1\n";
        const { snapshot, report } = await parse(snap);
        expect(report.errorCount).toBe(0);
        expect(snapshot.ids.toArray()).toEqual([1, 2, 3]);
        expect(edgeStrings(snapshot)).toEqual(["1->2", "2->3", "3->1"]);
    });

    it("DEFECT: KONECT files start with % header lines (design 8.4 and 16.5, note 07 2.6) and must import", async () => {
        // konect.cc TSV: "% sym unweighted" / "% asym positive", "% M N1 N2", then "u v [w [t]]".
        const konect = "% asym positive\n% 2 3 3\n1 2 16777217 1000000\n2 3 0.1 1000001\n";
        const { snapshot } = await parse(konect);
        expect(snapshot.ids.toArray()).toEqual([1, 2, 3]);
        expect(edgeStrings(snapshot)).toEqual(["1->2", "2->3"]);
        expect(snapshot.edges.byRole("weight")?.value(0)).toBe(16777217);
        expect(snapshot.edges.byRole("weight")?.value(1)).toBe(0.1);
    });

    it("reports a # comment line in the body as a field-count error rather than silently misreading it", async () => {
        const { snapshot, report } = await parse("source,target\na,b\n# comment\nb,c\n");
        expect(snapshot.edgeCount).toBe(2);
        expect(report.issues.map((i) => i.code)).toContain("E_CSV_FIELD_COUNT");
    });

    it("does not read adjacency lists (CSVDataSource did) but never misreads them silently: every row is an issue", async () => {
        const { snapshot, report } = await parse("a b c\nb d\n");
        expect(snapshot.edgeCount).toBe(0);
        expect(report.errorCount).toBe(2);
        const weighted = await parse("a b:1.5 c:2\nb d:1\n");
        expect(weighted.snapshot.edgeCount).toBe(0);
        expect(weighted.report.errorCount).toBe(2);
    });

    it("reads a Cytoscape interaction table as an edge list with interaction as an attribute", async () => {
        const { snapshot } = await parse("source,interaction,target\na,pp,b\nb,pd,c\n");
        expect(edgeStrings(snapshot)).toEqual(["a->b", "b->c"]);
        expect([0, 1].map((e) => snapshot.edges.value("interaction", e))).toEqual(["pp", "pd"]);
    });

    it("routes neo4j-admin headers to the neo4j importer when sniffing, even with a .csv name", async () => {
        const rels = ":START_ID,:END_ID,:TYPE\n1,2,KNOWS\n2,3,KNOWS\n";
        const auto = await importGraph(rels, { filename: "rels.csv" });
        expect(auto.format).toBe("neo4j");
        expect(auto.snapshot.edges.byRole("kind")?.value(0)).toBe("KNOWS");
        const forced = await parse(rels);
        expect(forced.snapshot.edgeCount).toBe(2);
        expect(forced.snapshot.edges.has(":TYPE")).toBe(true);
    });

    it("reads a node list (id,label / Id,Label) alone and infers column dtypes per column", async () => {
        const list = await parse("Id,Label,age,flag\n1,A,30,true\n2,B,,false\n");
        expect(list.snapshot.nodeCount).toBe(2);
        expect(list.snapshot.ids.toArray()).toEqual([1, 2]);
        expect(list.snapshot.nodes.byRole("label")?.value(1)).toBe("B");
        expect(list.snapshot.nodes.get("age")?.dtype).toBe("i32");
        expect(list.snapshot.nodes.isSet("age", 1)).toBe(false);
        expect(list.snapshot.nodes.get("flag")?.dtype).toBe("bool");
        const widened = await parse("source,target,v\na,b,1\nb,c,01\n");
        expect(widened.snapshot.edges.get("v")?.dtype).toBe("string");
        expect([0, 1].map((e) => widened.snapshot.edges.value("v", e))).toEqual(["1", "01"]);
    });

    it("keeps self-loops and parallel rows, and reports blank endpoints and bad Type cells explicitly", async () => {
        const kept = await parse("source,target\na,a\na,b\na,b\n");
        expect(kept.snapshot.selfLoopCount).toBe(1);
        expect(kept.snapshot.flags.multigraph).toBe(true);
        const bad = await parse("Source,Target,Type\na,,Directed\nb,c,Sideways\nc,d,Directed\n");
        expect(bad.snapshot.edgeCount).toBe(1);
        expect(bad.report.issues.map((i) => i.code)).toEqual(
            expect.arrayContaining(["E_MISSING_ENDPOINT", "E_CSV_BAD_TYPE"]),
        );
    });

    it("refuses a header without endpoint or id columns, an empty input and non-UTF-8 bytes explicitly", async () => {
        await expect(parse("foo,bar\n1,2\n")).rejects.toMatchObject({
            code: "E_IMPORT",
            report: { issues: [{ code: "E_CSV_NO_ENDPOINT_COLUMNS" }] },
        });
        await expect(parse("")).rejects.toMatchObject({
            code: "E_IMPORT",
            report: { issues: [{ code: "E_EMPTY_INPUT" }] },
        });
        const latin1 = Uint8Array.from(
            `source,target\ncaf${String.fromCharCode(0xe9)},b\n`,
            (c) => c.charCodeAt(0) & 0xff,
        );
        await expect(importGraph(latin1, { format: "csv" })).rejects.toMatchObject({
            report: { issues: [{ code: "E_INVALID_UTF8" }] },
        });
    });

    it("keeps a Gephi timeset interval cell as text (no interval parsing) without pretending otherwise", async () => {
        const { snapshot } = await parse('Source,Target,timeset\na,b,"<[2007, 2010]>"\n');
        expect(snapshot.edges.get("timeset")?.dtype).toBe("string");
        expect(snapshot.edges.value("timeset", 0)).toBe("<[2007, 2010]>");
        expect(snapshot.edges.byRole("spells")).toBeNull();
    });
});
