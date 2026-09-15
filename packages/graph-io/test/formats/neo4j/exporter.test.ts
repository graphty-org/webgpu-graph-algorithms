import { type ColumnDecl, GraphBuilder, GraphFormatError, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { DirectionResolver } from "../../../src/common/direction.js";
import { LOSS } from "../../../src/common/export.js";
import { ImportReportBuilder } from "../../../src/common/report.js";
import { decodeChunks } from "../../../src/common/writer.js";
import {
    ARRAY_DELIMITER_LOSS,
    DECLARED_TYPE_CHANGED_LOSS,
    ID_COLUMN_TAKEN_LOSS,
    ID_TEXT_COLLISION_LOSS,
    ID_TEXT_TYPE_LOSS,
    MULTIPLE_ID_PROPERTIES_LOSS,
    NEO4J_CAPABILITIES,
    neo4jExporter,
    type Neo4jExportOptions,
    UNDIRECTED_LOSS,
    WEIGHT_COLUMN_TAKEN_LOSS,
} from "../../../src/formats/neo4j/exporter.js";
import { neo4jImporter } from "../../../src/formats/neo4j/importer.js";
import { type CommonExportOptions, type LossNote } from "../../../src/types.js";

type Options = Neo4jExportOptions & CommonExportOptions;

function codes(notes: readonly LossNote[]): string[] {
    return notes.map((n) => n.code);
}

function note(notes: readonly LossNote[], code: string): LossNote {
    const found = notes.find((n) => n.code === code);
    if (found === undefined) {
        throw new Error(`no note ${code} in ${codes(notes).join(", ")}`);
    }
    return found;
}

function build(fill: (b: GraphBuilder) => void, directed = true): GraphSnapshot {
    const b = new GraphBuilder({ directed, weightDtype: "f64" });
    fill(b);
    return b.freeze();
}

async function imported(text: string): Promise<GraphSnapshot> {
    const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
    await neo4jImporter.import(text, b);
    return b.freeze();
}

function lines(text: string): string[] {
    return text.split("\n").filter((l) => l.length > 0);
}

async function exportError(snapshot: GraphSnapshot, options?: Options): Promise<GraphFormatError> {
    try {
        await neo4jExporter.exportToString(snapshot, options);
    } catch (err) {
        expect(err).toBeInstanceOf(GraphFormatError);
        return err as GraphFormatError;
    }
    throw new Error("expected export() to throw");
}

describe("neo4jExporter (design 8.5)", () => {
    it("declares the format and its capabilities", () => {
        expect(neo4jExporter.format).toBe("neo4j");
        expect(neo4jExporter.capabilities).toBe(NEO4J_CAPABILITIES);
        expect(NEO4J_CAPABILITIES).toMatchObject({
            mixedDirection: false,
            multiEdges: true,
            selfLoops: true,
            edgeIds: "none",
            idCharset: "any",
            components: false,
            lists: true,
            json: false,
            defaults: false,
            options: false,
            hierarchy: false,
            temporal: "none",
            graphAttributes: false,
            positions: false,
            viz: false,
        });
        // a dict column reads back as string (the header has no enumeration type)
        expect([...NEO4J_CAPABILITIES.dtypes]).toEqual(["f32", "f64", "i32", "bool", "string"]);
    });

    describe("output", () => {
        it("writes a node section and a relationship section", async () => {
            const s = build((b) => {
                b.addEdge("a", "b", 2.5);
                b.addEdge("b", "c");
                b.setNodeValue("name", 0, "Alice");
                b.setNodeValue("name", 1, "Bob");
                b.setEdgeValue("since", 0, 2015);
            });
            expect(neo4jExporter.check(s)).toEqual([]);
            const text = await neo4jExporter.exportToString(s);
            expect(lines(text)).toEqual([
                ":ID,name:string",
                "a,Alice",
                "b,Bob",
                "c,",
                ":START_ID,:END_ID,weight:double,since:int",
                "a,b,2.5,2015",
                "b,c,,",
            ]);
        });

        it("streams the same bytes as exportToString", async () => {
            const s = await imported(":ID,:LABEL,name\n1,P,x\n:START_ID,:END_ID,:TYPE\n1,1,SELF\n");
            const text = await neo4jExporter.exportToString(s);
            expect(await decodeChunks(neo4jExporter.export(s))).toBe(text);
            expect(text).toBe(":ID,:LABEL,name\n1,P,x\n:START_ID,:END_ID,:TYPE\n1,1,SELF\n");
        });

        it("quotes cells per RFC 4180 and writes a set empty string as a quoted empty cell", async () => {
            const s = build((b) => {
                b.addNode("a, b");
                b.addNode(" c");
                b.addNode("d");
                b.setNodeValue("text", 0, 'say "hi"');
                b.setNodeValue("text", 1, "two\nlines");
                b.setNodeValue("text", 2, "");
            });
            expect(lines(await neo4jExporter.exportToString(s, { part: "nodes" }))).toEqual([
                ":ID,text:string",
                '"a, b","say ""hi"""',
                '" c","two',
                'lines"',
                'd,""',
            ]);
        });

        it("writes only the requested part", async () => {
            const s = await imported(":ID\n1\n:START_ID,:END_ID\n1,1\n");
            expect(await neo4jExporter.exportToString(s, { part: "nodes" })).toBe(":ID\n1\n");
            expect(await neo4jExporter.exportToString(s, { part: "relationships" })).toBe(":START_ID,:END_ID\n1,1\n");
        });

        it("writes headers alone for an empty snapshot", async () => {
            const s = build(() => undefined);
            expect(await neo4jExporter.exportToString(s)).toBe(":ID\n:START_ID,:END_ID\n");
            const declared = build((b) => {
                b.declareNodeColumn({ name: "labels", dtype: "list", itemDtype: "dict", role: "labels" });
                b.declareNodeColumn({ name: "n", dtype: "i32" });
                b.declareEdgeColumn({ name: "type", dtype: "dict", role: "kind" });
                b.declareEdgeColumn({ name: "w", dtype: "f64" });
            });
            expect(await neo4jExporter.exportToString(declared, { idColumn: "key" })).toBe(
                "key:ID,:LABEL,n:int\n:START_ID,:END_ID,:TYPE,w:double\n",
            );
        });

        it("flattens a visual column with components and reports it", async () => {
            const s = build((b) => {
                b.addNode("n");
                b.declareNodeColumn({ name: "color", dtype: "u8", components: 4, role: "color" });
                b.setNodeValue("color", 0, [255, 0, 0, 128]);
            });
            const notes = neo4jExporter.check(s);
            // the role is lost (a plain property), then the dtype and stride notes of any other column
            expect(notes.map((n) => [n.code, n.column])).toEqual([
                [LOSS.VIZ, "color"],
                [LOSS.DTYPE, "color"],
                [LOSS.COMPONENTS, "color"],
            ]);
            expect(await neo4jExporter.exportToString(s, { part: "nodes" })).toBe(":ID,color:int[]\nn,255;0;0;128\n");
        });

        it("reports a visual json column and writes it as text", async () => {
            const s = build((b) => {
                b.addNode("n");
                b.declareNodeColumn({ name: "color", dtype: "json", role: "color" });
                b.setNodeValue("color", 0, { r: 1, g: 2, b: 3 });
            });
            expect(neo4jExporter.check(s).map((n) => [n.code, n.column])).toEqual([
                [LOSS.VIZ, "color"],
                [LOSS.JSON, "color"],
            ]);
            expect(await neo4jExporter.exportToString(s, { part: "nodes" })).toBe(
                ':ID,color:string\nn,"{""r"":1,""g"":2,""b"":3}"\n',
            );
        });

        it("writes labels and kind columns of unusual dtypes as text", async () => {
            const s = build((b) => {
                b.addEdge("a", "b");
                b.declareNodeColumn({ name: "label", dtype: "dict", role: "labels" });
                b.declareEdgeColumn({ name: "kind", dtype: "i32", role: "kind" });
                b.setNodeValue("label", 0, "Person");
                b.setEdgeValue("kind", 0, 7);
            });
            expect(await neo4jExporter.exportToString(s)).toBe(
                ":ID,:LABEL\na,Person\nb,\n:START_ID,:END_ID,:TYPE\na,b,7\n",
            );
        });

        it("uses the delimiter, array delimiter and quote options", async () => {
            const s = await imported(":ID,:LABEL,tags:string[]\n1,A;B,x;y\n");
            const text = await neo4jExporter.exportToString(s, {
                delimiter: "\t",
                arrayDelimiter: "|",
                quote: "'",
                part: "nodes",
            });
            expect(text).toBe(":ID\t:LABEL\ttags:string[]\n1\tA|B\tx|y\n");
        });

        it("restores declared types, temporal companions and points", async () => {
            const source =
                "id:ID,i:int,by:byte,sh:short,l:long,f:float,d:double,c:char,du:duration,dt:date,zdt:datetime,t:time,p:point,li:int[],ld:date[]\n" +
                "1,1,2,3,4,0.1,0.1,x,P1D,2020-01-02,2020-01-02T03:04:05+01:00,10:00:00Z,\"{x:1, y:2.5, crs:'cartesian'}\",1;2,2020-01-01;2020-01-02\n" +
                "2,,,,,,,,,,2020-01-02T02:04:05Z,,,,\n";
            const s = await imported(source);
            const text = await neo4jExporter.exportToString(s, { part: "nodes" });
            expect(lines(text)).toEqual([
                "id:ID,i:int,by:byte,sh:short,l:long,f:float,d:double,c:char,du:duration,dt:date,zdt:datetime,t:time,p:point,li:int[],ld:date[]",
                "1,1,2,3,4,0.1,0.1,x,P1D,2020-01-02,2020-01-02T03:04:05+01:00,10:00:00Z,\"{x:1, y:2.5, crs:'cartesian'}\",1;2,2020-01-01;2020-01-02",
                "2,,,,,,,,,,2020-01-02T02:04:05Z,,,,",
            ]);
            expect(neo4jExporter.check(s)).toEqual([]);
        });

        it("derives header types from dtypes when there is no compatible declaration", async () => {
            const s = build((b) => {
                b.addNode("n");
                const decls: ColumnDecl[] = [
                    { name: "f32", dtype: "f32" },
                    { name: "f64", dtype: "f64" },
                    { name: "i32", dtype: "i32" },
                    { name: "u32", dtype: "u32" },
                    { name: "u8", dtype: "u8" },
                    { name: "bool", dtype: "bool" },
                    { name: "dict", dtype: "dict" },
                    { name: "str", dtype: "string" },
                    { name: "json", dtype: "json" },
                    { name: "lst", dtype: "list", itemDtype: "f64" },
                    { name: "gexfInt", dtype: "i32", origin: { format: "gexf", type: "integer" } },
                    { name: "gexfLong", dtype: "f64", origin: { format: "gexf", type: "long" } },
                    { name: "gexfDate", dtype: "f64", origin: { format: "gexf", type: "date" } },
                    { name: "textLong", dtype: "string", origin: { format: "neo4j", type: "long" } },
                ];
                for (const decl of decls) {
                    b.declareNodeColumn(decl);
                }
                b.setNodeValue("f32", 0, 0.1);
                b.setNodeValue("f64", 0, 0.1);
                b.setNodeValue("i32", 0, -5);
                b.setNodeValue("u32", 0, 4000000000);
                b.setNodeValue("u8", 0, 255);
                b.setNodeValue("bool", 0, false);
                b.setNodeValue("dict", 0, "cat");
                b.setNodeValue("str", 0, "s");
                b.setNodeValue("json", 0, { a: [1, "x"] });
                b.setNodeValue("lst", 0, [1.5, 2]);
                b.setNodeValue("gexfInt", 0, 7);
                b.setNodeValue("gexfLong", 0, 9007199254740992);
                b.setNodeValue("gexfDate", 0, Date.UTC(2021, 0, 1));
                b.setNodeValue("textLong", 0, "123456789012345678901");
            });
            const text = await neo4jExporter.exportToString(s, { part: "nodes" });
            expect(lines(text)).toEqual([
                ":ID,f32:float,f64:double,i32:int,u32:long,u8:int,bool:boolean,dict:string,str:string,json:string,lst:double[],gexfInt:int,gexfLong:long,gexfDate:date,textLong:string",
                'n,0.1,0.1,-5,4000000000,255,false,cat,s,"{""a"":[1,""x""]}",1.5;2,7,9007199254740992,2021-01-01,123456789012345678901',
            ]);
            const notes = neo4jExporter.check(s, { part: "nodes" });
            expect(codes(notes).sort()).toEqual([LOSS.DTYPE, LOSS.DTYPE, LOSS.DTYPE, LOSS.JSON].sort());
            expect(notes.filter((n) => n.code === LOSS.DTYPE).map((n) => n.column)).toEqual(["u32", "u8", "dict"]);
        });

        it("flattens multi-component columns into arrays and reports it", async () => {
            const s = build((b) => {
                b.addNode("n");
                b.declareNodeColumn({ name: "pos", dtype: "f32", components: 3, role: "position" });
                b.declareNodeColumn({ name: "vec", dtype: "f64", components: 2 });
                b.declareNodeColumn({ name: "pairs", dtype: "list", itemDtype: "i32", itemComponents: 2 });
                b.setNodeValue("pos", 0, [1, 2, 3]);
                b.setNodeValue("vec", 0, [0.5, 1.5]);
                b.setNodeValue("pairs", 0, [
                    [1, 2],
                    [3, 4],
                ]);
            });
            const notes = neo4jExporter.check(s);
            expect(notes.filter((n) => n.code === LOSS.COMPONENTS).map((n) => n.column)).toEqual([
                "vec",
                "pos",
                "pairs",
            ]);
            const text = await neo4jExporter.exportToString(s, { part: "nodes" });
            expect(lines(text)).toEqual([":ID,pos:float[],vec:double[],pairs:int[]", "n,1;2;3,0.5;1.5,1;2;3;4"]);
        });

        it("writes explicit weights only, from the shadow column or the arc array", async () => {
            const shadow = await imported(":START_ID,:END_ID,weight:double\n1,2,0.1\n2,3,\n3,1,2\n");
            expect(lines(await neo4jExporter.exportToString(shadow, { part: "relationships" }))).toEqual([
                ":START_ID,:END_ID,weight:double",
                "1,2,0.1",
                "2,3,",
                "3,1,2",
            ]);
            const arcs = build((b) => {
                b.addEdge("a", "b", 0.5);
                b.addEdge("b", "a", 3);
            });
            expect(arcs.edges.byRole("weight")).toBeNull();
            expect(lines(await neo4jExporter.exportToString(arcs, { part: "relationships" }))).toEqual([
                ":START_ID,:END_ID,weight:double",
                "a,b,0.5",
                "b,a,3",
            ]);
            const f32 = new GraphBuilder({ directed: true, weightDtype: "f32" });
            f32.addEdge("a", "b", 0.1);
            expect(lines(await neo4jExporter.exportToString(f32.freeze(), { part: "relationships" }))).toEqual([
                ":START_ID,:END_ID,weight:double",
                "a,b,0.1",
            ]);
            const unweighted = build((b) => {
                b.addEdge("a", "b");
            });
            expect(lines(await neo4jExporter.exportToString(unweighted, { part: "relationships" }))).toEqual([
                ":START_ID,:END_ID",
                "a,b",
            ]);
        });

        it("honours weightColumn and idColumn", async () => {
            const s = build((b) => {
                b.addEdge("a", "b", 2);
            });
            const text = await neo4jExporter.exportToString(s, { weightColumn: "cost", idColumn: "key" });
            expect(lines(text)).toEqual(["key:ID", "a", "b", ":START_ID,:END_ID,cost:double", "a,b,2"]);
            const none = await neo4jExporter.exportToString(s, { weightColumn: null, part: "relationships" });
            expect(none).toBe(":START_ID,:END_ID\na,b\n");
            expect(neo4jExporter.check(s, { weightColumn: null })).toEqual([]);
        });

        it("groups nodes and relationships into sections by id space in index order", async () => {
            const s = await imported(
                "pid:ID(Person),name\np1,Ann\np2,Ben\nmid:ID(Movie),title\nm1,Film\n" +
                    ":START_ID(Person),:END_ID(Movie),:TYPE\np1,m1,ACTED_IN\n" +
                    ":START_ID(Person),:END_ID(Person),:TYPE\np1,p2,KNOWS\n" +
                    ":START_ID(Person),:END_ID(Movie),:TYPE\np2,m1,DIRECTED\n",
            );
            expect(neo4jExporter.check(s)).toEqual([]);
            expect(lines(await neo4jExporter.exportToString(s))).toEqual([
                "pid:ID(Person),name,title",
                "p1,Ann,",
                "p2,Ben,",
                "mid:ID(Movie),name,title",
                "m1,,Film",
                ":START_ID(Person),:END_ID(Movie),:TYPE",
                "p1,m1,ACTED_IN",
                ":START_ID(Person),:END_ID(Person),:TYPE",
                "p1,p2,KNOWS",
                ":START_ID(Person),:END_ID(Movie),:TYPE",
                "p2,m1,DIRECTED",
            ]);
        });

        it("writes a kind column and a labels column of any dtype", async () => {
            const s = build((b) => {
                b.addEdge("a", "b");
                b.declareNodeColumn({ name: "tag", dtype: "string", role: "labels" });
                b.declareEdgeColumn({ name: "rel", dtype: "string", role: "kind" });
                b.setNodeValue("tag", 0, "Person");
                b.setEdgeValue("rel", 0, "KNOWS");
            });
            expect(lines(await neo4jExporter.exportToString(s))).toEqual([
                ":ID,:LABEL",
                "a,Person",
                "b,",
                ":START_ID,:END_ID,:TYPE",
                "a,b,KNOWS",
            ]);
        });

        it("writes the static value of a dynamic column and skips graph attributes", async () => {
            const s = build((b) => {
                b.addNode("a");
                b.declareNodeColumn({ name: "price", dtype: "f64", dynamic: true });
                b.setNodeValue("price", 0, 2);
                b.setGraphValue("title", "graph");
            });
            const notes = neo4jExporter.check(s);
            expect(codes(notes)).toEqual([LOSS.DYNAMIC_VALUES, LOSS.GRAPH_ATTRIBUTES]);
            expect(await neo4jExporter.exportToString(s, { part: "nodes" })).toBe(":ID,price:double\na,2\n");
        });
    });

    describe("check() (design 8.5)", () => {
        it("reports an undirected snapshot and writes every edge once", async () => {
            const s = build((b) => {
                b.addEdge("a", "b");
                b.addEdge("b", "b");
            }, false);
            const notes = neo4jExporter.check(s);
            expect(note(notes, UNDIRECTED_LOSS)).toMatchObject({ column: null, count: 2 });
            expect(lines(await neo4jExporter.exportToString(s, { part: "relationships" }))).toEqual([
                ":START_ID,:END_ID",
                "a,b",
                "b,b",
            ]);
            expect(codes(neo4jExporter.check(s, { part: "nodes" }))).toEqual([]);
        });

        it("reports expanded mixed direction per onMixedDirection", async () => {
            const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
            const r = new DirectionResolver(b, new ImportReportBuilder("t", 10), "expand");
            r.setHeader(true);
            r.addEdge("a", "b", "directed");
            r.addEdge("b", "c", "undirected", 2);
            r.addEdge("c", "c", "undirected");
            const s = b.freeze();
            expect(s.edgeCount).toBe(4);

            const refused = neo4jExporter.check(s);
            expect(note(refused, LOSS.MIXED_DIRECTION_ERROR).count).toBe(2);
            const err = await exportError(s);
            expect(err.code).toBe("E_DIRECTED");
            expect(codes(neo4jExporter.check(s, { part: "nodes" }))).toEqual([]);
            expect(await neo4jExporter.exportToString(s, { part: "nodes" })).toBe(":ID\na\nb\nc\n");

            const folded = neo4jExporter.check(s, { onMixedDirection: "directed" });
            expect(note(folded, LOSS.MIXED_DIRECTION).count).toBe(2);
            expect(
                lines(await neo4jExporter.exportToString(s, { onMixedDirection: "directed", part: "relationships" })),
            ).toEqual([":START_ID,:END_ID,weight:double", "a,b,", "b,c,2", "c,c,"]);

            // "undirected" is honoured as far as Neo4j can: the pairs fold and every relationship is
            // directed, which the notes say
            const asUndirected = neo4jExporter.check(s, { onMixedDirection: "undirected" });
            expect(codes(asUndirected)).toEqual([LOSS.MIXED_DIRECTION, UNDIRECTED_LOSS]);
            expect(note(asUndirected, UNDIRECTED_LOSS).count).toBe(2);
            expect(
                lines(await neo4jExporter.exportToString(s, { onMixedDirection: "undirected", part: "relationships" })),
            ).toEqual([":START_ID,:END_ID,weight:double", "a,b,", "b,c,2", "c,c,"]);
        });

        it("reports and drops an edge id column", async () => {
            const s = build((b) => {
                b.addEdge("a", "b");
                b.declareEdgeColumn({ name: "id", dtype: "string", role: "id", unique: true });
                b.setEdgeValue("id", 0, "e0");
            });
            expect(note(neo4jExporter.check(s), LOSS.EDGE_IDS_DROPPED).column).toBe("id");
            expect(await neo4jExporter.exportToString(s, { part: "relationships" })).toBe(":START_ID,:END_ID\na,b\n");
        });

        it("reports ids whose text re-imports as another type", () => {
            const strings = build((b) => {
                b.addNode("1");
                b.addNode("x");
                b.addNode("01");
            });
            expect(note(neo4jExporter.check(strings), ID_TEXT_TYPE_LOSS).count).toBe(1);
            const numbers = build((b) => {
                b.addNode(1.5);
                b.addNode(2);
            });
            expect(note(neo4jExporter.check(numbers), ID_TEXT_TYPE_LOSS).count).toBe(1);
            const dense = build((b) => {
                b.addNode(1);
                b.addNode(2);
            });
            expect(neo4jExporter.check(dense)).toEqual([]);
        });

        it("refuses ids that share their text, for every part", async () => {
            const s = build((b) => {
                b.addNode(1);
                b.addNode("1");
                b.addNode("a");
            });
            expect(note(neo4jExporter.check(s), ID_TEXT_COLLISION_LOSS).count).toBe(1);
            expect(note(neo4jExporter.check(s, { part: "relationships" }), ID_TEXT_COLLISION_LOSS).count).toBe(1);
            const err = await exportError(s);
            expect(err.code).toBe("E_INVALID_ID");
            expect(err.details.reason).toBe("text collision");
            await expect(neo4jExporter.exportToString(s, { part: "relationships" })).rejects.toThrow(GraphFormatError);
        });

        it("refuses a weight column name an edge property already uses", async () => {
            const s = build((b) => {
                b.addEdge("a", "b", 2);
                b.setEdgeValue("weight", 0, "heavy");
            });
            expect(note(neo4jExporter.check(s), WEIGHT_COLUMN_TAKEN_LOSS).column).toBe("weight");
            expect((await exportError(s)).code).toBe("E_COLUMN_EXISTS");
            expect(codes(neo4jExporter.check(s, { part: "nodes" }))).toEqual([]);
            expect(codes(neo4jExporter.check(s, { weightColumn: "w" }))).toEqual([]);
            expect(lines(await neo4jExporter.exportToString(s, { weightColumn: "w", part: "relationships" }))).toEqual([
                ":START_ID,:END_ID,w:double,weight:string",
                "a,b,2,heavy",
            ]);
        });

        it("refuses an idColumn a node property already uses", async () => {
            const s = build((b) => {
                b.addNode("a");
                b.setNodeValue("key", 0, "k");
            });
            expect(note(neo4jExporter.check(s, { idColumn: "key" }), ID_COLUMN_TAKEN_LOSS).column).toBe("key");
            expect((await exportError(s, { idColumn: "key" })).code).toBe("E_COLUMN_EXISTS");
            expect(codes(neo4jExporter.check(s, { idColumn: "key", part: "relationships" }))).toEqual([]);
            const stored = await imported("pid:ID\n1\n");
            expect(note(neo4jExporter.check(stored, { idColumn: "pid" }), ID_COLUMN_TAKEN_LOSS).column).toBe("pid");
        });

        it("reports nodes with more than one stored-id column set", async () => {
            const s = await imported("a:ID\n1\nb:ID\n2\n");
            expect(neo4jExporter.check(s)).toEqual([]);
            const b = GraphBuilder.from(s);
            b.setNodeValue("b", 0, "1");
            const both = b.freeze();
            expect(note(neo4jExporter.check(both), MULTIPLE_ID_PROPERTIES_LOSS).count).toBe(1);
            expect(lines(await neo4jExporter.exportToString(both, { part: "nodes" }))).toEqual([
                "a:ID",
                "1",
                "b:ID",
                "2",
            ]);
        });

        it("writes a declared integer type as double when values are not integral", async () => {
            const s = build((b) => {
                b.addNode("n");
                b.declareNodeColumn({ name: "l", dtype: "f64", origin: { format: "neo4j", type: "long" } });
                b.declareNodeColumn({
                    name: "ll",
                    dtype: "list",
                    itemDtype: "f64",
                    origin: { format: "neo4j", type: "long[]" },
                });
                b.setNodeValue("l", 0, 1.5);
                b.setNodeValue("ll", 0, [1, 2.5]);
            });
            const notes = neo4jExporter.check(s);
            expect(notes.filter((n) => n.code === DECLARED_TYPE_CHANGED_LOSS).map((n) => n.column)).toEqual([
                "l",
                "ll",
            ]);
            expect(lines(await neo4jExporter.exportToString(s, { part: "nodes" }))).toEqual([
                ":ID,l:double,ll:double[]",
                "n,1.5,1;2.5",
            ]);
        });

        it("reports list items that contain the array delimiter", async () => {
            const s = build((b) => {
                b.addNode("n");
                b.addNode("m");
                b.declareNodeColumn({ name: "tags", dtype: "list", itemDtype: "string" });
                b.setNodeValue("tags", 0, ["a;b", "c"]);
                b.setNodeValue("tags", 1, ["d"]);
            });
            expect(note(neo4jExporter.check(s), ARRAY_DELIMITER_LOSS)).toMatchObject({ column: "tags", count: 1 });
            expect(neo4jExporter.check(s, { arrayDelimiter: "|" })).toEqual([]);
            expect(lines(await neo4jExporter.exportToString(s, { arrayDelimiter: "|", part: "nodes" }))).toEqual([
                ":ID,tags:string[]",
                "n,a;b|c",
                "m,d",
            ]);
        });

        it("reports json columns unless they hold points", async () => {
            const s = build((b) => {
                b.addNode("n");
                b.declareNodeColumn({ name: "p", dtype: "json", origin: { format: "neo4j", type: "point" } });
                b.declareNodeColumn({ name: "j", dtype: "json" });
                b.setNodeValue("p", 0, { x: 1, y: 2 });
                b.setNodeValue("j", 0, { x: 1 });
            });
            const notes = neo4jExporter.check(s);
            expect(notes.filter((n) => n.code === LOSS.JSON).map((n) => n.column)).toEqual(["j"]);
            expect(lines(await neo4jExporter.exportToString(s, { part: "nodes" }))).toEqual([
                ":ID,p:point,j:string",
                'n,"{x:1, y:2}","{""x"":1}"',
            ]);
        });

        it("writes a non-point value of a point column as JSON text", async () => {
            const s = build((b) => {
                b.addNode("n");
                b.addNode("m");
                b.declareNodeColumn({ name: "p", dtype: "json", origin: { format: "neo4j", type: "point" } });
                b.setNodeValue("p", 0, [1, 2]);
                b.setNodeValue("p", 1, { x: 1, flag: true });
            });
            expect(await neo4jExporter.exportToString(s, { part: "nodes" })).toBe(
                ':ID,p:point\nn,"[1,2]"\nm,"{""x"":1,""flag"":true}"\n',
            );
        });

        it("reports defaults, options, hierarchy, lifetimes, spells, open intervals and extension tables", () => {
            const s = build((b) => {
                b.addEdge("a", "b");
                b.declareNodeColumn({ name: "d", dtype: "i32", default: 1 });
                b.declareNodeColumn({ name: "o", dtype: "string", options: ["x", "y"] });
                b.declareNodeColumn({ name: "parent", dtype: "u32", role: "parent", refersTo: "node" });
                b.declareNodeColumn({ name: "start", dtype: "f64", role: "start" });
                b.declareEdgeColumn({
                    name: "spells",
                    dtype: "list",
                    itemDtype: "f64",
                    itemComponents: 2,
                    role: "spells",
                });
                b.declareEdgeColumn({ name: "open", dtype: "u8", role: "open" });
                b.setNodeValue("d", 0, 2);
                b.setNodeValue("o", 0, "x");
                b.setNodeValue("parent", 1, 0);
                b.setNodeValue("start", 0, 1);
                b.setEdgeValue("spells", 0, [[1, 2]]);
                b.setEdgeValue("open", 0, 1);
                const t = b.addExtensionTable("temporal:node:price", [
                    { name: "row", dtype: "u32" },
                    { name: "value", dtype: "f64" },
                ]);
                b.addExtensionRow(t, [0, 1]);
                const x = b.addExtensionTable("other", [{ name: "v", dtype: "i32" }]);
                b.addExtensionRow(x, [1]);
            });
            const notes = neo4jExporter.check(s);
            expect(codes(notes).sort()).toEqual(
                [
                    LOSS.DEFAULT,
                    LOSS.OPTIONS,
                    LOSS.HIERARCHY,
                    LOSS.TEMPORAL,
                    LOSS.SPELLS,
                    LOSS.OPEN_INTERVAL,
                    LOSS.DYNAMIC_VALUES,
                    LOSS.EXTENSION_TABLE,
                ].sort(),
            );
            expect(note(notes, LOSS.HIERARCHY).column).toBe("parent");
            expect(note(notes, LOSS.SPELLS).column).toBe("spells");
            const nodesOnly = neo4jExporter.check(s, { part: "nodes" });
            expect(codes(nodesOnly).sort()).toEqual(
                [
                    LOSS.DEFAULT,
                    LOSS.OPTIONS,
                    LOSS.HIERARCHY,
                    LOSS.TEMPORAL,
                    LOSS.DYNAMIC_VALUES,
                    LOSS.EXTENSION_TABLE,
                ].sort(),
            );
            const edgesOnly = neo4jExporter.check(s, { part: "relationships" });
            expect(codes(edgesOnly).sort()).toEqual(
                [LOSS.SPELLS, LOSS.OPEN_INTERVAL, LOSS.DYNAMIC_VALUES, LOSS.EXTENSION_TABLE].sort(),
            );
        });

        it("skips the columns it reports as unwritable", async () => {
            const s = build((b) => {
                b.addEdge("a", "b");
                b.declareNodeColumn({ name: "parent", dtype: "u32", role: "parent", refersTo: "node" });
                b.declareNodeColumn({ name: "start", dtype: "f64", role: "start" });
                b.declareNodeColumn({ name: "graphty.originalId", dtype: "string", role: "originalId" });
                b.declareEdgeColumn({
                    name: "spells",
                    dtype: "list",
                    itemDtype: "f64",
                    itemComponents: 2,
                    role: "spells",
                });
                b.declareEdgeColumn({ name: "open", dtype: "u8", role: "open" });
                b.declareEdgeColumn({ name: "graphty.mutual", dtype: "bool", role: "mutual" });
                b.setNodeValue("parent", 1, 0);
                b.setNodeValue("start", 0, 1);
                b.setNodeValue("graphty.originalId", 0, "A");
                b.setEdgeValue("spells", 0, [[1, 2]]);
                b.setEdgeValue("open", 0, 1);
                b.setEdgeValue("graphty.mutual", 0, true);
            });
            expect(await neo4jExporter.exportToString(s)).toBe(":ID\na\nb\n:START_ID,:END_ID\na,b\n");
        });

        it("returns a frozen list and never writes anything", () => {
            const s = build((b) => {
                b.addNode(1);
                b.addNode("1");
            });
            const notes = neo4jExporter.check(s);
            expect(Object.isFrozen(notes)).toBe(true);
            expect(notes.every((n) => Object.isFrozen(n))).toBe(true);
        });
    });

    describe("options", () => {
        it.each([
            [{ part: "edges" as never }],
            [{ arrayDelimiter: ":" as never }],
            [{ delimiter: ";", arrayDelimiter: ";" as const }],
            [{ delimiter: "" }],
            [{ quote: "''" }],
            [{ weightColumn: "" }],
            [{ idColumn: "" }],
            [{ sanitizeIds: "rename" as never }],
        ])("rejects %j", (options) => {
            const s = build((b) => {
                b.addNode("a");
            });
            expect(() => neo4jExporter.check(s, options)).toThrow(GraphFormatError);
            expect(() => neo4jExporter.export(s, options)).toThrow(GraphFormatError);
        });

        it("accepts sanitizeIds: mangle as a no-op (every id is representable)", async () => {
            const s = build((b) => {
                b.addNode("with space & punct!");
            });
            expect(neo4jExporter.check(s, { sanitizeIds: "mangle" })).toEqual([]);
            expect(await neo4jExporter.exportToString(s, { sanitizeIds: "mangle", part: "nodes" })).toBe(
                ":ID\nwith space & punct!\n",
            );
        });
    });
});
