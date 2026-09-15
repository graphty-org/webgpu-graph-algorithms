import {
    GraphBuilder,
    type GraphBuilderOptions,
    GraphFormatError,
    type GraphSnapshot,
    INVALID_INDEX,
} from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { PRECISION_CODE, RENAMED_CODE, UNKNOWN_TYPE_CODE } from "../../../src/common/attributes.js";
import { DIRECTION_FORCED_CODE, DIRECTION_REFUSED_CODE } from "../../../src/common/direction.js";
import { INVALID_UTF8_CODE } from "../../../src/common/input.js";
import { SINK_OPTION_CODE } from "../../../src/common/options.js";
import {
    COLUMN_COUNT_CODE,
    DUPLICATE_NODE_CODE,
    HEADER_CODE,
    HEADER_OPTION_CODE,
    ID_MERGED_CODE,
    ID_SPACE_COLLISION_CODE,
    ID_SPACE_COLUMN,
    IGNORED_COLUMNS_LOSS,
    LABELS_COLUMN,
    MISSING_ENDPOINT_CODE,
    MISSING_ID_CODE,
    neo4jImporter,
    type Neo4jImportOptions,
    ROLE_TAKEN_CODE,
    TYPE_COLUMN,
} from "../../../src/formats/neo4j/importer.js";
import { type CommonImportOptions, ImportError, type ImportInput, type ImportReport } from "../../../src/types.js";
import { inputShapes } from "../../helpers/corpus.js";
import { expectSameSnapshot } from "../../helpers/roundtrip.js";
import { neo4jText } from "./fixtures.js";

interface Imported {
    snapshot: GraphSnapshot;
    report: ImportReport;
    builder: GraphBuilder;
}

async function importText(
    input: ImportInput,
    options?: Neo4jImportOptions & CommonImportOptions,
    builderOptions: Partial<GraphBuilderOptions> = {},
): Promise<Imported> {
    const builder = new GraphBuilder({ directed: true, weightDtype: "f64", ...builderOptions });
    const report = await neo4jImporter.import(input, builder, options);
    return { snapshot: builder.freeze(), report, builder };
}

async function importError(
    input: ImportInput,
    options?: Neo4jImportOptions & CommonImportOptions,
): Promise<ImportError> {
    const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
    try {
        await neo4jImporter.import(input, builder, options);
    } catch (err) {
        expect(err).toBeInstanceOf(ImportError);
        return err as ImportError;
    }
    throw new Error("expected an ImportError");
}

function codes(report: ImportReport): string[] {
    return report.issues.map((i) => i.code);
}

const NODES = "id:ID,:LABEL,name,age:int\n1,Person,Alice,30\n2,Person;Admin,Bob,\n3,,Carol,41\n";
const RELS = ":START_ID,:END_ID,:TYPE,since:int,weight:double\n1,2,KNOWS,2015,2.5\n2,3,KNOWS,2016,\n3,1,LIKES,,0.5\n";

describe("neo4jImporter (design 8.4)", () => {
    it("declares the format", () => {
        expect(neo4jImporter.format).toBe("neo4j");
        expect(neo4jImporter.extensions).toEqual([".csv", ".tsv"]);
        expect(neo4jImporter.mimeTypes).toEqual(["text/csv", "text/tab-separated-values"]);
    });

    it("sniffs a header on the first line", () => {
        const encode = (s: string): Uint8Array => new TextEncoder().encode(s);
        expect(neo4jImporter.sniff?.(encode("id:ID,name\n1,Alice\n"))).toBe(0.95);
        expect(neo4jImporter.sniff?.(encode(":START_ID,:END_ID\n"))).toBe(0.95);
        expect(neo4jImporter.sniff?.(encode(":START_ID\t:END_ID\t:TYPE\r\n1\t2\tX"))).toBe(0.95);
        expect(neo4jImporter.sniff?.(encode("name;age:int;:ID"))).toBe(0.95);
        expect(neo4jImporter.sniff?.(encode(`${String.fromCharCode(0xfeff)}:ID\n`))).toBe(0.95);
        expect(neo4jImporter.sniff?.(encode("source,target\n1,2\n"))).toBe(0);
        expect(neo4jImporter.sniff?.(encode("Source,Target,Type\n"))).toBe(0);
        expect(neo4jImporter.sniff?.(encode("<graphml>"))).toBe(0);
        expect(neo4jImporter.sniff?.(new Uint8Array([0xff, 0xfe, 0x00]))).toBe(0);
    });

    describe("nodes", () => {
        it("pushes nodes with canonical ids, labels and typed properties", async () => {
            const { snapshot, report } = await importText(NODES);
            expect(snapshot.nodeCount).toBe(3);
            expect(snapshot.edgeCount).toBe(0);
            expect(snapshot.directed).toBe(true);
            expect(snapshot.ids.toArray()).toEqual([1, 2, 3]);
            expect(report.counts).toEqual({ nodes: 3, edges: 0, skippedNodes: 0, skippedEdges: 0, expandedMixed: 0 });
            expect(report.issues).toEqual([]);
            expect(report.lossy).toEqual([]);

            const labels = snapshot.nodes.requireTyped(LABELS_COLUMN, "list");
            expect(labels.meta.role).toBe("labels");
            expect(labels.meta.itemDtype).toBe("dict");
            expect(labels.meta.origin).toEqual({
                format: "neo4j",
                id: ":LABEL",
                title: null,
                type: "LABEL",
                namespace: null,
            });
            expect([...labels.sliceOf(0)]).toEqual(["Person"]);
            expect([...labels.sliceOf(1)]).toEqual(["Person", "Admin"]);
            expect(labels.isSet(2)).toBe(false);

            const id = snapshot.nodes.requireTyped("id", "string");
            expect(id.meta.origin).toEqual({ format: "neo4j", id: "id", title: null, type: "ID", namespace: null });
            expect(id.valueAt(0)).toBe("1");

            const name = snapshot.nodes.requireTyped("name", "string");
            expect(name.meta.origin?.type).toBeNull();
            expect(name.valueAt(2)).toBe("Carol");

            const age = snapshot.nodes.requireTyped("age", "i32");
            expect(age.meta.origin?.type).toBe("int");
            expect(age.value(0)).toBe(30);
            expect(age.isSet(1)).toBe(false);
            expect(age.value(2)).toBe(41);
        });

        it("keeps a bare :ID out of the properties", async () => {
            const { snapshot } = await importText(":ID,name\nx,Alice\n");
            expect(snapshot.nodes.names()).toEqual(["name"]);
            expect(snapshot.ids.toArray()).toEqual(["x"]);
        });

        it("records id spaces in the idSpace column and the stored id's namespace", async () => {
            const { snapshot } = await importText(neo4jText("movies-nodes.csv"));
            const space = snapshot.nodes.requireTyped(ID_SPACE_COLUMN, "dict");
            expect(space.meta.role).toBe("idSpace");
            expect(space.value(0)).toBe("Movie");
            expect(space.value(3)).toBe("Person");
            expect(space.dictionary).toEqual(["Movie", "Person"]);
            const movieId = snapshot.nodes.requireTyped("movieId", "string");
            expect(movieId.meta.origin).toEqual({
                format: "neo4j",
                id: "movieId",
                title: null,
                type: "ID",
                namespace: "Movie",
            });
            expect(movieId.valueAt(0)).toBe("m1");
            expect(movieId.isSet(3)).toBe(false);
            const personId = snapshot.nodes.requireTyped("personId", "string");
            expect(personId.meta.origin?.namespace).toBe("Person");
            expect(personId.valueAt(3)).toBe("p1");
        });

        it("reports an id declared in two id spaces and skips the later row", async () => {
            const text = ":ID(A),name\n1,a\n:ID(B),name\n1,b\n2,c\n";
            const { snapshot, report } = await importText(text);
            expect(snapshot.nodeCount).toBe(2);
            expect(snapshot.nodes.value("name", 0)).toBe("a");
            expect(report.counts.skippedNodes).toBe(1);
            expect(report.issues).toHaveLength(1);
            expect(report.issues[0]).toMatchObject({
                category: "validation-error",
                severity: "error",
                code: ID_SPACE_COLLISION_CODE,
                line: 4,
                element: "1",
            });
        });

        it("warns on a duplicate node and lets the later properties win", async () => {
            const { snapshot, report } = await importText(":ID,name\n1,a\n1,b\n");
            expect(snapshot.nodeCount).toBe(1);
            expect(snapshot.nodes.value("name", 0)).toBe("b");
            expect(report.counts.nodes).toBe(2);
            expect(report.issues[0]).toMatchObject({
                category: "merged",
                severity: "warning",
                code: DUPLICATE_NODE_CODE,
                line: 3,
            });
        });

        it("detects duplicates and collisions beyond the first thousand nodes", async () => {
            const rows = Array.from({ length: 1100 }, (_, i) => `n${i}`);
            const text = `:ID(A)\n${rows.join("\n")}\nn1050\n:ID(B)\nn1099\nfresh\n`;
            const { snapshot, report } = await importText(text);
            expect(snapshot.nodeCount).toBe(1101);
            expect(report.issues.map((i) => [i.code, i.element])).toEqual([
                [DUPLICATE_NODE_CODE, "n1050"],
                [ID_SPACE_COLLISION_CODE, "n1099"],
            ]);
        });

        it("does not treat a node created by a relationship as a duplicate", async () => {
            const { snapshot, report } = await importText(`${RELS}:ID,name\n1,Alice\n`);
            expect(snapshot.nodeCount).toBe(3);
            expect(snapshot.nodes.value("name", 0)).toBe("Alice");
            expect(codes(report)).toEqual([]);
        });

        it("applies the {label:...} header option and reports other options", async () => {
            const { snapshot, report } = await importText('":ID{label:Thing, id-type:string}",:LABEL\n1,Extra\n2,\n');
            const labels = snapshot.nodes.requireTyped(LABELS_COLUMN, "list");
            expect([...labels.sliceOf(0)]).toEqual(["Extra", "Thing"]);
            expect([...labels.sliceOf(1)]).toEqual(["Thing"]);
            expect(report.issues).toHaveLength(1);
            expect(report.issues[0]).toMatchObject({
                code: HEADER_OPTION_CODE,
                category: "unsupported",
                severity: "warning",
            });
            expect(report.issues[0].message).toContain("id-type:string");
        });

        it("counts :IGNORE columns in a loss note", async () => {
            const { snapshot, report } = await importText(":ID,x:IGNORE,name,:IGNORE\n1,skip,Alice,skip\n");
            expect(snapshot.nodes.names()).toEqual(["name"]);
            expect(report.lossy).toEqual([
                {
                    code: IGNORED_COLUMNS_LOSS,
                    message: "2 :IGNORE column(s) were skipped as the header instructs",
                    column: null,
                    count: 2,
                },
            ]);
        });

        it("reads a quoted empty cell as an empty value and an unquoted one as unset", async () => {
            const text = ':ID,s:string,i:int,l:string[],b:boolean\n1,"","","",""\n2,,,,\n';
            const { snapshot } = await importText(text);
            const s = snapshot.nodes.requireTyped("s", "string");
            expect(s.isSet(0)).toBe(true);
            expect(s.valueAt(0)).toBe("");
            expect(s.isSet(1)).toBe(false);
            expect(snapshot.nodes.isSet("i", 0)).toBe(false);
            const l = snapshot.nodes.requireTyped("l", "list");
            expect(l.isSet(0)).toBe(true);
            expect([...l.sliceOf(0)]).toEqual([]);
            expect(l.isSet(1)).toBe(false);
            expect(snapshot.nodes.isSet("b", 0)).toBe(false);
        });
    });

    describe("declared types (design 5.1)", () => {
        it("maps every neo4j-admin type", async () => {
            const { snapshot, report } = await importText(neo4jText("typed-properties.csv"));
            const { nodes } = snapshot;
            const dtypes = Object.fromEntries(nodes.names().map((n) => [n, nodes.require(n).dtype]));
            expect(dtypes).toEqual({
                id: "string",
                b: "bool",
                i: "i32",
                l: "f64",
                f: "f32",
                d: "f64",
                s: "string",
                c: "string",
                by: "i32",
                sh: "i32",
                p: "json",
                dt: "f64",
                lt: "f64",
                t: "f64",
                ldt: "f64",
                zdt: "f64",
                du: "string",
                li: "list",
                ld: "list",
                lb: "list",
                ls: "list",
                untyped: "string",
                "zdt.text": "string",
                "t.text": "string",
            });
            expect(nodes.require("li").meta.itemDtype).toBe("i32");
            expect(nodes.require("ld").meta.itemDtype).toBe("f64");
            expect(nodes.require("lb").meta.itemDtype).toBe("bool");
            expect(nodes.require("ls").meta.itemDtype).toBe("string");
            expect(nodes.require("l").meta.origin?.type).toBe("long");
            expect(nodes.require("li").meta.origin?.type).toBe("int[]");

            expect(nodes.value("b", 0)).toBe(true);
            expect(nodes.value("b", 1)).toBe(false);
            expect(nodes.value("b", 2)).toBe(true);
            expect(nodes.value("i", 0)).toBe(42);
            expect(nodes.value("l", 0)).toBe(9007199254740992);
            expect(nodes.value("f", 0)).toBe(0.5);
            expect(nodes.value("d", 0)).toBe(0.1);
            expect(nodes.value("d", 1)).toBe(2.5e10);
            expect(nodes.value("s", 1)).toBe("multi\nline");
            expect(nodes.value("s", 2)).toBe("");
            expect(nodes.value("c", 0)).toBe("x");
            expect(nodes.value("by", 0)).toBe(127);
            expect(nodes.value("sh", 0)).toBe(-32768);
            expect(nodes.value("p", 0)).toEqual({ x: 1.5, y: 2, crs: "cartesian" });
            expect(nodes.value("dt", 0)).toBe(Date.UTC(2024, 1, 29));
            expect(nodes.value("lt", 0)).toBe((13 * 60 + 45) * 60 * 1000);
            expect(nodes.value("t", 1)).toBe(((23 * 60 + 59) * 60 + 59) * 1000 + 500);
            expect(nodes.value("ldt", 1)).toBe(Date.UTC(1999, 11, 31, 23, 59, 59, 123));
            expect(nodes.value("zdt", 0)).toBe(Date.UTC(2024, 1, 29, 11, 45, 0));
            expect(nodes.value("du", 0)).toBe("P1Y2M3D");
            expect(nodes.value("li", 0)).toEqual([1, 2, 3]);
            expect(nodes.value("ld", 0)).toEqual([0.5, 1.5]);
            expect(nodes.value("lb", 0)).toEqual([true, false]);
            expect(nodes.value("ls", 0)).toEqual(["a", "b", "c"]);
            expect(nodes.value("ls", 2)).toEqual([]);
            expect(nodes.value("untyped", 0)).toBe("plain");
            expect(nodes.value("untyped", 2)).toBe("");

            // companions hold the source text only where the canonical form differs
            const zdtText = nodes.requireTyped("zdt.text", "string");
            expect(zdtText.meta.role).toBe("timeText");
            expect(zdtText.meta.extra).toEqual({ for: "zdt" });
            expect(zdtText.valueAt(0)).toBe("2024-02-29T13:45:00+02:00");
            expect(zdtText.isSet(1)).toBe(false);
            const tText = nodes.requireTyped("t.text", "string");
            expect(tText.meta.role).toBeNull();
            expect(tText.valueAt(1)).toBe("23:59:59.5Z");
            expect(tText.isSet(0)).toBe(false);
            expect(nodes.has("dt.text")).toBe(false);

            // long beyond 2^53
            const precision = report.issues.filter((i) => i.code === PRECISION_CODE);
            expect(precision).toHaveLength(1);
            expect(precision[0]).toMatchObject({ category: "precision", severity: "warning", line: 2, element: "1" });

            // relationships
            expect(snapshot.edgeCount).toBe(3);
            expect(snapshot.flags.weighted).toBe(true);
            const weights = snapshot.edges.byRole("weight");
            expect(weights).not.toBeNull();
            expect(weights?.isSet(0)).toBe(true);
            expect(weights?.isSet(1)).toBe(false);
            expect(weights?.isSet(2)).toBe(true);
            expect(weights?.value(0)).toBe(1.5);
            expect(weights?.value(2)).toBe(0);
            expect(snapshot.edges.value("d", 2)).toBe(1e21);
            expect(snapshot.edges.value("ls", 0)).toEqual(["q", "r"]);
            expect(snapshot.edges.value("dt", 0)).toBe(Date.UTC(2020, 4, 5));
            expect(snapshot.edges.isSet("dt", 1)).toBe(false);
        });

        it('keeps declared long columns as text under long: "string"', async () => {
            const { snapshot, report } = await importText(":ID,l:long\n1,9007199254740993\n", { long: "string" });
            const l = snapshot.nodes.requireTyped("l", "string");
            expect(l.valueAt(0)).toBe("9007199254740993");
            expect(l.meta.origin?.type).toBe("long");
            expect(codes(report)).toEqual([]);
        });

        it("keeps an unknown declared type as text and warns", async () => {
            const { snapshot, report } = await importText(":ID,x:uuid\n1,abc\n");
            expect(snapshot.nodes.require("x").dtype).toBe("string");
            expect(snapshot.nodes.require("x").meta.origin?.type).toBe("uuid");
            expect(report.issues[0]).toMatchObject({
                code: UNKNOWN_TYPE_CODE,
                category: "unsupported",
                severity: "warning",
                line: 1,
            });
        });

        it("shares a property column between sections with the same declaration", async () => {
            const text = ":ID,name,age:int\n1,a,1\n:ID,age:int,name\n2,2,b\n";
            const { snapshot, report } = await importText(text);
            expect(snapshot.nodes.names()).toEqual(["name", "age"]);
            expect(snapshot.nodes.value("age", 1)).toBe(2);
            expect(snapshot.nodes.value("name", 1)).toBe("b");
            expect(codes(report)).toEqual([]);
        });

        it("renames a property declared with another shape in a later section", async () => {
            const text = ":ID,age:int\n1,1\n:ID,age:string\n2,x\n";
            const { snapshot, report } = await importText(text);
            expect(snapshot.nodes.names()).toEqual(["age", "age#age"]);
            expect(snapshot.nodes.value("age#age", 1)).toBe("x");
            expect(report.issues[0]).toMatchObject({ code: RENAMED_CODE, category: "coercion", line: 3 });
        });

        it("renames a property that collides with a reserved column, and a reserved column that comes later", async () => {
            const rels = ":START_ID,:END_ID,:TYPE,type:string\n1,2,KNOWS,x\n";
            const first = await importText(rels);
            expect(first.snapshot.edges.names()).toEqual([TYPE_COLUMN, "type#type"]);
            expect(first.snapshot.edges.value(TYPE_COLUMN, 0)).toBe("KNOWS");
            expect(first.snapshot.edges.value("type#type", 0)).toBe("x");
            expect(first.report.issues[0].code).toBe(RENAMED_CODE);

            const later = ":START_ID,:END_ID,type:string\n1,2,x\n:START_ID,:END_ID,:TYPE\n2,3,KNOWS\n";
            const second = await importText(later);
            expect(second.snapshot.edges.names()).toEqual(["type", "type#:TYPE"]);
            expect(second.snapshot.edges.byRole("kind")?.meta.name).toBe("type#:TYPE");
            expect(second.snapshot.edges.value("type#:TYPE", 1)).toBe("KNOWS");
        });

        it("drops the reserved role when the sink already holds it", async () => {
            const builder = new GraphBuilder({ directed: true });
            builder.declareNodeColumn({ name: "tags", dtype: "list", itemDtype: "string", role: "labels" });
            const report = await neo4jImporter.import(":ID,:LABEL\n1,Person\n", builder);
            const snapshot = builder.freeze();
            expect(snapshot.nodes.require(LABELS_COLUMN).meta.role).toBeNull();
            expect(snapshot.nodes.value(LABELS_COLUMN, 0)).toEqual(["Person"]);
            expect(report.issues[0]).toMatchObject({ code: ROLE_TAKEN_CODE, category: "coercion" });
        });
    });

    describe("relationships", () => {
        it("pushes directed edges with type, weight and properties", async () => {
            const { snapshot, report } = await importText(NODES + RELS);
            expect(snapshot.edgeCount).toBe(3);
            expect(report.counts).toEqual({ nodes: 3, edges: 3, skippedNodes: 0, skippedEdges: 0, expandedMixed: 0 });
            const list = snapshot.edgeList();
            expect([...list.src]).toEqual([0, 1, 2]);
            expect([...list.dst]).toEqual([1, 2, 0]);
            const type = snapshot.edges.requireTyped(TYPE_COLUMN, "dict");
            expect(type.meta.role).toBe("kind");
            expect(type.dictionary).toEqual(["KNOWS", "LIKES"]);
            expect(type.value(2)).toBe("LIKES");
            expect(snapshot.edges.value("since", 0)).toBe(2015);
            expect(snapshot.edges.isSet("since", 2)).toBe(false);
            expect(snapshot.edges.names()).toEqual([TYPE_COLUMN, "since", "graphty.weight"]);
            const weights = snapshot.edges.byRole("weight");
            expect(weights?.value(0)).toBe(2.5);
            expect(weights?.isSet(1)).toBe(false);
            expect([...(list.weights ?? [])]).toEqual([2.5, 1, 0.5]);
        });

        it("creates missing endpoints by default and reports them when the sink refuses", async () => {
            const withMissing = await importText(RELS);
            expect(withMissing.snapshot.nodeCount).toBe(3);
            expect(withMissing.snapshot.ids.toArray()).toEqual([1, 2, 3]);

            const strict = await importText(`:ID\n1\n2\n${RELS}`, undefined, { addMissingNodes: false });
            expect(strict.snapshot.edgeCount).toBe(1);
            expect(strict.report.counts.skippedEdges).toBe(2);
            const unknown = strict.report.issues.filter((i) => i.code === "E_UNKNOWN_NODE");
            expect(unknown).toHaveLength(2);
            expect(unknown[0]).toMatchObject({
                category: "missing-value",
                severity: "error",
                line: 6,
                element: "2->3",
            });
        });

        it("honours weightFrom and reads no weight under null", async () => {
            const named = await importText(":START_ID,:END_ID,cost:double,weight:double\n1,2,3,4\n", {
                weightFrom: "cost",
            });
            expect(named.snapshot.edgeList().weights?.[0]).toBe(3);
            expect(named.snapshot.edges.value("weight", 0)).toBe(4);
            expect(named.snapshot.edges.has("cost")).toBe(false);

            const none = await importText(":START_ID,:END_ID,weight:double\n1,2,4\n", { weightFrom: null });
            expect(none.snapshot.flags.weighted).toBe(false);
            expect(none.snapshot.edges.value("weight", 0)).toBe(4);
        });

        it("reports header options on a relationship section", async () => {
            const { report } = await importText('":START_ID{id-type:string}",:END_ID\n1,2\n');
            expect(report.issues.map((i) => i.code)).toEqual([HEADER_OPTION_CODE]);
            expect(report.issues[0].element).toBe(":START_ID{id-type:string}");
        });

        it('skips a relationship whose endpoint is not an id under ids: "number"', async () => {
            const { snapshot, report } = await importText(":START_ID,:END_ID\nabc,1\n1,xyz\n1,2\n", { ids: "number" });
            expect(snapshot.edgeCount).toBe(1);
            expect(report.counts.skippedEdges).toBe(2);
            expect(report.issues.map((i) => [i.code, i.element])).toEqual([
                ["E_INVALID_ID", "abc"],
                ["E_INVALID_ID", "xyz"],
            ]);
        });

        it("keeps the companion text of a relationship temporal property", async () => {
            const { snapshot } = await importText(
                ":START_ID,:END_ID,at:datetime\n1,2,2020-01-01T00:00:00+01:00\n2,3,2020-01-01T00:00:00Z\n",
            );
            const text = snapshot.edges.requireTyped("at.text", "string");
            expect(text.meta.role).toBe("timeText");
            expect(text.valueAt(0)).toBe("2020-01-01T00:00:00+01:00");
            expect(text.isSet(1)).toBe(false);
            expect(snapshot.edges.value("at", 0)).toBe(Date.UTC(2019, 11, 31, 23));
        });

        it("reads a relationship section without :TYPE", async () => {
            const { snapshot } = await importText(":START_ID,:END_ID\na,b\n");
            expect(snapshot.edges.has(TYPE_COLUMN)).toBe(false);
            expect(snapshot.edgeCount).toBe(1);
        });

        it("keeps self-loops and parallel edges", async () => {
            const { snapshot } = await importText(":START_ID,:END_ID\n1,1\n1,2\n1,2\n");
            expect(snapshot.edgeCount).toBe(3);
            expect(snapshot.selfLoopCount).toBe(1);
            expect(snapshot.flags.multigraph).toBe(true);
        });
    });

    describe("direction (design 8.4 rules 1 and 2)", () => {
        it("reports builder-policy options the caller's sink does not use (design 8.4 precedence)", async () => {
            const { report } = await importText(
                RELS,
                { weightDtype: "f32", duplicateEdges: "sum", selfLoops: "keep", addMissingNodes: true },
                { weightDtype: "f64", duplicateEdges: "keep", selfLoops: "keep", addMissingNodes: true },
            );
            expect(report.issues.map((i) => [i.code, i.element, i.category])).toEqual([
                [SINK_OPTION_CODE, "duplicateEdges", "coercion"],
                [SINK_OPTION_CODE, "weightDtype", "coercion"],
            ]);
            // options left undefined are defaults, never requests
            const silent = await importText(RELS, undefined, { weightDtype: "f32", duplicateEdges: "sum" });
            expect(codes(silent.report)).toEqual([]);
        });

        it("sets an empty sink directed", async () => {
            const { snapshot, report } = await importText(RELS, undefined, { directed: false });
            expect(snapshot.directed).toBe(true);
            expect(codes(report)).toEqual([]);
        });

        it('reads the file as undirected under onMixedDirection "undirected" and reports it', async () => {
            const { snapshot, report } = await importText(RELS, { onMixedDirection: "undirected" });
            expect(snapshot.directed).toBe(false);
            expect(snapshot.edgeCount).toBe(3);
            expect(report.issues[0]).toMatchObject({
                code: DIRECTION_FORCED_CODE,
                category: "coercion",
                severity: "warning",
            });
        });

        it("leaves the sink's direction alone when the file has no relationships", async () => {
            const builder = new GraphBuilder({ directed: false, weightDtype: "f64" });
            builder.addEdge("x", "y");
            const report = await neo4jImporter.import(NODES, builder);
            expect(builder.directed).toBe(false);
            expect(report.counts.expandedMixed).toBe(0);
            expect(codes(report)).toEqual([]);
            const snapshot = builder.freeze();
            expect(snapshot.directed).toBe(false);
            expect(snapshot.nodeCount).toBe(5);
            const empty = await importText(NODES, undefined, { directed: false });
            expect(empty.snapshot.directed).toBe(false);
        });

        it("expands a non-empty undirected sink", async () => {
            const builder = new GraphBuilder({ directed: false, weightDtype: "f64" });
            builder.addEdge("x", "y");
            const report = await neo4jImporter.import(RELS, builder);
            const snapshot = builder.freeze();
            expect(snapshot.directed).toBe(true);
            expect(snapshot.edgeCount).toBe(5);
            expect(report.counts.expandedMixed).toBe(1);
            expect(snapshot.edges.byRole("pair")).not.toBeNull();
        });

        it('aborts on a locked undirected sink under "expand" and reads it as undirected under "undirected"', async () => {
            const locked = (): GraphBuilder => {
                const builder = new GraphBuilder({ directed: false, weightDtype: "f64" });
                builder.lockDirected();
                return builder;
            };
            await expect(neo4jImporter.import(RELS, locked())).rejects.toThrow(ImportError);
            try {
                await neo4jImporter.import(RELS, locked());
            } catch (err) {
                const { report } = err as ImportError;
                expect(report.issues[0]).toMatchObject({
                    code: DIRECTION_REFUSED_CODE,
                    category: "coercion",
                    severity: "warning",
                });
                expect(report.issues[1]).toMatchObject({ code: DIRECTION_REFUSED_CODE, severity: "error", line: 2 });
            }
            const builder = locked();
            const report = await neo4jImporter.import(RELS, builder, { onMixedDirection: "undirected" });
            expect(builder.directed).toBe(false);
            expect(builder.edgeCount).toBe(3);
            expect(report.issues.length).toBeGreaterThan(0);
            expect(report.issues.every((i) => i.code === DIRECTION_FORCED_CODE)).toBe(true);
        });
    });

    describe("id coercion (design 4.1)", () => {
        it('keeps every cell a string under ids: "string"', async () => {
            const { snapshot } = await importText(RELS, { ids: "string" });
            expect(snapshot.ids.toArray()).toEqual(["1", "2", "3"]);
        });

        it("keeps non-canonical integer text and integers beyond 2^53 as strings", async () => {
            const { snapshot } = await importText(":ID\n01\n1\n-0\n+1\n1.0\n9007199254740993\n-9007199254740991\n");
            expect(snapshot.ids.toArray()).toEqual(["01", 1, "-0", "+1", "1.0", "9007199254740993", -9007199254740991]);
        });

        it('reports merges under ids: "number" and rejects non-numeric text', async () => {
            const { snapshot, report } = await importText(":ID\n1\n01\nabc\n", { ids: "number" });
            expect(snapshot.ids.toArray()).toEqual([1]);
            expect(report.counts).toMatchObject({ nodes: 2, skippedNodes: 1 });
            expect(report.issues.map((i) => [i.code, i.category, i.severity, i.line])).toEqual([
                [ID_MERGED_CODE, "coercion", "warning", 3],
                [DUPLICATE_NODE_CODE, "merged", "warning", 3],
                ["E_INVALID_ID", "validation-error", "error", 4],
            ]);
        });

        it("rejects an unknown ids option", async () => {
            await expect(importText(RELS, { ids: "float" as never })).rejects.toThrow(GraphFormatError);
        });
    });

    describe("per-row errors (design 8.6)", () => {
        it("skips rows with the wrong cell count", async () => {
            const { snapshot, report } = await importText(
                ":ID,name\n1,a,extra\n2\n3,c\n:START_ID,:END_ID\n1,2,3\n3,1\n",
            );
            expect(snapshot.ids.toArray()).toEqual([3, 1]);
            expect(snapshot.edgeCount).toBe(1);
            expect(report.counts).toMatchObject({ nodes: 1, edges: 1, skippedNodes: 2, skippedEdges: 1 });
            expect(report.issues.map((i) => [i.code, i.line])).toEqual([
                [COLUMN_COUNT_CODE, 2],
                [COLUMN_COUNT_CODE, 3],
                [COLUMN_COUNT_CODE, 6],
            ]);
            expect(report.issues[0].category).toBe("validation-error");
        });

        it("skips rows with an empty id or endpoint", async () => {
            // (a one-cell empty row would be a blank line, which the reader skips)
            const { snapshot, report } = await importText(":ID,name\n,a\n1,b\n:START_ID,:END_ID\n,1\n1,\n1,1\n");
            expect(snapshot.nodeCount).toBe(1);
            expect(snapshot.edgeCount).toBe(1);
            expect(report.issues.map((i) => [i.code, i.category, i.line])).toEqual([
                [MISSING_ID_CODE, "missing-value", 2],
                [MISSING_ENDPOINT_CODE, "missing-value", 5],
                [MISSING_ENDPOINT_CODE, "missing-value", 6],
            ]);
            expect(report.issues[1].message).toContain(":START_ID");
            expect(report.issues[2].message).toContain(":END_ID");
        });

        it('reads a quoted empty cell as the id "" (an unquoted one is missing)', async () => {
            const { snapshot, report } = await importText(':ID\n""\n1\n:START_ID,:END_ID\n"",1\n1,""\n');
            expect(report.issues).toEqual([]);
            expect(snapshot.ids.toArray()).toEqual(["", 1]);
            expect(snapshot.edgeCount).toBe(2);
            const list = snapshot.edgeList();
            expect([...list.src]).toEqual([0, 1]);
            expect([...list.dst]).toEqual([1, 0]);
        });

        it("skips a row whose typed cell does not parse, without touching the sink", async () => {
            const text =
                ":ID,age:int,flag:boolean\n1,x,true\n2,3,maybe\n3,4,false\n:START_ID,:END_ID,n:int\n1,3,1.5\n3,3,2\n";
            const { snapshot, report } = await importText(text);
            expect(snapshot.ids.toArray()).toEqual([3]);
            expect(snapshot.edgeCount).toBe(1);
            expect(report.counts).toMatchObject({ nodes: 1, edges: 1, skippedNodes: 2, skippedEdges: 1 });
            expect(report.issues.map((i) => [i.code, i.category, i.line, i.element])).toEqual([
                ["E_COLUMN_TYPE", "validation-error", 2, "1 age"],
                ["E_COLUMN_TYPE", "validation-error", 3, "2 flag"],
                ["E_COLUMN_TYPE", "validation-error", 6, "1->3 n"],
            ]);
        });

        it("rejects NaN and non-numeric weights before the sink is touched", async () => {
            const { snapshot, report } = await importText(
                ":START_ID,:END_ID,weight:double\n1,2,NaN\n2,3,heavy\n3,1,Infinity\n",
            );
            expect(snapshot.edgeCount).toBe(1);
            expect(snapshot.edgeList().weights?.[0]).toBe(Infinity);
            expect(report.issues.map((i) => i.code)).toEqual(["E_INVALID_WEIGHT", "E_INVALID_WEIGHT"]);
            expect(report.counts.skippedEdges).toBe(2);
        });

        it("aborts with the partial report beyond the error limit", async () => {
            const rows = Array.from({ length: 6 }, (_, i) => `${i},bad`).join("\n");
            const err = await importError(`:ID,n:int\n${rows}\n`, { errorLimit: 3 });
            expect(err.code).toBe("E_IMPORT");
            expect(err.report.truncated).toBe(true);
            expect(err.report.errorCount).toBe(4);
            expect(err.report.counts.skippedNodes).toBe(3);
            expect(err.report.format).toBe("neo4j");
        });

        it("tolerates every error under errorLimit Infinity", async () => {
            const rows = Array.from({ length: 150 }, (_, i) => `${i},bad`).join("\n");
            const { report } = await importText(`:ID,n:int\n${rows}\n`, { errorLimit: Infinity });
            expect(report.errorCount).toBe(150);
            expect(report.truncated).toBe(false);
        });
    });

    describe("fatal errors", () => {
        it.each([
            ["", "no header row"],
            ["a,b\n1,2\n", ":ID column"],
            [":ID,:ID\n1,1\n", "more than one :ID"],
            [":START_ID\n1\n", "exactly one :START_ID and one :END_ID"],
            [":ID,:START_ID,:END_ID\n", "mixes :ID"],
            [":START_ID,:END_ID,:LABEL\n", "cannot have a :LABEL"],
            [":ID,:TYPE\n", "cannot have a :TYPE"],
            [":START_ID,:END_ID,:TYPE,:TYPE\n", "more than one :TYPE"],
            [":ID,name,name:int\n", "declared twice"],
            ["pid:ID,pid\n", "declared twice"],
            [":ID,x:string(S)\n", "id space"],
            [":ID,:double\n", "needs a name"],
            [":ID,name\n1,a\n:START_ID\n", "exactly one :START_ID"],
        ])("aborts on the malformed header %j", async (text, message) => {
            const err = await importError(text);
            expect(err.code).toBe("E_IMPORT");
            expect(err.report.issues).toHaveLength(1);
            expect(err.report.issues[0]).toMatchObject({
                code: HEADER_CODE,
                category: "parse-error",
                severity: "error",
            });
            expect(err.report.issues[0].message).toContain(message);
            expect(err.message).toContain(message);
        });

        it("keeps the rows read before a malformed later header", async () => {
            const err = await importError(":ID\n1\n2\n:START_ID,:TYPE\n1,x\n");
            expect(err.report.counts.nodes).toBe(2);
            expect(err.report.issues[0].line).toBe(4);
        });

        it("aborts on invalid UTF-8", async () => {
            const err = await importError(new Uint8Array([0x3a, 0x49, 0x44, 0x0a, 0xff, 0xfe, 0x0a]));
            expect(err.report.issues[0].code).toBe(INVALID_UTF8_CODE);
        });
    });

    describe("several inputs", () => {
        it("reads the primary input, then nodes, then relationships", async () => {
            const nodes = neo4jText("movies-nodes.csv");
            const rels = neo4jText("movies-rels.csv");
            const { snapshot, report } = await importText(nodes, { relationships: rels });
            expect(snapshot.nodeCount).toBe(7);
            expect(snapshot.edgeCount).toBe(6);
            expect(snapshot.ids.toArray()).toEqual(["m1", "m2", "m3", "p1", "p2", "p3", "p4"]);
            expect(report.counts).toMatchObject({ nodes: 7, edges: 6 });
            const roles = snapshot.edges.requireTyped("roles", "list");
            expect([...roles.sliceOf(3)]).toEqual(["Zachry", "Dr. Henry Goose"]);
            expect(roles.isSet(4)).toBe(false);

            const split = await importText(":ID\nm1\n", {
                nodes: [":ID\nm2\n", new TextEncoder().encode(":ID\nm3\n")],
                relationships: [":START_ID,:END_ID\nm1,m2\n", ":START_ID,:END_ID\nm2,m3\n"],
            });
            expect(split.snapshot.ids.toArray()).toEqual(["m1", "m2", "m3"]);
            expect(split.snapshot.edgeCount).toBe(2);
        });

        it("is equal to the same sections in one file", async () => {
            const bundle = await importText(neo4jText("movies-nodes.csv") + neo4jText("movies-rels.csv"));
            const paired = await importText(neo4jText("movies-nodes.csv"), {
                relationships: neo4jText("movies-rels.csv"),
            });
            expectSameSnapshot(bundle.snapshot, paired.snapshot);
        });

        it("rejects an entry that is not an ImportInput", async () => {
            await expect(importText(":ID\n1\n", { relationships: [42 as never] })).rejects.toThrow(GraphFormatError);
            await expect(importText(":ID\n1\n", { nodes: {} as never })).rejects.toThrow(GraphFormatError);
        });
    });

    describe("syntax options", () => {
        it("reads tab-separated CRLF input with a BOM, blank lines and a quoted tab", async () => {
            const { snapshot, report } = await importText(neo4jText("crlf-tabs.tsv"), { delimiter: "\t" });
            expect(snapshot.ids.toArray()).toEqual(["a", "b", "c"]);
            expect(snapshot.nodes.value("name", 2)).toBe("Gam\tma");
            expect(snapshot.nodes.value("n", 1)).toBe(2);
            expect(snapshot.nodes.value(LABELS_COLUMN, 0)).toEqual(["Thing"]);
            expect(snapshot.nodes.value("nodeId", 0)).toBe("a");
            expect(snapshot.edgeCount).toBe(2);
            expect(codes(report)).toEqual([]);
        });

        it("splits arrays and labels on the array delimiter", async () => {
            const { snapshot } = await importText(":ID,:LABEL,t:int[]\n1,A|B,1|2\n", { arrayDelimiter: "|" });
            expect(snapshot.nodes.value(LABELS_COLUMN, 0)).toEqual(["A", "B"]);
            expect(snapshot.nodes.value("t", 0)).toEqual([1, 2]);
            const comma = await importText(":ID;:LABEL\n1;A,B\n", { delimiter: ";", arrayDelimiter: "," });
            expect(comma.snapshot.nodes.value(LABELS_COLUMN, 0)).toEqual(["A", "B"]);
        });

        it("accepts another quote character", async () => {
            const { snapshot } = await importText(":ID,name\n1,'a,b'\n", { quote: "'" });
            expect(snapshot.nodes.value("name", 0)).toBe("a,b");
        });

        it.each([
            [{ delimiter: ";;" }],
            [{ quote: "" }],
            [{ arrayDelimiter: ":" as never }],
            [{ delimiter: ";", arrayDelimiter: ";" as const }],
            [{ delimiter: '"' }],
        ])("rejects the syntax options %j", async (options) => {
            await expect(importText(":ID\n1\n", options)).rejects.toThrow(GraphFormatError);
            try {
                await importText(":ID\n1\n", options);
            } catch (err) {
                expect((err as GraphFormatError).code).toBe("E_UNSUPPORTED");
            }
        });
    });

    describe("streaming (design 8.4)", () => {
        it("produces the same snapshot from every input shape", async () => {
            const bytes = new TextEncoder().encode(neo4jText("social-bundle.csv"));
            const reference = await importText(bytes);
            for (const shape of inputShapes(bytes)) {
                const { snapshot, report } = await importText(shape.make());
                expectSameSnapshot(reference.snapshot, snapshot, { allowExtraColumns: false });
                expect(report.counts).toEqual(reference.report.counts);
            }
        });

        it("reports cumulative progress over several in-memory inputs", async () => {
            const calls: [number, number | undefined][] = [];
            const a = ":ID\n1\n";
            const b = new TextEncoder().encode(":START_ID,:END_ID\n1,2\n");
            await importText(a, { relationships: b, onProgress: (done, total) => calls.push([done, total]) });
            const total = a.length + b.byteLength;
            expect(calls.every(([, t]) => t === total)).toBe(true);
            expect(calls[calls.length - 1][0]).toBe(total);
            expect(calls.map(([d]) => d)).toEqual([...calls.map(([d]) => d)].sort((x, y) => x - y));
        });

        it("leaves the total unknown when an input is a stream", async () => {
            const calls: (number | undefined)[] = [];
            const stream = inputShapes(new TextEncoder().encode(":ID\n2\n")).find((s) =>
                s.name.startsWith("ReadableStream"),
            );
            await importText(":ID\n1\n", { nodes: stream?.make(), onProgress: (_d, total) => calls.push(total) });
            expect(calls.some((t) => t === undefined)).toBe(true);
        });

        it("stops when the signal aborts", async () => {
            const controller = new AbortController();
            controller.abort();
            await expect(importText(":ID\n1\n", { signal: controller.signal })).rejects.toMatchObject({
                name: "AbortError",
            });
        });
    });

    it("declares nothing on an empty relationship section and nothing when no node has properties", async () => {
        const { snapshot } = await importText(":ID\n1\n:START_ID,:END_ID\n");
        expect(snapshot.nodes.names()).toEqual([]);
        expect(snapshot.edges.names()).toEqual([]);
        expect(snapshot.edgeCount).toBe(0);
        expect(snapshot.nodes.byRole("labels")).toBeNull();
        expect(snapshot.nodeCount).toBe(1);
        expect(snapshot.ids.indexOf(1)).not.toBe(INVALID_INDEX);
    });
});
