import { GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { PRECISION_CODE } from "../../../src/common/attributes.js";
import { INVALID_UTF8_CODE } from "../../../src/common/input.js";
import { UNCLOSED_QUOTE_CODE } from "../../../src/formats/csv/records.js";
import {
    HEADER_CODE,
    ID_SPACE_COLUMN,
    IGNORED_COLUMNS_LOSS,
    LABELS_COLUMN,
    neo4jImporter,
    type Neo4jImportOptions,
    TYPE_COLUMN,
} from "../../../src/formats/neo4j/importer.js";
import { type CommonImportOptions, ImportError, type ImportInput, type ImportReport } from "../../../src/types.js";
import { inputShapes } from "../../helpers/corpus.js";
import { expectSameSnapshot } from "../../helpers/roundtrip.js";
import {
    neo4jBytes,
    type Neo4jCorpusFile,
    neo4jFiles,
    neo4jMalformedBytes,
    neo4jMalformedFiles,
    neo4jManifest,
    neo4jText,
} from "./fixtures.js";

async function load(
    input: ImportInput,
    options?: Neo4jImportOptions & CommonImportOptions,
): Promise<{ snapshot: GraphSnapshot; report: ImportReport }> {
    const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
    const report = await neo4jImporter.import(input, builder, options);
    return { snapshot: builder.freeze(), report };
}

function optionsOf(entry: Neo4jCorpusFile): Neo4jImportOptions & CommonImportOptions {
    return entry.options ?? {};
}

describe("neo4j corpus (design 16.5)", () => {
    const manifest = neo4jManifest();

    it("has a manifest of the shared shape", () => {
        expect(manifest.format).toBe("neo4j");
        expect(manifest.files.length).toBeGreaterThan(0);
        for (const file of manifest.files) {
            expect(typeof file.path).toBe("string");
            expect(typeof file.source).toBe("string");
            expect(typeof file.license).toBe("string");
            expect(Number.isInteger(file.expectedNodes)).toBe(true);
            expect(Number.isInteger(file.expectedEdges)).toBe(true);
            expect(Array.isArray(file.features)).toBe(true);
        }
    });

    for (const entry of neo4jFiles()) {
        describe(entry.path, () => {
            it("imports with the expected counts, directed, without errors", async () => {
                const { snapshot, report } = await load(neo4jText(entry.path), optionsOf(entry));
                expect(snapshot.nodeCount).toBe(entry.expectedNodes);
                expect(snapshot.edgeCount).toBe(entry.expectedEdges);
                expect(snapshot.directed).toBe(true);
                expect(report.counts.nodes + report.counts.edges).toBeGreaterThan(0);
                expect(report.counts.skippedNodes).toBe(0);
                expect(report.counts.skippedEdges).toBe(0);
                expect(report.errorCount).toBe(0);
                expect(report.truncated).toBe(false);
                expect(report.format).toBe("neo4j");
                expect(report.durationMs).toBeGreaterThanOrEqual(0);
            });

            it("imports identically from every input shape", async () => {
                const bytes = neo4jBytes(entry.path);
                const reference = await load(bytes, optionsOf(entry));
                for (const shape of inputShapes(bytes)) {
                    const { snapshot } = await load(shape.make(), optionsOf(entry));
                    expectSameSnapshot(reference.snapshot, snapshot, { allowExtraColumns: false });
                }
            });

            const paired = entry.with;
            if (paired !== undefined) {
                it("imports with its relationship files", async () => {
                    const { snapshot, report } = await load(neo4jText(entry.path), {
                        ...optionsOf(entry),
                        relationships: paired.relationships.map((name) => neo4jText(name)),
                    });
                    expect(snapshot.nodeCount).toBe(paired.expectedNodes);
                    expect(snapshot.edgeCount).toBe(paired.expectedEdges);
                    expect(report.errorCount).toBe(0);
                });
            }
        });
    }

    describe("spot checks", () => {
        it("social-bundle.csv", async () => {
            const { snapshot, report } = await load(neo4jText("social-bundle.csv"));
            const { nodes, edges } = snapshot;
            expect(snapshot.ids.toArray()).toEqual(["u1", "u2", "u3", "u4", "u5", "u6"]);
            expect(nodes.value("name", 2)).toBe("Carol, Jr.");
            expect(nodes.value("name", 4)).toBe('Eve "the" Great');
            expect(nodes.value("name", 5)).toBe(`Zo${String.fromCharCode(0xeb)}\nLine two`);
            expect(nodes.value(LABELS_COLUMN, 1)).toEqual(["Person", "Admin"]);
            expect(nodes.isSet(LABELS_COLUMN, 3)).toBe(false);
            expect(nodes.value("userId", 0)).toBe("u1");
            expect(nodes.value("age", 0)).toBe(34);
            expect(nodes.isSet("age", 2)).toBe(false);
            expect(nodes.value("score", 1)).toBe(1.25);
            expect(nodes.value("score", 5)).toBe(1000);
            expect(nodes.value("active", 1)).toBe(false);
            expect(nodes.value("tags", 0)).toEqual(["admin", "editor"]);
            expect(nodes.isSet("tags", 1)).toBe(false);
            expect(nodes.value("tags", 3)).toEqual([]);
            expect(nodes.value("tags", 4)).toEqual(["x", "y", "z"]);
            expect(nodes.value("joined", 0)).toBe(Date.UTC(2020, 0, 15));
            expect(nodes.isSet("joined", 2)).toBe(false);
            expect(nodes.has("joined.text")).toBe(false);
            expect(nodes.require("joined").meta.origin?.type).toBe("date");

            const list = snapshot.edgeList();
            expect([...list.src]).toEqual([0, 1, 2, 0, 3, 4, 5, 0]);
            expect([...list.dst]).toEqual([1, 2, 0, 3, 4, 5, 5, 1]);
            expect(edges.value(TYPE_COLUMN, 2)).toBe("FOLLOWS");
            expect(edges.value(TYPE_COLUMN, 7)).toBe("LIKES");
            expect(edges.value("since", 0)).toBe(2015);
            expect(edges.isSet("since", 4)).toBe(false);
            expect(snapshot.flags.weighted).toBe(true);
            expect(snapshot.flags.multigraph).toBe(true);
            expect(snapshot.selfLoopCount).toBe(1);
            const weight = edges.byRole("weight");
            expect(weight?.value(0)).toBe(0.5);
            expect(weight?.isSet(1)).toBe(false);
            expect(weight?.value(6)).toBe(7.5);
            expect([...(list.weights ?? [])]).toEqual([0.5, 1, 2, 1, 0.25, 1, 7.5, 1]);
            expect(report.issues).toEqual([]);
            expect(report.lossy).toEqual([]);
        });

        it("movies-nodes.csv with movies-rels.csv", async () => {
            const { snapshot, report } = await load(neo4jText("movies-nodes.csv"), {
                relationships: neo4jText("movies-rels.csv"),
            });
            const { nodes, edges } = snapshot;
            expect(nodes.value(ID_SPACE_COLUMN, 0)).toBe("Movie");
            expect(nodes.value(ID_SPACE_COLUMN, 6)).toBe("Person");
            expect(nodes.value("title", 1)).toBe("Matrix, Reloaded");
            expect(nodes.value(LABELS_COLUMN, 1)).toEqual(["Movie", "Sequel"]);
            expect(nodes.value("released", 2)).toBe(2012);
            expect(nodes.value("name", 3)).toBe("Keanu Reeves");
            expect(nodes.value("born", 6)).toBe(1956);
            expect(nodes.isSet("title", 3)).toBe(false);
            expect(nodes.isSet("name", 0)).toBe(false);
            const list = snapshot.edgeList();
            expect([...list.src]).toEqual([3, 3, 4, 6, 5, 4]);
            expect([...list.dst]).toEqual([0, 1, 0, 2, 3, 3]);
            expect(edges.value(TYPE_COLUMN, 0)).toBe("ACTED_IN");
            expect(edges.value(TYPE_COLUMN, 4)).toBe("DIRECTED_WITH");
            expect(edges.value("roles", 0)).toEqual(["Neo"]);
            expect(edges.value("roles", 3)).toEqual(["Zachry", "Dr. Henry Goose"]);
            expect(edges.isSet("roles", 5)).toBe(false);
            expect(snapshot.flags.weighted).toBe(false);
            expect(report.issues).toEqual([]);
        });

        it("movies-rels.csv alone creates its endpoints", async () => {
            const { snapshot } = await load(neo4jText("movies-rels.csv"));
            expect(snapshot.ids.toArray()).toEqual(["p1", "m1", "m2", "p2", "p4", "m3", "p3"]);
            expect(snapshot.nodes.names()).toEqual([]);
        });

        it("typed-properties.csv", async () => {
            const { snapshot, report } = await load(neo4jText("typed-properties.csv"));
            expect(snapshot.ids.toArray()).toEqual([1, 2, 3]);
            expect(snapshot.nodes.value("p", 0)).toEqual({ x: 1.5, y: 2, crs: "cartesian" });
            expect(snapshot.nodes.value("zdt.text", 0)).toBe("2024-02-29T13:45:00+02:00");
            expect(snapshot.nodes.value("l", 1)).toBe(-12);
            expect(snapshot.nodes.value("ld", 0)).toEqual([0.5, 1.5]);
            expect(report.issues.map((i) => i.code)).toEqual([PRECISION_CODE]);
            expect(report.lossy.map((n) => n.code)).toEqual([IGNORED_COLUMNS_LOSS]);
        });

        it("karate-neo4j.csv", async () => {
            const { snapshot } = await load(neo4jText("karate-neo4j.csv"));
            expect(snapshot.ids.kind).toBe("identity");
            expect(snapshot.ids.offset).toBe(1);
            expect(snapshot.ids.idOf(33)).toBe(34);
            expect(snapshot.nodes.value("club", 0)).toBe("Mr. Hi");
            expect(snapshot.nodes.value("club", 33)).toBe("Officer");
            expect(snapshot.nodes.value(LABELS_COLUMN, 16)).toEqual(["Member"]);
            const type = snapshot.edges.requireTyped(TYPE_COLUMN, "dict");
            expect(type.dictionary).toEqual(["TIES"]);
            expect(snapshot.edgeSource(0)).toBe(1);
            expect(snapshot.edgeTarget(0)).toBe(0);
            expect(snapshot.hasArc(33, 32)).toBe(true);
            expect(snapshot.hasArc(32, 33)).toBe(false);
            expect(snapshot.flags.weighted).toBe(false);
        });

        it("crlf-tabs.tsv", async () => {
            const { snapshot, report } = await load(neo4jText("crlf-tabs.tsv"), { delimiter: "\t" });
            expect(snapshot.ids.toArray()).toEqual(["a", "b", "c"]);
            expect(snapshot.nodes.value("nodeId", 1)).toBe("b");
            expect(snapshot.nodes.value("name", 2)).toBe("Gam\tma");
            expect(snapshot.nodes.value("n", 2)).toBe(3);
            expect(snapshot.nodes.value(LABELS_COLUMN, 2)).toEqual(["Thing"]);
            expect(snapshot.edges.value(TYPE_COLUMN, 1)).toBe("NEXT");
            expect(report.issues).toEqual([]);
        });
    });

    describe("malformed cases", () => {
        const expectedCodes: Readonly<Record<string, string>> = {
            "binary-content.csv": INVALID_UTF8_CODE,
            "duplicate-property.csv": HEADER_CODE,
            "empty-file.csv": HEADER_CODE,
            "header-only-relationship.csv": HEADER_CODE,
            "label-on-relationship.csv": HEADER_CODE,
            "mixed-header.csv": HEADER_CODE,
            "no-header.csv": HEADER_CODE,
            "relationship-without-end.csv": HEADER_CODE,
            "space-on-property.csv": HEADER_CODE,
            "too-many-bad-cells.csv": "E_COLUMN_TYPE",
            "two-id-columns.csv": HEADER_CODE,
            "unclosed-quote.csv": UNCLOSED_QUOTE_CODE,
            "unnamed-property.csv": HEADER_CODE,
        };

        it("covers every malformed file", () => {
            expect(neo4jMalformedFiles()).toEqual(Object.keys(expectedCodes).sort());
        });

        for (const name of neo4jMalformedFiles()) {
            it(`${name} produces an ImportError with a report`, async () => {
                const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
                let caught: unknown = null;
                try {
                    await neo4jImporter.import(neo4jMalformedBytes(name), builder);
                } catch (err) {
                    caught = err;
                }
                expect(caught).toBeInstanceOf(ImportError);
                const err = caught as ImportError;
                expect(err.code).toBe("E_IMPORT");
                expect(err.name).toBe("ImportError");
                expect(err.report.format).toBe("neo4j");
                expect(err.report.errorCount).toBeGreaterThan(0);
                const first = err.report.issues.find((i) => i.severity === "error");
                expect(first?.code).toBe(expectedCodes[name]);
                if (name === "too-many-bad-cells.csv") {
                    expect(err.report.truncated).toBe(true);
                    expect(err.report.errorCount).toBe(101);
                    expect(err.report.counts.skippedNodes).toBe(100);
                } else {
                    expect(err.report.truncated).toBe(false);
                }
            });
        }
    });
});
