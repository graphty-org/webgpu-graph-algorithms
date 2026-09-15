import { describe, expect, it } from "vitest";

import { registry } from "../src/registry.js";
import {
    extensionOf,
    GRAPH_FORMATS,
    headBytes,
    normalizeMimeType,
    rankFormats,
    SNIFF_HEAD_BYTES,
    sniffFormat,
    sniffJsonDialectHead,
} from "../src/sniff.js";
import { type GraphImporter } from "../src/types.js";
import { CORPUS_FORMATS, corpusFiles, readCorpusBytes, readCorpusText } from "./helpers/corpus.js";

const IMPORTERS = registry.importers();

/**
 * A fake importer that claims the given extensions and answers a fixed content confidence.
 * @param format - the format name
 * @param extensions - the extensions
 * @param content - the sniff answer
 * @returns the importer
 */
function fake(format: string, extensions: string[], content: number): GraphImporter {
    return {
        format,
        extensions,
        mimeTypes: [`application/x-${format}`],
        sniff: () => content,
        import: () => Promise.reject(new Error("not implemented")),
    };
}

describe("extensionOf / normalizeMimeType / headBytes", () => {
    it("takes the lower-cased last extension of a name or path", () => {
        expect(extensionOf("graph.GEXF")).toBe(".gexf");
        expect(extensionOf("/data/sets/karate.gml")).toBe(".gml");
        expect(extensionOf("C:\\\\data\\\\net.NET")).toBe(".net");
        expect(extensionOf("archive.tar.gz")).toBe(".gz");
        expect(extensionOf("noext")).toBeNull();
        expect(extensionOf(".hidden")).toBeNull();
        expect(extensionOf("trailing.")).toBeNull();
        expect(extensionOf("dir.with.dots/name")).toBeNull();
        expect(extensionOf("")).toBeNull();
    });

    it("strips MIME parameters and case", () => {
        expect(normalizeMimeType("Text/CSV; charset=utf-8")).toBe("text/csv");
        expect(normalizeMimeType("  application/json ")).toBe("application/json");
        expect(normalizeMimeType("text/plain")).toBe("text/plain");
    });

    it("encodes a string head and caps the head at SNIFF_HEAD_BYTES", () => {
        expect(Array.from(headBytes("ab"))).toEqual([0x61, 0x62]);
        const long = new Uint8Array(SNIFF_HEAD_BYTES * 2);
        expect(headBytes(long).byteLength).toBe(SNIFF_HEAD_BYTES);
        expect(headBytes("x".repeat(SNIFF_HEAD_BYTES * 2)).byteLength).toBe(SNIFF_HEAD_BYTES);
        const short = new Uint8Array(3);
        expect(headBytes(short)).toBe(short);
    });
});

describe("rankFormats / sniffFormat (design 8.2)", () => {
    it("lists the eight built-in formats in the default registry's order", () => {
        expect(IMPORTERS.map((i) => i.format)).toEqual([...GRAPH_FORMATS]);
        expect(GRAPH_FORMATS).toEqual(["json", "graphml", "gexf", "csv", "gml", "dot", "pajek", "neo4j"]);
    });

    it("recognises every corpus file from its content alone and from its name alone", () => {
        for (const format of CORPUS_FORMATS) {
            for (const file of corpusFiles(format)) {
                const bytes = readCorpusBytes(format, file.path);
                const byContent = sniffFormat({ head: bytes.subarray(0, SNIFF_HEAD_BYTES) }, IMPORTERS);
                expect(byContent?.format, `${format}/${file.path} by content`).toBe(format);
                expect(byContent?.confidence, `${format}/${file.path}`).toBeGreaterThanOrEqual(0.5);
                expect(byContent?.content).toBeGreaterThan(0);
                expect(byContent?.extension).toBe(false);
                const both = sniffFormat({ head: bytes, filename: file.path }, IMPORTERS);
                expect(both?.format, `${format}/${file.path} by content and name`).toBe(format);
                expect(both?.extension, `${format}/${file.path}`).toBe(true);
                expect(both?.confidence).toBeGreaterThan(byContent?.confidence ?? 1);
                if (format !== "neo4j") {
                    // a neo4j-admin file is a .csv; only its header tells it from a plain CSV
                    const byName = sniffFormat({ filename: file.path }, IMPORTERS);
                    expect(byName?.format, `${format}/${file.path} by name`).toBe(format);
                    expect(byName?.confidence).toBeLessThan(0.5);
                    expect(byName?.content).toBe(0);
                }
            }
        }
    });

    it("lets the content beat a misleading extension and decide a shared one", () => {
        const gml = readCorpusText("gml", "karate.gml");
        expect(sniffFormat({ head: gml, filename: "karate.csv" }, IMPORTERS)?.format).toBe("gml");
        expect(sniffFormat({ head: gml, filename: "karate.txt" }, IMPORTERS)?.format).toBe("gml");
        const neo4j = readCorpusText("neo4j", "karate-neo4j.csv");
        expect(sniffFormat({ head: neo4j, filename: "karate.csv" }, IMPORTERS)?.format).toBe("neo4j");
        expect(
            sniffFormat({ head: readCorpusText("csv", "simple-edges.csv"), filename: "x.csv" }, IMPORTERS)?.format,
        ).toBe("csv");
        const gexf = readCorpusText("gexf", "minimal.gexf");
        expect(sniffFormat({ head: gexf, filename: "graph.xml" }, IMPORTERS)?.format).toBe("gexf");
        const graphml = readCorpusText("graphml", "simple.graphml");
        expect(sniffFormat({ head: graphml, filename: "graph.xml" }, IMPORTERS)?.format).toBe("graphml");
    });

    it("breaks extension ties by registration order and uses the MIME type as a weak hint", () => {
        expect(sniffFormat({ filename: "graph.xml" }, IMPORTERS)?.format).toBe("graphml");
        expect(sniffFormat({ filename: "table.csv" }, IMPORTERS)?.format).toBe("csv");
        expect(rankFormats({ filename: "table.csv" }, IMPORTERS).map((r) => r.format)).toEqual(["csv", "neo4j"]);
        expect(sniffFormat({ filename: "graph.edges" }, IMPORTERS)?.format).toBe("csv");
        expect(sniffFormat({ filename: "graph.gv" }, IMPORTERS)?.format).toBe("dot");
        expect(sniffFormat({ filename: "graph.paj" }, IMPORTERS)?.format).toBe("pajek");
        expect(sniffFormat({ mimeType: "application/gexf+xml" }, IMPORTERS)?.format).toBe("gexf");
        expect(sniffFormat({ mimeType: "text/vnd.graphviz" }, IMPORTERS)?.format).toBe("dot");
        expect(sniffFormat({ mimeType: "Text/CSV; charset=utf-8" }, IMPORTERS)?.format).toBe("csv");
        const csvHint = sniffFormat({ filename: "a.csv", mimeType: "text/csv" }, IMPORTERS);
        expect(csvHint).toMatchObject({ format: "csv", extension: true, mimeType: true, content: 0 });
        expect(csvHint?.confidence).toBeCloseTo(0.4, 10);
        expect(sniffFormat({ mimeType: "application/xml" }, IMPORTERS)?.format).toBe("graphml");
    });

    it("returns null for nothing, for unknown hints and for content no importer claims", () => {
        expect(sniffFormat({}, IMPORTERS)).toBeNull();
        expect(sniffFormat({ filename: "x.unknown", mimeType: "image/png" }, IMPORTERS)).toBeNull();
        expect(sniffFormat({ head: new Uint8Array([0, 1, 2, 3, 0xff]) }, IMPORTERS)).toBeNull();
        expect(sniffFormat({ head: new Uint8Array(0), filename: null, mimeType: null }, IMPORTERS)).toBeNull();
        expect(rankFormats({ head: "" }, IMPORTERS)).toEqual([]);
    });

    it("scores by the documented formula and clamps an importer's answer into 0..1", () => {
        const importers = [fake("one", [".x"], 2), fake("two", [".x"], NaN), fake("three", [".y"], 0.5)];
        const ranked = rankFormats({ head: "anything", filename: "f.x" }, importers);
        expect(ranked.map((r) => [r.format, r.confidence, r.content])).toEqual([
            ["one", 0.95, 1],
            ["three", 0.675, 0.5],
            ["two", 0.3, 0],
        ]);
        expect(ranked[0].dialect).toBeNull();
        // ties: registration order
        const tie = rankFormats({ filename: "f.x" }, [fake("b", [".x"], 0), fake("a", [".x"], 0)]);
        expect(tie.map((r) => r.format)).toEqual(["b", "a"]);
        // an importer without sniff() is a candidate by its hints only
        const noSniff: GraphImporter = {
            format: "n",
            extensions: [".n"],
            mimeTypes: [],
            import: fake("n", [], 0).import,
        };
        expect(rankFormats({ head: "n", filename: "f.n" }, [noSniff])[0]).toMatchObject({
            confidence: 0.3,
            content: 0,
        });
        expect(rankFormats({ head: "n" }, [noSniff])).toEqual([]);
    });

    it("attaches the JSON dialect sniffed from the head to the json candidate", () => {
        const cyto = sniffFormat({ head: '{"elements": {"nodes": [{"data": {"id": "a"}}]}}' }, IMPORTERS);
        expect(cyto).toMatchObject({ format: "json", dialect: "cytoscape" });
        const csv = sniffFormat({ head: "source,target\na,b\n" }, IMPORTERS);
        expect(csv?.dialect).toBeNull();
        expect(sniffFormat({ filename: "g.json" }, IMPORTERS)?.dialect).toBeNull();
    });
});

describe("sniffJsonDialectHead", () => {
    const NODE_LINK = JSON.stringify({
        directed: true,
        multigraph: false,
        graph: {},
        nodes: Array.from({ length: 200 }, (_, i) => ({ id: `node-${i}`, group: i % 3 })),
        links: Array.from({ length: 200 }, (_, i) => ({ source: `node-${i}`, target: `node-${(i + 1) % 200}` })),
    });
    const D3 = JSON.stringify({
        nodes: Array.from({ length: 200 }, (_, i) => ({ name: `n${i}` })),
        links: Array.from({ length: 200 }, (_, i) => ({ source: i, target: (i + 1) % 200, value: 1 })),
    });
    const JGF = JSON.stringify({
        graph: {
            id: "g",
            type: "test",
            label: "a graph",
            metadata: { creator: "test", tags: ["x", "y"] },
            directed: true,
            nodes: Object.fromEntries(Array.from({ length: 200 }, (_, i) => [`n${i}`, { label: `node ${i}` }])),
            edges: Array.from({ length: 200 }, (_, i) => ({ source: `n${i}`, target: `n${(i + 1) % 200}` })),
        },
    });
    const CYTO = JSON.stringify({
        elements: {
            nodes: Array.from({ length: 200 }, (_, i) => ({ data: { id: `n${i}` }, position: { x: i, y: 0 } })),
            edges: [],
        },
        style: [],
    });
    const CYTO_ARRAY = JSON.stringify(
        Array.from({ length: 200 }, (_, i) => ({ group: "nodes", data: { id: `n${i}` } })),
    );
    const GRAPHOLOGY = JSON.stringify({
        options: { type: "mixed", multi: false, allowSelfLoops: true },
        attributes: { name: "g" },
        nodes: Array.from({ length: 200 }, (_, i) => ({ key: `n${i}`, attributes: { x: i } })),
        edges: Array.from({ length: 200 }, (_, i) => ({
            source: `n${i}`,
            target: `n${(i + 1) % 200}`,
            undirected: true,
        })),
    });
    const VIS = JSON.stringify({
        nodes: Array.from({ length: 200 }, (_, i) => ({ id: i, label: `n${i}` })),
        edges: Array.from({ length: 200 }, (_, i) => ({ from: i, to: (i + 1) % 200 })),
    });

    const CASES: [string, string, string][] = [
        ["node-link", NODE_LINK, "node-link"],
        ["d3", D3, "d3"],
        ["jgf", JGF, "jgf"],
        ["cytoscape", CYTO, "cytoscape"],
        ["cytoscape array", CYTO_ARRAY, "cytoscape"],
        ["graphology", GRAPHOLOGY, "graphology"],
        ["vis", VIS, "vis"],
    ];

    it.each(CASES)("classifies a whole %s document", (_name, text, dialect) => {
        expect(sniffJsonDialectHead(text)).toBe(dialect);
        expect(sniffJsonDialectHead(new TextEncoder().encode(text))).toBe(dialect);
        expect(sniffJsonDialectHead(`\uFEFF  ${text}`)).toBe(dialect);
    });

    it.each(CASES)("classifies a truncated %s head from its key skeleton", (_name, text, dialect) => {
        // the graphology / vis rules read the first node and the first edge; cut after both exist
        const edgesAt = Math.max(text.indexOf('"edges"'), text.indexOf('"links"'));
        const cut = text.slice(0, Math.min(text.length - 1, Math.max(400, edgesAt + 160)));
        expect(JSON.parse.bind(JSON, cut)).toThrow();
        expect(sniffJsonDialectHead(cut)).toBe(dialect);
        expect(sniffJsonDialectHead(cut.slice(0, cut.lastIndexOf('"') + 1))).toBe(dialect);
    });

    it("returns null for non-JSON, for JSON without a graph shape and for a scalar", () => {
        expect(sniffJsonDialectHead("<gexf/>")).toBeNull();
        expect(sniffJsonDialectHead("")).toBeNull();
        expect(sniffJsonDialectHead('{"foo": 1, "bar": [1, 2]}')).toBeNull();
        expect(sniffJsonDialectHead('{"foo": 1, "bar": [1, 2')).toBeNull();
        expect(sniffJsonDialectHead("42")).toBeNull();
        expect(sniffJsonDialectHead("[1, 2, 3]")).toBeNull();
        expect(sniffJsonDialectHead("[1, 2, 3")).toBeNull();
        expect(sniffJsonDialectHead("{")).toBeNull();
        expect(sniffJsonDialectHead("[")).toBeNull();
        expect(sniffJsonDialectHead("[]")).toBe("cytoscape");
        expect(sniffJsonDialectHead('[{"data": {"id": "a"}}, {"data"')).toBe("cytoscape");
        expect(sniffJsonDialectHead('[{"id": "a"}, {"id"')).toBeNull();
    });

    it("survives keys with escapes, strings holding braces and a cut inside a string", () => {
        const head =
            '{"no\\"des": [{"id": "{[not a key]}"}], "nodes": [{"id": "a", "k\\u0065y": "x"}], "links": [{"source": "a", "target": "cut he';
        expect(sniffJsonDialectHead(head)).toBe("d3");
        expect(
            sniffJsonDialectHead(
                '{"nodes": [{"key": "a"}], "edges": [{"source": "a", "target": "b", "attributes": {"w": 1',
            ),
        ).toBe("graphology");
        expect(sniffJsonDialectHead('{"elements": [')).toBe("cytoscape");
        expect(sniffJsonDialectHead('{"graphs": [{"nodes": {')).toBe("jgf");
        expect(sniffJsonDialectHead('{"graph": {"nodes": {"a": {}}}')).toBe("jgf");
        expect(sniffJsonDialectHead('{"graph": {"name": "g"}, "nodes": [], "links": []}')).toBe("node-link");
    });
});
