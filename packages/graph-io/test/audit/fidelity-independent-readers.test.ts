/**
 * Fidelity audit, independent-reader lens: the XML exporters' output must be well-formed XML that
 * an INDEPENDENT reader (fast-xml-parser's validator, and its parser under a configuration the
 * GEXF importer does not use: no preserveOrder, entities processed, values trimmed, attributes
 * prefixed with "@") reads to the same node and edge sets the snapshot holds, with every data
 * element referencing a declared key / attribute; the CSV exporters' output must be RFC 4180 (a
 * strict hand-written reader: quoted fields with doubled quotes only, no bare quotes, CRLF or LF
 * records, a constant field count) with one row per written edge or node whose endpoint cells are
 * the ids' text. Plus the conformance probes a corpus of well-behaved files never exercises: XML 1.0
 * illegal characters, carriage returns, empty ids.
 */

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { GraphBuilder, type GraphSnapshot } from "@graphty/graph-format";
import { XMLParser, XMLValidator } from "fast-xml-parser";
import { describe, expect, it } from "vitest";

import { csvExporter } from "../../src/formats/csv/exporter.js";
import { csvImporter } from "../../src/formats/csv/importer.js";
import { dotImporter } from "../../src/formats/dot/importer.js";
import { gexfExporter } from "../../src/formats/gexf/exporter.js";
import { gexfImporter } from "../../src/formats/gexf/importer.js";
import { gmlImporter } from "../../src/formats/gml/importer.js";
import { graphmlExporter } from "../../src/formats/graphml/exporter.js";
import { graphmlImporter } from "../../src/formats/graphml/importer.js";
import { jsonImporter } from "../../src/formats/json/importer.js";
import { neo4jExporter } from "../../src/formats/neo4j/exporter.js";
import { neo4jImporter } from "../../src/formats/neo4j/importer.js";
import { pajekImporter } from "../../src/formats/pajek/importer.js";
import { type CommonImportOptions, type GraphImporter } from "../../src/types.js";
import { DYNAMIC_1_3, OPEN_1_2 } from "../formats/gexf/fixtures.js";
import { CORPUS_FORMATS, CORPUS_ROOT, type CorpusFormat } from "../helpers/corpus.js";

type AnyImportOptions = Record<string, unknown> & CommonImportOptions;

const IMPORTERS: Readonly<Record<CorpusFormat, GraphImporter<AnyImportOptions>>> = {
    csv: csvImporter as GraphImporter<AnyImportOptions>,
    dot: dotImporter as GraphImporter<AnyImportOptions>,
    gexf: gexfImporter as GraphImporter<AnyImportOptions>,
    gml: gmlImporter as GraphImporter<AnyImportOptions>,
    graphml: graphmlImporter as GraphImporter<AnyImportOptions>,
    json: jsonImporter as GraphImporter<AnyImportOptions>,
    neo4j: neo4jImporter as GraphImporter<AnyImportOptions>,
    pajek: pajekImporter as GraphImporter<AnyImportOptions>,
};

const NOT_A_GRAPH: ReadonlySet<string> = new Set(["graphml/got-social-network.graphml"]);

interface Input {
    readonly label: string;
    readonly format: CorpusFormat;
    readonly text: string;
    readonly importOptions: AnyImportOptions;
}

function corpusText(format: CorpusFormat, name: string): string {
    return readFileSync(join(CORPUS_ROOT, format, name), "utf-8");
}

function importOptionsFor(format: CorpusFormat, name: string): AnyImportOptions {
    if (format === "neo4j" && name === "crlf-tabs.tsv") {
        return { delimiter: "\t" };
    }
    if (format === "neo4j" && name === "movies-nodes.csv") {
        return { relationships: [corpusText("neo4j", "movies-rels.csv")] };
    }
    if (format === "csv" && name === "got-edges.csv") {
        return { nodes: corpusText("csv", "got-nodes.csv") };
    }
    return {};
}

function allInputs(): Input[] {
    const inputs: Input[] = [];
    for (const format of CORPUS_FORMATS) {
        for (const name of readdirSync(join(CORPUS_ROOT, format)).sort()) {
            const label = `${format}/${name}`;
            if (name === "manifest.json" || NOT_A_GRAPH.has(label)) {
                continue;
            }
            inputs.push({
                label,
                format,
                text: corpusText(format, name),
                importOptions: importOptionsFor(format, name),
            });
        }
    }
    inputs.push({ label: "synthetic/DYNAMIC_1_3", format: "gexf", text: DYNAMIC_1_3, importOptions: {} });
    inputs.push({ label: "synthetic/OPEN_1_2", format: "gexf", text: OPEN_1_2, importOptions: {} });
    return inputs;
}

const INPUTS = allInputs();
const originals = new Map<string, Promise<GraphSnapshot>>();

function original(input: Input): Promise<GraphSnapshot> {
    let loaded = originals.get(input.label);
    if (loaded === undefined) {
        loaded = (async (): Promise<GraphSnapshot> => {
            const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
            await IMPORTERS[input.format].import(input.text, builder, input.importOptions);
            return builder.freeze();
        })();
        originals.set(input.label, loaded);
    }
    return loaded;
}

/** Logical edges the exporters write: every edge minus the mirror halves of undirected pairs (mutual pairs stay two). */
function writtenEdges(snapshot: GraphSnapshot, mutualAsOne: boolean): number {
    const pair = snapshot.edges.byRole("pair");
    const directed = snapshot.edges.byRole("directed");
    if (pair === null || pair.dtype !== "u32") {
        return snapshot.edgeCount;
    }
    let count = snapshot.edgeCount;
    for (let e = 0; e < snapshot.edgeCount; e++) {
        if (!pair.isSet(e) || pair.data[e] >= e) {
            continue;
        }
        const undirected =
            directed !== null && directed.dtype === "bool" && directed.isSet(e) && directed.value(e) === false;
        if (undirected || mutualAsOne) {
            count--;
        }
    }
    return count;
}

// ============================================================ an independent XML reader

const cp = String.fromCharCode;

/** XML 1.0 illegal characters (C0 controls except tab, LF and CR; the two non-characters), built at runtime. */
const XML_1_0_ILLEGAL = new RegExp(
    `[${cp(0)}-${cp(8)}${cp(0x0b)}${cp(0x0c)}${cp(0x0e)}-${cp(0x1f)}${cp(0xfffe)}${cp(0xffff)}]`,
);

const independentXml = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: "@",
    preserveOrder: false,
    parseTagValue: false,
    parseAttributeValue: false,
    processEntities: true,
    trimValues: true,
    allowBooleanAttributes: false,
    removeNSPrefix: false,
    isArray: (name: string): boolean =>
        ["node", "edge", "attribute", "attvalue", "key", "data", "graph", "attributes"].includes(name),
});

type XmlNode = Record<string, unknown>;

function asArray(value: unknown): XmlNode[] {
    return Array.isArray(value) ? (value as XmlNode[]) : [];
}

function asNode(value: unknown): XmlNode {
    return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as XmlNode) : {};
}

/** Parse with the validator first, then the independent configuration; throws on either failure. */
function readXml(text: string): XmlNode {
    const verdict = XMLValidator.validate(text, { allowBooleanAttributes: false, unpairedTags: [] });
    if (verdict !== true) {
        throw new Error(`not well-formed: ${JSON.stringify(verdict)}`);
    }
    return asNode(independentXml.parse(text));
}

interface GexfWalk {
    nodeIds: string[];
    edges: { source: string; target: string }[];
    undeclared: string[];
    declaredCount: string | undefined;
}

function walkGexf(doc: XmlNode): GexfWalk {
    const root = asNode(doc.gexf);
    const graph = asArray(root.graph)[0] ?? {};
    const declared = new Set<string>();
    for (const group of asArray(graph.attributes)) {
        for (const attribute of asArray(group.attribute)) {
            declared.add(`${String(group["@class"])}:${String(attribute["@id"])}`);
        }
    }
    const out: GexfWalk = { nodeIds: [], edges: [], undeclared: [], declaredCount: undefined };
    const nodes = asNode(graph.nodes);
    out.declaredCount = typeof nodes["@count"] === "string" ? nodes["@count"] : undefined;
    const walkNodes = (list: XmlNode[]): void => {
        for (const node of list) {
            out.nodeIds.push(String(node["@id"]));
            for (const av of asArray(asNode(node.attvalues).attvalue)) {
                if (!declared.has(`node:${String(av["@for"])}`)) {
                    out.undeclared.push(`node attvalue for=${String(av["@for"])}`);
                }
            }
            walkNodes(asArray(asNode(node.nodes).node));
        }
    };
    walkNodes(asArray(nodes.node));
    for (const edge of asArray(asNode(graph.edges).edge)) {
        out.edges.push({ source: String(edge["@source"]), target: String(edge["@target"]) });
        for (const av of asArray(asNode(edge.attvalues).attvalue)) {
            if (!declared.has(`edge:${String(av["@for"])}`)) {
                out.undeclared.push(`edge attvalue for=${String(av["@for"])}`);
            }
        }
    }
    return out;
}

interface GraphmlWalk {
    nodeIds: string[];
    edges: { source: string; target: string }[];
    undeclared: string[];
}

function walkGraphml(doc: XmlNode): GraphmlWalk {
    const root = asNode(doc.graphml);
    const keys = new Set(asArray(root.key).map((k) => String(k["@id"])));
    const out: GraphmlWalk = { nodeIds: [], edges: [], undeclared: [] };
    const checkData = (element: XmlNode, what: string): void => {
        for (const data of asArray(element.data)) {
            if (!keys.has(String(data["@key"]))) {
                out.undeclared.push(`${what} data key=${String(data["@key"])}`);
            }
        }
    };
    const walkGraph = (graph: XmlNode): void => {
        checkData(graph, "graph");
        for (const node of asArray(graph.node)) {
            out.nodeIds.push(String(node["@id"]));
            checkData(node, "node");
            for (const nested of asArray(node.graph)) {
                walkGraph(nested);
            }
        }
        for (const edge of asArray(graph.edge)) {
            out.edges.push({ source: String(edge["@source"]), target: String(edge["@target"]) });
            checkData(edge, "edge");
        }
    };
    for (const graph of asArray(root.graph)) {
        walkGraph(graph);
    }
    return out;
}

// ============================================================ a strict RFC 4180 reader

/**
 * Parse RFC 4180 text: fields separated by `delimiter`, a field containing the delimiter, a quote,
 * CR or LF must be quoted, quotes inside quoted fields are doubled, records end with CRLF or LF
 * (a lone CR is an error), every record has the header's field count.
 */
function parseRfc4180(text: string, delimiter = ","): string[][] {
    const records: string[][] = [];
    let record: string[] = [];
    let i = 0;
    const n = text.length;
    while (i < n) {
        let field = "";
        if (text[i] === '"') {
            i++;
            for (;;) {
                if (i >= n) {
                    throw new Error(`unterminated quoted field at offset ${i}`);
                }
                const ch = text[i];
                if (ch === '"') {
                    if (text[i + 1] === '"') {
                        field += '"';
                        i += 2;
                        continue;
                    }
                    i++;
                    break;
                }
                field += ch;
                i++;
            }
            if (i < n && text[i] !== delimiter && text[i] !== "\r" && text[i] !== "\n") {
                throw new Error(`text after a closing quote at offset ${i}: ${JSON.stringify(text.slice(i, i + 12))}`);
            }
        } else {
            while (i < n && text[i] !== delimiter && text[i] !== "\r" && text[i] !== "\n") {
                if (text[i] === '"') {
                    throw new Error(`bare quote inside an unquoted field at offset ${i}`);
                }
                field += text[i];
                i++;
            }
        }
        record.push(field);
        if (i >= n) {
            break;
        }
        if (text[i] === delimiter) {
            i++;
            if (i >= n) {
                record.push("");
            }
            continue;
        }
        if (text[i] === "\r") {
            if (text[i + 1] !== "\n") {
                throw new Error(`a lone CR at offset ${i}`);
            }
            i += 2;
        } else {
            i++;
        }
        records.push(record);
        record = [];
    }
    if (record.length > 0) {
        records.push(record);
    }
    const width = records[0]?.length ?? 0;
    records.forEach((r, k) => {
        if (r.length !== width) {
            throw new Error(`record ${k} has ${r.length} field(s), the header ${width}`);
        }
    });
    return records;
}

/** Split a neo4j-admin bundle into its sections: a header row is one whose cells carry a `:ID` / `:START_ID` marker. */
function neo4jSections(text: string): string[] {
    const lines = text.split("\n");
    const sections: string[][] = [];
    for (const line of lines) {
        if (/(^|,)[^,]*:(ID|START_ID)(\(|,|$)/.test(line)) {
            sections.push([line]);
        } else if (sections.length > 0) {
            sections[sections.length - 1].push(line);
        }
    }
    return sections.map((s) => s.join("\n"));
}

function idText(snapshot: GraphSnapshot, index: number): string {
    return String(snapshot.ids.idOf(index));
}

// ============================================================ XML formats over the corpus

describe("independent XML reader: GEXF 1.3, GEXF 1.2 and GraphML exports of every corpus file", () => {
    for (const input of INPUTS) {
        for (const version of ["1.3", "1.2"] as const) {
            it(`${input.label} as GEXF ${version} is well-formed and reads to the snapshot's nodes, edges and declarations`, async () => {
                const snapshot = await original(input);
                const text = await gexfExporter.exportToString(snapshot, { version });
                expect(XML_1_0_ILLEGAL.exec(text), "an XML 1.0 illegal character is written literally").toBeNull();
                const walk = walkGexf(readXml(text));
                expect(walk.undeclared, "attvalues naming undeclared attributes").toEqual([]);
                expect(walk.nodeIds.length).toBe(snapshot.nodeCount);
                expect(new Set(walk.nodeIds).size, "duplicate node ids in the document").toBe(snapshot.nodeCount);
                if (walk.declaredCount !== undefined) {
                    expect(Number(walk.declaredCount)).toBe(snapshot.nodeCount);
                }
                for (let i = 0; i < snapshot.nodeCount; i++) {
                    expect(walk.nodeIds).toContain(idText(snapshot, i));
                }
                expect(walk.edges.length).toBe(writtenEdges(snapshot, true));
                const ids = new Set(walk.nodeIds);
                for (const edge of walk.edges) {
                    expect(
                        ids.has(edge.source) && ids.has(edge.target),
                        `dangling edge ${edge.source} -> ${edge.target}`,
                    ).toBe(true);
                }
            });
        }

        it(`${input.label} as GraphML is well-formed and reads to the snapshot's nodes, edges and keys`, async () => {
            const snapshot = await original(input);
            const text = await graphmlExporter.exportToString(snapshot, { sanitizeIds: "mangle" });
            expect(XML_1_0_ILLEGAL.exec(text), "an XML 1.0 illegal character is written literally").toBeNull();
            const walk = walkGraphml(readXml(text));
            expect(walk.undeclared, "data elements naming undeclared keys").toEqual([]);
            expect(walk.nodeIds.length).toBe(snapshot.nodeCount);
            expect(new Set(walk.nodeIds).size, "duplicate node ids in the document").toBe(snapshot.nodeCount);
            expect(walk.edges.length).toBe(writtenEdges(snapshot, true));
            const ids = new Set(walk.nodeIds);
            for (const edge of walk.edges) {
                expect(
                    ids.has(edge.source) && ids.has(edge.target),
                    `dangling edge ${edge.source} -> ${edge.target}`,
                ).toBe(true);
            }
        });
    }
});

// ============================================================ CSV formats over the corpus

describe("RFC 4180 reader: CSV and Neo4j exports of every corpus file", () => {
    for (const input of INPUTS) {
        it(`${input.label} as a Gephi edge table, a node table and a generic edge table is RFC 4180 with the right rows`, async () => {
            const snapshot = await original(input);
            const edges = parseRfc4180(await csvExporter.exportToString(snapshot));
            expect(edges.length - 1, "edge rows").toBe(writtenEdges(snapshot, false));
            const header = edges[0];
            const source = header.indexOf("Source");
            const target = header.indexOf("Target");
            expect(source).toBeGreaterThanOrEqual(0);
            expect(target).toBeGreaterThanOrEqual(0);
            const ids = new Set<string>();
            for (let i = 0; i < snapshot.nodeCount; i++) {
                ids.add(idText(snapshot, i));
            }
            for (const row of edges.slice(1)) {
                expect(
                    ids.has(row[source]) && ids.has(row[target]),
                    `endpoint cells ${row[source]} -> ${row[target]} are not node ids`,
                ).toBe(true);
            }
            const nodes = parseRfc4180(await csvExporter.exportToString(snapshot, { table: "nodes" }));
            expect(nodes.length - 1, "node rows").toBe(snapshot.nodeCount);
            const idColumn = nodes[0].indexOf("Id");
            expect(idColumn).toBeGreaterThanOrEqual(0);
            expect(nodes.slice(1).map((row) => row[idColumn])).toEqual([...ids]);
            const generic = parseRfc4180(await csvExporter.exportToString(snapshot, { dialect: "generic" }));
            expect(generic.length - 1).toBe(writtenEdges(snapshot, false));
        });

        it(`${input.label} as neo4j-admin CSV: every section is RFC 4180 with the right rows`, async () => {
            const snapshot = await original(input);
            let text: string;
            try {
                text = await neo4jExporter.exportToString(snapshot, { onMixedDirection: "directed" });
            } catch (err) {
                const notes = neo4jExporter.check(snapshot, { onMixedDirection: "directed" });
                expect(
                    notes.some((n) => n.code.startsWith("E_")),
                    `export threw ${String(err)} without an E_ note`,
                ).toBe(true);
                return;
            }
            const sections = neo4jSections(text);
            expect(sections.length).toBeGreaterThan(0);
            let nodeRows = 0;
            let relationshipRows = 0;
            for (const section of sections) {
                const records = parseRfc4180(section.endsWith("\n") ? section : `${section}\n`);
                const isRelationship = records[0].some((cell) => cell.includes(":START_ID"));
                if (isRelationship) {
                    relationshipRows += records.length - 1;
                } else {
                    nodeRows += records.length - 1;
                }
            }
            expect(nodeRows).toBe(snapshot.nodeCount);
            expect(relationshipRows).toBe(writtenEdges(snapshot, false));
        });
    }
});

// ============================================================ conformance probes

const c = String.fromCharCode;

/** Text values a well-behaved corpus never carries. */
const HOSTILE_TEXT: readonly string[] = [
    "plain",
    'with "quotes"',
    "comma, separated",
    "line\nbreak",
    "cr\rreturn",
    "crlf\r\nboth",
    "tab\tbed",
    "<tag> & entity",
    "]]> cdata end",
    "  padded  ",
    "",
    "single ' quote",
    `unicode ${c(0xe9)} ${c(0x4e2d)} ${c(0xd83d)}${c(0xde00)}`,
    `nbsp ${c(0xa0)} here`,
    "semi;colon",
    "pipe|bar",
    "equals=sign",
    "brackets [a, b]",
    "true",
    "123",
    "1e5",
    "-0",
    "00123",
    "null",
];

function hostileSnapshot(values: readonly string[], asIds: boolean): GraphSnapshot {
    const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
    b.declareNodeColumn({ name: "s", dtype: "string" });
    b.declareEdgeColumn({ name: "es", dtype: "string" });
    values.forEach((v, i) => {
        b.addNode(asIds ? v : `n${i}`);
        b.setNodeValue("s", i, v);
    });
    for (let i = 1; i < values.length; i++) {
        b.addEdge(asIds ? values[i - 1] : `n${i - 1}`, asIds ? values[i] : `n${i}`);
        b.setEdgeValue("es", i - 1, values[i]);
    }
    return b.freeze();
}

describe("conformance probes: hostile text through the XML and CSV exporters", () => {
    for (const asIds of [false, true]) {
        const where = asIds ? "as ids and cells" : "as cells";

        it(`GEXF keeps hostile text ${where} byte-exact through an independent reader and its own importer`, async () => {
            const snapshot = hostileSnapshot(HOSTILE_TEXT, asIds);
            const text = await gexfExporter.exportToString(snapshot);
            const walk = walkGexf(readXml(text));
            expect(walk.nodeIds.length).toBe(snapshot.nodeCount);
            const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
            await gexfImporter.import(text, b, { ids: "string" });
            const back = b.freeze();
            expect(back.ids.toArray()).toEqual(snapshot.ids.toArray());
            const s = back.nodes.require("s");
            expect(HOSTILE_TEXT.map((_, i) => s.value(i))).toEqual(HOSTILE_TEXT);
        });

        it(`GraphML keeps hostile text ${where} byte-exact through an independent reader and its own importer`, async () => {
            const snapshot = hostileSnapshot(HOSTILE_TEXT, asIds);
            const text = await graphmlExporter.exportToString(snapshot, { sanitizeIds: "mangle" });
            const walk = walkGraphml(readXml(text));
            expect(walk.nodeIds.length).toBe(snapshot.nodeCount);
            const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
            await graphmlImporter.import(text, b, { ids: "string" });
            const back = b.freeze();
            expect(back.ids.toArray()).toEqual(snapshot.ids.toArray());
            const s = back.nodes.require("s");
            expect(HOSTILE_TEXT.map((_, i) => s.value(i))).toEqual(HOSTILE_TEXT);
        });

        it(`CSV keeps hostile text ${where} RFC 4180-valid and byte-exact (a quoted empty cell is the empty string)`, async () => {
            const snapshot = hostileSnapshot(HOSTILE_TEXT, asIds);
            const notes = csvExporter.check(snapshot, { table: "nodes" });
            const nodeTable = await csvExporter.exportToString(snapshot, { table: "nodes" });
            const records = parseRfc4180(nodeTable);
            expect(records.length - 1).toBe(snapshot.nodeCount);
            const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
            const report = await csvImporter.import(nodeTable, b, { table: "nodes", ids: "string" });
            const back = b.freeze();
            const dropped = snapshot.nodeCount - back.nodeCount;
            if (dropped > 0) {
                // a node lost on re-import must have been predicted
                expect(
                    notes.filter((n) => n.column === null).map((n) => n.code),
                    `${dropped} node(s) vanished (${report.issues.map((i) => i.code).join(", ")}) and check() said:\n${JSON.stringify(notes, null, 1)}`,
                ).not.toEqual([]);
            }
            expect(
                back.nodeCount,
                `${dropped} node(s) vanished: ${report.issues.map((i) => `${i.code} ${i.message}`).join("; ")}`,
            ).toBe(snapshot.nodeCount);
            expect(back.ids.toArray()).toEqual(snapshot.ids.toArray());
            const s = back.nodes.require("s");
            expect(HOSTILE_TEXT.map((_, i) => (s.isSet(i) ? s.value(i) : undefined))).toEqual(HOSTILE_TEXT);
        });

        it(`Neo4j keeps hostile text ${where} RFC 4180-valid and byte-exact`, async () => {
            const snapshot = hostileSnapshot(HOSTILE_TEXT, asIds);
            const notes = neo4jExporter.check(snapshot);
            const text = await neo4jExporter.exportToString(snapshot);
            for (const section of neo4jSections(text)) {
                parseRfc4180(section.endsWith("\n") ? section : `${section}\n`);
            }
            const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
            const report = await neo4jImporter.import(text, b, { ids: "string" });
            const back = b.freeze();
            const dropped = snapshot.nodeCount - back.nodeCount;
            if (dropped > 0) {
                expect(
                    notes.filter((n) => n.column === null).map((n) => n.code),
                    `${dropped} node(s) vanished (${report.issues.map((i) => i.code).join(", ")}) and check() said:\n${JSON.stringify(notes, null, 1)}`,
                ).not.toEqual([]);
            }
            expect(
                back.nodeCount,
                `${dropped} node(s) vanished: ${report.issues.map((i) => `${i.code} ${i.message}`).join("; ")}`,
            ).toBe(snapshot.nodeCount);
            expect(back.ids.toArray()).toEqual(snapshot.ids.toArray());
            expect(back.edgeCount).toBe(snapshot.edgeCount);
            const s = back.nodes.require("s");
            expect(HOSTILE_TEXT.map((_, i) => s.value(i))).toEqual(HOSTILE_TEXT);
        });
    }

    it("GEXF and GraphML never write an XML 1.0 illegal character literally (a conforming parser rejects the file)", async () => {
        const b = new GraphBuilder({ directed: true, weightDtype: "f64" });
        b.declareNodeColumn({ name: "s", dtype: "string" });
        b.addNode("a");
        b.addNode("b");
        b.addEdge("a", "b");
        b.setNodeValue("s", 0, `ctrl ${c(1)} char`);
        b.setNodeValue("s", 1, `form ${c(0x0c)} feed`);
        const snapshot = b.freeze();
        // either escaped as a character reference (still illegal in XML 1.0, but at least visible) or
        // refused: check() names the cells (E_XML_ILLEGAL_CHAR) and export() throws before writing
        for (const [name, exporter] of [
            ["GEXF", gexfExporter],
            ["GraphML", graphmlExporter],
        ] as const) {
            const notes = exporter.check(snapshot);
            let text: string | null = null;
            let refused: unknown = null;
            try {
                text = await exporter.exportToString(snapshot);
            } catch (err) {
                refused = err;
            }
            if (text !== null) {
                expect(
                    XML_1_0_ILLEGAL.exec(text),
                    `${name} writes ${JSON.stringify(XML_1_0_ILLEGAL.exec(text)?.[0])} literally; check(): ${JSON.stringify(notes)}`,
                ).toBeNull();
            } else {
                expect(refused, name).toMatchObject({ code: "E_COLUMN_TYPE" });
                expect(
                    notes.filter((n) => n.code === "E_XML_ILLEGAL_CHAR").map((n) => [n.column, n.count]),
                    name,
                ).toEqual([["s", 2]]);
            }
        }
    });
});
