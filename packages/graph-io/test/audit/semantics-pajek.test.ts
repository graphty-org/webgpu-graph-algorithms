/**
 * Semantic audit of the Pajek importer against the corpus files themselves (facts derived here by
 * manual line parsing of the raw text) and against the quirks graphty-element's PajekDataSource,
 * NetworkX's pajek.py and research note 07 section 2.5 describe. Tests named "DEFECT:" pin a
 * defect and fail until the importer is fixed.
 */

import { type GraphSnapshot } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { importGraph, type ImportGraphOptions, type ImportGraphResult } from "../../src/registry.js";
import { readCorpusBytes, readCorpusText } from "../helpers/corpus.js";

/** The byte-order mark, built from its code so this file stays ASCII. */
const BOM = String.fromCharCode(0xfeff);

async function load(name: string, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
    return importGraph(readCorpusBytes("pajek", name), { format: "pajek", ...options });
}

async function parse(text: string, options: ImportGraphOptions = {}): Promise<ImportGraphResult> {
    return importGraph(text, { format: "pajek", ...options });
}

interface RawNet {
    readonly declared: number;
    readonly vertices: { number: number; label: string | null; coords: number[] }[];
    readonly lines: { section: "arcs" | "edges"; u: number; v: number; weight: number | null }[];
}

/** A manual line walk of a .net file: the *Vertices count, the vertex lines and the *Arcs / *Edges lines. */
function walk(text: string): RawNet {
    const vertices: RawNet["vertices"] = [];
    const lines: RawNet["lines"] = [];
    let declared = -1;
    let section: "none" | "vertices" | "arcs" | "edges" = "none";
    for (const raw of text.split(/\r?\n/)) {
        const line = raw.trim();
        if (line.length === 0) {
            continue;
        }
        const header = /^\*(\w+)\s*(\d+)?/i.exec(line);
        if (header !== null) {
            const keyword = header[1].toLowerCase();
            if (keyword === "vertices") {
                section = "vertices";
                declared = Number(header[2]);
            } else if (keyword === "arcs") {
                section = "arcs";
            } else if (keyword === "edges") {
                section = "edges";
            } else {
                section = "none";
            }
            continue;
        }
        if (section === "vertices") {
            const m = /^(\d+)(?:\s+"([^"]*)")?((?:\s+-?[\d.]+)*)/.exec(line);
            if (m === null) {
                throw new Error(`unparsed vertex line ${line}`);
            }
            const coords = m[3].trim().length === 0 ? [] : m[3].trim().split(/\s+/).map(Number);
            vertices.push({ number: Number(m[1]), label: m[2] ?? null, coords });
        } else if (section === "arcs" || section === "edges") {
            const parts = line.split(/\s+/);
            lines.push({
                section,
                u: Number(parts[0]),
                v: Number(parts[1]),
                weight: parts.length > 2 ? Number(parts[2]) : null,
            });
        }
    }
    return { declared, vertices, lines };
}

function endpoints(snapshot: GraphSnapshot, e: number): [number, number] {
    const list = snapshot.edgeList();
    return [snapshot.ids.idOf(list.src[e]) as number, snapshot.ids.idOf(list.dst[e]) as number];
}

function checkUndirectedNet(snapshot: GraphSnapshot, net: RawNet): void {
    expect(net.lines.every((l) => l.section === "edges")).toBe(true);
    expect(snapshot.directed).toBe(false);
    expect(snapshot.nodeCount).toBe(net.declared);
    expect(snapshot.edgeCount).toBe(net.lines.length);
    for (let i = 0; i < net.vertices.length; i++) {
        expect(snapshot.ids.idOf(i)).toBe(net.vertices[i].number);
        expect(snapshot.nodes.value("label", i)).toBe(net.vertices[i].label);
    }
    const degree = new Map<number, number>();
    const seen = new Set<string>();
    for (let e = 0; e < net.lines.length; e++) {
        const l = net.lines[e];
        expect(endpoints(snapshot, e)).toEqual([l.u, l.v]);
        expect(l.u).not.toBe(l.v);
        const key = [l.u, l.v].sort((a, b) => a - b).join("|");
        expect(seen.has(key)).toBe(false);
        seen.add(key);
        degree.set(l.u, (degree.get(l.u) ?? 0) + 1);
        degree.set(l.v, (degree.get(l.v) ?? 0) + 1);
    }
    expect(snapshot.selfLoopCount).toBe(0);
    expect(snapshot.flags.multigraph).toBe(false);
    const degrees = snapshot.degree();
    for (const [id, d] of degree) {
        expect(degrees[snapshot.ids.requireIndex(id)]).toBe(d);
    }
}

describe("Pajek corpus facts: simple.net (mixed *Arcs and *Edges, coordinates)", () => {
    const text = readCorpusText("pajek", "simple.net");
    const net = walk(text);

    it("the raw file declares 5 labelled 3D vertices, three arcs then two edges, all weighted", () => {
        expect(net.declared).toBe(5);
        expect(net.vertices.map((v) => [v.number, v.label, v.coords])).toEqual([
            [1, "Node A", [0.1, 0.2, 0]],
            [2, "Node B", [0.3, 0.4, 0]],
            [3, "Node C", [0.5, 0.6, 0]],
            [4, "Node D", [0.7, 0.8, 0]],
            [5, "Node E", [0.9, 1, 0]],
        ]);
        expect(net.lines).toEqual([
            { section: "arcs", u: 1, v: 2, weight: 1 },
            { section: "arcs", u: 2, v: 3, weight: 2 },
            { section: "arcs", u: 3, v: 4, weight: 1.5 },
            { section: "edges", u: 4, v: 5, weight: 1 },
            { section: "edges", u: 1, v: 3, weight: 0.5 },
        ]);
    });

    it("imports a directed graph with the two undirected edges expanded into pairs", async () => {
        const { snapshot, report } = await load("simple.net");
        expect(report.errorCount).toBe(0);
        expect(snapshot.directed).toBe(true);
        expect(snapshot.nodeCount).toBe(5);
        expect(snapshot.edgeCount).toBe(7);
        expect(snapshot.arcCount).toBe(7);
        expect(report.counts.expandedMixed).toBe(2);
        expect(snapshot.ids.toArray()).toEqual([1, 2, 3, 4, 5]);
        expect([0, 1, 2].map((e) => endpoints(snapshot, e))).toEqual([
            [1, 2],
            [2, 3],
            [3, 4],
        ]);
        const directed = snapshot.edges.byRole("directed");
        const pair = snapshot.edges.byRole("pair");
        expect([0, 1, 2].map((e) => directed?.value(e))).toEqual([true, true, true]);
        const rest = [3, 4, 5, 6].map((e) => endpoints(snapshot, e).join("-"));
        expect(rest).toEqual(["4-5", "5-4", "1-3", "3-1"]);
        expect([3, 4, 5, 6].map((e) => directed?.value(e))).toEqual([false, false, false, false]);
        expect(pair?.value(3)).toBe(4);
        expect(pair?.value(6)).toBe(5);
        const weights = Array.from(snapshot.edgeList().weights ?? []);
        expect(weights).toEqual([1, 2, 1.5, 1, 1, 0.5, 0.5]);
    });

    it("reads labels and coordinates into the label and position roles", async () => {
        const { snapshot } = await load("simple.net");
        const position = snapshot.nodes.byRole("position");
        expect(position?.meta.components).toBe(3);
        expect(position?.meta.extra).toMatchObject({ sourceDims: 3, units: "file" });
        for (let i = 0; i < 5; i++) {
            expect(snapshot.nodes.value("label", i)).toBe(net.vertices[i].label);
            expect(Array.from(position?.value(i) as ArrayLike<number>)).toEqual(
                net.vertices[i].coords.map(Math.fround),
            );
        }
        expect(snapshot.nodes.get("label")?.meta.role).toBe("label");
    });
});

describe("Pajek corpus facts: football.net and dolphins.net (zero-based numbering)", () => {
    it("football: 115 vertices numbered from 0, 613 unweighted edges, a zero-based warning", async () => {
        const net = walk(readCorpusText("pajek", "football.net"));
        expect(net.declared).toBe(115);
        expect(net.vertices).toHaveLength(115);
        expect(net.vertices[0]).toEqual({ number: 0, label: "BrighamYoung", coords: [] });
        expect(net.vertices[4].label).toBe("NewMexico");
        expect(net.vertices[114]).toEqual({ number: 114, label: "Hawaii", coords: [] });
        expect(net.lines).toHaveLength(613);
        expect(net.lines[0]).toMatchObject({ u: 1, v: 0, weight: null });
        expect(net.lines[612]).toMatchObject({ u: 114, v: 104 });
        const { snapshot, report } = await load("football.net");
        checkUndirectedNet(snapshot, net);
        expect(snapshot.ids.kind).toBe("identity");
        expect(snapshot.ids.offset).toBe(0);
        expect(snapshot.flags.weighted).toBe(false);
        expect(report.issues.map((i) => i.code)).toEqual(["W_PAJEK_ZERO_BASED"]);
        expect(snapshot.degree()[snapshot.ids.requireIndex(0)]).toBe(12);
    });

    it("dolphins: 62 vertices numbered from 0 and 159 edges in file order", async () => {
        const net = walk(readCorpusText("pajek", "dolphins.net"));
        expect(net.declared).toBe(62);
        expect(net.vertices[0].label).toBe("Beak");
        expect(net.vertices[61].label).toBe("Zipfel");
        expect(net.lines).toHaveLength(159);
        expect(net.lines[0]).toMatchObject({ u: 8, v: 3 });
        expect(net.lines[158]).toMatchObject({ u: 61, v: 53 });
        const { snapshot } = await load("dolphins.net");
        checkUndirectedNet(snapshot, net);
        expect(snapshot.degree()[snapshot.ids.requireIndex(14)]).toBe(12);
    });

    it("karate.net: 34 vertices from 1 with Actor labels, 78 edges each carrying weight 1", async () => {
        const net = walk(readCorpusText("pajek", "karate.net"));
        expect(net.declared).toBe(34);
        expect(net.vertices[0]).toEqual({ number: 1, label: "Actor 1", coords: [] });
        expect(net.vertices[33].label).toBe("Actor 34");
        expect(net.lines.every((l) => l.weight === 1)).toBe(true);
        expect(net.lines[0]).toMatchObject({ u: 1, v: 2 });
        expect(net.lines[77]).toMatchObject({ u: 33, v: 34 });
        const { snapshot, report } = await load("karate.net");
        checkUndirectedNet(snapshot, net);
        expect(report.warningCount).toBe(0);
        expect(snapshot.ids.offset).toBe(1);
        expect(snapshot.flags.weighted).toBe(true);
        expect(snapshot.flags.allWeightsOne).toBe(true);
        expect(snapshot.degree()[snapshot.ids.requireIndex(34)]).toBe(17);
    });
});

describe("Pajek quirks from PajekDataSource, NetworkX pajek.py and research note 07", () => {
    it("reads lower-case section headers, tab separation, CRLF, a BOM, blank lines and % comment lines", async () => {
        const { snapshot } = await parse(
            `${BOM}% a comment\r\n*vertices 2\r\n\r\n1\t"a"\r\n% mid\r\n2\t"b"\r\n*edges\r\n1\t2\r\n`,
        );
        expect(snapshot.ids.toArray()).toEqual([1, 2]);
        expect(snapshot.edgeCount).toBe(1);
        expect([0, 1].map((i) => snapshot.nodes.value("label", i))).toEqual(["a", "b"]);
    });

    it("records the first-mode size of a two-mode *Vertices N N1 header", async () => {
        const { snapshot } = await parse('*Vertices 3 2\n1 "a"\n2 "b"\n3 "c"\n*Edges\n1 3\n2 3\n');
        expect(snapshot.nodeCount).toBe(3);
        expect(snapshot.meta.extra.pajek).toMatchObject({ firstMode: 2 });
    });

    it('reads *Arcs :k "name" relation headers into a relation column', async () => {
        const { snapshot } = await parse('*Vertices 2\n1 "a"\n2 "b"\n*Arcs :1 "friend"\n1 2\n*Arcs :2 "enemy"\n2 1\n');
        expect(snapshot.directed).toBe(true);
        expect([0, 1].map((e) => snapshot.edges.value("relation", e))).toEqual(["friend", "enemy"]);
    });

    it("reads *Edgeslist / *Arcslist adjacency lists and *Matrix rows", async () => {
        const edgeslist = await parse('*Vertices 4\n1 "a"\n2 "b"\n3 "c"\n4 "d"\n*Edgeslist\n1 2 3\n2 4\n');
        expect(edgeslist.snapshot.directed).toBe(false);
        expect([0, 1, 2].map((e) => endpoints(edgeslist.snapshot, e))).toEqual([
            [1, 2],
            [1, 3],
            [2, 4],
        ]);
        const arcslist = await parse("*Vertices 3\n*Arcslist\n1 2 3\n2 3\n");
        expect(arcslist.snapshot.directed).toBe(true);
        expect(arcslist.snapshot.edgeCount).toBe(3);
        const matrix = await parse('*Vertices 3\n1 "a"\n2 "b"\n3 "c"\n*Matrix\n0 1 0\n0 0 1\n1 0 0\n');
        expect(matrix.snapshot.directed).toBe(true);
        expect([0, 1, 2].map((e) => endpoints(matrix.snapshot, e))).toEqual([
            [1, 2],
            [2, 3],
            [3, 1],
        ]);
    });

    it("accepts vertex lines with only a number, no vertex lines at all, and unquoted one-word labels", async () => {
        const numbers = await parse("*Vertices 3\n1\n2\n3\n*Edges\n1 2\n");
        expect(numbers.snapshot.nodeCount).toBe(3);
        expect(numbers.snapshot.nodes.byRole("label")).toBeNull();
        const none = await parse("*Vertices 3\n*Edges\n1 2\n2 3\n");
        expect(none.snapshot.ids.toArray()).toEqual([1, 2, 3]);
        expect(none.snapshot.edgeCount).toBe(2);
        const unquoted = await parse("*Vertices 2\n1 Alice\n2 42\n*Edges\n1 2\n");
        expect([0, 1].map((i) => unquoted.snapshot.nodes.value("label", i))).toEqual(["Alice", "42"]);
    });

    it("reads the shape keyword, ic / bc colours and line parameters (c, l, w) as attributes", async () => {
        const { snapshot } = await parse(
            '*Vertices 2\n1 "a" 0.1 0.2 0.3 box ic Red bc Black\n2 "b" 0.4 0.5 0.6 ellipse ic Blue\n*Edges\n1 2 3 c Blue l "lab" w 2\n',
        );
        expect([0, 1].map((i) => snapshot.nodes.value("shape", i))).toEqual(["box", "ellipse"]);
        expect(snapshot.nodes.value("ic", 0)).toBe("Red");
        expect(snapshot.nodes.value("bc", 0)).toBe("Black");
        expect(snapshot.nodes.isSet("bc", 1)).toBe(false);
        expect(Array.from(snapshot.edgeList().weights ?? [])).toEqual([3]);
        expect(snapshot.edges.value("c", 0)).toBe("Blue");
        expect(snapshot.edges.value("l", 0)).toBe("lab");
        expect(snapshot.edges.value("w", 0)).toBe(2);
    });

    it("reads [1-5,7-*] time interval tokens into the spells role with an open end as Infinity", async () => {
        const { snapshot } = await parse('*Vertices 2\n1 "a" [1-5,7-*]\n2 "b" [3-*]\n*Edges\n1 2 1 [2-4]\n');
        const spells = snapshot.nodes.byRole("spells");
        const rows = (spells?.value(0) as readonly ArrayLike<number>[]).map((s) => Array.from(s));
        expect(rows).toEqual([
            [1, 5],
            [7, Infinity],
        ]);
        expect(
            (snapshot.edges.byRole("spells")?.value(0) as readonly ArrayLike<number>[]).map((s) => Array.from(s)),
        ).toEqual([[2, 4]]);
    });

    it("takes *Network name into the meta and uses the label as the id under nodeIdFrom: label", async () => {
        const named = await parse('*Network test net\n*Vertices 2\n1 "a"\n2 "b"\n*Edges\n1 2\n');
        expect(named.snapshot.meta.name).toBe("test net");
        const byLabel = await parse('*Vertices 2\n1 "a"\n2 "b"\n*Edges\n1 2\n', { nodeIdFrom: "label" });
        expect(byLabel.snapshot.ids.toArray()).toEqual(["a", "b"]);
        expect(byLabel.snapshot.edgeCount).toBe(1);
    });

    it("DEFECT: a .paj project with a *Partition after the network must not abort on the partition's *Vertices line", async () => {
        // Pajek project files (.paj, an advertised extension) write every partition / vector as
        // "*Partition name" followed by its own "*Vertices N" and N value lines. The importer marks
        // *Partition unsupported and skips its lines, but then reads that *Vertices as a second
        // network and aborts the whole import, so no .paj with a partition or vector can be read.
        const paj =
            '*Network n\n*Vertices 2\n1 "a"\n2 "b"\n*Edges\n1 2\n*Partition C1\n*Vertices 2\n1\n2\n*Vector v\n*Vertices 2\n0.5\n0.3\n';
        const { snapshot, report } = await parse(paj);
        expect(snapshot.ids.toArray()).toEqual([1, 2]);
        expect(snapshot.edgeCount).toBe(1);
        expect(report.issues.filter((i) => i.code === "E_PAJEK_UNSUPPORTED_SECTION")).toHaveLength(2);
        expect(report.issues.map((i) => i.code)).not.toContain("E_PAJEK_MULTIPLE_NETWORKS");
    });

    it("reports a vertex count mismatch, an endpoint outside 1..N and a bad weight token, each explicitly", async () => {
        const mismatch = await parse('*Vertices 3\n1 "a"\n2 "b"\n*Edges\n1 2\n');
        expect(mismatch.snapshot.nodeCount).toBe(3);
        expect(mismatch.report.issues.map((i) => i.code)).toContain("E_PAJEK_VERTEX_COUNT");
        const outside = await parse('*Vertices 2\n1 "a"\n2 "b"\n*Edges\n1 3\n');
        expect(outside.snapshot.edgeCount).toBe(0);
        expect(outside.report.issues.map((i) => i.code)).toContain("E_UNKNOWN_NODE");
        const badWeight = await parse('*Vertices 2\n1 "a"\n2 "b"\n*Edges\n1 2 x\n');
        expect(badWeight.snapshot.edgeCount).toBe(0);
        expect(badWeight.report.errorCount).toBe(1);
    });

    it("keeps third-column weights exact (0.1, 16777217) in the f64 shadow and reads negative arc weights", async () => {
        const { snapshot } = await parse('*Vertices 2\n1 "a"\n2 "b"\n*Arcs\n1 2 16777217\n2 1 0.1\n1 1 -1\n');
        const weight = snapshot.edges.byRole("weight");
        expect(weight?.dtype).toBe("f64");
        expect([0, 1, 2].map((e) => weight?.value(e))).toEqual([16777217, 0.1, -1]);
        expect(snapshot.edgeList().weights?.[0]).toBe(16777216);
        expect(snapshot.selfLoopCount).toBe(1);
    });

    it("keeps parallel lines, reads *Edges then *Arcs as a mixed graph, and warns when no line section follows", async () => {
        const parallel = await parse('*Vertices 2\n1 "a"\n2 "b"\n*Edges\n1 2\n1 2\n');
        expect(parallel.snapshot.edgeCount).toBe(2);
        expect(parallel.snapshot.flags.multigraph).toBe(true);
        const mixed = await parse('*Vertices 3\n1 "a"\n2 "b"\n3 "c"\n*Edges\n2 3\n*Arcs\n1 2\n');
        expect(mixed.snapshot.directed).toBe(true);
        expect(mixed.snapshot.edgeCount).toBe(3);
        expect(mixed.report.counts.expandedMixed).toBe(1);
        const onlyVertices = await parse('*Vertices 1\n1 "a"\n');
        expect(onlyVertices.report.issues.map((i) => i.code)).toContain("W_PAJEK_NO_LINES");
    });

    it("refuses a line section before *Vertices and a truly second network with explicit errors", async () => {
        await expect(parse("*Edges\n1 2\n")).rejects.toMatchObject({ code: "E_IMPORT" });
        await expect(parse('*Vertices 1\n1 "a"\n*Edges\n*Network other\n*Vertices 1\n1 "b"\n')).rejects.toMatchObject({
            code: "E_IMPORT",
            report: { issues: [{ code: "E_PAJEK_MULTIPLE_NETWORKS" }] },
        });
    });
});
