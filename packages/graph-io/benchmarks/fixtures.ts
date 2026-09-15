/**
 * Deterministic large fixtures for the streaming / performance audit (design sections 8.4 and
 * 15.5 item 5): edge lists of one million edges and XML documents of hundreds of thousands of
 * edges, generated once under `<workspace>/tmp/io-bench/` (two levels above this package) and
 * reused by `test/audit/streaming-*.test.ts` (gated on IO_BENCH=1) and `benchmarks/import.bench.ts`.
 *
 * Every file is generated from a linear congruential generator seeded by the file name, so the
 * same name always yields the same bytes and a missing file is rebuilt identically. Generation is
 * synchronous and buffered a megabyte at a time; a 1M-edge file takes about a second.
 */

import { closeSync, existsSync, mkdirSync, openSync, renameSync, statSync, writeSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/** The directory the fixtures live in: `<workspace>/tmp/io-bench`. */
const FIXTURE_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "tmp", "io-bench");

/** A pseudo-random sequence in [0, 1) from a seed (Numerical Recipes LCG, 32-bit). */
class Lcg {
    private state: number;

    /**
     * Create a generator.
     * @param seed - any 32-bit integer
     */
    constructor(seed: number) {
        this.state = seed >>> 0;
    }

    /**
     * The next value.
     * @returns a number in [0, 1)
     */
    next(): number {
        this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
        return this.state / 4294967296;
    }

    /**
     * A random integer below a bound.
     * @param bound - the exclusive upper bound
     * @returns an integer in [0, bound)
     */
    int(bound: number): number {
        return Math.floor(this.next() * bound);
    }
}

/**
 * A 32-bit hash of a name, the seed of its generator.
 * @param name - the file name
 * @returns the seed
 */
function seedOf(name: string): number {
    let h = 0x811c9dc5;
    for (let i = 0; i < name.length; i++) {
        h = Math.imul(h ^ name.charCodeAt(i), 0x01000193);
    }
    return h >>> 0;
}

/** A line sink that buffers about a megabyte before each write. */
class LineWriter {
    private readonly fd: number;

    private parts: string[] = [];

    private units = 0;

    /**
     * Open the file for writing.
     * @param path - the file path
     */
    constructor(path: string) {
        this.fd = openSync(path, "w");
    }

    /**
     * Append text.
     * @param text - the text (usually one line with its terminator)
     */
    write(text: string): void {
        this.parts.push(text);
        this.units += text.length;
        if (this.units >= 1048576) {
            this.flush();
        }
    }

    /** Write the buffered text and close the file. */
    close(): void {
        this.flush();
        closeSync(this.fd);
    }

    private flush(): void {
        if (this.parts.length > 0) {
            writeSync(this.fd, this.parts.join(""));
            this.parts = [];
            this.units = 0;
        }
    }
}

/**
 * Generate a file once: the generator runs only when the file is absent or empty, writing to a
 * `.partial` name that is renamed into place when complete (a crashed generation never leaves a
 * truncated fixture behind under the real name).
 * @param name - the file name under FIXTURE_DIR
 * @param generate - writes the content
 * @returns the absolute path
 */
function fixture(name: string, generate: (out: LineWriter, rnd: Lcg) => void): string {
    mkdirSync(FIXTURE_DIR, { recursive: true });
    const path = join(FIXTURE_DIR, name);
    if (existsSync(path) && statSync(path).size > 0) {
        return path;
    }
    const partial = `${path}.partial`;
    const out = new LineWriter(partial);
    generate(out, new Lcg(seedOf(name)));
    out.close();
    renameSync(partial, path);
    return path;
}

/**
 * How endpoint ids are spelled in an edge list.
 * @public part of the csvEdgeList signature
 */
export type IdStyle = "numeric" | "short-string" | "long-string";

/**
 * The text of a node id under a style.
 * @param index - the node number
 * @param style - numeric (`123`), short-string (12 characters, below V8's sliced-string minimum of
 * 13) or long-string (15 characters, above it)
 * @returns the id text
 */
function idText(index: number, style: IdStyle): string {
    switch (style) {
        case "numeric":
            return String(index);
        case "short-string":
            return `n${String(index).padStart(11, "0")}`;
        case "long-string":
            return `node_${String(index).padStart(10, "0")}`;
        default:
            return String(index);
    }
}

/**
 * A generic CSV edge list `source,target,weight` with `nodes` distinct endpoints drawn uniformly
 * (about 21 MB per million edges with numeric ids).
 * @param edges - the edge count
 * @param nodes - the node count
 * @param style - how ids are spelled
 * @returns the path
 */
export function csvEdgeList(edges: number, nodes: number, style: IdStyle = "numeric"): string {
    return fixture(`csv-${style}-${edges}.csv`, (out, rnd) => {
        out.write("source,target,weight\n");
        for (let i = 0; i < edges; i++) {
            const s = idText(rnd.int(nodes), style);
            const t = idText(rnd.int(nodes), style);
            out.write(`${s},${t},${(rnd.next() * 100).toFixed(6)}\n`);
        }
    });
}

/**
 * A Pajek network: `*Vertices n` with a short label per vertex, then `*Arcs` with a value per arc.
 * @param arcs - the arc count
 * @param vertices - the vertex count
 * @returns the path
 */
export function pajekNetwork(arcs: number, vertices: number): string {
    return fixture(`pajek-${vertices}-${arcs}.net`, (out, rnd) => {
        out.write(`*Vertices ${vertices}\n`);
        for (let i = 1; i <= vertices; i++) {
            out.write(`${i} "v${i}"\n`);
        }
        out.write("*Arcs\n");
        for (let i = 0; i < arcs; i++) {
            out.write(`${rnd.int(vertices) + 1} ${rnd.int(vertices) + 1} ${(rnd.next() * 100).toFixed(6)}\n`);
        }
    });
}

/**
 * A neo4j-admin node file `id:ID,name,:LABEL`.
 * @param nodes - the node count
 * @returns the path
 */
export function neo4jNodes(nodes: number): string {
    return fixture(`neo4j-nodes-${nodes}.csv`, (out) => {
        out.write("id:ID,name,:LABEL\n");
        for (let i = 0; i < nodes; i++) {
            out.write(`${i},name${i},Person\n`);
        }
    });
}

/**
 * A neo4j-admin relationship file `:START_ID,:END_ID,:TYPE,weight:double`.
 * @param rels - the relationship count
 * @param nodes - the node count the endpoints are drawn from
 * @returns the path
 */
export function neo4jRelationships(rels: number, nodes: number): string {
    return fixture(`neo4j-rels-${nodes}-${rels}.csv`, (out, rnd) => {
        out.write(":START_ID,:END_ID,:TYPE,weight:double\n");
        for (let i = 0; i < rels; i++) {
            out.write(`${rnd.int(nodes)},${rnd.int(nodes)},KNOWS,${(rnd.next() * 100).toFixed(6)}\n`);
        }
    });
}

/**
 * A GEXF 1.3 document: one double node attribute, edges with ids and weights, one element per line.
 * @param edges - the edge count
 * @param nodes - the node count
 * @returns the path
 */
export function gexfDocument(edges: number, nodes: number): string {
    return fixture(`gexf-${nodes}-${edges}.gexf`, (out, rnd) => {
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n');
        out.write('<gexf xmlns="http://gexf.net/1.3" version="1.3">\n');
        out.write('<graph mode="static" defaultedgetype="directed">\n');
        out.write('<attributes class="node"><attribute id="0" title="score" type="double"/></attributes>\n');
        out.write("<nodes>\n");
        for (let i = 0; i < nodes; i++) {
            out.write(
                `<node id="${i}" label="n${i}"><attvalues><attvalue for="0" value="${(i % 97) / 7}"/></attvalues></node>\n`,
            );
        }
        out.write("</nodes>\n<edges>\n");
        for (let k = 0; k < edges; k++) {
            out.write(
                `<edge id="${k}" source="${rnd.int(nodes)}" target="${rnd.int(nodes)}" weight="${(rnd.next() * 100).toFixed(6)}"/>\n`,
            );
        }
        out.write("</edges>\n</graph>\n</gexf>\n");
    });
}

/**
 * A GraphML document: a double node key and a double edge key, edges with ids, one element per
 * line, or (`oneLine`) the same document without any line break, as non-pretty-printing writers
 * emit it.
 * @param edges - the edge count
 * @param nodes - the node count
 * @param oneLine - whether to omit every line break
 * @returns the path
 */
export function graphmlDocument(edges: number, nodes: number, oneLine = false): string {
    const nl = oneLine ? "" : "\n";
    return fixture(`graphml-${nodes}-${edges}${oneLine ? "-oneline" : ""}.graphml`, (out, rnd) => {
        out.write(`<?xml version="1.0" encoding="UTF-8"?>${nl}`);
        out.write(`<graphml xmlns="http://graphml.graphdrawing.org/xmlns">${nl}`);
        out.write(`<key id="d0" for="node" attr.name="score" attr.type="double"/>${nl}`);
        out.write(`<key id="d1" for="edge" attr.name="weight" attr.type="double"/>${nl}`);
        out.write(`<graph id="G" edgedefault="directed">${nl}`);
        for (let i = 0; i < nodes; i++) {
            out.write(`<node id="n${i}"><data key="d0">${(i % 97) / 7}</data></node>${nl}`);
        }
        for (let k = 0; k < edges; k++) {
            const s = rnd.int(nodes);
            const t = rnd.int(nodes);
            out.write(
                `<edge id="e${k}" source="n${s}" target="n${t}"><data key="d1">${(rnd.next() * 100).toFixed(6)}</data></edge>${nl}`,
            );
        }
        out.write(`</graph>${nl}</graphml>${nl}`);
    });
}

/**
 * The size of a fixture in bytes.
 * @param path - the fixture path
 * @returns the byte length
 */
export function fixtureBytes(path: string): number {
    return statSync(path).size;
}
