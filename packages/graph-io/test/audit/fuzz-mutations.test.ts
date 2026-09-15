/**
 * Fuzz audit, mutation lens (design sections 8.4, 8.6 and 11.1): whatever bytes an importer is
 * handed, it must either leave the sink holding a graph that passes validate("full") or throw
 * ImportError with a report. A TypeError, a RangeError (stack overflow, a Map or typed array
 * beyond its limit), a hang or unbounded memory is a defect.
 *
 * Part 1 mutates every well-formed corpus file with fast-check: truncation at a random offset,
 * flipped and overwritten bytes, injected NUL / control / invalid-UTF-8 bytes, duplicated lines
 * and spans, swapped delimiters, unbalanced quotes and tags (one of the pairing characters
 * removed), format tokens spliced in, and compositions of up to four of those.
 *
 * Part 2 is the structural attack list: JSON nested 10k deep, XML with an internal DTD entity
 * expansion attempt, a 50 MB attribute value in every format, declared counts beyond MAX_COUNT,
 * deep DOT nesting, `__proto__` keys, huge and non-finite numbers, repeated ids. The tests marked
 * FAILS pin the defects found: a repeated edge id in GEXF, Gephi CSV and three JSON dialects is
 * pushed without an issue and the core's freeze() then throws a raw E_DUPLICATE_EDGE_ID (the
 * fast-check runs over the GEXF and Cytoscape corpus files find it through the duplicated-line
 * mutation); a declared count beyond MAX_COUNT lets reserve()'s E_TOO_LARGE escape GEXF and
 * GraphML import() as a raw GraphFormatError; the DOT parser recurses per nesting level and
 * overflows the stack (a RangeError escapes); the Pajek importer keeps materialising vertices
 * after reserve() refused the declared count; JSON merges repeated node ids silently; GEXF keeps
 * an undefined entity reference as literal id text without an issue; and the GraphML tokenizer
 * accepts literal control characters XML forbids.
 */

import { GraphBuilder, GraphFormatError, type GraphSink, type NodeId } from "@graphty/graph-format";
import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { registry } from "../../src/registry.js";
import { type CommonImportOptions, ImportError, type ImportReport } from "../../src/types.js";
import { CORPUS_FORMATS, corpusFiles, type CorpusFormat, readCorpusBytes } from "../helpers/corpus.js";

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/** Largest corpus file mutated by the property tests (root.gv at 23 KB is in; nothing is skipped today). */
const MAX_FUZZED_BYTES = 200 * 1024;

/** The wall-clock budget of one import of a mutated corpus file. */
const HANG_MS = 10_000;

type Outcome =
    | { readonly kind: "snapshot"; readonly report: ImportReport }
    | { readonly kind: "error"; readonly error: ImportError };

/**
 * Import bytes into a fresh builder; a valid snapshot or an ImportError is the only acceptable
 * outcome (anything else propagates and fails the test). Returns the outcome and the wall time.
 */
async function classify(
    format: string,
    input: string | Uint8Array,
    options: CommonImportOptions = {},
): Promise<{ outcome: Outcome; ms: number; sink: GraphBuilder }> {
    const sink = new GraphBuilder({ directed: true, weightDtype: "f64" });
    const t0 = performance.now();
    let report: ImportReport;
    try {
        report = await registry.importer(format).import(input, sink, options);
    } catch (err) {
        if (err instanceof ImportError) {
            expect(err.report.issues.length, "an ImportError must carry at least one issue").toBeGreaterThan(0);
            expect(err.report.errorCount).toBeGreaterThanOrEqual(1);
            return { outcome: { kind: "error", error: err }, ms: performance.now() - t0, sink };
        }
        throw err;
    }
    try {
        const snapshot = sink.freeze();
        snapshot.validate({ level: "full" });
    } catch (err) {
        if (err instanceof GraphFormatError) {
            // the importer returned a report over a sink the core refuses to freeze: the design's
            // per-element catch (8.6) did not run for this element
            throw new Error(
                `${format}: import() returned a report (issues: ${report.issues.map((i) => i.code).join(", ") || "none"}) ` +
                    `but freeze()/validate() throws ${err.code}: ${err.message}`,
            );
        }
        throw err;
    }
    return { outcome: { kind: "snapshot", report }, ms: performance.now() - t0, sink };
}

// ============================================================ part 1: fast-check mutations

const TOKENS: Readonly<Record<CorpusFormat, readonly string[]>> = {
    csv: [
        ",",
        ";",
        "\t",
        '"',
        "\n",
        "\r",
        " ",
        "source",
        "target",
        "weight",
        "Type",
        "Directed",
        "Undirected",
        '""',
        "0",
        "-",
        "e",
        ".",
    ],
    neo4j: [
        ",",
        ";",
        '"',
        "\n",
        ":ID",
        ":LABEL",
        ":TYPE",
        ":START_ID",
        ":END_ID",
        ":int",
        ":string[]",
        "(",
        ")",
        "{",
        "}",
        '""',
    ],
    dot: [
        "{",
        "}",
        "[",
        "]",
        "->",
        "--",
        ";",
        "=",
        '"',
        "<",
        ">",
        "subgraph",
        "digraph",
        "graph",
        "strict",
        "node",
        "edge",
        ":",
        "\\",
        "/*",
        "*/",
        "//",
        "\n",
        "+",
    ],
    gml: [
        "[",
        "]",
        '"',
        "id",
        "node",
        "edge",
        "graph",
        "source",
        "target",
        "directed",
        "label",
        "\n",
        "-",
        ".",
        "e",
        "#",
        "&",
        ";",
        "0",
    ],
    gexf: [
        "<",
        ">",
        "</",
        "/>",
        '"',
        "'",
        "=",
        "&",
        ";",
        "<node",
        "<edge",
        "</node>",
        "</edge>",
        "id=",
        "source=",
        "target=",
        "<attvalue",
        "for=",
        "<!--",
        "-->",
        "<![CDATA[",
        "]]>",
        "&amp;",
        "&#",
        "<?",
        "?>",
        "\n",
        "type=",
        "pid=",
    ],
    graphml: [
        "<",
        ">",
        "</",
        "/>",
        '"',
        "'",
        "=",
        "&",
        ";",
        "<node",
        "<edge",
        "</node>",
        "</edge>",
        "id=",
        "source=",
        "target=",
        "<data",
        "key=",
        "<key",
        "<graph",
        "</graph>",
        "<!--",
        "-->",
        "<![CDATA[",
        "]]>",
        "&amp;",
        "&#",
        "<?",
        "?>",
        "\n",
        "<hyperedge",
        "<endpoint",
        "<port",
    ],
    json: [
        "{",
        "}",
        "[",
        "]",
        '"',
        ":",
        ",",
        "null",
        "true",
        "1e999",
        "-",
        "\\",
        "\\u",
        "id",
        "nodes",
        "links",
        "edges",
        "source",
        "target",
        "elements",
        "data",
        "__proto__",
        "\n",
        "0",
    ],
    pajek: [
        "*Vertices",
        "*Edges",
        "*Arcs",
        "*Matrix",
        "*Edgeslist",
        "*Arcslist",
        '"',
        "\n",
        " ",
        "-",
        ".",
        "0",
        "1",
        "%",
        "[",
        "]",
        "*",
        "ic",
        "bc",
        "l",
        "c",
        "w",
        "p",
    ],
};

type Mutation =
    | { readonly op: "truncate"; readonly at: number }
    | { readonly op: "flip"; readonly at: number; readonly bit: number }
    | { readonly op: "byte"; readonly at: number; readonly value: number }
    | { readonly op: "insert"; readonly at: number; readonly token: string }
    | { readonly op: "insertByte"; readonly at: number; readonly value: number }
    | { readonly op: "delete"; readonly at: number; readonly count: number }
    | { readonly op: "dupLine"; readonly which: number; readonly times: number }
    | { readonly op: "dupSpan"; readonly at: number; readonly count: number; readonly times: number }
    | { readonly op: "swapDelim"; readonly from: string; readonly to: string }
    | { readonly op: "dropChar"; readonly ch: string; readonly which: number };

function mutationArb(format: CorpusFormat, length: number): fc.Arbitrary<readonly Mutation[]> {
    const at = fc.nat({ max: Math.max(0, length) });
    const index = fc.nat({ max: Math.max(0, length - 1) });
    const one: fc.Arbitrary<Mutation> = fc.oneof(
        fc.record({ op: fc.constant("truncate" as const), at }),
        fc.record({ op: fc.constant("flip" as const), at: index, bit: fc.nat({ max: 7 }) }),
        fc.record({ op: fc.constant("byte" as const), at: index, value: fc.nat({ max: 255 }) }),
        fc.record({ op: fc.constant("insert" as const), at, token: fc.constantFrom(...TOKENS[format]) }),
        // NUL, BEL, BS, VT, FF, ESC, DEL, and bytes that are never valid UTF-8 or start a truncated sequence
        fc.record({
            op: fc.constant("insertByte" as const),
            at,
            value: fc.constantFrom(0, 1, 7, 8, 11, 12, 27, 127, 0xfe, 0xff, 0xc0, 0xed, 0xf5),
        }),
        fc.record({ op: fc.constant("delete" as const), at: index, count: fc.nat({ max: 64 }) }),
        fc.record({
            op: fc.constant("dupLine" as const),
            which: fc.nat({ max: 2000 }),
            times: fc.integer({ min: 2, max: 8 }),
        }),
        fc.record({
            op: fc.constant("dupSpan" as const),
            at: index,
            count: fc.nat({ max: 200 }),
            times: fc.integer({ min: 2, max: 20 }),
        }),
        fc.record({
            op: fc.constant("swapDelim" as const),
            from: fc.constantFrom(",", "\t", ";", " ", "\n"),
            to: fc.constantFrom(",", "\t", ";", " ", "\n", ""),
        }),
        fc.record({
            op: fc.constant("dropChar" as const),
            ch: fc.constantFrom('"', "<", ">", "[", "]", "{", "}", "'", "/", "=", "&", ";"),
            which: fc.nat({ max: 2000 }),
        }),
    );
    return fc.array(one, { minLength: 1, maxLength: 4 });
}

function splice(bytes: Uint8Array, at: number, insert: Uint8Array, remove: number): Uint8Array {
    const start = Math.min(at, bytes.length);
    const end = Math.min(start + remove, bytes.length);
    const out = new Uint8Array(bytes.length - (end - start) + insert.length);
    out.set(bytes.subarray(0, start));
    out.set(insert, start);
    out.set(bytes.subarray(end), start + insert.length);
    return out;
}

function applyMutation(bytes: Uint8Array, m: Mutation): Uint8Array {
    const n = bytes.length;
    switch (m.op) {
        case "truncate":
            return bytes.subarray(0, Math.min(m.at, n));
        case "flip": {
            if (n === 0) {
                return bytes;
            }
            const out = new Uint8Array(bytes);
            out[m.at % n] ^= 1 << m.bit;
            return out;
        }
        case "byte": {
            if (n === 0) {
                return bytes;
            }
            const out = new Uint8Array(bytes);
            out[m.at % n] = m.value;
            return out;
        }
        case "insert":
            return splice(bytes, m.at, encoder.encode(m.token), 0);
        case "insertByte":
            return splice(bytes, m.at, Uint8Array.of(m.value), 0);
        case "delete":
            return n === 0 ? bytes : splice(bytes, m.at % n, new Uint8Array(0), m.count);
        case "dupLine": {
            const lines = decoder.decode(bytes).split("\n");
            const which = m.which % lines.length;
            const copies = new Array<string>(m.times).fill(lines[which]);
            lines.splice(which, 0, ...copies);
            return encoder.encode(lines.join("\n"));
        }
        case "dupSpan": {
            if (n === 0) {
                return bytes;
            }
            const start = m.at % n;
            const span = bytes.subarray(start, Math.min(start + m.count, n));
            const repeated = new Uint8Array(span.length * (m.times - 1));
            for (let k = 0; k < m.times - 1; k++) {
                repeated.set(span, k * span.length);
            }
            return splice(bytes, start + span.length, repeated, 0);
        }
        case "swapDelim":
            return encoder.encode(decoder.decode(bytes).split(m.from).join(m.to));
        case "dropChar": {
            const text = decoder.decode(bytes);
            const positions: number[] = [];
            for (let i = 0; i < text.length; i++) {
                if (text[i] === m.ch) {
                    positions.push(i);
                }
            }
            if (positions.length === 0) {
                return bytes;
            }
            const i = positions[m.which % positions.length];
            return encoder.encode(text.slice(0, i) + text.slice(i + 1));
        }
        default:
            return bytes;
    }
}

describe("fuzz audit: mutated corpus files import to a valid snapshot or throw ImportError", () => {
    for (const format of CORPUS_FORMATS) {
        for (const entry of corpusFiles(format)) {
            const bytes = readCorpusBytes(format, entry.path);
            if (bytes.byteLength > MAX_FUZZED_BYTES) {
                continue;
            }
            it(
                `${format}/${entry.path}: 60 fast-check mutation sets`,
                async () => {
                    let slowest = 0;
                    await fc.assert(
                        fc.asyncProperty(mutationArb(format, bytes.byteLength), async (mutations) => {
                            let mutated = bytes;
                            for (const m of mutations) {
                                mutated = applyMutation(mutated, m);
                            }
                            if (mutated.byteLength > 4 * MAX_FUZZED_BYTES) {
                                return;
                            }
                            const { ms } = await classify(format, mutated);
                            slowest = Math.max(slowest, ms);
                            expect(ms, `hang: ${format}/${entry.path} with ${JSON.stringify(mutations)}`).toBeLessThan(
                                HANG_MS,
                            );
                        }),
                        { numRuns: 60, seed: 20260914, endOnFailure: true },
                    );
                    expect(slowest).toBeLessThan(HANG_MS);
                },
                { timeout: 120_000 },
            );
        }
    }
});

describe("fuzz audit: mutated corpus files, looking past the pinned E_DUPLICATE_EDGE_ID freeze defect", () => {
    // The GEXF and Cytoscape files fail the property above through one defect (a duplicated edge
    // line). This run tolerates exactly that freeze error so every other outcome of the same
    // mutations is still checked; delete it once the defect is fixed.
    const affected: readonly { readonly format: CorpusFormat; readonly path: string }[] = [
        { format: "gexf", path: "minimal.gexf" },
        { format: "gexf", path: "lesmiserables.gexf" },
        { format: "gexf", path: "airlines-sample.gexf" },
        { format: "json", path: "cytoscape-format.json" },
    ];
    for (const { format, path } of affected) {
        const bytes = readCorpusBytes(format, path);
        it(
            `${format}/${path}: 60 fast-check mutation sets`,
            async () => {
                await fc.assert(
                    fc.asyncProperty(mutationArb(format, bytes.byteLength), async (mutations) => {
                        let mutated = bytes;
                        for (const m of mutations) {
                            mutated = applyMutation(mutated, m);
                        }
                        if (mutated.byteLength > 4 * MAX_FUZZED_BYTES) {
                            return;
                        }
                        try {
                            const { ms } = await classify(format, mutated);
                            expect(ms).toBeLessThan(HANG_MS);
                        } catch (err) {
                            if (err instanceof Error && err.message.includes("E_DUPLICATE_EDGE_ID")) {
                                return;
                            }
                            throw err;
                        }
                    }),
                    { numRuns: 60, seed: 20260914, endOnFailure: true },
                );
            },
            { timeout: 120_000 },
        );
    }
});

// ============================================================ part 2: structural attacks

const MB = 1024 * 1024;

function xmlGraphml(inner: string, keys = ""): string {
    return `<?xml version="1.0"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns">${keys}<graph id="G" edgedefault="directed">${inner}</graph></graphml>`;
}

function xmlGexf(inner: string, attributes = "", graphAttrs = ""): string {
    return `<?xml version="1.0"?><gexf xmlns="http://www.gexf.net/1.2draft" version="1.2"><graph${graphAttrs}>${attributes}${inner}</graph></gexf>`;
}

/** An internal DTD subset with a nine-level entity expansion (10^9 "lol" when expanded). */
const BILLION_LAUGHS = ((): string => {
    const entities: string[] = ['<!ENTITY lol "lol">'];
    for (let level = 2; level <= 9; level++) {
        const previous = level === 2 ? "lol" : `lol${level - 1}`;
        entities.push(`<!ENTITY lol${level} "${`&${previous};`.repeat(10)}">`);
    }
    return `<!DOCTYPE x [${entities.join("")}]>`;
})();

async function expectSnapshotOrImportError(
    format: string,
    input: string | Uint8Array,
    options?: CommonImportOptions,
): Promise<Outcome> {
    const { outcome, ms } = await classify(format, input, options);
    expect(ms, `${format} took ${ms.toFixed(0)} ms`).toBeLessThan(HANG_MS);
    return outcome;
}

describe("fuzz audit: structural attacks", () => {
    describe("deep nesting", () => {
        it("JSON nested 10k deep inside an attribute value, at the top level and in a Cytoscape data record", async () => {
            const deep = `${"[".repeat(10_000)}${"]".repeat(10_000)}`;
            await expectSnapshotOrImportError("json", `{"nodes":[{"id":"a","v":${deep}}],"links":[]}`);
            await expectSnapshotOrImportError("json", deep);
            await expectSnapshotOrImportError(
                "json",
                `{"elements":{"nodes":[{"data":{"id":"a","x":${deep}}}],"edges":[]}}`,
            );
            const objects = `${'{"a":'.repeat(10_000)}1${"}".repeat(10_000)}`;
            await expectSnapshotOrImportError("json", `{"nodes":[{"id":"a","v":${objects}}],"links":[]}`);
            await expectSnapshotOrImportError(
                "json",
                `{"directed":true,"graph":{"v":${objects}},"nodes":[{"id":"a"}],"links":[]}`,
            );
        });

        it("GML lists nested 10k deep, at the top level and inside a node", async () => {
            await expectSnapshotOrImportError(
                "gml",
                `graph [ node [ id 1 v ${"[ a ".repeat(10_000)}1${" ]".repeat(10_000)} ] ]`,
            );
            await expectSnapshotOrImportError("gml", `${"a [ ".repeat(10_000)}${" ]".repeat(10_000)}`);
        });

        it("GraphML graphs nested 10k deep inside nodes, and 10k unknown elements inside a data element", async () => {
            const ids = Array.from({ length: 10_000 }, (_, i) => `n${i}`);
            await expectSnapshotOrImportError(
                "graphml",
                xmlGraphml(
                    `${ids.map((id) => `<node id="${id}"><graph id="g${id}" edgedefault="directed">`).join("")}${"</graph></node>".repeat(10_000)}`,
                ),
            );
            await expectSnapshotOrImportError(
                "graphml",
                xmlGraphml(
                    `<node id="n"><data key="d">${"<a>".repeat(10_000)}${"</a>".repeat(10_000)}</data></node>`,
                    '<key id="d" for="node" attr.name="d" attr.type="string"/>',
                ),
            );
        });

        it("GEXF elements nested 10k deep (fast-xml-parser's maxNestedTags) and nodes nested 2k deep", async () => {
            await expectSnapshotOrImportError(
                "gexf",
                xmlGexf(
                    `<nodes><node id="a" label="a">${"<x>".repeat(10_000)}${"</x>".repeat(10_000)}</node></nodes><edges/>`,
                ),
            );
            const ids = Array.from({ length: 2000 }, (_, i) => `n${i}`);
            await expectSnapshotOrImportError(
                "gexf",
                xmlGexf(
                    `<nodes>${ids.map((id) => `<node id="${id}" label="${id}"><nodes>`).join("")}${"</nodes></node>".repeat(2000)}</nodes><edges/>`,
                ),
            );
        });

        it("DOT subgraphs nested 1000 deep", async () => {
            await expectSnapshotOrImportError(
                "dot",
                `digraph { ${"subgraph { ".repeat(1000)} a -> b ${"}".repeat(1000)} }`,
            );
            await expectSnapshotOrImportError("dot", `digraph { a -> ${"{ ".repeat(1000)} b ${"}".repeat(1000)} }`);
        });

        it("DOT subgraphs nested 10k deep are an ImportError, never a stack overflow", async () => {
            // FAILS: statementList -> statement -> subgraph -> statementList recurses once per
            // nesting level with no depth limit (src/formats/dot/importer.ts), so ~3000 levels
            // exhaust the default Node stack and a RangeError ("Maximum call stack size exceeded")
            // escapes import() instead of an ImportError; browsers with smaller stacks fail sooner.
            await expectSnapshotOrImportError(
                "dot",
                `digraph { ${"subgraph { ".repeat(10_000)} a -> b ${"}".repeat(10_000)} }`,
            );
            await expectSnapshotOrImportError("dot", `digraph { ${"{ ".repeat(10_000)} a -> b ${"}".repeat(10_000)} }`);
            await expectSnapshotOrImportError("dot", `digraph { a -> ${"{ ".repeat(10_000)} b ${"}".repeat(10_000)} }`);
        });
    });

    describe("XML internal DTD entity expansion", () => {
        it("GraphML: the entity reference is refused (E_XML_SYNTAX), nothing is expanded", async () => {
            const doc = `<?xml version="1.0"?>${BILLION_LAUGHS}<graphml xmlns="http://graphml.graphdrawing.org/xmlns"><graph id="G" edgedefault="directed"><node id="&lol9;"/></graph></graphml>`;
            const outcome = await expectSnapshotOrImportError("graphml", doc);
            expect(outcome.kind).toBe("error");
            if (outcome.kind === "error") {
                expect(outcome.error.report.issues[0].code).toBe("E_XML_SYNTAX");
            }
        });

        it("GEXF: the entity is not expanded", async () => {
            const doc = `<?xml version="1.0"?>${BILLION_LAUGHS}<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2"><graph><nodes><node id="&lol9;" label="&lol9;"/></nodes><edges/></graph></gexf>`;
            const { outcome, sink } = await classify("gexf", doc);
            if (outcome.kind === "snapshot") {
                const snapshot = sink.freeze();
                for (const id of snapshot.ids.toArray()) {
                    expect(String(id).length).toBeLessThan(100);
                }
            }
        });

        it("GEXF: an undefined entity reference is not silently kept as literal id text", async () => {
            // FAILS: processEntities is off and decodeXmlEntities() leaves an unknown named entity
            // as written, so the node id becomes the six characters "&lol9;" with no issue in the
            // report; the same document is E_XML_SYNTAX ("unknown entity") for the GraphML importer.
            const doc = `<?xml version="1.0"?>${BILLION_LAUGHS}<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2"><graph><nodes><node id="&lol9;" label="x"/></nodes><edges/></graph></gexf>`;
            const { outcome, sink } = await classify("gexf", doc);
            if (outcome.kind === "snapshot") {
                expect(
                    outcome.report.issues.length,
                    "an undefined entity kept as text must at least be reported",
                ).toBeGreaterThan(0);
                expect(sink.freeze().ids.toArray()).not.toContain("&lol9;");
            }
        });
    });

    describe("a 50 MB attribute value as one in-memory document", () => {
        const big = "x".repeat(50 * MB);
        const documents: Readonly<Record<CorpusFormat, string>> = {
            json: `{"nodes":[{"id":"a","v":"${big}"}],"links":[]}`,
            graphml: xmlGraphml(
                `<node id="a"><data key="d">${big}</data></node>`,
                '<key id="d" for="node" attr.name="d" attr.type="string"/>',
            ),
            gexf: xmlGexf(`<nodes><node id="a" label="${big}"/></nodes><edges/>`),
            gml: `graph [ node [ id 1 label "${big}" ] ]`,
            dot: `digraph { a [label="${big}"] }`,
            csv: `source,target,label\na,b,"${big}"\n`,
            neo4j: `:ID,name,:LABEL\n1,"${big}",P\n`,
            pajek: `*Vertices 1\n1 "${big}"\n*Edges\n1 1\n`,
        };
        for (const format of CORPUS_FORMATS) {
            it(
                `${format}: completes within 10 s with a snapshot or an ImportError`,
                async () => {
                    const outcome = await expectSnapshotOrImportError(format, documents[format]);
                    expect(outcome.kind).toBe("snapshot");
                },
                { timeout: 60_000 },
            );
        }
    });

    describe("declared counts and numbers beyond what the sink can hold", () => {
        it("Pajek: after reserve() refuses the declared vertex count the importer creates no vertex", async () => {
            // FAILS: `*Vertices 4294967295` makes sink.reserve() throw E_TOO_LARGE, which the
            // importer records as a per-element issue and then continues with vertexCount set, so
            // finishVertices() materialises implicit vertices one addNode() at a time. With a real
            // GraphBuilder that runs for ~9 s and >1 GB until V8's Map limit throws RangeError
            // ("Map maximum size exceeded", recorded as E_PARSE) and the later freeze() throws the
            // same RangeError; a 40-byte file is enough. The declared count must be fatal.
            const inner = new GraphBuilder({ directed: true, weightDtype: "f64" });
            let reserveRefused = false;
            let addedAfterRefusal = 0;
            const sink: GraphSink = new Proxy(inner, {
                get(target, prop, receiver): unknown {
                    if (prop === "reserve") {
                        return (nodes?: number, edges?: number): void => {
                            try {
                                target.reserve(nodes, edges);
                            } catch (err) {
                                reserveRefused = true;
                                throw err;
                            }
                        };
                    }
                    if (prop === "addNode") {
                        return (id: NodeId): number => {
                            if (reserveRefused) {
                                addedAfterRefusal++;
                                if (addedAfterRefusal > 1000) {
                                    throw new RangeError("Map maximum size exceeded");
                                }
                            }
                            return target.addNode(id);
                        };
                    }
                    const value: unknown = Reflect.get(target, prop, receiver);
                    return typeof value === "function"
                        ? (value as (...args: unknown[]) => unknown).bind(target)
                        : value;
                },
            });
            let error: unknown = null;
            try {
                await registry.importer("pajek").import('*Vertices 4294967295\n1 "a"\n*Edges\n1 1\n', sink, {});
            } catch (err) {
                error = err;
            }
            expect(reserveRefused).toBe(true);
            expect(addedAfterRefusal, "vertices materialised after reserve() refused the count").toBe(0);
            expect(error).toBeInstanceOf(ImportError);
        });

        it("Pajek: a vertex count that is not a safe integer, negative or non-numeric is an ImportError", async () => {
            for (const count of ["1e10", "-1", "0x10", "NaN", "Infinity", "3.5", "2 3 4", ""]) {
                const outcome = await expectSnapshotOrImportError("pajek", `*Vertices ${count}\n1 "a"\n*Edges\n1 1\n`);
                if (outcome.kind === "snapshot") {
                    expect(outcome.report.issues.length, `*Vertices ${count} accepted silently`).toBeGreaterThan(0);
                }
            }
        });

        it("GEXF: a declared count beyond MAX_COUNT is an issue or an ImportError, never a raw GraphFormatError", async () => {
            // FAILS: `<nodes count="4294967296">` reaches sink.reserve() unchecked; the E_TOO_LARGE
            // GraphFormatError it throws escapes import() as is (no ImportError, no report).
            const gexf = xmlGexf(
                '<nodes count="4294967296"><node id="a" label="a"/></nodes><edges count="99999999999999999999"/>',
            );
            const a = await expectSnapshotOrImportError("gexf", gexf);
            if (a.kind === "snapshot") {
                expect(a.report.counts.nodes).toBe(1);
            }
        });

        it("GraphML: parse.nodes / parse.edges beyond MAX_COUNT or negative are an issue or an ImportError", async () => {
            const graphml = xmlGraphml('<node id="a"/>').replace(
                "<graph ",
                '<graph parse.nodes="4294967296" parse.edges="-5" ',
            );
            const b = await expectSnapshotOrImportError("graphml", graphml);
            if (b.kind === "snapshot") {
                expect(b.report.counts.nodes).toBe(1);
            }
        });

        it("Pajek: edge endpoints beyond the vertex range, zero, negative, float or huge are skipped with an issue", async () => {
            const outcome = await expectSnapshotOrImportError(
                "pajek",
                '*Vertices 2\n1 "a"\n2 "b"\n*Edges\n1 4294967295\n0 1\n-1 2\n1e3 2\n1.5 2\n1 99999999999999999999\n',
            );
            expect(outcome.kind).toBe("snapshot");
            if (outcome.kind === "snapshot") {
                expect(outcome.report.counts.edges).toBe(0);
                expect(outcome.report.counts.skippedEdges).toBe(6);
            }
        });

        it("JSON: non-finite, huge and non-integer numeric ids and weights", async () => {
            await expectSnapshotOrImportError(
                "json",
                '{"nodes":[{"id":1e400},{"id":-0},{"id":1e21},{"id":9007199254740993}],"links":[{"source":1e21,"target":-0,"weight":1e400},{"source":4294967295,"target":-1},{"source":1e300,"target":0.5}]}',
            );
            await expectSnapshotOrImportError(
                "json",
                '{"nodes":[{"id":null},{"id":true},{"id":{}},{"id":[]}],"links":[{"source":null,"target":true}]}',
            );
            await expectSnapshotOrImportError("json", '{"nodes":1e9,"links":[]}');
        });

        it("GML, CSV and Pajek: huge, hex, non-finite and signed-zero numbers", async () => {
            await expectSnapshotOrImportError(
                "gml",
                "graph [ node [ id 99999999999999999999999 ] node [ id 1e400 ] node [ id -0 ] edge [ source 99999999999999999999999 target 1e400 value 1e999 ] ]",
            );
            await expectSnapshotOrImportError(
                "csv",
                "source,target,weight\na,b,1e400\na,b,NaN\na,b,-0\na,b,0x10\na,b,Infinity\na,b,1_000\na,b,+1\na,b,.5\na,b,5.\n",
            );
            await expectSnapshotOrImportError(
                "pajek",
                '*Vertices 2\n1 "a" 0x10 1e400 -0 NaN Infinity\n2 "b"\n*Edges\n1 2 1e400\n1 2 NaN\n1 2 -0\n1 2 0x10\n',
            );
        });
    });

    describe("duplicate ids the file repeats", () => {
        // FAILS for gexf, csv and the JSON Cytoscape / graphology / vis dialects: the importer
        // declares the edge id column `unique` but pushes the second edge with the same id without
        // an issue, so the core's freeze() throws a raw E_DUPLICATE_EDGE_ID GraphFormatError (no
        // ImportError, nothing in the report); through importGraph() the caller gets that
        // GraphFormatError with no partial report at all. GraphML records the issue and skips.
        const duplicateEdgeIds: Readonly<Record<string, string>> = {
            gexf: xmlGexf(
                '<nodes><node id="a" label="a"/><node id="b" label="b"/></nodes><edges><edge id="0" source="a" target="b"/><edge id="0" source="b" target="a"/></edges>',
            ),
            graphml: xmlGraphml(
                '<node id="a"/><node id="b"/><edge id="e0" source="a" target="b"/><edge id="e0" source="b" target="a"/>',
            ),
            csv: "Source,Target,Type,Id,Label,Weight\na,b,Directed,1,x,1\nb,a,Directed,1,y,2\n",
            "json (cytoscape)":
                '{"elements":{"nodes":[{"data":{"id":"a"}},{"data":{"id":"b"}}],"edges":[{"data":{"id":"e","source":"a","target":"b"}},{"data":{"id":"e","source":"b","target":"a"}}]}}',
            "json (graphology)":
                '{"attributes":{},"nodes":[{"key":"a"},{"key":"b"}],"edges":[{"key":"e","source":"a","target":"b"},{"key":"e","source":"b","target":"a"}]}',
            "json (vis)":
                '{"nodes":[{"id":"a"},{"id":"b"}],"edges":[{"id":"e","from":"a","to":"b"},{"id":"e","from":"b","to":"a"}]}',
        };
        for (const [name, text] of Object.entries(duplicateEdgeIds)) {
            const format = name.split(" ")[0];
            it(`${name}: a repeated edge id is an issue or an ImportError, and the sink still freezes`, async () => {
                const outcome = await expectSnapshotOrImportError(format, text);
                if (outcome.kind === "snapshot") {
                    expect(outcome.report.issues.map((i) => i.code)).toContain("E_DUPLICATE_EDGE_ID");
                }
            });
        }

        it("JSON: a repeated node id is reported, as every other importer does (W_ / E_DUPLICATE_NODE)", async () => {
            // FAILS: node-link and Cytoscape documents with two nodes of the same id import to one
            // node with an empty report; GraphML, GML, CSV, Neo4j and Pajek all record the merge.
            for (const text of [
                '{"nodes":[{"id":"a","v":1},{"id":"a","v":2}],"links":[]}',
                '{"elements":{"nodes":[{"data":{"id":"a"}},{"data":{"id":"a"}}],"edges":[]}}',
            ]) {
                const outcome = await expectSnapshotOrImportError("json", text);
                if (outcome.kind === "snapshot") {
                    // the shared convention (GEXF, GraphML, CSV, Neo4j): a repeated declaration is merged into
                    // the existing node and counted as neither a node nor a skipped node, with a W_DUPLICATE_NODE
                    expect(outcome.report.counts).toMatchObject({ nodes: 1, skippedNodes: 0 });
                    expect(
                        outcome.report.issues.map((i) => i.code),
                        "duplicate node id merged silently",
                    ).toEqual(["W_DUPLICATE_NODE"]);
                }
            }
        });
    });

    describe("hostile attribute names and control characters", () => {
        it("__proto__ / constructor / prototype names never pollute Object.prototype and become ordinary columns", async () => {
            const probe = {} as Record<string, unknown>;
            await expectSnapshotOrImportError(
                "json",
                '{"nodes":[{"id":"a","__proto__":{"polluted":true},"constructor":2,"prototype":3}],"links":[{"source":"a","target":"a","__proto__":{"polluted":true}}]}',
            );
            await expectSnapshotOrImportError(
                "json",
                '{"nodes":[{"id":"__proto__"},{"id":"constructor"}],"links":[{"source":"__proto__","target":"constructor"}]}',
            );
            await expectSnapshotOrImportError(
                "csv",
                "source,target,__proto__,constructor,prototype\n__proto__,constructor,1,2,3\n",
            );
            await expectSnapshotOrImportError("neo4j", ":ID,__proto__,constructor:int,:LABEL\n1,x,2,__proto__\n");
            await expectSnapshotOrImportError(
                "gml",
                'graph [ node [ id 1 __proto__ 2 constructor "x" prototype [ a 1 ] ] ]',
            );
            await expectSnapshotOrImportError(
                "dot",
                "digraph { __proto__ [constructor=1 __proto__=2] __proto__ -> constructor }",
            );
            await expectSnapshotOrImportError(
                "pajek",
                '*Vertices 1\n1 "__proto__" __proto__ 1 constructor 2\n*Edges\n1 1 1 __proto__ 3\n',
            );
            await expectSnapshotOrImportError(
                "graphml",
                xmlGraphml(
                    '<node id="a"><data key="__proto__">x</data><data key="constructor">1</data></node>',
                    '<key id="__proto__" for="node" attr.name="__proto__" attr.type="string"/><key id="constructor" for="node" attr.name="constructor" attr.type="int"/>',
                ),
            );
            await expectSnapshotOrImportError(
                "gexf",
                xmlGexf(
                    '<nodes><node id="a" label="a"><attvalues><attvalue for="__proto__" value="x"/><attvalue for="constructor" value="1"/></attvalues></node></nodes><edges/>',
                    '<attributes class="node"><attribute id="__proto__" title="__proto__" type="string"/><attribute id="constructor" title="constructor" type="integer"/></attributes>',
                ),
            );
            expect(probe.polluted).toBeUndefined();
            expect(Object.prototype.hasOwnProperty.call(Object.prototype, "polluted")).toBe(false);
        });

        it("literal NUL and C0 control characters in ids and values of the text formats", async () => {
            const nul = String.fromCharCode(0);
            const soh = String.fromCharCode(1);
            await expectSnapshotOrImportError("csv", `source,target\na${nul},b${soh}\n`);
            await expectSnapshotOrImportError("neo4j", `:ID,name\n1${nul},a${soh}\n`);
            await expectSnapshotOrImportError("gml", `graph [ node [ id 1 label "a${nul}b" ] ]`);
            await expectSnapshotOrImportError("dot", `digraph { "a${nul}b" -> "c${soh}d" }`);
            await expectSnapshotOrImportError("pajek", `*Vertices 1\n1 "a${nul}b"\n*Edges\n1 1\n`);
            await expectSnapshotOrImportError("json", `{"nodes":[{"id":"a\\u0000"}],"links":[]}`);
        });

        it("GraphML: a literal U+0000 in an attribute value is not well-formed XML (an ImportError)", async () => {
            // FAILS: the tokenizer checks character REFERENCES against the XML Char production
            // (&#0; and &#xD800; are E_XML_SYNTAX) but never the literal characters of the input,
            // so a raw NUL or U+0001 in an id or a data value is accepted and lands in the snapshot.
            const nul = String.fromCharCode(0);
            const outcome = await expectSnapshotOrImportError("graphml", xmlGraphml(`<node id="a${nul}b"/>`));
            expect(outcome.kind).toBe("error");
            const ref = await expectSnapshotOrImportError("graphml", xmlGraphml('<node id="a&#0;b"/>'));
            expect(ref.kind).toBe("error");
        });

        it("GEXF: character references to surrogates and NUL, and literal control characters", async () => {
            const nul = String.fromCharCode(0);
            await expectSnapshotOrImportError(
                "gexf",
                xmlGexf('<nodes><node id="a&#xD800;" label="b&#0;"/></nodes><edges/>'),
            );
            await expectSnapshotOrImportError("gexf", xmlGexf(`<nodes><node id="a${nul}" label="b"/></nodes><edges/>`));
        });

        it("lone surrogates in JSON ids, values and keys", async () => {
            await expectSnapshotOrImportError("json", '{"nodes":[{"id":"\\ud800"}],"links":[]}');
            await expectSnapshotOrImportError("json", '{"nodes":[{"id":"a","v":"\\udc00"}],"links":[]}');
            await expectSnapshotOrImportError("json", '{"nodes":[{"id":"a","\\ud800":1}],"links":[]}');
        });
    });

    describe("wide and repetitive input", () => {
        it(
            "GraphML: 100k attributes on one element and 20k declared keys",
            async () => {
                const attrs = Array.from({ length: 100_000 }, (_, i) => `a${i}="v"`).join(" ");
                await expectSnapshotOrImportError("graphml", xmlGraphml(`<node id="a" ${attrs}/>`));
            },
            { timeout: 60_000 },
        );

        it("GEXF: 100k attributes on one element", async () => {
            const attrs = Array.from({ length: 100_000 }, (_, i) => `x${i}="v"`).join(" ");
            await expectSnapshotOrImportError(
                "gexf",
                xmlGexf(`<nodes><node id="a" label="a" ${attrs}/></nodes><edges/>`),
            );
        });

        it("DOT: a 100k-node edge chain, 50k attribute statements and a 100k-part string concatenation", async () => {
            const chain = Array.from({ length: 100_000 }, (_, i) => `n${i}`).join(" -> ");
            const a = await expectSnapshotOrImportError("dot", `digraph { ${chain} }`);
            expect(a.kind === "snapshot" && a.report.counts.edges).toBe(99_999);
            await expectSnapshotOrImportError(
                "dot",
                `digraph { ${"node [shape=box]; edge [color=red]; graph [x=1]; ".repeat(50_000)} a -> b }`,
            );
            await expectSnapshotOrImportError("dot", `digraph { a [label=${'"x" + '.repeat(100_000)}"y"] }`);
            await expectSnapshotOrImportError(
                "dot",
                `digraph { a [label=<${"<b>".repeat(100_000)}x${"</b>".repeat(100_000)}>] }`,
            );
        });

        it("Pajek: 10k alternating *Edges / *Arcs sections and a 100k-neighbour *Edgeslist row", async () => {
            await expectSnapshotOrImportError(
                "pajek",
                `*Vertices 2\n1 "a"\n2 "b"\n${"*Edges\n1 2\n*Arcs\n2 1\n".repeat(10_000)}`,
            );
            await expectSnapshotOrImportError(
                "pajek",
                `*Vertices 2\n1 "a"\n2 "b"\n*Edgeslist\n1 ${new Array<string>(100_000).fill("2").join(" ")}\n`,
            );
        });

        it("CSV: 100k blank lines, a lone-CR file of 2M rows and a 100k-label Neo4j :LABEL cell", async () => {
            await expectSnapshotOrImportError("csv", "\n".repeat(100_000));
            await expectSnapshotOrImportError("csv", `source,target\r${"a,b\r".repeat(200_000)}`);
            await expectSnapshotOrImportError(
                "neo4j",
                `:ID,:LABEL\n1,${Array.from({ length: 100_000 }, (_, i) => `L${i}`).join(";")}\n`,
            );
        });
    });
});
