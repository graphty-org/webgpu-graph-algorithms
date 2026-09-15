/**
 * Differential tests against the legacy `Graph` class, part 2 (design section 16.2): edge
 * enumeration once per logical edge (`edgeList()`, the arc filter of section 3.3, `coo()`,
 * `edgeSource` / `edgeTarget`) against the accepted command log and against legacy `edges()`, the
 * legacy string-coercing enumeration bug the design retires (section 3.3), the builder's dirty-path
 * incidence queries (section 6.6) against legacy `neighbors()`, and the String(id)-keyed result
 * shapes of section 14.2 (`toRecord` / `toStringMap` / `stringIndex`) against the way the legacy
 * algorithms build them.
 *
 * Two tests in this file FAIL deliberately; each pins a finding of the audit:
 *
 * - "outEdgesOf / inEdgesOf on an undirected builder ..." -- the builder's `outEdgesOf(u)` on an
 *   UNDIRECTED builder returns only the edges whose DECLARED source is u, so the dirty-path answer to
 *   "which edges touch this node" (design 6.6) disagrees with legacy `neighbors()` and with the
 *   snapshot's own row `outArcs(u)`, which holds both orientations (I7).
 * - "toRecord / toStringMap collisions ..." -- when two ids share a String() form (`1` and `"1"`,
 *   legal per 4.1) the legacy `Record<string, number>` results keep the LAST node's value
 *   (assignment in iteration order), while `ids.toRecord()` / `toStringMap()` keep the FIRST
 *   (STATUS.md records "lower index wins"; the design is silent), so the 14.2 facade conversion is
 *   not byte-identical to the legacy result for such graphs.
 */

import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { type GraphSnapshot, type NodeId } from "../../src/types/index.js";
import { LegacyGraph } from "../helpers/legacy-graph.js";

const RUNS = Number(process.env.FC_RUNS ?? "60");

// ============================================================ generators (self-contained)

type IdKind = "ints" | "strings" | "mixed";

const WEIGHTS: readonly (number | undefined)[] = [undefined, 1, 0, -1, 2, 0.5, 0.1, 3.7, 16777217, Infinity];

const STRING_POOL: readonly string[] = [
    "",
    "0",
    "1",
    "01",
    "1.0",
    "-0",
    "a",
    "b",
    "c",
    "z",
    "NaN",
    "__proto__",
    "n-10",
    "n-9",
];
const INT_POOL: readonly number[] = [0, 1, 2, 3, 4, 5, 9, 10, 11, 100, -1, -5, 2 ** 31, 2 ** 53 - 1];

function idOf(kind: IdKind, i: number): NodeId {
    switch (kind) {
        case "ints":
            return i < INT_POOL.length ? INT_POOL[i] : i * 7 + 1000;
        case "strings":
            return i < STRING_POOL.length ? STRING_POOL[i] : `s${i}`;
        case "mixed":
            if (i % 2 === 0) {
                return idOf("ints", i / 2);
            }
            return idOf("strings", (i - 1) / 2);
        default:
            throw new Error(`unknown kind ${String(kind)}`);
    }
}

interface EnumCase {
    readonly directed: boolean;
    readonly allowSelfLoops: boolean;
    readonly allowParallelEdges: boolean;
    readonly idKind: IdKind;
    readonly n: number;
    readonly edges: readonly (readonly [number, number, number | undefined])[];
}

const enumCaseArb: fc.Arbitrary<EnumCase> = fc
    .record({
        directed: fc.boolean(),
        allowSelfLoops: fc.boolean(),
        allowParallelEdges: fc.boolean(),
        idKind: fc.constantFrom<IdKind>("ints", "strings", "mixed"),
        n: fc.oneof(
            { weight: 5, arbitrary: fc.integer({ min: 0, max: 14 }) },
            { weight: 2, arbitrary: fc.integer({ min: 15, max: 300 }) },
            { weight: 1, arbitrary: fc.integer({ min: 1500, max: 2000 }) },
        ),
    })
    .chain((head) => {
        const index = fc.nat({ max: Math.max(0, head.n - 1) });
        return fc.record({
            head: fc.constant(head),
            edges: fc.array(fc.tuple(index, index, fc.constantFrom(...WEIGHTS)), {
                maxLength: head.n === 0 ? 0 : Math.min(4000, 3 * head.n),
                size: "max",
            }),
        });
    })
    .map(({ head, edges }) => ({ ...head, edges }));

interface AcceptedEdge {
    readonly source: NodeId;
    readonly target: NodeId;
    readonly weight: number;
}

function buildBoth(c: EnumCase): { legacy: LegacyGraph; builder: GraphBuilder; accepted: AcceptedEdge[] } {
    const legacy = new LegacyGraph({
        directed: c.directed,
        allowSelfLoops: c.allowSelfLoops,
        allowParallelEdges: c.allowParallelEdges,
    });
    const builder = new GraphBuilder({ directed: c.directed });
    const accepted: AcceptedEdge[] = [];
    for (const [si, ti, w] of c.edges) {
        const source = idOf(c.idKind, si);
        const target = idOf(c.idKind, ti);
        builder.addNode(source);
        builder.addNode(target);
        try {
            legacy.addEdge(source, target, w);
        } catch {
            continue;
        }
        builder.addEdge(source, target, w);
        accepted.push({ source, target, weight: w ?? 1 });
    }
    return { legacy, builder, accepted };
}

function keyOf(id: NodeId): string {
    return typeof id === "number" ? `n:${String(id)}` : `s:${id}`;
}

/** Multiset key of an edge; undirected edges are normalised to an unordered pair. */
function edgeKey(directed: boolean, source: NodeId, target: NodeId, weight: number): string {
    const a = keyOf(source);
    const b = keyOf(target);
    const w = String(Math.fround(weight));
    if (directed || a <= b) {
        return `${a}|${b}|${w}`;
    }
    return `${b}|${a}|${w}`;
}

function sortedKeys(keys: Iterable<string>): string[] {
    return [...keys].sort();
}

// ============================================================ edgeList() vs the command log

describe("differential: edge enumeration once per logical edge (design sections 3.1, 3.3, 7.2)", () => {
    it("edgeList() reproduces the accepted addEdge log: order, declared orientation and weight", () => {
        fc.assert(
            fc.property(enumCaseArb, (c) => {
                const { builder, accepted } = buildBoth(c);
                const s = builder.freeze();
                const list = s.edgeList();
                expect(list.src.length).toBe(accepted.length);
                expect(list.dst.length).toBe(accepted.length);
                expect(list.arc.length).toBe(accepted.length);
                expect(list.arc).toBe(s.edgeToArc);
                const coo = s.coo();
                for (let e = 0; e < accepted.length; e++) {
                    // I14: logical edge index = addEdge order; 3.1: declared orientation kept even when undirected
                    expect(s.ids.idOf(list.src[e])).toBe(accepted[e].source === 0 ? 0 : accepted[e].source);
                    expect(s.ids.idOf(list.dst[e])).toBe(accepted[e].target === 0 ? 0 : accepted[e].target);
                    expect(s.edgeSource(e)).toBe(list.src[e]);
                    expect(s.edgeTarget(e)).toBe(list.dst[e]);
                    const arc = s.edgeToArc[e];
                    expect(s.arcToEdge[arc]).toBe(e);
                    expect(coo.src[arc]).toBe(list.src[e]);
                    expect(coo.dst[arc]).toBe(list.dst[e]);
                    expect(s.arcSource(arc)).toBe(list.src[e]);
                    expect(s.colIdx[arc]).toBe(list.dst[e]);
                    if (list.weights !== null) {
                        expect(Object.is(list.weights[e], Math.fround(accepted[e].weight))).toBe(true);
                    } else {
                        expect(accepted[e].weight).toBe(1);
                    }
                }
                // the arc-side "each edge once" rule of 3.3 selects exactly edgeCount arcs, one per edge
                const seen = new Uint8Array(s.edgeCount);
                let kept = 0;
                for (let a = 0; a < s.arcCount; a++) {
                    if (a === s.edgeToArc[s.arcToEdge[a]]) {
                        kept++;
                        seen[s.arcToEdge[a]] = 1;
                    }
                }
                expect(kept).toBe(s.edgeCount);
                expect(seen.every((flag) => flag === 1)).toBe(true);
                // coo().src is the row of every arc
                for (let a = 0; a < s.arcCount; a++) {
                    expect(coo.src[a]).toBe(s.arcSource(a));
                }
                expect(coo.dst).toBe(s.colIdx);
            }),
            { numRuns: RUNS },
        );
    });

    it("legacy edges() and edgeList() enumerate the same edge multiset on simple graphs with homogeneous ids", () => {
        fc.assert(
            fc.property(enumCaseArb, (c) => {
                fc.pre(!c.allowParallelEdges && c.idKind !== "mixed");
                const { legacy, builder } = buildBoth(c);
                const s = builder.freeze();
                const list = s.edgeList();
                const fromFormat: string[] = [];
                for (let e = 0; e < s.edgeCount; e++) {
                    const w = list.weights === null ? 1 : list.weights[e];
                    fromFormat.push(edgeKey(c.directed, s.ids.idOf(list.src[e]), s.ids.idOf(list.dst[e]), w));
                }
                const fromLegacy: string[] = [];
                for (const edge of legacy.edges()) {
                    fromLegacy.push(edgeKey(c.directed, edge.source, edge.target, edge.weight ?? 1));
                }
                expect(sortedKeys(fromFormat)).toEqual(sortedKeys(fromLegacy));
                expect(legacy.uniqueEdgeCount).toBe(s.edgeCount);
                if (c.directed) {
                    // directed: the legacy yields the stored orientation; the format keeps it as well
                    const oriented = [...legacy.edges()]
                        .map((edge) => `${keyOf(edge.source)}>${keyOf(edge.target)}`)
                        .sort();
                    const formatOriented: string[] = [];
                    for (let e = 0; e < s.edgeCount; e++) {
                        formatOriented.push(`${keyOf(s.ids.idOf(list.src[e]))}>${keyOf(s.ids.idOf(list.dst[e]))}`);
                    }
                    expect(formatOriented.sort()).toEqual(oriented);
                }
            }),
            { numRuns: RUNS },
        );
    });

    it("legacy edges() yields an undirected edge between a number id and a non-numeric string TWICE; the format once (design 3.3)", () => {
        // The legacy `source > edge.target` skip coerces operands: 1 > "a" and "a" > 1 are both false
        // (NaN comparisons), so neither side skips and uniqueEdgeCount over-counts. Design 3.3 retires
        // this comparison; the format enumerates logical edges through edgeToArc instead.
        const legacy = new LegacyGraph({ directed: false });
        legacy.addEdge(1, "a", 2);
        legacy.addEdge(1, "1", 3);
        legacy.addEdge("b", "c", 4);
        expect(legacy.totalEdgeCount).toBe(3);
        expect([...legacy.edges()].length).toBe(5);
        expect(legacy.uniqueEdgeCount).toBe(5);
        const b = new GraphBuilder({ directed: false });
        b.addEdge(1, "a", 2);
        b.addEdge(1, "1", 3);
        b.addEdge("b", "c", 4);
        const s = b.freeze();
        expect(s.edgeCount).toBe(3);
        expect(s.arcCount).toBe(6);
        const list = s.edgeList();
        expect(list.src.length).toBe(3);
        const pairs: string[] = [];
        for (let e = 0; e < s.edgeCount; e++) {
            pairs.push(`${keyOf(s.ids.idOf(list.src[e]))}>${keyOf(s.ids.idOf(list.dst[e]))}`);
        }
        expect(pairs).toEqual(["n:1>s:a", "n:1>s:1", "s:b>s:c"]);
    });

    it("an undirected edge keeps the caller's orientation in edgeList() while the legacy yields it in id order", () => {
        const legacy = new LegacyGraph({ directed: false });
        legacy.addEdge("z", "a", 5);
        legacy.addEdge(10, 9, 6);
        const b = new GraphBuilder({ directed: false });
        b.addEdge("z", "a", 5);
        b.addEdge(10, 9, 6);
        const s = b.freeze();
        const list = s.edgeList();
        expect([s.ids.idOf(list.src[0]), s.ids.idOf(list.dst[0])]).toEqual(["z", "a"]);
        expect([s.ids.idOf(list.src[1]), s.ids.idOf(list.dst[1])]).toEqual([10, 9]);
        // the legacy yields from the row whose key is not greater: ("a", "z") and (9, 10)
        expect([...legacy.edges()].map((e) => [e.source, e.target])).toEqual([
            ["a", "z"],
            [9, 10],
        ]);
    });
});

// ============================================================ builder dirty-path incidence (6.6)

describe("differential: builder incidence lists vs legacy neighbours before a freeze (design section 6.6)", () => {
    /** The ids reachable from index u through the builder's edge lists. */
    function targetsOf(builder: GraphBuilder, edges: Uint32Array, u: number): Set<string> {
        const out = new Set<string>();
        for (const e of edges) {
            const [a, b] = builder.edgeEndpoints(e);
            out.add(keyOf(builder.idOf(a === u ? b : a)));
        }
        return out;
    }

    it("directed: outEdgesOf / inEdgesOf targets equal legacy neighbors() / inNeighbors()", () => {
        fc.assert(
            fc.property(enumCaseArb, (c) => {
                fc.pre(c.directed);
                const { legacy, builder } = buildBoth(c);
                for (let u = 0; u < builder.nodeBound; u++) {
                    const id = builder.idOf(u);
                    const out = new Set<string>();
                    for (const e of builder.outEdgesOf(u)) {
                        out.add(keyOf(builder.idOf(builder.edgeEndpoints(e)[1])));
                    }
                    const inn = new Set<string>();
                    for (const e of builder.inEdgesOf(u)) {
                        inn.add(keyOf(builder.idOf(builder.edgeEndpoints(e)[0])));
                    }
                    expect(out).toEqual(new Set([...legacy.neighbors(id)].map(keyOf)));
                    expect(inn).toEqual(new Set([...legacy.inNeighbors(id)].map(keyOf)));
                }
            }),
            { numRuns: RUNS },
        );
    });

    it("undirected: the union of outEdgesOf and inEdgesOf equals legacy neighbors(), and findEdges sees both orientations", () => {
        fc.assert(
            fc.property(enumCaseArb, (c) => {
                fc.pre(!c.directed);
                const { legacy, builder } = buildBoth(c);
                for (let u = 0; u < builder.nodeBound; u++) {
                    const id = builder.idOf(u);
                    const union = new Set<string>([
                        ...targetsOf(builder, builder.outEdgesOf(u), u),
                        ...targetsOf(builder, builder.inEdgesOf(u), u),
                    ]);
                    expect(union).toEqual(new Set([...legacy.neighbors(id)].map(keyOf)));
                }
            }),
            { numRuns: RUNS },
        );
    });

    it("FINDING: outEdgesOf / inEdgesOf on an undirected builder answer with the declared orientation only, unlike legacy neighbors() and the snapshot row", () => {
        // Design 6.6: the incidence queries exist so an owner can answer "which edges touch this node"
        // while dirty, switching to the snapshot when clean. On an undirected graph the snapshot's
        // row (outArcs / outDegreeOf / outDegree) holds BOTH orientations (I7, 3.3), and inDegree()
        // aliases outDegree() (3.4), so the pre-freeze and post-freeze answers must agree.
        const legacy = new LegacyGraph({ directed: false });
        legacy.addEdge("a", "b");
        legacy.addEdge("c", "a");
        legacy.addEdge("a", "a");
        const b = new GraphBuilder({ directed: false });
        b.addEdge("a", "b");
        b.addEdge("c", "a");
        b.addEdge("a", "a");
        const a = b.indexOf("a");
        const s = b.freeze();
        const rowTargets = Array.from(s.colIdx.subarray(s.rowPtr[a], s.rowPtr[a + 1])).map((v) => s.ids.idOf(v));
        expect(rowTargets).toEqual(["a", "b", "c"]);
        expect([...legacy.neighbors("a")].sort()).toEqual(["a", "b", "c"]);
        const viaOut = Array.from(b.outEdgesOf(a))
            .map((e) => {
                const [x, y] = b.edgeEndpoints(e);
                return b.idOf(x === a ? y : x);
            })
            .sort();
        // legacy neighbors(a) === {a, b, c}; the builder's "edges leaving a" on an undirected graph
        // must be the same set (an undirected graph has no in / out distinction, design 3.4)
        expect(viaOut).toEqual(["a", "b", "c"]);
        expect(b.outEdgesOf(a).length).toBe(s.outDegreeOf(a));
        expect(Array.from(b.inEdgesOf(a)).sort()).toEqual(Array.from(b.outEdgesOf(a)).sort());
    });
});

// ============================================================ String(id)-keyed result shapes (14.2)

describe("differential: String(id)-keyed result shapes vs the legacy algorithms (design section 14.2)", () => {
    /** How the legacy algorithms build a Record result: assignment in node iteration order. */
    function legacyRecord(legacy: LegacyGraph, valueOf: (id: NodeId) => number): Record<string, number> {
        const out: Record<string, number> = {};
        for (const node of legacy.nodes()) {
            out[String(node.id)] = valueOf(node.id);
        }
        return out;
    }

    function legacyStringMap(legacy: LegacyGraph, valueOf: (id: NodeId) => number): Map<string, number> {
        const out = new Map<string, number>();
        for (const node of legacy.nodes()) {
            out.set(String(node.id), valueOf(node.id));
        }
        return out;
    }

    function degreeVector(s: GraphSnapshot): Float64Array {
        const out = new Float64Array(s.nodeCount);
        const degree = s.degree();
        for (let i = 0; i < s.nodeCount; i++) {
            out[i] = degree[i];
        }
        return out;
    }

    it("toRecord / toStringMap / toMap / stringIndex equal the legacy shapes when no two ids share a String() form", () => {
        fc.assert(
            fc.property(enumCaseArb, (c) => {
                const { legacy, builder } = buildBoth(c);
                const s = builder.freeze();
                const strings = new Set<string>();
                for (const id of s.ids) {
                    strings.add(String(id));
                }
                fc.pre(strings.size === s.nodeCount);
                // a plain-object legacy record LOSES an id "__proto__" (the assignment sets the prototype);
                // toRecord() returns a null-prototype object and keeps it -- documented below
                fc.pre(!strings.has("__proto__"));
                const vec = degreeVector(s);
                // the legacy degree is the format's degree() on a directed graph and outDegree() on an
                // undirected one (3.4); use the format's own vector on both sides so only the keying is tested
                const valueOf = (id: NodeId): number => vec[s.ids.requireIndex(id)];
                expect(Object.entries(s.ids.toRecord(vec))).toEqual(Object.entries(legacyRecord(legacy, valueOf)));
                expect([...s.ids.toStringMap(vec)]).toEqual([...legacyStringMap(legacy, valueOf)]);
                const asMap = s.ids.toMap(vec);
                for (const node of legacy.nodes()) {
                    const id = node.id === 0 ? 0 : node.id;
                    expect(asMap.get(id)).toBe(valueOf(id));
                    expect(s.ids.stringIndex().get(String(id))).toBe(s.ids.requireIndex(id));
                }
                expect([...s.ids.entries(vec)].length).toBe(s.nodeCount);
            }),
            { numRuns: RUNS },
        );
    });

    it("FINDING: toRecord / toStringMap collisions keep the FIRST id's value; the legacy record keeps the LAST", () => {
        const legacy = new LegacyGraph({ directed: true });
        legacy.addNode(1);
        legacy.addNode("1");
        legacy.addNode("x");
        const b = new GraphBuilder({ directed: true });
        b.addNode(1);
        b.addNode("1");
        b.addNode("x");
        const s = b.freeze();
        const vec = [10, 20, 30];
        const valueOf = (id: NodeId): number => vec[s.ids.requireIndex(id)];
        const expected = legacyRecord(legacy, valueOf);
        expect(expected).toEqual({ "1": 20, x: 30 });
        // the 14.2 facade conversion of a Record<string, number> result must reproduce the legacy result
        expect(Object.entries(s.ids.toRecord(vec))).toEqual(Object.entries(expected));
        expect([...s.ids.toStringMap(vec)]).toEqual([...legacyStringMap(legacy, valueOf)]);
    });

    it('an id "__proto__" is an ordinary key of toRecord() but is lost by the legacy plain-object record', () => {
        // Legacy: `centrality[node.id.toString()] = v` on a `{}` literal sets the prototype for the id
        // "__proto__", so the value never appears in Object.entries(). The format's null-prototype
        // record keeps it; this is a deliberate improvement, not a mismatch to preserve.
        const legacy = new LegacyGraph({ directed: true });
        legacy.addNode("__proto__");
        legacy.addNode("k");
        const b = new GraphBuilder({ directed: true });
        b.addNode("__proto__");
        b.addNode("k");
        const s = b.freeze();
        const vec = [7, 8];
        const valueOf = (id: NodeId): number => vec[s.ids.requireIndex(id)];
        expect(Object.entries(legacyRecord(legacy, valueOf))).toEqual([["k", 8]]);
        expect(Object.entries(s.ids.toRecord(vec))).toEqual([
            ["__proto__", 7],
            ["k", 8],
        ]);
        expect(Object.getPrototypeOf(s.ids.toRecord(vec))).toBeNull();
    });

    it("stringIndex() resolves a String(id) the way the legacy string-typed parameters did for homogeneous ids", () => {
        const legacy = new LegacyGraph({ directed: false });
        legacy.addEdge(5, 6);
        legacy.addEdge(6, 7);
        const b = new GraphBuilder({ directed: false });
        b.addEdge(5, 6);
        b.addEdge(6, 7);
        const s = b.freeze();
        // 14.2: a numeric-id graph called with "5" resolves through stringIndex() after indexOf misses
        expect(s.ids.indexOf("5")).toBe(INVALID_INDEX);
        expect(s.ids.stringIndex().get("5")).toBe(0);
        expect(legacy.hasNode("5")).toBe(false);
        expect(legacy.hasNode(5)).toBe(true);
    });
});
