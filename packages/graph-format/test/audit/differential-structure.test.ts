/**
 * Differential tests against the legacy `Graph` class of `@graphty/algorithms` (design section 16.2;
 * the correspondences of sections 2, 3.3, 3.4, 3.5, 3.9, 4 and 14.2). Random directed and undirected
 * graphs -- with and without self-loops and parallel edges, numeric / string / mixed ids, 0 to 2000
 * nodes -- are fed to a copy of the legacy class (test/helpers/legacy-graph.ts) and to GraphBuilder,
 * and every structural answer the two can both give is compared:
 *
 * - node count, edge count, the counting vocabulary of section 2 (edgeCount vs arcCount vs
 *   selfLoopCount) against the accepted command log;
 * - degree / inDegree / outDegree per node, documenting the self-loop convention difference of
 *   section 3.4 exactly where the design declares it;
 * - neighbour sets from a row scan vs legacy `neighbors()` / `inNeighbors()`;
 * - `hasEdge` vs `findArc` / `hasArc`, `getEdge().weight` vs `weights[arc]`;
 * - the builder's own dirty-path queries (section 6.6) vs the legacy class before any freeze;
 * - id round trips through the id map, including the kind selection rules of section 4.2.
 *
 * Legacy quirks the comparison has to route around are documented at the comparison site: the
 * legacy adjacency Map cannot hold parallel edges (so its degrees count DISTINCT neighbours and
 * `getEdge` returns the LAST parallel), and an undirected self-loop is counted once by the legacy
 * class whereas `degree()` counts it twice (section 3.4).
 */

import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { type GraphSnapshot, type NodeId, type NodeIdMapKind } from "../../src/types/index.js";
import { fromWire } from "../../src/wire/from-wire.js";
import { assertInvariants } from "../helpers/invariants.js";
import { LegacyGraph } from "../helpers/legacy-graph.js";

/** Property runs per test; FC_RUNS raises it for a soak. */
const RUNS = Number(process.env.FC_RUNS ?? "60");

// ============================================================ generators

/** The id pools: each exercises one storage kind of design section 4.2 or a documented edge case. */
type IdKind =
    "identity" | "oneBased" | "hugeOffset" | "negOffset" | "sparseInts" | "floats" | "strings" | "unicode" | "mixed";

const ID_KINDS: readonly IdKind[] = [
    "identity",
    "oneBased",
    "hugeOffset",
    "negOffset",
    "sparseInts",
    "floats",
    "strings",
    "unicode",
    "mixed",
];

/** Weights covering the f32-exact and non-exact cases, zero, negatives and the infinities (3.7). */
const WEIGHTS: readonly (number | undefined)[] = [
    undefined,
    1,
    0,
    -0,
    -1,
    2,
    0.5,
    0.1,
    3.7,
    16777217,
    1e10,
    -1e-3,
    Infinity,
    -Infinity,
    1e300,
    5e-324,
];

const INTERESTING_STRINGS: readonly string[] = [
    "",
    "0",
    "1",
    "01",
    "1.0",
    "-0",
    "-1",
    "a",
    "NaN",
    "Infinity",
    "true",
    "null",
    "undefined",
    "__proto__",
    "constructor",
    "\u00e9",
    "\u65e5\u672c",
    "\ud83d\ude00",
    "a\u0000b",
];

const INTERESTING_NUMBERS: readonly number[] = [
    0,
    -0,
    1,
    -1,
    2.5,
    0.1,
    1e15,
    2 ** 31,
    2 ** 32,
    2 ** 53 - 1,
    -(2 ** 53 - 1),
    4294967294,
    4294967295,
    1e-300,
    5e-324,
    -Number.MAX_VALUE,
];

/** A deterministic distinct id for pools too large to draw uniquely. */
function deterministicId(kind: IdKind, i: number): NodeId {
    switch (kind) {
        case "identity":
            return i;
        case "oneBased":
            return i + 1;
        case "hugeOffset":
            return 2 ** 40 + i;
        case "negOffset":
            return i - 7;
        case "sparseInts":
            return i * 3 - 1000;
        case "floats":
            return i + 0.5;
        case "strings":
            return `node-${i}`;
        case "unicode":
            return `\u00e9${i}\u65e5`;
        case "mixed":
            return i % 2 === 0 ? i / 2 : `${(i - 1) / 2}`;
        default:
            throw new Error(`unknown id kind ${String(kind)}`);
    }
}

/** Whether a string contains a lone UTF-16 surrogate (rejected by the builder, design 4.1). */
function hasLoneSurrogate(s: string): boolean {
    return /[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/u.test(s);
}

/** A random single id of a kind (only used for pools of up to 300 nodes). */
function randomId(kind: IdKind): fc.Arbitrary<NodeId> {
    const sparse = fc.oneof(fc.integer({ min: -1_000_000, max: 1_000_000 }), fc.constantFrom(...INTERESTING_NUMBERS));
    const strings = fc.oneof(fc.string({ maxLength: 6 }), fc.constantFrom(...INTERESTING_STRINGS));
    switch (kind) {
        case "sparseInts":
            return sparse;
        case "floats":
            return fc.oneof(
                fc.double({ noNaN: true, noDefaultInfinity: true }),
                fc.constantFrom(...INTERESTING_NUMBERS),
            );
        case "strings":
            return strings;
        case "unicode":
            return fc
                .oneof(fc.string({ unit: "grapheme", maxLength: 4 }), fc.constantFrom(...INTERESTING_STRINGS))
                .filter((s) => !hasLoneSurrogate(s));
        case "mixed":
            return fc.oneof(sparse, strings);
        default:
            return fc.constant(deterministicId(kind, 0));
    }
}

/** A pool of `n` ids, distinct under SameValueZero, in a fixed pool order. */
function poolArb(kind: IdKind, n: number): fc.Arbitrary<readonly NodeId[]> {
    const deterministic = kind === "identity" || kind === "oneBased" || kind === "hugeOffset" || kind === "negOffset";
    if (deterministic || n > 300) {
        return fc.constant(Array.from({ length: n }, (_, i) => deterministicId(kind, i)));
    }
    return fc.uniqueArray(randomId(kind), { minLength: n, maxLength: n, comparator: "SameValueZero" });
}

/** One raw edge request: either a fresh pair or a repeat (possibly flipped) of an earlier request. */
interface RawEdge {
    readonly s: number;
    readonly t: number;
    readonly w: number | undefined;
    readonly dup: number | null;
    readonly flip: boolean;
    readonly loop: boolean;
}

/** A generated scenario. Indices refer to the pool. */
interface GraphCase {
    readonly directed: boolean;
    readonly allowSelfLoops: boolean;
    readonly allowParallelEdges: boolean;
    readonly weightDtype: "f32" | "f64";
    readonly idKind: IdKind;
    /** Whether every pool id is added with addNode (in pool order) before the edges. */
    readonly preAddNodes: boolean;
    readonly pool: readonly NodeId[];
    readonly rawEdges: readonly RawEdge[];
}

function rawEdgeArb(n: number): fc.Arbitrary<RawEdge> {
    const index = fc.nat({ max: Math.max(0, n - 1) });
    return fc.record({
        s: index,
        t: index,
        w: fc.constantFrom(...WEIGHTS),
        dup: fc.oneof({ weight: 5, arbitrary: fc.constant(null) }, { weight: 2, arbitrary: fc.nat({ max: 4000 }) }),
        flip: fc.boolean(),
        loop: fc.oneof({ weight: 7, arbitrary: fc.constant(false) }, { weight: 1, arbitrary: fc.constant(true) }),
    });
}

const nodeCountArb: fc.Arbitrary<number> = fc.oneof(
    { weight: 6, arbitrary: fc.integer({ min: 0, max: 12 }) },
    { weight: 3, arbitrary: fc.integer({ min: 0, max: 200 }) },
    { weight: 1, arbitrary: fc.integer({ min: 1000, max: 2000 }) },
);

const graphCaseArb: fc.Arbitrary<GraphCase> = fc
    .record({
        directed: fc.boolean(),
        allowSelfLoops: fc.boolean(),
        allowParallelEdges: fc.boolean(),
        weightDtype: fc.constantFrom<"f32" | "f64">("f32", "f64"),
        idKind: fc.constantFrom(...ID_KINDS),
        preAddNodes: fc.boolean(),
        n: nodeCountArb,
    })
    .chain((head) => {
        const maxEdges = head.n === 0 ? 0 : Math.min(4000, 2 * head.n + 4);
        return fc.record({
            head: fc.constant(head),
            pool: poolArb(head.idKind, head.n),
            // size "max": without it fast-check keeps every array near its small default length
            rawEdges: fc.array(rawEdgeArb(head.n), { maxLength: maxEdges, size: "max" }),
        });
    })
    .map(({ head, pool, rawEdges }) => ({
        directed: head.directed,
        allowSelfLoops: head.allowSelfLoops,
        allowParallelEdges: head.allowParallelEdges,
        weightDtype: head.weightDtype,
        idKind: head.idKind,
        preAddNodes: head.preAddNodes,
        pool,
        rawEdges,
    }));

// ============================================================ building both sides

/** An edge the legacy class accepted, in acceptance order (= logical edge index). */
interface AcceptedEdge {
    readonly source: NodeId;
    readonly target: NodeId;
    /** The legacy weight (1 when omitted). */
    readonly weight: number;
    readonly explicit: boolean;
}

interface Built {
    readonly legacy: LegacyGraph;
    readonly builder: GraphBuilder;
    readonly accepted: readonly AcceptedEdge[];
    /** Every id the scenario mentioned, whether or not it ended up in the graph. */
    readonly mentioned: readonly NodeId[];
}

/** Resolve the raw requests into concrete (source, target, weight) requests. */
function resolveRequests(c: GraphCase): { source: NodeId; target: NodeId; weight: number | undefined }[] {
    const out: { source: NodeId; target: NodeId; weight: number | undefined }[] = [];
    for (const raw of c.rawEdges) {
        let { s } = raw;
        let t = raw.loop ? raw.s : raw.t;
        if (raw.dup !== null && raw.dup < out.length) {
            const earlier = out[raw.dup];
            const su = c.pool.indexOf(earlier.source);
            const tu = c.pool.indexOf(earlier.target);
            s = raw.flip ? tu : su;
            t = raw.flip ? su : tu;
        }
        out.push({ source: c.pool[s], target: c.pool[t], weight: raw.w });
    }
    return out;
}

/**
 * Build the legacy graph and the builder from the same requests. The legacy class is the oracle for
 * acceptance: a request it throws on (self-loop or parallel refused by its config) is skipped on
 * both sides, so both structures see the same accepted command log.
 */
function buildBoth(c: GraphCase, options: { weighted?: boolean | "auto" } = {}): Built {
    const legacy = new LegacyGraph({
        directed: c.directed,
        allowSelfLoops: c.allowSelfLoops,
        allowParallelEdges: c.allowParallelEdges,
    });
    const builder = new GraphBuilder({
        directed: c.directed,
        weightDtype: c.weightDtype,
        weighted: options.weighted ?? "auto",
    });
    const mentioned: NodeId[] = [];
    if (c.preAddNodes) {
        for (const id of c.pool) {
            legacy.addNode(id);
            builder.addNode(id);
            mentioned.push(id);
        }
    }
    const accepted: AcceptedEdge[] = [];
    for (const request of resolveRequests(c)) {
        mentioned.push(request.source, request.target);
        // The legacy class adds both endpoints BEFORE it checks its own policy, so a refused request
        // still creates the nodes; mirror that so the node sets stay comparable.
        builder.addNode(request.source);
        builder.addNode(request.target);
        try {
            legacy.addEdge(request.source, request.target, request.weight);
        } catch {
            continue;
        }
        builder.addEdge(request.source, request.target, request.weight);
        accepted.push({
            source: request.source,
            target: request.target,
            weight: request.weight ?? 1,
            explicit: request.weight !== undefined,
        });
    }
    return { legacy, builder, accepted, mentioned };
}

// ============================================================ comparison helpers

/** SameValueZero normalisation: the format stores -0 as 0 (design 4.1); the legacy Node keeps -0. */
function norm(id: NodeId): NodeId {
    return typeof id === "number" && id === 0 ? 0 : id;
}

/** Type-tagged key so that 1 and "1" stay distinct in string-keyed sets. */
function keyOf(id: NodeId): string {
    return typeof id === "number" ? `n:${String(id)}` : `s:${id}`;
}

/** The legacy node ids in insertion order, -0 normalised. */
function legacyIds(legacy: LegacyGraph): NodeId[] {
    return [...legacy.nodes()].map((node) => norm(node.id));
}

/** The exact weight of logical edge e: the f64 shadow when it holds the row, else the arc value. */
function exactWeight(s: GraphSnapshot, e: number): number {
    const shadow = s.edges.byRole("weight");
    if (shadow !== null && shadow.dtype === "f64" && shadow.isSet(e)) {
        return shadow.value(e) as number;
    }
    return s.weights === null ? 1 : s.weights[s.edgeToArc[e]];
}

/** The arc weight (1 when the snapshot is unweighted). */
function arcWeight(s: GraphSnapshot, a: number): number {
    return s.weights === null ? 1 : s.weights[a];
}

/** Distinct targets of a row (the adjacent-skip idiom of design 3.5), as ids. */
function distinctRowIds(s: GraphSnapshot, rowPtr: Uint32Array, colIdx: Uint32Array, u: number): Set<string> {
    const out = new Set<string>();
    for (let a = rowPtr[u]; a < rowPtr[u + 1]; a++) {
        out.add(keyOf(s.ids.idOf(colIdx[a])));
    }
    return out;
}

function keySet(ids: Iterable<NodeId>): Set<string> {
    const out = new Set<string>();
    for (const id of ids) {
        out.add(keyOf(id));
    }
    return out;
}

/** Unordered-pair key for undirected multiplicity counting. */
function pairKey(directed: boolean, u: number, v: number): string {
    if (directed || u <= v) {
        return `${u}-${v}`;
    }
    return `${v}-${u}`;
}

// ============================================================ the comparisons

/** Counts and the counting vocabulary of design section 2 (I3 / I6 / I7). */
function checkCounts(c: GraphCase): void {
    const { legacy, builder, accepted } = buildBoth(c);
    // builder-level counts before any freeze (the dirty path of design 6.6)
    expect(builder.nodeCount).toBe(legacy.nodeCount);
    expect(builder.edgeCount).toBe(legacy.totalEdgeCount);
    const s = builder.freeze();
    assertInvariants(s);
    expect(s.directed).toBe(legacy.isDirected);
    expect(s.nodeCount).toBe(legacy.nodeCount);
    // edgeCount is the number of logical edges = every accepted addEdge (legacy totalEdgeCount)
    expect(s.edgeCount).toBe(legacy.totalEdgeCount);
    expect(s.edgeCount).toBe(accepted.length);
    const loops = accepted.filter((e) => norm(e.source) === norm(e.target)).length;
    expect(s.selfLoopCount).toBe(loops);
    expect(s.flags.hasSelfLoops).toBe(loops > 0);
    // arcCount per the counting vocabulary of section 2
    expect(s.arcCount).toBe(c.directed ? s.edgeCount : 2 * s.edgeCount - loops);
    expect(s.arcCount).toBe(s.colIdx.length);
    expect(s.rowPtr[s.nodeCount]).toBe(s.arcCount);
    // multigraph flag vs the log
    const seen = new Set<string>();
    let multigraph = false;
    for (const e of accepted) {
        const key = pairKey(c.directed, s.ids.requireIndex(e.source), s.ids.requireIndex(e.target));
        if (seen.has(key)) {
            multigraph = true;
        }
        seen.add(key);
    }
    expect(s.flags.multigraph).toBe(multigraph);
    if (!c.allowParallelEdges) {
        expect(s.flags.multigraph).toBe(false);
    }
    // legacy uniqueEdgeCount equals edgeCount for a simple graph whose ids are homogeneous
    // (mixed ids trip the legacy string-coercing comparison; see differential-enumeration)
    if (!c.allowParallelEdges && c.idKind !== "mixed") {
        expect(legacy.uniqueEdgeCount).toBe(s.edgeCount);
    }
}

/** Degrees per node (design section 3.4), with the self-loop convention difference where declared. */
function checkDegrees(c: GraphCase): void {
    const { legacy, builder } = buildBoth(c);
    const s = builder.freeze();
    const outDegree = s.outDegree();
    const inDegree = s.inDegree();
    const degree = s.degree();
    const loopsPerNode = s.selfLoopsPerNode();
    const reverse = s.reverse();
    for (let u = 0; u < s.nodeCount; u++) {
        const id = s.ids.idOf(u);
        expect(outDegree[u]).toBe(s.outDegreeOf(u));
        expect(outDegree[u]).toBe(s.rowPtr[u + 1] - s.rowPtr[u]);
        expect(loopsPerNode[u]).toBe(s.selfLoopsAt(u));
        // The legacy adjacency Map holds one entry per DISTINCT neighbour, so on a multigraph
        // its degrees are distinct-neighbour counts; the row scan's distinct target count is
        // the comparable quantity (design 3.5, adjacent-skip idiom).
        const distinctOut = distinctRowIds(s, s.rowPtr, s.colIdx, u).size;
        const distinctIn = distinctRowIds(s, reverse.rowPtr, reverse.colIdx, u).size;
        expect(distinctOut).toBe(legacy.outDegree(id));
        expect(distinctIn).toBe(legacy.inDegree(id));
        if (!c.allowParallelEdges) {
            // simple graph: the arc counts ARE the legacy counts
            expect(outDegree[u]).toBe(legacy.outDegree(id));
            expect(inDegree[u]).toBe(legacy.inDegree(id));
            if (c.directed) {
                // directed: legacy degree = in + out; a loop is one out-arc and one in-arc
                // on both sides, so the two agree exactly (3.4 table, directed column).
                expect(degree[u]).toBe(legacy.degree(id));
            } else {
                // undirected: THE DOCUMENTED DIFFERENCE (3.4). The legacy class stores an
                // undirected loop once and counts it once (its degree === outDegree); the
                // format's degree() is the NetworkX convention, loop counted twice:
                // degree[u] === outDegree[u] + selfLoopsPerNode[u].
                expect(outDegree[u]).toBe(legacy.degree(id));
                expect(degree[u]).toBe(legacy.degree(id) + loopsPerNode[u]);
                expect(inDegree[u]).toBe(legacy.inDegree(id));
            }
        }
        if (c.directed) {
            expect(degree[u]).toBe(outDegree[u] + inDegree[u]);
        } else {
            expect(inDegree).toBe(outDegree);
            expect(degree[u]).toBe(outDegree[u] + loopsPerNode[u]);
        }
    }
    // sum identities: sum(outDegree) === arcCount; undirected sum(degree) === 2 * edgeCount
    let sumOut = 0;
    let sumDegree = 0;
    for (let u = 0; u < s.nodeCount; u++) {
        sumOut += outDegree[u];
        sumDegree += degree[u];
    }
    expect(sumOut).toBe(s.arcCount);
    expect(sumDegree).toBe(2 * s.edgeCount);
}

/** Neighbour sets from a row scan vs legacy neighbors() / inNeighbors() (design 3.3, 3.5, 7.2). */
function checkNeighbours(c: GraphCase): void {
    const { legacy, builder } = buildBoth(c);
    const s = builder.freeze();
    const reverse = s.reverse();
    for (let u = 0; u < s.nodeCount; u++) {
        const id = s.ids.idOf(u);
        expect(distinctRowIds(s, s.rowPtr, s.colIdx, u)).toEqual(keySet(legacy.neighbors(id)));
        expect(distinctRowIds(s, s.rowPtr, s.colIdx, u)).toEqual(keySet(legacy.outNeighbors(id)));
        expect(distinctRowIds(s, reverse.rowPtr, reverse.colIdx, u)).toEqual(keySet(legacy.inNeighbors(id)));
        // I4: the row is sorted by target index; parallels adjacent
        for (let a = s.rowPtr[u] + 1; a < s.rowPtr[u + 1]; a++) {
            expect(s.colIdx[a]).toBeGreaterThanOrEqual(s.colIdx[a - 1]);
        }
    }
    // the builder's incidence lists must give the same neighbour sets while dirty (6.6):
    // findEdges(u, v) is non-empty exactly when the legacy adjacency has the pair
    for (let u = 0; u < s.nodeCount; u++) {
        const id = s.ids.idOf(u);
        const viaFindEdges = new Set<string>();
        for (let v = 0; v < s.nodeCount; v++) {
            if (s.nodeCount > 64 && v > 64) {
                break;
            }
            if (builder.findEdges(u, v).length > 0) {
                viaFindEdges.add(keyOf(s.ids.idOf(v)));
            }
        }
        const expected = keySet(legacy.neighbors(id));
        for (const key of viaFindEdges) {
            expect(expected.has(key)).toBe(true);
        }
        if (s.nodeCount <= 64) {
            expect(viaFindEdges).toEqual(expected);
        }
    }
}

/** hasEdge vs findArc / hasArc / multiplicity (design 3.5, 3.9) over random and every accepted pair. */
function checkHasEdge(c: GraphCase, pairs: readonly (readonly [number, number])[]): void {
    const { legacy, builder, accepted, mentioned } = buildBoth(c);
    const s = builder.freeze();
    const candidates: NodeId[] = [...mentioned, "not-a-node", -12345.5, 999_999_999];
    const multiplicity = new Map<string, number>();
    for (const e of accepted) {
        const key = pairKey(c.directed, s.ids.requireIndex(e.source), s.ids.requireIndex(e.target));
        multiplicity.set(key, (multiplicity.get(key) ?? 0) + 1);
    }
    const check = (a: NodeId, b: NodeId): void => {
        const u = s.ids.indexOf(a);
        const v = s.ids.indexOf(b);
        const legacyHas = legacy.hasEdge(a, b);
        if (u === INVALID_INDEX || v === INVALID_INDEX) {
            expect(legacyHas).toBe(false);
            return;
        }
        expect(s.hasArc(u, v)).toBe(legacyHas);
        expect(s.findArc(u, v) !== INVALID_INDEX).toBe(legacyHas);
        const [lo, hi] = s.arcsBetween(u, v);
        expect(hi - lo).toBe(s.multiplicity(u, v));
        expect(hi - lo).toBe(multiplicity.get(pairKey(c.directed, u, v)) ?? 0);
        if (legacyHas) {
            const arc = s.findArc(u, v);
            expect(arc).toBe(lo);
            expect(s.colIdx[arc]).toBe(v);
            expect(s.arcSource(arc)).toBe(u);
        }
        // the builder's dirty-path answer (6.6): findEdges(u, v) non-empty iff legacy hasEdge
        expect(builder.findEdges(u, v).length > 0).toBe(legacyHas);
        expect(builder.findEdges(u, v).length).toBe(multiplicity.get(pairKey(c.directed, u, v)) ?? 0);
    };
    if (candidates.length === 0) {
        return;
    }
    for (const [i, j] of pairs) {
        check(candidates[i % candidates.length], candidates[j % candidates.length]);
    }
    // every accepted edge, both orientations
    for (const e of accepted) {
        check(e.source, e.target);
        check(e.target, e.source);
    }
}

/** Weights via getEdge vs weights[arc] (design 3.7): f32 arc values, exact f64 shadow, truthful flags. */
function checkWeights(c: GraphCase): void {
    const { legacy, builder, accepted } = buildBoth(c);
    const s = builder.freeze();
    const anyExplicit = accepted.some((e) => e.explicit);
    // weighted: "auto" (3.7): the arc array exists iff some addEdge passed a weight
    expect(s.weights !== null).toBe(anyExplicit);
    expect(s.flags.weighted).toBe(anyExplicit);
    const edgeList = s.edgeList();
    for (let e = 0; e < accepted.length; e++) {
        const expected = accepted[e].weight;
        const arc = s.edgeToArc[e];
        // the arc array is f32 (3.7); both arcs of an undirected edge carry the same value
        expect(Object.is(arcWeight(s, arc), Math.fround(expected))).toBe(true);
        if (edgeList.weights !== null) {
            expect(Object.is(edgeList.weights[e], Math.fround(expected))).toBe(true);
        }
        if (!c.directed) {
            const mate = s.mate()[arc];
            expect(Object.is(arcWeight(s, mate), Math.fround(expected))).toBe(true);
        }
        if (c.weightDtype === "f64") {
            // the f64 staging keeps the exact legacy value through the shadow column
            expect(Object.is(exactWeight(s, e), expected)).toBe(true);
        }
        // the builder's own read-back before / after the freeze
        const staged = builder.edgeWeight(e);
        expect(Object.is(staged, c.weightDtype === "f64" ? expected : Math.fround(expected))).toBe(true);
    }
    // legacy getEdge(u, v) holds the LAST parallel added (Map overwrite); by I4 the parallels
    // of a row are ordered by ascending logical edge index, so the last arc of arcsBetween
    // is the comparable one. On a simple graph that is findArc itself.
    for (const e of accepted) {
        const u = s.ids.requireIndex(e.source);
        const v = s.ids.requireIndex(e.target);
        const legacyEdge = legacy.getEdge(e.source, e.target);
        expect(legacyEdge).toBeDefined();
        const legacyWeight = legacyEdge?.weight ?? Number.NaN;
        const [lo, hi] = s.arcsBetween(u, v);
        expect(hi).toBeGreaterThan(lo);
        expect(Object.is(arcWeight(s, hi - 1), Math.fround(legacyWeight))).toBe(true);
        if (!c.allowParallelEdges) {
            expect(Object.is(arcWeight(s, s.findArc(u, v)), Math.fround(legacyWeight))).toBe(true);
        }
        if (!c.directed) {
            const reverseEdge = legacy.getEdge(e.target, e.source);
            const [rlo, rhi] = s.arcsBetween(v, u);
            expect(rhi).toBeGreaterThan(rlo);
            expect(Object.is(arcWeight(s, rhi - 1), Math.fround(reverseEdge?.weight ?? Number.NaN))).toBe(true);
        }
    }
    // flags are truthful over the legacy weights
    const values = accepted.map((e) => Math.fround(e.weight));
    expect(s.flags.allWeightsOne).toBe(values.every((w) => w === 1));
    expect(s.flags.nonNegativeWeights).toBe(values.every((w) => w >= 0));
    expect(s.flags.finiteWeights).toBe(values.every((w) => Number.isFinite(w)));
}

/** The storage kind design 4.2 prescribes for an id array. */
function expectedKind(ids: readonly NodeId[]): NodeIdMapKind {
    const n = ids.length;
    const numbers = ids.filter((id): id is number => typeof id === "number");
    if (numbers.length === n) {
        // an identity offset is a SAFE integer (index + offset must stay exact, design 4.2 / I11)
        if (n > 0 && Number.isSafeInteger(numbers[0]) && numbers.every((id, i) => id === i + numbers[0])) {
            return "identity";
        }
        if (n === 0) {
            return "identity";
        }
        const max = Math.max(...numbers);
        if (numbers.every((id) => Number.isInteger(id) && id >= 0 && id < 0xfffffffe) && max + 1 <= 2 * n) {
            return "dense";
        }
        return "numeric";
    }
    if (ids.every((id) => typeof id === "string")) {
        return "string";
    }
    return "mixed";
}

/** Id round trip through the id map (design 4.1, 4.2, I11) and the builder's own lookups. */
function checkIds(c: GraphCase): void {
    const { legacy, builder, mentioned } = buildBoth(c);
    // builder-level lookups before the freeze
    for (const id of mentioned) {
        expect(builder.hasNode(id)).toBe(legacy.hasNode(id));
        const index = builder.indexOf(id);
        expect(index !== INVALID_INDEX).toBe(legacy.hasNode(id));
        if (index !== INVALID_INDEX) {
            expect(norm(builder.idOf(index))).toBe(norm(id));
        }
    }
    const s = builder.freeze();
    const expected = legacyIds(legacy);
    // I14: node index = order of first appearance, which is the legacy Map insertion order
    expect(s.ids.toArray().map(norm)).toEqual(expected);
    expect(s.ids.idsSlice().map(norm)).toEqual(expected);
    expect([...s.ids].map(norm)).toEqual(expected);
    expect(s.ids.size).toBe(legacy.nodeCount);
    for (let i = 0; i < s.nodeCount; i++) {
        const id = s.ids.idOf(i);
        expect(s.ids.indexOf(id)).toBe(i);
        expect(s.ids.requireIndex(id)).toBe(i);
        expect(s.ids.has(id)).toBe(true);
        expect(legacy.hasNode(id)).toBe(true);
        // the legacy Node object is looked up under the same key
        expect(norm(legacy.getNode(id)?.id ?? "<missing>")).toBe(norm(id));
        // the builder agrees with its own snapshot (4.4: same numbers between freezes)
        expect(builder.indexOf(id)).toBe(i);
        expect(norm(builder.idOf(i))).toBe(norm(id));
    }
    // absent ids and cross-type lookups miss on both sides (4.1: 1 !== "1")
    for (const id of mentioned) {
        const other: NodeId = typeof id === "number" ? String(id) : Number(id);
        if (typeof other === "number" && Number.isNaN(other)) {
            continue;
        }
        expect(s.ids.has(other)).toBe(legacy.hasNode(other));
        expect(s.ids.indexOf(other) !== INVALID_INDEX).toBe(legacy.hasNode(other));
    }
    expect(s.ids.indexOf("definitely-not-present")).toBe(INVALID_INDEX);
    expect(s.ids.indexOf(-987654321.5)).toBe(INVALID_INDEX);
    // kind selection (4.2)
    expect(s.ids.kind).toBe(expectedKind(s.ids.toArray()));
    // indicesOf round trip in bulk
    const indices = s.ids.indicesOf(s.ids.toArray());
    for (let i = 0; i < s.nodeCount; i++) {
        expect(indices[i]).toBe(i);
    }
}

/** The id map through GraphBuilder.from and through toWire / fromWire. */
function checkIdRoundTrips(c: GraphCase): void {
    const { legacy, builder } = buildBoth(c);
    const s = builder.freeze();
    const expected = legacyIds(legacy);
    const again = GraphBuilder.from(s).freeze();
    expect(again.ids.toArray().map(norm)).toEqual(expected);
    expect(again.ids.kind).toBe(s.ids.kind);
    expect(again.nodeCount).toBe(s.nodeCount);
    expect(again.edgeCount).toBe(s.edgeCount);
    expect(again.arcCount).toBe(s.arcCount);
    expect(Array.from(again.rowPtr)).toEqual(Array.from(s.rowPtr));
    expect(Array.from(again.colIdx)).toEqual(Array.from(s.colIdx));
    expect(again.contentHash()).toBe(s.contentHash());
    const back = fromWire(s.toWire(), { validate: "full" });
    expect(back.ids.toArray().map(norm)).toEqual(expected);
    expect(back.ids.kind).toBe(s.ids.kind);
    for (let i = 0; i < s.nodeCount; i++) {
        expect(back.ids.indexOf(s.ids.idOf(i))).toBe(i);
    }
}

// ============================================================ deterministic fixtures

/** A deterministic 2000-node case: a cycle plus chords, loops and repeats, per id kind. */
function bigCase(idKind: IdKind, directed: boolean, allowSelfLoops: boolean, allowParallelEdges: boolean): GraphCase {
    const n = 2000;
    const rawEdges: RawEdge[] = [];
    for (let i = 0; i < n; i++) {
        rawEdges.push({ s: i, t: (i + 1) % n, w: WEIGHTS[i % WEIGHTS.length], dup: null, flip: false, loop: false });
        if (i % 7 === 0) {
            rawEdges.push({
                s: i,
                t: (i * 37 + 11) % n,
                w: WEIGHTS[(i + 3) % WEIGHTS.length],
                dup: null,
                flip: false,
                loop: false,
            });
        }
        if (i % 53 === 0) {
            rawEdges.push({ s: i, t: i, w: 2, dup: null, flip: false, loop: true });
        }
        if (i % 41 === 0) {
            rawEdges.push({ s: 0, t: 0, w: 3, dup: rawEdges.length - 1, flip: i % 2 === 0, loop: false });
        }
    }
    return {
        directed,
        allowSelfLoops,
        allowParallelEdges,
        weightDtype: idKind === "strings" ? "f64" : "f32",
        idKind,
        preAddNodes: idKind === "identity" || idKind === "oneBased",
        pool: Array.from({ length: n }, (_, i) => deterministicId(idKind, i)),
        rawEdges,
    };
}

const BIG_CASES: readonly GraphCase[] = [
    bigCase("identity", true, true, true),
    bigCase("identity", false, true, false),
    bigCase("oneBased", false, true, true),
    bigCase("sparseInts", true, false, false),
    bigCase("strings", false, true, true),
    bigCase("unicode", true, true, false),
    bigCase("mixed", false, false, true),
    bigCase("mixed", true, true, true),
];

/** Small hand-written cases the random generator reaches only by chance. */
function smallCase(
    idKind: IdKind,
    directed: boolean,
    pool: readonly NodeId[],
    edges: readonly (readonly [number, number, number | undefined])[],
    preAddNodes = false,
): GraphCase {
    return {
        directed,
        allowSelfLoops: true,
        allowParallelEdges: true,
        weightDtype: "f64",
        idKind,
        preAddNodes,
        pool,
        rawEdges: edges.map(([s, t, w]) => ({ s, t, w, dup: null, flip: false, loop: false })),
    };
}

const SMALL_CASES: readonly (readonly [string, GraphCase])[] = [
    ["empty graph", smallCase("identity", true, [], [])],
    ["one isolated node", smallCase("strings", false, ["only"], [], true)],
    [
        "undirected graph whose edges are all self-loops",
        smallCase(
            "identity",
            false,
            [0, 1, 2],
            [
                [0, 0, 1],
                [2, 2, 2],
                [0, 0, 3],
            ],
            true,
        ),
    ],
    [
        "directed graph whose edges are all self-loops",
        smallCase(
            "identity",
            true,
            [0, 1, 2],
            [
                [1, 1, 1],
                [1, 1, 2],
            ],
            true,
        ),
    ],
    [
        "directed input already grouped by source and sorted by target (identity permutation)",
        smallCase(
            "identity",
            true,
            [0, 1, 2, 3],
            [
                [0, 1, 1],
                [0, 2, 2],
                [1, 1, 3],
                [1, 3, 4],
                [3, 0, 5],
            ],
            true,
        ),
    ],
    [
        "dense ids out of order",
        smallCase(
            "sparseInts",
            true,
            [3, 0, 1, 2],
            [
                [0, 1, 1],
                [1, 2, 1],
                [2, 3, 1],
            ],
        ),
    ],
    [
        "dense boundary: max + 1 === 2n",
        smallCase(
            "sparseInts",
            false,
            [0, 2, 4, 7],
            [
                [0, 1, 1],
                [1, 2, 1],
                [2, 3, 1],
            ],
            true,
        ),
    ],
    [
        "numeric just past the dense boundary: max + 1 === 2n + 1",
        smallCase(
            "sparseInts",
            false,
            [0, 2, 4, 8],
            [
                [0, 1, 1],
                [1, 2, 1],
                [2, 3, 1],
            ],
            true,
        ),
    ],
    [
        "identity with a negative offset",
        smallCase(
            "negOffset",
            true,
            [-3, -2, -1, 0],
            [
                [0, 1, 1],
                [3, 2, 1],
            ],
            true,
        ),
    ],
    [
        '1 and "1", 0 and "0", -0 (mixed)',
        smallCase(
            "mixed",
            false,
            [1, "1", -0, "0", "", 2.5],
            [
                [0, 1, 1],
                [1, 2, 2],
                [2, 3, 3],
                [3, 4, 4],
                [4, 5, 5],
                [5, 0, 6],
                [0, 0, 7],
            ],
        ),
    ],
    [
        "floats and extreme numbers",
        smallCase(
            "floats",
            true,
            [0.1, 2 ** 53 - 1, -(2 ** 53 - 1), 5e-324, 1e300, 4294967295],
            [
                [0, 1, 1],
                [1, 2, 1],
                [2, 3, 1],
                [3, 4, 1],
                [4, 5, 1],
                [5, 0, 1],
            ],
        ),
    ],
    [
        "unicode ids",
        smallCase(
            "unicode",
            false,
            ["\u00e9", "\u65e5\u672c", "\ud83d\ude00", "a\u0000b", ""],
            [
                [0, 1, 1],
                [1, 2, 1],
                [2, 3, 1],
                [3, 4, 1],
                [4, 0, 1],
            ],
        ),
    ],
];

const RANDOM_PAIRS: readonly (readonly [number, number])[] = Array.from(
    { length: 200 },
    (_, i) => [i * 7919, i * 104729] as const,
);

/** Every comparison of this file over one case. */
function checkEverything(c: GraphCase): void {
    checkCounts(c);
    checkDegrees(c);
    checkNeighbours(c);
    checkHasEdge(c, RANDOM_PAIRS);
    checkWeights(c);
    checkIds(c);
    checkIdRoundTrips(c);
}

// ============================================================ the properties

describe("differential: counts and the counting vocabulary (design sections 2, 3.2 I3 / I6 / I7)", () => {
    it("nodeCount, edgeCount, arcCount and selfLoopCount agree with the legacy class and the accepted log", () => {
        fc.assert(fc.property(graphCaseArb, checkCounts), { numRuns: RUNS });
    });
});

describe("differential: degrees per node (design section 3.4)", () => {
    it("outDegree / inDegree / degree agree with the legacy class up to the documented self-loop convention", () => {
        fc.assert(fc.property(graphCaseArb, checkDegrees), { numRuns: RUNS });
    });
});

describe("differential: neighbour sets from a row scan (design sections 3.3, 3.5, 7.2)", () => {
    it("the row of every node equals legacy neighbors() as a set, and the reverse row equals inNeighbors()", () => {
        fc.assert(fc.property(graphCaseArb, checkNeighbours), { numRuns: RUNS });
    });
});

describe("differential: hasEdge vs findArc / hasArc / multiplicity (design sections 3.5, 3.9)", () => {
    it("hasArc agrees with legacy hasEdge for every ordered pair of mentioned ids, including absent ids", () => {
        fc.assert(
            fc.property(
                graphCaseArb,
                fc.array(fc.tuple(fc.nat(), fc.nat()), { maxLength: 200, size: "max" }),
                checkHasEdge,
            ),
            { numRuns: RUNS },
        );
    });
});

describe("differential: id round trip through the id map (design sections 4.1, 4.2, I11)", () => {
    it("ids.toArray() is the legacy insertion order and every id maps back to its index", () => {
        fc.assert(fc.property(graphCaseArb, checkIds), { numRuns: RUNS });
    });

    it("the id map survives a builder round trip and a wire round trip unchanged", () => {
        fc.assert(fc.property(graphCaseArb, checkIdRoundTrips), { numRuns: Math.max(10, Math.floor(RUNS / 2)) });
    });
});

describe("differential: deterministic 2000-node fixtures (every comparison)", () => {
    it.each(
        BIG_CASES.map(
            (c) =>
                [
                    `${c.idKind} ${c.directed ? "directed" : "undirected"} loops=${c.allowSelfLoops} parallels=${c.allowParallelEdges}`,
                    c,
                ] as const,
        ),
    )("%s", (_name, c) => {
        expect(c.pool.length).toBe(2000);
        checkEverything(c);
    });
});

describe("differential: hand-written boundary fixtures (every comparison)", () => {
    it.each(SMALL_CASES)("%s", (_name, c) => {
        checkEverything(c);
    });

    it("kind boundaries (design 4.2): identity / dense / numeric are chosen exactly at the documented thresholds", () => {
        const kindOf = (ids: readonly NodeId[]): NodeIdMapKind => {
            const b = new GraphBuilder({ directed: true });
            for (const id of ids) {
                b.addNode(id);
            }
            return b.freeze().ids.kind;
        };
        expect(kindOf([])).toBe("identity");
        expect(kindOf([0, 1, 2])).toBe("identity");
        expect(kindOf([1, 2, 3])).toBe("identity");
        expect(kindOf([-3, -2, -1])).toBe("identity");
        expect(kindOf([2 ** 40, 2 ** 40 + 1])).toBe("identity");
        expect(kindOf([3, 0, 1, 2])).toBe("dense");
        expect(kindOf([0, 2, 4, 7])).toBe("dense");
        expect(kindOf([0, 2, 4, 8])).toBe("numeric");
        expect(kindOf([1, 3])).toBe("dense");
        expect(kindOf([0, 4])).toBe("numeric");
        expect(kindOf([0.5, 1.5])).toBe("numeric");
        expect(kindOf([-1, 0])).toBe("identity");
        expect(kindOf([0, -1])).toBe("numeric");
        expect(kindOf(["a"])).toBe("string");
        expect(kindOf([1, "1"])).toBe("mixed");
    });

    it("an undirected graph whose edges are all self-loops has arcCount === edgeCount and arcToEdgeIsIdentity === false (3.1)", () => {
        const { builder, legacy } = buildBoth(SMALL_CASES[2][1]);
        const s = builder.freeze();
        expect(s.arcCount).toBe(s.edgeCount);
        expect(s.flags.arcToEdgeIsIdentity).toBe(false);
        expect(Array.from(s.arcToEdge)).toEqual([0, 2, 1]);
        // the legacy counts each loop once; degree() twice per loop (3.4)
        expect(legacy.degree(0)).toBe(1);
        expect(s.outDegree()[0]).toBe(2);
        expect(s.degree()[0]).toBe(4);
    });

    it("a directed input already grouped and sorted has the identity permutation and still matches the legacy", () => {
        const { builder } = buildBoth(SMALL_CASES[4][1]);
        const s = builder.freeze();
        expect(s.flags.arcToEdgeIsIdentity).toBe(true);
        expect(Array.from(s.arcToEdge)).toEqual([0, 1, 2, 3, 4]);
        expect(Array.from(s.edgeToArc)).toEqual([0, 1, 2, 3, 4]);
    });
});

describe("differential: weights via getEdge vs weights[arc] (design section 3.7)", () => {
    it("weights[arc] is the f32 value of the legacy weight; the f64 shadow keeps the exact value", () => {
        fc.assert(fc.property(graphCaseArb, checkWeights), { numRuns: RUNS });
    });

    it("weighted: true materialises an all-ones array that still matches legacy weights", () => {
        fc.assert(
            fc.property(graphCaseArb, (c) => {
                const { legacy, builder, accepted } = buildBoth(c, { weighted: true });
                const s = builder.freeze();
                expect(s.weights).not.toBeNull();
                expect(s.weights?.length).toBe(s.arcCount);
                for (let e = 0; e < accepted.length; e++) {
                    const legacyWeight = legacy.getEdge(accepted[e].source, accepted[e].target)?.weight;
                    expect(legacyWeight).toBeDefined();
                    expect(Object.is(arcWeight(s, s.edgeToArc[e]), Math.fround(accepted[e].weight))).toBe(true);
                }
            }),
            { numRuns: Math.max(10, Math.floor(RUNS / 3)) },
        );
    });
});
