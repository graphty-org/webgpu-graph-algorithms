/**
 * View and derived-graph benchmarks (design section 15.5 item 2) over the 100k-node / 1M-edge
 * random graph: `reverse`, `coo`, `edgeList`, the degree views, `mate`, `degreeOrder`,
 * `isSymmetric`, `toUndirected`, `inducedSubgraph` (50% of the nodes), `contract` (1000 blocks),
 * `withoutSelfLoops`, `filterEdges` (50%), and the wire round trip (`toBytes` / `fromBytes` at each
 * validation level, `toWire` / `fromWire`). Each view is measured cold (a fresh snapshot per run).
 */

import { fromEdgeArrays } from "../src/populate/from-edge-arrays.js";
import { type GraphSnapshot } from "../src/snapshot/graph-snapshot.js";
import { type ValidationLevel } from "../src/types/index.js";
import { fromBytes } from "../src/wire/bytes.js";
import { fromWire } from "../src/wire/from-wire.js";
import { makeMask, maskSet } from "../src/util/mask.js";
import { randomEdges } from "./datasets.js";
import { bench, type BenchResult, makeRandom } from "./harness.js";

const NODES = 100_000;
const EDGES = 1_000_000;

/**
 * A fresh snapshot of the benchmark graph (no view cached).
 * @param directed - the orientation
 * @returns the snapshot
 */
function freshSnapshot(directed: boolean): GraphSnapshot {
    const edges = randomEdges(NODES, EDGES);
    return fromEdgeArrays({
        directed,
        nodeCount: edges.nodeCount,
        src: edges.src,
        dst: edges.dst,
        weights: edges.weights,
    });
}

/**
 * Run the view benchmarks.
 * @returns the results
 */
export function runViewBenchmarks(): BenchResult[] {
    const results: BenchResult[] = [];
    const directed = freshSnapshot(true);
    const undirected = freshSnapshot(false);
    const views: { name: string; directed: boolean; run: (s: GraphSnapshot) => unknown }[] = [
        { name: "reverse() directed", directed: true, run: (s) => s.reverse() },
        { name: "coo() directed", directed: true, run: (s) => s.coo() },
        { name: "edgeList() directed", directed: true, run: (s) => s.edgeList() },
        { name: "edgeList() undirected", directed: false, run: (s) => s.edgeList() },
        { name: "outDegree() directed", directed: true, run: (s) => s.outDegree() },
        { name: "inDegree() directed (includes reverse())", directed: true, run: (s) => s.inDegree() },
        { name: "degree() undirected", directed: false, run: (s) => s.degree() },
        { name: "weightedOutDegree() directed", directed: true, run: (s) => s.weightedOutDegree() },
        { name: "weightedDegree() undirected", directed: false, run: (s) => s.weightedDegree() },
        { name: "selfLoopArcs() undirected", directed: false, run: (s) => s.selfLoopArcs() },
        { name: "mate() undirected", directed: false, run: (s) => s.mate() },
        { name: "degreeOrder() directed", directed: true, run: (s) => s.degreeOrder() },
        {
            name: "degreeOrder({ of: reverse }) directed (includes reverse())",
            directed: true,
            run: (s) => s.degreeOrder({ of: "reverse" }),
        },
        { name: "isSymmetric() directed (includes reverse())", directed: true, run: (s) => s.isSymmetric() },
        { name: "totalWeight() undirected", directed: false, run: (s) => s.totalWeight() },
    ];
    for (const view of views) {
        results.push(
            bench(
                "views",
                view.name,
                { setup: () => freshSnapshot(view.directed), run: view.run },
                { items: EDGES, unit: "edges" },
            ),
        );
    }
    const half = makeMask(NODES);
    for (let i = 0; i < NODES; i += 2) {
        maskSet(half, i, true);
    }
    const halfEdges = makeMask(EDGES);
    const random = makeRandom(5);
    for (let e = 0; e < EDGES; e++) {
        if (random() < 0.5) {
            maskSet(halfEdges, e, true);
        }
    }
    const partition = new Uint32Array(NODES);
    for (let i = 0; i < NODES; i++) {
        partition[i] = i % 1000;
    }
    const derived: { name: string; snapshot: GraphSnapshot; run: (s: GraphSnapshot) => unknown }[] = [
        { name: "toUndirected() directed", snapshot: directed, run: (s) => s.toUndirected() },
        { name: "transpose() directed (zero-copy of reverse())", snapshot: directed, run: (s) => s.transpose() },
        { name: "withoutSelfLoops() undirected", snapshot: undirected, run: (s) => s.withoutSelfLoops() },
        { name: "filterEdges(50%) directed", snapshot: directed, run: (s) => s.filterEdges(halfEdges) },
        {
            name: "inducedSubgraph(50% mask) directed",
            snapshot: directed,
            run: (s) => s.inducedSubgraph({ mask: half }),
        },
        {
            name: "inducedSubgraph(50% mask) undirected",
            snapshot: undirected,
            run: (s) => s.inducedSubgraph({ mask: half }),
        },
        {
            name: "contract(1000 blocks, weights: sum) undirected",
            snapshot: undirected,
            run: (s) => s.contract(partition, { weights: "sum" }),
        },
        { name: "simplified() undirected", snapshot: undirected, run: (s) => s.simplified() },
    ];
    for (const d of derived) {
        results.push(bench("views", d.name, { setup: () => d.snapshot, run: d.run }, { items: EDGES, unit: "edges" }));
    }
    results.push(
        bench(
            "views",
            "toWire() directed",
            { setup: () => directed, run: (s) => s.toWire() },
            { items: EDGES, unit: "edges" },
        ),
    );
    const wire = directed.toWire();
    for (const level of ["none", "structure", "full"] as const) {
        results.push(
            bench(
                "views",
                `fromWire(validate: ${level}) directed`,
                { setup: () => wire, run: (w) => fromWire(w, { validate: level as ValidationLevel }) },
                { items: EDGES, unit: "edges" },
            ),
        );
    }
    results.push(
        bench(
            "views",
            "toBytes() directed",
            { setup: () => directed, run: (s) => s.toBytes() },
            { items: EDGES, unit: "edges" },
        ),
    );
    const bytes = directed.toBytes();
    for (const level of ["none", "structure", "full"] as const) {
        results.push(
            bench(
                "views",
                `fromBytes(validate: ${level}) directed`,
                { setup: () => bytes, run: (b) => fromBytes(b, { validate: level as ValidationLevel }) },
                { items: EDGES, unit: "edges" },
            ),
        );
    }
    results.push(
        bench(
            "views",
            "validate({ level: full }) directed",
            {
                setup: () => directed,
                run: (s) => {
                    s.validate({ level: "full" });
                    return s;
                },
            },
            { items: EDGES, unit: "edges" },
        ),
    );
    return results;
}
