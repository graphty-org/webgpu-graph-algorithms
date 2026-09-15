/**
 * Id-map benchmarks (design section 15.5 item 3): interning 100k / 1M string ids through the
 * builder (`addNode` per id plus the freeze that detects the kind and shares the map), the same for
 * sparse numeric ids, then `indexOf` throughput per kind (identity, dense, numeric, string), `toMap`,
 * `entries` and `idsSlice` over the 1M-id maps.
 */

import { GraphBuilder } from "../src/builder/graph-builder.js";
import { type GraphSnapshot } from "../src/snapshot/graph-snapshot.js";
import { type NodeId } from "../src/types/index.js";
import { sparseNumericIds, stringIds } from "./datasets.js";
import { bench, type BenchResult, makeRandom } from "./harness.js";

/** Lookups per indexOf run. */
const LOOKUPS = 1_000_000;

/**
 * A snapshot of `count` isolated nodes with the given ids (edges do not matter to the id map).
 * @param ids - the ids
 * @returns the snapshot
 */
function idsSnapshot(ids: readonly NodeId[]): GraphSnapshot {
    const builder = new GraphBuilder({ directed: true, expectedNodes: ids.length });
    builder.addNodes(ids);
    return builder.freeze();
}

/**
 * Random probe ids drawn from the map's own ids (every lookup hits).
 * @param ids - the ids
 * @param count - how many probes
 * @param seed - the generator seed
 * @returns the probes
 */
function probes<T>(ids: readonly T[], count: number, seed: number): T[] {
    const random = makeRandom(seed);
    const out = new Array<T>(count);
    for (let i = 0; i < count; i++) {
        out[i] = ids[Math.floor(random() * ids.length)];
    }
    return out;
}

/**
 * Sum of indexOf over the probes (kept alive so the loop is not eliminated).
 * @param snapshot - the snapshot
 * @param ids - the probes
 * @returns the checksum
 */
function lookupAll(snapshot: GraphSnapshot, ids: readonly NodeId[]): number {
    let sum = 0;
    for (const id of ids) {
        sum += snapshot.ids.indexOf(id);
    }
    return sum;
}

/**
 * Run the id-map benchmarks.
 * @returns the results
 */
export function runIdBenchmarks(): BenchResult[] {
    const results: BenchResult[] = [];
    for (const count of [100_000, 1_000_000]) {
        const label = count >= 1_000_000 ? `${count / 1_000_000}M` : `${count / 1000}k`;
        const strings = stringIds(count);
        const numbers = sparseNumericIds(count);
        results.push(
            bench(
                "ids",
                `intern ${label} string ids (addNode x n + freeze)`,
                { setup: () => strings, run: (ids) => idsSnapshot(ids) },
                { items: count, unit: "ids" },
            ),
        );
        results.push(
            bench(
                "ids",
                `intern ${label} string ids (addNode x n only)`,
                {
                    setup: () => strings,
                    run: (ids) => {
                        const builder = new GraphBuilder({ directed: true, expectedNodes: ids.length });
                        for (const id of ids) {
                            builder.addNode(id);
                        }
                        return builder;
                    },
                },
                { items: count, unit: "ids" },
            ),
        );
        results.push(
            bench(
                "ids",
                `intern ${label} sparse numeric ids (addNode x n + freeze)`,
                { setup: () => numbers, run: (ids) => idsSnapshot(ids) },
                { items: count, unit: "ids" },
            ),
        );
    }
    const n = 1_000_000;
    const kinds: { name: string; ids: NodeId[] }[] = [
        { name: "identity", ids: Array.from({ length: n }, (_, i) => i) },
        { name: "dense", ids: Array.from({ length: n }, (_, i) => n - 1 - i) },
        { name: "numeric", ids: sparseNumericIds(n) },
        { name: "string", ids: stringIds(n) },
    ];
    for (const { name, ids } of kinds) {
        const snapshot = idsSnapshot(ids);
        if (snapshot.ids.kind !== name) {
            throw new Error(`expected a ${name} id map, got ${snapshot.ids.kind}`);
        }
        const hits = probes(ids, LOOKUPS, 4242);
        results.push(
            bench(
                "ids",
                `indexOf 1M lookups on a 1M ${name} map (cold: first call builds the reverse map)`,
                { setup: () => idsSnapshot(ids), run: (s) => lookupAll(s, hits) },
                { items: LOOKUPS, unit: "lookups" },
            ),
        );
        snapshot.ids.indexOf(ids[0]);
        results.push(
            bench(
                "ids",
                `indexOf 1M lookups on a 1M ${name} map (warm)`,
                { setup: () => snapshot, run: (s) => lookupAll(s, hits) },
                { items: LOOKUPS, unit: "lookups" },
            ),
        );
    }
    const stringSnapshot = idsSnapshot(kinds[3].ids);
    const values = new Float32Array(n);
    results.push(
        bench(
            "ids",
            "toMap over 1M string ids",
            { setup: () => stringSnapshot, run: (s) => s.ids.toMap(values) },
            { items: n, unit: "ids" },
        ),
    );
    results.push(
        bench(
            "ids",
            "entries over 1M string ids (consumed)",
            {
                setup: () => stringSnapshot,
                run: (s) => {
                    let count = 0;
                    for (const entry of s.ids.entries(values)) {
                        count += entry[0] === "" ? 0 : 1;
                    }
                    return count;
                },
            },
            { items: n, unit: "ids" },
        ),
    );
    results.push(
        bench(
            "ids",
            "idsSlice of 1M string ids",
            { setup: () => stringSnapshot, run: (s) => s.ids.idsSlice() },
            { items: n, unit: "ids" },
        ),
    );
    results.push(
        bench(
            "ids",
            "toWire of a 1M string id map (encodes the Utf8 store once)",
            { setup: () => idsSnapshot(kinds[3].ids), run: (s) => s.toWire({ includeColumns: false }) },
            { items: n, unit: "ids" },
        ),
    );
    return results;
}
