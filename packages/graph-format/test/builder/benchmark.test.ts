/**
 * A benchmark-style sanity test (design section 15.4; not a performance gate): 100k nodes and 1M
 * directed edges pushed from typed arrays and frozen must complete under a generous ceiling and yield
 * a valid snapshot. The measured times are printed for the integrator's notes.
 */

import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";

/** A deterministic xorshift generator so the graph is the same on every run. */
function makeRandom(seed: number): () => number {
    let state = seed >>> 0 || 1;
    return () => {
        state ^= state << 13;
        state >>>= 0;
        state ^= state >>> 17;
        state ^= state << 5;
        state >>>= 0;
        return state / 0x100000000;
    };
}

describe("freeze at scale (sanity, not a gate)", () => {
    it("freezes 100k nodes / 1M directed edges pushed from typed arrays", () => {
        const nodeCount = 100_000;
        const edgeCount = 1_000_000;
        const random = makeRandom(12345);
        const src = new Uint32Array(edgeCount);
        const dst = new Uint32Array(edgeCount);
        const weights = new Float32Array(edgeCount);
        for (let e = 0; e < edgeCount; e++) {
            src[e] = Math.floor(random() * nodeCount);
            dst[e] = Math.floor(random() * nodeCount);
            weights[e] = 1 + Math.floor(random() * 10);
        }
        const start = performance.now();
        const builder = new GraphBuilder({ directed: true, expectedNodes: nodeCount, expectedEdges: edgeCount });
        builder.addAnonymousNodes(nodeCount);
        builder.addEdges(src, dst, weights);
        const pushed = performance.now();
        const { snapshot, report } = builder.freezeWithReport({ profile: true });
        const frozen = performance.now();
        expect(snapshot.nodeCount).toBe(nodeCount);
        expect(snapshot.edgeCount).toBe(edgeCount);
        expect(snapshot.arcCount).toBe(edgeCount);
        expect(snapshot.flags.arcToEdgeIsIdentity).toBe(false);
        expect(snapshot.flags.weighted).toBe(true);
        expect(snapshot.arena?.byteLength).toBe(400_128 + 4 * 4_000_000);
        expect(snapshot.ids.kind).toBe("identity");
        snapshot.validate({ level: "full" });
        const validated = performance.now();
        // a second, undirected freeze of the same edges through the builder path
        const undirected = new GraphBuilder({ directed: false, expectedNodes: nodeCount, expectedEdges: edgeCount });
        undirected.addAnonymousNodes(nodeCount);
        undirected.addEdges(src, dst, weights);
        const undirectedStart = performance.now();
        const u = undirected.freeze();
        const undirectedEnd = performance.now();
        expect(u.arcCount).toBe(2 * edgeCount - u.selfLoopCount);
        const pushMs = (pushed - start).toFixed(1);
        const freezeMs = (frozen - pushed).toFixed(1);
        const validateMs = (validated - frozen).toFixed(1);
        const undirectedMs = (undirectedEnd - undirectedStart).toFixed(1);
        console.log(
            `[benchmark] 100k nodes / 1M directed edges: push ${pushMs} ms, freeze ${freezeMs} ms ` +
                `(sort ${report.timings.sort.toFixed(1)} ms), validate(full) ${validateMs} ms; ` +
                `undirected freeze ${undirectedMs} ms`,
        );
        expect(frozen - pushed).toBeLessThan(10_000);
    }, 120_000);
});
