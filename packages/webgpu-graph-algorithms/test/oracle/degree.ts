/**
 * The CPU reference of `degree` (spec 11.3, contract 5.3): out-degree per node as `rowPtr` differences, written as an
 * explicit loop so the reference is independent of graph-format's `outDegree()` view (which the GPU result must also
 * equal, by construction of the CSR). Self-loops count once per arc, parallels once per copy (I1).
 */

import { type GraphSnapshot, type U32 } from "@graphty/graph-format";

/**
 * rowPtr differences; equals snapshot.outDegree() by construction (the independent reference is the loop, not the view).
 * @param s - the snapshot
 * @returns the out-degree of every node, index-aligned
 */
export function outDegreeOracle(s: GraphSnapshot): U32 {
    const n = s.nodeCount;
    const { rowPtr } = s;
    const out = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
        out[i] = rowPtr[i + 1] - rowPtr[i];
    }
    return out;
}
