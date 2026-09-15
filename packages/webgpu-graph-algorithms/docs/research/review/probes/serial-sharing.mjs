// Probe: do withColumns() siblings share `serial`, and does toUndirected() of a directed
// snapshot produce a NEW snapshot with a different serial? (plan 4.1 refs: Set<serial>; plan 9.4 item 2)
import { fromEdgeArrays } from "/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/dist/graph-format.js";

const src = new Uint32Array([0, 1, 2]);
const dst = new Uint32Array([1, 2, 0]);
const s = fromEdgeArrays({ directed: true, nodeCount: 3, src, dst });
const sib = s.withColumns({ mass: new Float32Array([1, 2, 3]) });
const und = s.toUndirected();
console.log(JSON.stringify({
    serial: s.serial,
    siblingSerial: sib.serial,
    siblingSharesRowPtr: sib.rowPtr === s.rowPtr,
    undirectedIsSameObject: und.snapshot === s,
    undirectedSerial: und.snapshot.serial,
    undirectedSharesRowPtr: und.snapshot.rowPtr === s.rowPtr,
}));
