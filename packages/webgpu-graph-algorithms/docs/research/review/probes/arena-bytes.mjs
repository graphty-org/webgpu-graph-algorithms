// Probe: recompute the arena hot prefix / byteLength for the plan's tiers using the
// graph-format layout rule (256-byte aligned segments, order rowPtr, colIdx, weights, arcToEdge, edgeToArc).
import { layoutSegments } from "/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/dist/src/util/typed-array.js";
const ALIGN = 256;
for (const [n, E] of [[100_000, 1_000_000], [1_000_000, 10_000_000], [10_000_000, 100_000_000]]) {
    const A = 2 * E;
    const bytes = [4 * (n + 1), 4 * A, 4 * A, 4 * A, 4 * E];
    const l = layoutSegments(bytes, ALIGN);
    const hot = l.offsets[2] + bytes[2];
    console.log(`n=${n} E=${E}: hotByteLength=${hot.toLocaleString()} byteLength=${l.byteLength.toLocaleString()} offsets=${JSON.stringify(l.offsets)}`);
}
