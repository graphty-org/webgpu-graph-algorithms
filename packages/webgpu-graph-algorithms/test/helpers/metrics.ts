/**
 * Layout-quality metrics for the distributional parity of spec 11.4 (contract 5.2 test/helpers/metrics.ts): stress
 * against BFS graph distances, edge-length quantiles, the self-normalised nearest-neighbour distance histogram, the
 * inter-component separation and the spread. Pure CPU. Every function reads a stride-3 array (x, y, z per node) and
 * uses its first `dim` components, so scene-unit and layout-unit arrays both work as long as both sides of a
 * comparison use the same units. Callers never compare coordinates (spec 7.16) -- only these summaries.
 */

import type { GraphSnapshot } from "@graphty/graph-format";

/** Bins of the nearest-neighbour histogram in layoutMetrics(). */
const NN_BINS = 8;
/** The histogram spans nn / mean(nn) in [0, NN_RANGE); larger ratios land in the last bin. */
const NN_RANGE = 2;

/**
 * Squared Euclidean distance between rows i and j over the first `dim` components.
 * @param positions - stride-3 positions
 * @param i - first row
 * @param j - second row
 * @param dim - components compared
 * @returns |p_i - p_j|^2
 */
function dist2(positions: ArrayLike<number>, i: number, j: number, dim: 2 | 3): number {
    let acc = 0;
    for (let k = 0; k < dim; k++) {
        const d = positions[3 * i + k] - positions[3 * j + k];
        acc += d * d;
    }
    return acc;
}

/**
 * Breadth-first hop distances from `source` over the CSR rows (-1 = unreachable).
 * @param s - the snapshot
 * @param source - the source node
 * @param dist - output, length nodeCount
 * @param queue - scratch, length nodeCount
 */
function bfs(s: GraphSnapshot, source: number, dist: Int32Array, queue: Int32Array): void {
    dist.fill(-1);
    dist[source] = 0;
    let head = 0;
    let tail = 0;
    queue[tail] = source;
    tail++;
    while (head < tail) {
        const u = queue[head];
        head++;
        const du = dist[u];
        for (let a = s.rowPtr[u]; a < s.rowPtr[u + 1]; a++) {
            const v = s.colIdx[a];
            if (dist[v] < 0) {
                dist[v] = du + 1;
                queue[tail] = v;
                tail++;
            }
        }
    }
}

/**
 * Linear-interpolation quantile of an ascending array; 0 for an empty array.
 * @param sorted - ascending values
 * @param q - the quantile in [0, 1]
 * @returns the interpolated value
 */
function quantile(sorted: Float64Array, q: number): number {
    if (sorted.length === 0) {
        return 0;
    }
    const pos = q * (sorted.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(sorted.length - 1, lo + 1);
    const t = pos - lo;
    return sorted[lo] * (1 - t) + sorted[hi] * t;
}

/**
 * The distance from every node to its nearest other node (0 when n < 2).
 * @param positions - stride-3 positions
 * @param n - node count
 * @param dim - components compared
 * @returns one distance per node
 */
function nearestNeighbourDistances(positions: ArrayLike<number>, n: number, dim: 2 | 3): Float64Array {
    const out = new Float64Array(n);
    if (n < 2) {
        return out;
    }
    for (let i = 0; i < n; i++) {
        let best = Number.POSITIVE_INFINITY;
        for (let j = 0; j < n; j++) {
            if (j === i) {
                continue;
            }
            const d2 = dist2(positions, i, j, dim);
            if (d2 < best) {
                best = d2;
            }
        }
        out[i] = Math.sqrt(best);
    }
    return out;
}

/**
 * Component labels by BFS over the CSR rows.
 * @param s - the snapshot
 * @returns the label of every node and the component count
 */
function componentLabels(s: GraphSnapshot): { readonly labels: Int32Array; readonly count: number } {
    const n = s.nodeCount;
    const labels = new Int32Array(n).fill(-1);
    const queue = new Int32Array(n);
    let count = 0;
    for (let root = 0; root < n; root++) {
        if (labels[root] >= 0) {
            continue;
        }
        labels[root] = count;
        let head = 0;
        let tail = 0;
        queue[tail] = root;
        tail++;
        while (head < tail) {
            const u = queue[head];
            head++;
            for (let a = s.rowPtr[u]; a < s.rowPtr[u + 1]; a++) {
                const v = s.colIdx[a];
                if (labels[v] < 0) {
                    labels[v] = count;
                    queue[tail] = v;
                    tail++;
                }
            }
        }
        count++;
    }
    return { labels, count };
}

/**
 * Normalised stress: the mean over connected pairs (i < j) of ((|p_i - p_j| - d_ij) / d_ij)^2 where d_ij is the hop
 * distance; 0 when no pair is connected.
 * @param s - the snapshot
 * @param positions - stride-3 positions
 * @param dim - components compared
 * @returns the stress
 */
export function stress(s: GraphSnapshot, positions: ArrayLike<number>, dim: 2 | 3): number {
    const n = s.nodeCount;
    if (n < 2) {
        return 0;
    }
    const dist = new Int32Array(n);
    const queue = new Int32Array(n);
    let sum = 0;
    let pairs = 0;
    for (let i = 0; i < n; i++) {
        bfs(s, i, dist, queue);
        for (let j = i + 1; j < n; j++) {
            const d = dist[j];
            if (d <= 0) {
                continue;
            }
            const r = (Math.sqrt(dist2(positions, i, j, dim)) - d) / d;
            sum += r * r;
            pairs++;
        }
    }
    return pairs === 0 ? 0 : sum / pairs;
}

/**
 * Quantiles (linear interpolation) of the Euclidean length of every edge: each undirected edge once (v > u), every
 * arc of a directed snapshot, self-loops excluded.
 * @param s - the snapshot
 * @param positions - stride-3 positions
 * @param dim - components compared
 * @param quantiles - the quantiles in [0, 1]
 * @returns one length per requested quantile (0 when the graph has no edge)
 */
export function edgeLengthQuantiles(
    s: GraphSnapshot,
    positions: ArrayLike<number>,
    dim: 2 | 3,
    quantiles: readonly number[],
): number[] {
    const lengths: number[] = [];
    for (let u = 0; u < s.nodeCount; u++) {
        for (let a = s.rowPtr[u]; a < s.rowPtr[u + 1]; a++) {
            const v = s.colIdx[a];
            if (v === u) {
                continue;
            }
            if (!s.directed && v < u) {
                continue;
            }
            lengths.push(Math.sqrt(dist2(positions, u, v, dim)));
        }
    }
    const sorted = Float64Array.from(lengths).sort();
    return quantiles.map((q) => quantile(sorted, q));
}

/**
 * The self-normalised nearest-neighbour histogram: the fraction of nodes whose nn / mean(nn) falls in each of `bins`
 * equal bins over [0, 2) (ratios >= 2 land in the last bin; a zero mean puts every node in bin 0). Scale-free, so
 * two layouts of different extent but the same local structure agree.
 * @param positions - stride-3 positions
 * @param n - node count
 * @param dim - components compared
 * @param bins - number of bins
 * @returns the fractions (sum 1 when n >= 2)
 */
export function nearestNeighbourHistogram(positions: ArrayLike<number>, n: number, dim: 2 | 3, bins: number): number[] {
    const out = new Array<number>(bins).fill(0);
    if (n < 2 || bins < 1) {
        return out;
    }
    const nn = nearestNeighbourDistances(positions, n, dim);
    let mean = 0;
    for (let i = 0; i < n; i++) {
        mean += nn[i];
    }
    mean /= n;
    const width = NN_RANGE / bins;
    for (let i = 0; i < n; i++) {
        const ratio = mean > 0 ? nn[i] / mean : 0;
        const bin = Math.min(bins - 1, Math.floor(ratio / width));
        out[bin] += 1 / n;
    }
    return out;
}

/**
 * The smallest distance between two nodes of different connected components; Infinity when the graph has fewer than
 * two components.
 * @param s - the snapshot
 * @param positions - stride-3 positions
 * @param dim - components compared
 * @returns the separation
 */
export function componentSeparation(s: GraphSnapshot, positions: ArrayLike<number>, dim: 2 | 3): number {
    const { labels, count } = componentLabels(s);
    if (count < 2) {
        return Number.POSITIVE_INFINITY;
    }
    let best = Number.POSITIVE_INFINITY;
    for (let i = 0; i < s.nodeCount; i++) {
        for (let j = i + 1; j < s.nodeCount; j++) {
            if (labels[i] !== labels[j]) {
                best = Math.min(best, dist2(positions, i, j, dim));
            }
        }
    }
    return Math.sqrt(best);
}

/**
 * The largest axis extent (max - min) over the first `dim` axes; 0 for an empty layout. The layout test's
 * "spread > 0.3 in width and height" pin is the 2D case.
 * @param positions - stride-3 positions
 * @param n - node count
 * @param dim - axes considered
 * @returns the extent
 */
export function spread(positions: ArrayLike<number>, n: number, dim: 2 | 3): number {
    if (n === 0) {
        return 0;
    }
    let best = 0;
    for (let k = 0; k < dim; k++) {
        let lo = Number.POSITIVE_INFINITY;
        let hi = Number.NEGATIVE_INFINITY;
        for (let i = 0; i < n; i++) {
            const v = positions[3 * i + k];
            lo = Math.min(lo, v);
            hi = Math.max(hi, v);
        }
        best = Math.max(best, hi - lo);
    }
    return best;
}

/**
 * Every metric above as one record, for the "within 10%" comparisons of spec 11.4: stress, edge-length quantiles
 * (edgeQ10 / edgeQ50 / edgeQ90), nearest-neighbour distance quantiles (nnQ10 / nnQ50 / nnQ90), spread, the
 * separation (only when finite, i.e. at least two components) and the 8 self-normalised nn bins (nnBin0 .. nnBin7).
 * @param s - the snapshot
 * @param positions - stride-3 positions
 * @param dim - components compared
 * @returns the record
 */
export function layoutMetrics(
    s: GraphSnapshot,
    positions: ArrayLike<number>,
    dim: 2 | 3,
): Readonly<Record<string, number>> {
    const n = s.nodeCount;
    const record: Record<string, number> = {};
    record.stress = stress(s, positions, dim);
    const [edgeQ10, edgeQ50, edgeQ90] = edgeLengthQuantiles(s, positions, dim, [0.1, 0.5, 0.9]);
    record.edgeQ10 = edgeQ10;
    record.edgeQ50 = edgeQ50;
    record.edgeQ90 = edgeQ90;
    const nn = nearestNeighbourDistances(positions, n, dim).sort();
    record.nnQ10 = quantile(nn, 0.1);
    record.nnQ50 = quantile(nn, 0.5);
    record.nnQ90 = quantile(nn, 0.9);
    record.spread = spread(positions, n, dim);
    const separation = componentSeparation(s, positions, dim);
    if (Number.isFinite(separation)) {
        record.separation = separation;
    }
    const hist = nearestNeighbourHistogram(positions, n, dim, NN_BINS);
    hist.forEach((fraction, bin) => {
        record[`nnBin${bin}`] = fraction;
    });
    return record;
}
