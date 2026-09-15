// Core type definitions for WebGPU Graph Algorithms

/**
 * Represents a graph in CSR (Compressed Sparse Row) format
 */
export interface CSRGraph {
    /** Number of vertices in the graph */
    numVertices: number;
    /** Number of edges in the graph */
    numEdges: number;
    /** Row pointers - indices where each vertex's edges start */
    rowPtr: Uint32Array;
    /** Column indices - destination vertices for each edge */
    colIdx: Uint32Array;
    /** Edge weights (optional) */
    edgeWeights?: Float32Array;
}

/**
 * Represents a graph in edge list format
 */
export interface EdgeListGraph {
    /** Number of vertices in the graph */
    numVertices: number;
    /** Source vertices for each edge */
    sources: Uint32Array;
    /** Destination vertices for each edge */
    destinations: Uint32Array;
    /** Edge weights (optional) */
    weights?: Float32Array;
}

/**
 * WebGPU device and adapter configuration
 */
export interface GPUConfig {
    /** GPU adapter */
    adapter: GPUAdapter;
    /** GPU device */
    device: GPUDevice;
    /** Compute workgroup size */
    workgroupSize?: number;
}

/**
 * Base interface for all graph algorithms
 */
export interface GraphAlgorithm<TInput, TOutput> {
    /** Algorithm name */
    readonly name: string;
    /** Initialize the algorithm with GPU configuration */
    initialize(config: GPUConfig): Promise<void>;
    /** Execute the algorithm on input data */
    execute(input: TInput): Promise<TOutput>;
    /** Clean up GPU resources */
    dispose(): void;
}

/**
 * Result of connected components algorithm
 */
export interface ConnectedComponentsResult {
    /** Component ID for each vertex */
    components: Uint32Array;
    /** Number of components found */
    numComponents: number;
}

/**
 * Result of shortest path algorithm
 */
export interface ShortestPathResult {
    /** Distance to each vertex from source */
    distances: Float32Array;
    /** Parent vertex for each vertex in shortest path tree */
    parents: Int32Array;
}

/**
 * Result of PageRank algorithm
 */
export interface PageRankResult {
    /** PageRank score for each vertex */
    scores: Float32Array;
    /** Number of iterations performed */
    iterations: number;
}