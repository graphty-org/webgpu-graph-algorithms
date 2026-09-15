/**
 * A TEST-ONLY copy of the legacy `Graph` class of `@graphty/algorithms` (algorithms/src/core/graph.ts
 * in the monorepo) together with the four types it needs, so the differential suite of design
 * section 16.2 can compare GraphBuilder output against the legacy Map-of-Maps semantics without
 * the format package taking a dependency on the algorithms package.
 *
 * Semantics are kept exactly, including the legacy quirks the differential tests document:
 *
 * - `allowParallelEdges: true` does not store parallels: the adjacency Map is keyed by the
 *   neighbour id, so a second `addEdge(u, v)` overwrites the stored Edge (last write wins) while
 *   `totalEdgeCount` is still incremented. `neighbors()` and `degree()` therefore count DISTINCT
 *   neighbours, and `getEdge(u, v)` returns the LAST parallel added.
 * - an undirected self-loop is stored once (no reverse entry), so `degree()` counts it once.
 * - `edges()` on an undirected graph skips the entry whose `source > edge.target` using the JS
 *   relational operator, which coerces mixed string / number ids (design section 3.3 calls this the
 *   "string-coercing comparison" that disappears with the format): an undirected edge between a
 *   number id and a non-numeric string id, or between `1` and `"1"`, is yielded TWICE.
 * - `removeNode` on an undirected graph scans every adjacency Map.
 *
 * The only edits are the local type declarations, the removal of the unused-var directive comment
 * and an equivalent spelling of `addEdge`'s parameter defaults (the house eslint config forbids a
 * defaulted parameter before an optional one).
 */

/** Node identifier type of the legacy class. */
export type LegacyNodeId = string | number;

/** Edge representation of the legacy class. */
interface LegacyEdge {
    source: LegacyNodeId;
    target: LegacyNodeId;
    weight?: number;
    id?: string;
    data?: Record<string, unknown> | undefined;
}

/** Node representation of the legacy class. */
interface LegacyNode {
    id: LegacyNodeId;
    data?: Record<string, unknown> | undefined;
}

/** Graph configuration of the legacy class. */
interface LegacyGraphConfig {
    directed: boolean;
    allowSelfLoops: boolean;
    allowParallelEdges: boolean;
}

/**
 * Core Graph data structure for the Graphty Algorithms library (legacy copy).
 *
 * Provides efficient graph representation with support for both directed and undirected graphs.
 * Uses adjacency lists for optimal performance with sparse graphs.
 */
export class LegacyGraph {
    private nodeMap: Map<LegacyNodeId, LegacyNode>;
    private adjacencyList: Map<LegacyNodeId, Map<LegacyNodeId, LegacyEdge>>;
    private incomingEdges: Map<LegacyNodeId, Map<LegacyNodeId, LegacyEdge>>; // For directed graphs
    private config: LegacyGraphConfig;
    private edgeCount: number;

    /**
     * Creates a new Graph instance.
     * @param config - Configuration options for the graph
     */
    constructor(config: Partial<LegacyGraphConfig> = {}) {
        this.config = {
            directed: false,
            allowSelfLoops: true,
            allowParallelEdges: false,
            ...config,
        };
        this.nodeMap = new Map();
        this.adjacencyList = new Map();
        this.incomingEdges = new Map();
        this.edgeCount = 0;
    }

    /**
     * Add a node to the graph
     * @param id - The unique identifier for the node
     * @param data - Optional key-value data to attach to the node
     */
    addNode(id: LegacyNodeId, data?: Record<string, unknown>): void {
        if (!this.nodeMap.has(id)) {
            this.nodeMap.set(id, { id, data });
            this.adjacencyList.set(id, new Map());

            if (this.config.directed) {
                this.incomingEdges.set(id, new Map());
            }
        }
    }

    /**
     * Remove a node from the graph
     * @param id - The unique identifier of the node to remove
     * @returns True if the node was removed, false if it did not exist
     */
    removeNode(id: LegacyNodeId): boolean {
        if (!this.nodeMap.has(id)) {
            return false;
        }

        // Remove all edges connected to this node
        const outgoingEdges = this.adjacencyList.get(id);
        if (outgoingEdges) {
            for (const targetId of Array.from(outgoingEdges.keys())) {
                this.removeEdge(id, targetId);
            }
        }

        if (this.config.directed) {
            const incomingEdges = this.incomingEdges.get(id);
            if (incomingEdges) {
                for (const sourceId of Array.from(incomingEdges.keys())) {
                    this.removeEdge(sourceId, id);
                }
            }
        } else {
            // For undirected graphs, also remove edges where this node is the target
            for (const [nodeId, edges] of Array.from(this.adjacencyList)) {
                if (edges.has(id)) {
                    this.removeEdge(nodeId, id);
                }
            }
        }

        // Remove the node itself
        this.nodeMap.delete(id);
        this.adjacencyList.delete(id);

        if (this.config.directed) {
            this.incomingEdges.delete(id);
        }

        return true;
    }

    /**
     * Add an edge to the graph
     * @param source - The source node identifier
     * @param target - The target node identifier
     * @param weight - The weight of the edge (defaults to 1)
     * @param data - Optional key-value data to attach to the edge
     */
    addEdge(source: LegacyNodeId, target: LegacyNodeId, weight?: number, data?: Record<string, unknown>): void {
        const resolvedWeight = weight === undefined ? 1 : weight;
        // Ensure both nodes exist
        this.addNode(source);
        this.addNode(target);

        // Check self-loops
        if (!this.config.allowSelfLoops && source === target) {
            throw new Error("Self-loops are not allowed in this graph");
        }

        // Check parallel edges
        if (!this.config.allowParallelEdges && this.hasEdge(source, target)) {
            throw new Error("Parallel edges are not allowed in this graph");
        }

        const edge: LegacyEdge = { source, target, weight: resolvedWeight, data };

        // Add to adjacency list
        const sourceAdjacency = this.adjacencyList.get(source);

        if (sourceAdjacency) {
            sourceAdjacency.set(target, edge);
        }

        if (this.config.directed) {
            // For directed graphs, add to incoming edges list
            const targetIncoming = this.incomingEdges.get(target);

            if (targetIncoming) {
                targetIncoming.set(source, edge);
            }
        } else {
            // For undirected graphs, add the reverse edge
            if (source !== target) {
                const reverseEdge: LegacyEdge = { source: target, target: source, weight: resolvedWeight, data };
                const targetAdjacency = this.adjacencyList.get(target);

                if (targetAdjacency) {
                    targetAdjacency.set(source, reverseEdge);
                }
            }
        }

        this.edgeCount++;
    }

    /**
     * Remove an edge from the graph
     * @param source - The source node identifier
     * @param target - The target node identifier
     * @returns True if the edge was removed, false if it did not exist
     */
    removeEdge(source: LegacyNodeId, target: LegacyNodeId): boolean {
        const sourceEdges = this.adjacencyList.get(source);
        if (!sourceEdges?.has(target)) {
            return false;
        }

        sourceEdges.delete(target);

        if (this.config.directed) {
            const targetIncoming = this.incomingEdges.get(target);
            if (targetIncoming) {
                targetIncoming.delete(source);
            }
        } else {
            // For undirected graphs, remove the reverse edge
            const targetEdges = this.adjacencyList.get(target);
            if (targetEdges) {
                targetEdges.delete(source);
            }
        }

        this.edgeCount--;
        return true;
    }

    /**
     * Check if a node exists in the graph
     * @param id - The unique identifier of the node to check
     * @returns True if the node exists, false otherwise
     */
    hasNode(id: LegacyNodeId): boolean {
        return this.nodeMap.has(id);
    }

    /**
     * Check if an edge exists in the graph
     * @param source - The source node identifier
     * @param target - The target node identifier
     * @returns True if the edge exists, false otherwise
     */
    hasEdge(source: LegacyNodeId, target: LegacyNodeId): boolean {
        const sourceEdges = this.adjacencyList.get(source);
        return sourceEdges ? sourceEdges.has(target) : false;
    }

    /**
     * Get a node by ID
     * @param id - The unique identifier of the node to retrieve
     * @returns The node if found, undefined otherwise
     */
    getNode(id: LegacyNodeId): LegacyNode | undefined {
        return this.nodeMap.get(id);
    }

    /**
     * Get an edge by source and target
     * @param source - The source node identifier
     * @param target - The target node identifier
     * @returns The edge if found, undefined otherwise
     */
    getEdge(source: LegacyNodeId, target: LegacyNodeId): LegacyEdge | undefined {
        const sourceEdges = this.adjacencyList.get(source);
        return sourceEdges ? sourceEdges.get(target) : undefined;
    }

    /**
     * Get the number of nodes in the graph
     * @returns The total count of nodes
     */
    get nodeCount(): number {
        return this.nodeMap.size;
    }

    /**
     * Get the number of edges in the graph
     * @returns The total count of edges
     */
    get totalEdgeCount(): number {
        return this.edgeCount;
    }

    /**
     * Check if the graph is directed
     * @returns True if the graph is directed, false otherwise
     */
    get isDirected(): boolean {
        return this.config.directed;
    }

    /**
     * Get all nodes in the graph
     * @returns An iterator over all nodes
     */
    nodes(): IterableIterator<LegacyNode> {
        return this.nodeMap.values();
    }

    /**
     * Get all edges in the graph
     * @yields Each unique edge in the graph
     */
    *edges(): IterableIterator<LegacyEdge> {
        for (const [source, edges] of this.adjacencyList) {
            for (const edge of edges.values()) {
                // For undirected graphs, only yield each edge once
                if (!this.config.directed && source > edge.target) {
                    continue;
                }

                yield edge;
            }
        }
    }

    /**
     * Get neighbors of a node (outgoing edges)
     * @param nodeId - The node identifier to get neighbors for
     * @returns An iterator over the neighbor node identifiers
     */
    neighbors(nodeId: LegacyNodeId): IterableIterator<LegacyNodeId> {
        const edges = this.adjacencyList.get(nodeId);
        return edges ? edges.keys() : new Map<LegacyNodeId, LegacyEdge>().keys();
    }

    /**
     * Get incoming neighbors of a node (directed graphs only)
     * @param nodeId - The node identifier to get incoming neighbors for
     * @returns An iterator over the incoming neighbor node identifiers
     */
    inNeighbors(nodeId: LegacyNodeId): IterableIterator<LegacyNodeId> {
        if (!this.config.directed) {
            return this.neighbors(nodeId);
        }

        const edges = this.incomingEdges.get(nodeId);
        return edges ? edges.keys() : new Map<LegacyNodeId, LegacyEdge>().keys();
    }

    /**
     * Get outgoing neighbors of a node
     * @param nodeId - The node identifier to get outgoing neighbors for
     * @returns An iterator over the outgoing neighbor node identifiers
     */
    outNeighbors(nodeId: LegacyNodeId): IterableIterator<LegacyNodeId> {
        return this.neighbors(nodeId);
    }

    /**
     * Get the degree of a node
     * @param nodeId - The node identifier to get the degree for
     * @returns The total degree of the node
     */
    degree(nodeId: LegacyNodeId): number {
        if (this.config.directed) {
            return this.inDegree(nodeId) + this.outDegree(nodeId);
        }

        const edges = this.adjacencyList.get(nodeId);
        return edges ? edges.size : 0;
    }

    /**
     * Get the in-degree of a node
     * @param nodeId - The node identifier to get the in-degree for
     * @returns The number of incoming edges to the node
     */
    inDegree(nodeId: LegacyNodeId): number {
        if (!this.config.directed) {
            return this.degree(nodeId);
        }

        const edges = this.incomingEdges.get(nodeId);
        return edges ? edges.size : 0;
    }

    /**
     * Get the out-degree of a node
     * @param nodeId - The node identifier to get the out-degree for
     * @returns The number of outgoing edges from the node
     */
    outDegree(nodeId: LegacyNodeId): number {
        const edges = this.adjacencyList.get(nodeId);
        return edges ? edges.size : 0;
    }

    /**
     * Create a copy of the graph
     * @returns A new Graph instance with the same nodes and edges
     */
    clone(): LegacyGraph {
        const cloned = new LegacyGraph(this.config);

        // Copy nodes
        for (const node of this.nodeMap.values()) {
            cloned.addNode(node.id, node.data ? { ...node.data } : undefined);
        }

        // Copy edges
        for (const edge of this.edges()) {
            cloned.addEdge(edge.source, edge.target, edge.weight, edge.data ? { ...edge.data } : undefined);
        }

        return cloned;
    }

    /**
     * Get graph configuration
     * @returns A copy of the graph configuration object
     */
    getConfig(): LegacyGraphConfig {
        return { ...this.config };
    }

    /**
     * Clear all nodes and edges from the graph
     */
    clear(): void {
        this.nodeMap.clear();
        this.adjacencyList.clear();
        this.incomingEdges.clear();
        this.edgeCount = 0;
    }

    /**
     * Get the number of unique edges in the graph
     * For undirected graphs, each edge is counted once
     * @returns The count of unique edges
     */
    get uniqueEdgeCount(): number {
        if (this.config.directed) {
            return this.edgeCount;
        }

        // For undirected graphs, we need to count each edge only once
        let count = 0;

        for (const _edge of this.edges()) {
            count++;
        }
        return count;
    }
}
