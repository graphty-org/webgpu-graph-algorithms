/**
 * The children CSR over a containment column (design sections 5.10, 7.1 and 8.2): containment
 * (GEXF `pid` / `<parents>`, Cytoscape `parent`, DOT clusters, GraphML nested graphs) is a node
 * column with the `parent` role (u32, refersTo node) or the `parents` role (list of u32) and never
 * enters the CSR of the snapshot. Importers and exporters that need to walk containers top-down
 * (nested subgraphs, cluster blocks, parent lists) build this inverse index once: for every node
 * `u`, `children[rowPtr[u] .. rowPtr[u + 1])` are its children in ascending index order, `roots`
 * are the nodes with no parent, and `depthFirst()` is the pre-order the nested writers use.
 *
 * The helper is deliberately outside the core (7.1: it computes a property the arrays do not
 * already imply and only io needs it). A reference is ignored when it is unset, INVALID_INDEX, out
 * of range, or the node itself; a cycle (a parent chain that never reaches a root) is reported
 * through `unreachable` and its members are appended to the depth-first order in index order so
 * every node is written exactly once.
 */

import { type Column, type GraphSnapshot, INVALID_INDEX } from "@graphty/graph-format";

/** The role of the column a children CSR is built over. */
export type ContainmentRole = "parent" | "parents";

/** How the CSR is built: from a role column of the snapshot (default `parent`, then `parents`) or an explicit column. */
export interface ChildrenOptions {
    /**
     * The column to read: a role name (`parent` for the u32 single-parent column, `parents` for
     * the list column) or a column object of either shape; "auto" (default) takes the `parent`
     * role column when the snapshot has one, else `parents`, else no containment; null builds
     * the empty CSR (every node a root) whatever the snapshot holds.
     */
    readonly column?: ContainmentRole | Column | "auto" | null | undefined;
}

/** One depth-first traversal of the containment forest. */
export interface DepthFirstOrder {
    /** Every node index exactly once: roots in index order, each followed by its descendants, cycle members last. */
    readonly order: Uint32Array;
    /** The depth of every node in the traversal (0 for a root or a cycle member forced to the top). */
    readonly depth: Uint32Array;
    /** Whether `order` differs from `0..n-1`. */
    readonly reordered: boolean;
}

/**
 * The children CSR of a snapshot's containment column (design sections 5.10 and 7.1): the inverse
 * of the parent relation as a compressed sparse row structure over node indices.
 */
export class ChildrenCsr {
    /** The number of nodes (the row count of the CSR). */
    readonly nodeCount: number;
    /** `rowPtr[u] .. rowPtr[u + 1]` is the range of `children` holding the children of `u`; length nodeCount + 1. */
    readonly rowPtr: Uint32Array;
    /** Child node indices, ascending within each parent's range. */
    readonly children: Uint32Array;
    /** The nodes with no (valid) parent, in index order. */
    readonly roots: Uint32Array;
    /** Every (child, parent) reference kept; equal to `children.length`. */
    readonly edgeCount: number;
    /** References dropped because they were INVALID_INDEX, out of range or self-references. */
    readonly dropped: number;
    /** Nodes whose parent chain never reaches a root (members of a cycle), counted once each. */
    readonly unreachable: number;
    /** The column the CSR was built from, or null when the snapshot has no containment. */
    readonly column: Column | null;

    private readonly parentCount: Uint32Array;

    /**
     * Build the CSR; use childrenCsr() rather than the constructor.
     * @param nodeCount - the node count
     * @param rowPtr - the row pointers
     * @param children - the child list
     * @param parentCount - parents per node (0 for a root)
     * @param dropped - references ignored
     * @param column - the source column, or null
     */
    constructor(
        nodeCount: number,
        rowPtr: Uint32Array,
        children: Uint32Array,
        parentCount: Uint32Array,
        dropped: number,
        column: Column | null,
    ) {
        this.nodeCount = nodeCount;
        this.rowPtr = rowPtr;
        this.children = children;
        this.parentCount = parentCount;
        this.edgeCount = children.length;
        this.dropped = dropped;
        this.column = column;
        let rootCount = 0;
        for (let u = 0; u < nodeCount; u++) {
            if (parentCount[u] === 0) {
                rootCount++;
            }
        }
        const roots = new Uint32Array(rootCount);
        let r = 0;
        for (let u = 0; u < nodeCount; u++) {
            if (parentCount[u] === 0) {
                roots[r++] = u;
            }
        }
        this.roots = roots;
        this.unreachable = nodeCount - this.reachableCount();
    }

    /**
     * The children of a node, as a view into `children` (do not modify).
     * @param node - the node index
     * @returns the child indices in ascending order; empty for a leaf
     */
    childrenOf(node: number): Uint32Array {
        return this.children.subarray(this.rowPtr[node], this.rowPtr[node + 1]);
    }

    /**
     * How many children a node has.
     * @param node - the node index
     * @returns the count
     */
    childCount(node: number): number {
        return this.rowPtr[node + 1] - this.rowPtr[node];
    }

    /**
     * Whether a node is a container (has at least one child).
     * @param node - the node index
     * @returns true when it has children
     */
    hasChildren(node: number): boolean {
        return this.rowPtr[node + 1] > this.rowPtr[node];
    }

    /**
     * How many parents a node has: 0 for a root, 1 under a `parent` column, any number under `parents`.
     * @param node - the node index
     * @returns the parent count
     */
    parentCountOf(node: number): number {
        return this.parentCount[node];
    }

    /**
     * Whether a node has no parent.
     * @param node - the node index
     * @returns true for a root
     */
    isRoot(node: number): boolean {
        return this.parentCount[node] === 0;
    }

    /**
     * Whether the snapshot has any containment at all.
     * @returns true when at least one child reference was kept
     */
    isEmpty(): boolean {
        return this.edgeCount === 0;
    }

    /**
     * A pre-order depth-first traversal from the roots in index order (children in ascending
     * order), then every node not yet visited (a cycle member, or a multi-parent node reached
     * only through a cycle) in index order at depth 0. Each node appears exactly once, so nested
     * writers emit every element once even for a `parents` column that shares a child.
     * @returns the order, the depth of every node and whether the order differs from index order
     */
    depthFirst(): DepthFirstOrder {
        const n = this.nodeCount;
        const order = new Uint32Array(n);
        const depth = new Uint32Array(n);
        const visited = new Uint8Array(n);
        const stackNode: number[] = [];
        const stackCursor: number[] = [];
        let position = 0;
        let reordered = false;
        const place = (u: number, d: number): void => {
            visited[u] = 1;
            if (u !== position) {
                reordered = true;
            }
            order[position++] = u;
            depth[u] = d;
        };
        const visit = (root: number): void => {
            place(root, 0);
            stackNode.push(root);
            stackCursor.push(this.rowPtr[root]);
            while (stackNode.length > 0) {
                const top = stackNode.length - 1;
                const u = stackNode[top];
                const cursor = stackCursor[top];
                if (cursor >= this.rowPtr[u + 1]) {
                    stackNode.pop();
                    stackCursor.pop();
                    continue;
                }
                stackCursor[top] = cursor + 1;
                const v = this.children[cursor];
                if (visited[v] === 1) {
                    continue;
                }
                place(v, stackNode.length);
                stackNode.push(v);
                stackCursor.push(this.rowPtr[v]);
            }
        };
        for (const root of this.roots) {
            visit(root);
        }
        for (let u = 0; u < n; u++) {
            if (visited[u] === 0) {
                visit(u);
            }
        }
        return { order, depth, reordered };
    }

    /**
     * The nodes reachable from the roots by following child references.
     * @returns how many nodes a traversal from the roots reaches
     */
    private reachableCount(): number {
        const visited = new Uint8Array(this.nodeCount);
        // every node is pushed at most once, so a stack of nodeCount entries never overflows
        const stack = new Uint32Array(this.nodeCount);
        let top = 0;
        let count = 0;
        for (const root of this.roots) {
            visited[root] = 1;
            count++;
            stack[top++] = root;
            while (top > 0) {
                const u = stack[--top];
                for (let k = this.rowPtr[u]; k < this.rowPtr[u + 1]; k++) {
                    const v = this.children[k];
                    if (visited[v] === 0) {
                        visited[v] = 1;
                        count++;
                        stack[top++] = v;
                    }
                }
            }
        }
        return count;
    }
}

/**
 * Build the children CSR of a snapshot (design section 7.1: a graph-io helper, not a core view).
 * @param snapshot - the snapshot
 * @param options - which column to read; the `parent` role, then `parents`, by default
 * @returns the CSR; empty (every node a root) when the snapshot has no containment column
 */
export function childrenCsr(snapshot: GraphSnapshot, options: ChildrenOptions = {}): ChildrenCsr {
    const column = resolveColumn(snapshot, options.column === undefined ? "auto" : options.column);
    return childrenFromColumn(snapshot.nodeCount, column);
}

/**
 * Build the children CSR over one containment column without a snapshot (an importer that holds
 * the column before freezing, a test).
 * @param nodeCount - the node count the references index into
 * @param column - a u32 `parent`-shaped column, a list-of-u32 `parents`-shaped column, or null for none
 * @returns the CSR
 */
export function childrenFromColumn(nodeCount: number, column: Column | null): ChildrenCsr {
    const rowPtr = new Uint32Array(nodeCount + 1);
    const parentCount = new Uint32Array(nodeCount);
    let dropped = 0;
    const references: number[] = [];
    if (column !== null) {
        const rows = Math.min(column.length, nodeCount);
        const keep = (child: number, parent: number): void => {
            if (!Number.isInteger(parent) || parent === INVALID_INDEX || parent >= nodeCount || parent === child) {
                dropped++;
                return;
            }
            references.push(child, parent);
            rowPtr[parent + 1]++;
            parentCount[child]++;
        };
        if (column.dtype === "u32") {
            const { data } = column;
            for (let i = 0; i < rows; i++) {
                if (column.isSet(i)) {
                    keep(i, data[i]);
                }
            }
        } else if (column.dtype === "list") {
            for (let i = 0; i < rows; i++) {
                if (!column.isSet(i)) {
                    continue;
                }
                for (const item of column.sliceOf(i)) {
                    if (typeof item === "number") {
                        keep(i, item);
                    } else {
                        dropped++;
                    }
                }
            }
        } else {
            throw new TypeError(
                `a containment column must be u32 or a list of u32, found ${column.dtype} column "${column.meta.name}"`,
            );
        }
    }
    for (let u = 0; u < nodeCount; u++) {
        rowPtr[u + 1] += rowPtr[u];
    }
    const children = new Uint32Array(rowPtr[nodeCount]);
    const fill = rowPtr.slice(0, nodeCount);
    // references were collected in ascending child order, so each parent's range fills ascending
    for (let k = 0; k < references.length; k += 2) {
        children[fill[references[k + 1]]++] = references[k];
    }
    return new ChildrenCsr(nodeCount, rowPtr, children, parentCount, dropped, column);
}

/**
 * The column a ChildrenOptions selects.
 * @param snapshot - the snapshot
 * @param choice - the option value
 * @returns the column, or null when the snapshot has none of that shape
 */
function resolveColumn(snapshot: GraphSnapshot, choice: ContainmentRole | Column | "auto" | null): Column | null {
    if (typeof choice !== "string") {
        return choice;
    }
    if (choice === "parent" || choice === "parents") {
        return snapshot.nodes.byRole(choice);
    }
    return snapshot.nodes.byRole("parent") ?? snapshot.nodes.byRole("parents");
}
