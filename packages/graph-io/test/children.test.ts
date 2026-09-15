import { type Column, GraphBuilder, type GraphSnapshot, INVALID_INDEX } from "@graphty/graph-format";
import { describe, expect, it } from "vitest";

import { ChildrenCsr, childrenCsr, childrenFromColumn } from "../src/children.js";
import { dotImporter } from "../src/formats/dot/importer.js";
import { readCorpusText } from "./helpers/corpus.js";

/**
 * A snapshot whose nodes are the given ids with a u32 `parent` role column holding the given
 * parent index (or undefined for a root).
 * @param ids - the node ids
 * @param parents - the parent index of every node, undefined for none
 * @returns the snapshot
 */
function withParents(ids: readonly string[], parents: readonly (number | undefined)[]): GraphSnapshot {
    const b = new GraphBuilder({ directed: true });
    const parent = b.declareNodeColumn({ name: "parent", dtype: "u32", role: "parent", refersTo: "node" });
    for (const id of ids) {
        b.addNode(id);
    }
    parents.forEach((p, i) => {
        if (p !== undefined) {
            b.setNodeValue(parent, i, p);
        }
    });
    return b.freeze();
}

describe("childrenCsr (design 7.1 / 5.10)", () => {
    it("inverts a parent column into a CSR with roots in index order", () => {
        // a(0) <- b(1), c(2); c <- d(3); e(4) root
        const s = withParents(["a", "b", "c", "d", "e"], [undefined, 0, 0, 2, undefined]);
        const csr = childrenCsr(s);
        expect(csr).toBeInstanceOf(ChildrenCsr);
        expect(csr.nodeCount).toBe(5);
        expect(Array.from(csr.rowPtr)).toEqual([0, 2, 2, 3, 3, 3]);
        expect(Array.from(csr.children)).toEqual([1, 2, 3]);
        expect(Array.from(csr.roots)).toEqual([0, 4]);
        expect(csr.edgeCount).toBe(3);
        expect(csr.dropped).toBe(0);
        expect(csr.unreachable).toBe(0);
        expect(csr.isEmpty()).toBe(false);
        expect(csr.column?.meta.name).toBe("parent");
        expect(Array.from(csr.childrenOf(0))).toEqual([1, 2]);
        expect(Array.from(csr.childrenOf(1))).toEqual([]);
        expect(Array.from(csr.childrenOf(2))).toEqual([3]);
        expect(csr.childCount(0)).toBe(2);
        expect(csr.childCount(4)).toBe(0);
        expect(csr.hasChildren(2)).toBe(true);
        expect(csr.hasChildren(3)).toBe(false);
        expect([0, 1, 2, 3, 4].map((u) => csr.isRoot(u))).toEqual([true, false, false, false, true]);
        expect([0, 1, 2, 3, 4].map((u) => csr.parentCountOf(u))).toEqual([0, 1, 1, 1, 0]);
    });

    it("walks the forest depth-first from the roots in index order, children ascending", () => {
        const s = withParents(["a", "b", "c", "d", "e"], [undefined, 0, 0, 2, undefined]);
        const order = childrenCsr(s).depthFirst();
        expect(Array.from(order.order)).toEqual([0, 1, 2, 3, 4]);
        expect(Array.from(order.depth)).toEqual([0, 1, 1, 2, 0]);
        expect(order.reordered).toBe(false);
        // a child declared before its parent is written after it
        const nested = withParents(["x", "y", "z"], [1, undefined, 0]);
        const walk = childrenCsr(nested).depthFirst();
        expect(Array.from(walk.order)).toEqual([1, 0, 2]);
        expect(Array.from(walk.depth)).toEqual([1, 0, 2]);
        expect(walk.reordered).toBe(true);
    });

    it("keeps cycle members reachable from no root at the top level, each exactly once", () => {
        // a <-> b (a cycle), c root, d child of a
        const s = withParents(["a", "b", "c", "d"], [1, 0, undefined, 0]);
        const csr = childrenCsr(s);
        expect(Array.from(csr.roots)).toEqual([2]);
        expect(csr.unreachable).toBe(3);
        const walk = csr.depthFirst();
        // c first (the only root), then a forced to the top with its children b and d in ascending order
        expect(Array.from(walk.order)).toEqual([2, 0, 1, 3]);
        expect(Array.from(walk.depth)).toEqual([0, 1, 0, 1]);
        expect(walk.reordered).toBe(true);
        expect(new Set(walk.order).size).toBe(4);
    });

    it("drops INVALID_INDEX, out-of-range and self references and treats their nodes as roots", () => {
        const b = new GraphBuilder({ directed: true });
        const parent = b.declareNodeColumn({ name: "parent", dtype: "u32", role: "parent", refersTo: "node" });
        for (const id of ["a", "b", "c", "d"]) {
            b.addNode(id);
        }
        b.setNodeValue(parent, 1, INVALID_INDEX);
        b.setNodeValue(parent, 2, 0);
        const s = b.freeze();
        const column = s.nodes.require("parent");
        // a self reference cannot be pushed through the builder's range checks, so build the CSR
        // over a synthetic column view with one more out-of-range reference and a self reference
        const csr = childrenFromColumn(4, {
            ...column,
            dtype: "u32",
            data: Uint32Array.from([0, INVALID_INDEX, 0, 3]),
            length: 4,
            isSet: (row: number): boolean => row !== 0,
        } as Column);
        expect(csr.dropped).toBe(2);
        expect(Array.from(csr.children)).toEqual([2]);
        expect(Array.from(csr.roots)).toEqual([0, 1, 3]);
        const fromSnapshot = childrenCsr(s);
        expect(Array.from(fromSnapshot.children)).toEqual([2]);
        expect(Array.from(fromSnapshot.roots)).toEqual([0, 1, 3]);
    });

    it("reads a parents list column (multi-parent containment) and visits a shared child once", () => {
        const b = new GraphBuilder({ directed: true });
        const parents = b.declareNodeColumn({
            name: "parents",
            dtype: "list",
            itemDtype: "u32",
            role: "parents",
            refersTo: "node",
        });
        for (const id of ["a", "b", "c", "d"]) {
            b.addNode(id);
        }
        b.setNodeValue(parents, 2, [0, 1]);
        b.setNodeValue(parents, 3, [2]);
        const s = b.freeze();
        const csr = childrenCsr(s);
        expect(csr.column?.meta.name).toBe("parents");
        expect(Array.from(csr.rowPtr)).toEqual([0, 1, 2, 3, 3]);
        expect(Array.from(csr.children)).toEqual([2, 2, 3]);
        expect(Array.from(csr.roots)).toEqual([0, 1]);
        expect(csr.parentCountOf(2)).toBe(2);
        const walk = csr.depthFirst();
        expect(Array.from(walk.order)).toEqual([0, 2, 3, 1]);
        expect(Array.from(walk.depth)).toEqual([0, 0, 1, 2]);
        expect(childrenCsr(s, { column: "parents" }).edgeCount).toBe(3);
        expect(childrenCsr(s, { column: "parent" }).isEmpty()).toBe(true);
    });

    it("prefers the parent role over parents under auto and accepts an explicit column", () => {
        const b = new GraphBuilder({ directed: true });
        const parent = b.declareNodeColumn({ name: "parent", dtype: "u32", role: "parent", refersTo: "node" });
        const parents = b.declareNodeColumn({
            name: "parents",
            dtype: "list",
            itemDtype: "u32",
            role: "parents",
            refersTo: "node",
        });
        b.addNode("a");
        b.addNode("b");
        b.addNode("c");
        b.setNodeValue(parent, 1, 0);
        b.setNodeValue(parents, 2, [0, 1]);
        const s = b.freeze();
        expect(Array.from(childrenCsr(s).children)).toEqual([1]);
        expect(Array.from(childrenCsr(s, { column: "auto" }).children)).toEqual([1]);
        expect(Array.from(childrenCsr(s, { column: s.nodes.require("parents") }).children)).toEqual([2, 2]);
        expect(childrenCsr(s, { column: null }).isEmpty()).toBe(true);
        expect(Array.from(childrenCsr(s, { column: null }).roots)).toEqual([0, 1, 2]);
    });

    it("gives every node as a root when the snapshot has no containment", () => {
        const b = new GraphBuilder({ directed: false });
        b.addEdge("a", "b");
        const s = b.freeze();
        const csr = childrenCsr(s);
        expect(csr.column).toBeNull();
        expect(csr.isEmpty()).toBe(true);
        expect(Array.from(csr.roots)).toEqual([0, 1]);
        expect(Array.from(csr.rowPtr)).toEqual([0, 0, 0]);
        expect(csr.depthFirst().reordered).toBe(false);
        expect(childrenFromColumn(0, null).nodeCount).toBe(0);
        expect(Array.from(childrenFromColumn(0, null).depthFirst().order)).toEqual([]);
    });

    it("refuses a column of another dtype and ignores non-numeric list items", () => {
        const b = new GraphBuilder({ directed: true });
        const tags = b.declareNodeColumn({ name: "tags", dtype: "list", itemDtype: "string" });
        b.addNode("a");
        b.setNodeValue("name", 0, "x");
        b.setNodeValue(tags, 0, ["p", "q"]);
        const s = b.freeze();
        expect(() => childrenFromColumn(1, s.nodes.require("name"))).toThrow(TypeError);
        const csr = childrenFromColumn(1, s.nodes.require("tags"));
        expect(csr.dropped).toBe(2);
        expect(csr.isEmpty()).toBe(true);
    });

    it("indexes the containers of an imported DOT cluster graph", async () => {
        const builder = new GraphBuilder({ directed: true, weightDtype: "f64" });
        await dotImporter.import(readCorpusText("dot", "cluster.gv"), builder);
        const s = builder.freeze();
        const csr = childrenCsr(s);
        expect(csr.column?.meta.role).toBe("parent");
        const containers = [...csr.roots].filter((u) => csr.hasChildren(u)).map((u) => s.ids.idOf(u));
        expect(containers).toEqual(["cluster_0", "cluster_1"]);
        expect(csr.edgeCount).toBe(8);
        expect(csr.unreachable).toBe(0);
        const walk = csr.depthFirst();
        expect(new Set(walk.order).size).toBe(s.nodeCount);
    });
});
