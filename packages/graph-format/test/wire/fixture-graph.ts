/**
 * The fixture graphs of the wire tests and of the golden fixture generator: a snapshot exercising
 * every dtype, every table kind and the tagged metadata values, and a small snapshot with caller
 * chosen ids. No vitest import here so tmp/make-golden.ts can run it under tsx.
 */

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { type GraphSnapshot } from "../../src/snapshot/graph-snapshot.js";
import { type NodeId } from "../../src/types/index.js";

/** Options of richSnapshot(). */
interface RichOptions {
    readonly ids?: readonly NodeId[];
    readonly directed?: boolean;
    readonly checksum?: boolean;
}

/**
 * A snapshot with string ids (by default), weights, a self-loop, a parallel edge, one column of every
 * dtype on the node table, edge / graph columns, an extension table and tagged metadata values.
 */
export function richSnapshot(options: RichOptions = {}): GraphSnapshot {
    const ids = options.ids ?? ["a", "b", "c", "d"];
    const b = new GraphBuilder({ directed: options.directed ?? false, weightDtype: "f64" });
    b.addNodes(ids);
    b.declareNodeColumn({ name: "pos", dtype: "f32", components: 3, role: "position", mutable: true });
    b.declareNodeColumn({ name: "end", dtype: "f64", role: "end", default: Infinity });
    b.declareNodeColumn({ name: "nanfill", dtype: "f64", fill: Number.NaN });
    b.declareNodeColumn({ name: "score", dtype: "i32", default: -1 });
    b.declareNodeColumn({ name: "parent", dtype: "u32", refersTo: "node", role: "parent" });
    b.declareNodeColumn({ name: "byte", dtype: "u8" });
    b.declareNodeColumn({ name: "flag", dtype: "bool" });
    b.declareNodeColumn({ name: "cat", dtype: "dict", options: ["x", "y"], role: "kind" });
    b.declareNodeColumn({ name: "label", dtype: "string", role: "label", origin: { format: "gexf", id: "0" } });
    b.declareNodeColumn({ name: "tags", dtype: "list", itemDtype: "string" });
    b.declareNodeColumn({ name: "spells", dtype: "list", itemDtype: "f64", itemComponents: 2, role: "spells" });
    b.declareNodeColumn({ name: "blob", dtype: "json", extra: { note: "nested", limit: -0 } });
    b.declareNodeColumn({ name: "solid", dtype: "u32", nullable: false });
    b.declareEdgeColumn({ name: "id", dtype: "string", role: "id", unique: true });
    b.declareEdgeColumn({ name: "kind", dtype: "dict" });
    b.declareEdgeColumn({ name: "level", dtype: "u8", default: 7 });
    const n = ids.length;
    for (let i = 0; i < n; i++) {
        b.setNodeValue("pos", i, [i, i * 2, i * 3]);
        if (i % 2 === 0) {
            b.setNodeValue("end", i, i * 10);
            b.setNodeValue("score", i, i - 2);
            b.setNodeValue("parent", i, (i + 1) % n);
            b.setNodeValue("byte", i, 200 + i);
            b.setNodeValue("flag", i, i % 4 === 0);
            b.setNodeValue("cat", i, i === 0 ? "y" : "z");
            b.setNodeValue("label", i, `node ${i} \u00e9`);
            b.setNodeValue("tags", i, [`t${i}`, "shared"]);
            b.setNodeValue("spells", i, [
                [i, i + 1],
                [i + 2, i + 3],
            ]);
            b.setNodeValue("blob", i, { deep: [1, "x", null, { k: i }], inf: Infinity, negZero: -0 });
        } else {
            b.setNodeValue("nanfill", i, i / 3);
            b.setNodeValue("tags", i, []);
            b.setNodeValue("blob", i, null);
        }
        b.setNodeValue("solid", i, i * 100);
    }
    const e0 = b.addEdge(ids[0], ids[1], 0.5);
    const e1 = b.addEdge(ids[1], ids[2], 2);
    const e2 = b.addEdge(ids[2], ids[2], 3);
    const e3 = b.addEdge(ids[0], ids[1], 0.1);
    const e4 = b.addEdge(ids[3], ids[0]);
    b.setEdgeValue("id", e0, "e0");
    b.setEdgeValue("id", e1, "e1");
    b.setEdgeValue("id", e2, "e2");
    b.setEdgeValue("id", e3, "e3");
    b.setEdgeValue("id", e4, "e4");
    b.setEdgeValue("kind", e0, "friend");
    b.setEdgeValue("kind", e1, "foe");
    b.setEdgeValue("kind", e3, "friend");
    b.setEdgeValue("level", e1, 9);
    b.setGraphValue("title", "rich");
    b.setGraphValue("count", 3, { dtype: "i32" });
    b.setGraphValue("payload", { a: [1, 2, { b: "c" }] });
    const t = b.addExtensionTable("temporal:node:price", [
        { name: "element", dtype: "u32", refersTo: "node" },
        { name: "start", dtype: "f64", role: "start" },
        { name: "end", dtype: "f64", role: "end", default: Infinity },
        { name: "value", dtype: "f64" },
        { name: "open", dtype: "u8", role: "open" },
    ]);
    b.addExtensionRow(t, [0, 1, 2, 10, 0]);
    b.addExtensionRow(t, [1, 2, undefined, 20, 1]);
    b.addExtensionRow(t, [2, 3, 4, 30, undefined]);
    b.setMeta({
        name: "rich graph",
        description: "every dtype",
        keywords: ["a", "b"],
        sourceFormat: "gexf",
        sourceVersion: "1.3",
        idType: "string",
        timeFormat: "double",
        timeRepresentation: "interval",
        mode: "dynamic",
        declaredMultigraph: true,
        weightOrigin: { format: "gexf", id: null, title: null, type: "double", namespace: null },
        extra: { nested: { inf: Infinity, ninf: -Infinity, nan: Number.NaN, nz: -0, list: [1, "two", null] } },
    });
    return b.freeze({ label: "rich", checksum: options.checksum });
}

/** A small snapshot with the given ids (identity / dense / numeric / string / mixed selection). */
export function idsSnapshot(ids: readonly NodeId[], directed = true): GraphSnapshot {
    const b = new GraphBuilder({ directed });
    b.addNodes(ids);
    for (let i = 0; i + 1 < ids.length; i++) {
        b.addEdge(ids[i], ids[i + 1]);
    }
    return b.freeze();
}
