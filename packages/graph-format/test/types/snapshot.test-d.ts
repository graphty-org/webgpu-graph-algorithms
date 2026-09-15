import {
    type AdjacencyView,
    type ArenaLayout,
    type AttributeTable,
    type CooView,
    type DegreeOrderView,
    type DerivedGraph,
    type EdgeListView,
    type F32,
    type F64,
    type GraphMeta,
    type GraphSnapshot,
    INVALID_INDEX,
    type NodeId,
    type NodeIdMap,
    type ReverseView,
    SNAPSHOT_BRAND,
    type SnapshotFlags,
    type U8,
    type U32,
    type ViewName,
    type WireSnapshot,
} from "@graphty/graph-format";
import { expectTypeOf } from "vitest";

declare const snapshot: GraphSnapshot;
declare const source: number;

// ---- the assignability the design promises (section 12.3): GraphSnapshot and ReverseView are
// AdjacencyViews; an AdjacencyView is not a GraphSnapshot
expectTypeOf<GraphSnapshot>().toMatchTypeOf<AdjacencyView>();
expectTypeOf<ReverseView>().toMatchTypeOf<AdjacencyView>();
expectTypeOf<AdjacencyView>().not.toMatchTypeOf<GraphSnapshot>();
expectTypeOf<AdjacencyView>().not.toMatchTypeOf<ReverseView>();
expectTypeOf(snapshot.reverse()).toMatchTypeOf<AdjacencyView>();

// ---- identity, counts, brand
expectTypeOf(snapshot[SNAPSHOT_BRAND]).toEqualTypeOf<true>();
expectTypeOf(snapshot.formatVersion).toEqualTypeOf<1>();
expectTypeOf(snapshot.serial).toBeNumber();
expectTypeOf(snapshot.label).toEqualTypeOf<string | null>();
expectTypeOf(snapshot.directed).toBeBoolean();
expectTypeOf(snapshot.nodeCount).toBeNumber();
expectTypeOf(snapshot.edgeCount).toBeNumber();
expectTypeOf(snapshot.arcCount).toBeNumber();
expectTypeOf(snapshot.selfLoopCount).toBeNumber();
expectTypeOf(snapshot.detached).toBeBoolean();

// ---- core arrays: plain U32 / F32 over ArrayBuffer, never Readonly wrappers, never optional
expectTypeOf(snapshot.rowPtr).toEqualTypeOf<U32>();
expectTypeOf(snapshot.colIdx).toEqualTypeOf<U32>();
expectTypeOf(snapshot.arcToEdge).toEqualTypeOf<U32>();
expectTypeOf(snapshot.edgeToArc).toEqualTypeOf<U32>();
expectTypeOf(snapshot.weights).toEqualTypeOf<F32 | null>();
expectTypeOf(snapshot.arena).toEqualTypeOf<ArenaLayout | null>();

// ---- side structures
expectTypeOf(snapshot.flags).toEqualTypeOf<SnapshotFlags>();
expectTypeOf(snapshot.ids).toEqualTypeOf<NodeIdMap>();
expectTypeOf(snapshot.nodes).toEqualTypeOf<AttributeTable>();
expectTypeOf(snapshot.edges).toEqualTypeOf<AttributeTable>();
expectTypeOf(snapshot.graph).toEqualTypeOf<AttributeTable>();
expectTypeOf(snapshot.extensions).toEqualTypeOf<ReadonlyMap<string, AttributeTable>>();
expectTypeOf(snapshot.meta).toEqualTypeOf<GraphMeta>();

// ---- output shapes have no optional members: "absent" is null (design section 12.1)
expectTypeOf<Required<SnapshotFlags>>().toEqualTypeOf<SnapshotFlags>();
expectTypeOf<Required<GraphMeta>>().toEqualTypeOf<GraphMeta>();
expectTypeOf<Required<DerivedGraph>>().toEqualTypeOf<DerivedGraph>();
expectTypeOf<Required<ArenaLayout>>().toEqualTypeOf<ArenaLayout>();
expectTypeOf<SnapshotFlags[keyof SnapshotFlags]>().toEqualTypeOf<boolean>();

// ---- arena
declare const arena: ArenaLayout;
expectTypeOf(arena.alignment).toEqualTypeOf<256>();
expectTypeOf(arena.segments.weights).toEqualTypeOf<{
    readonly byteOffset: number;
    readonly byteLength: number;
} | null>();
expectTypeOf(arena.buffer).toEqualTypeOf<ArrayBuffer>();

// ---- queries (3.9)
expectTypeOf(snapshot.outArcs(0)).toEqualTypeOf<readonly [start: number, end: number]>();
expectTypeOf(snapshot.arcsBetween(0, 1)).toEqualTypeOf<readonly [lo: number, hi: number]>();
expectTypeOf(snapshot.findArc(0, 1)).toBeNumber();
expectTypeOf(snapshot.hasArc(0, 1)).toBeBoolean();
expectTypeOf(snapshot.multiplicity(0, 1)).toBeNumber();
expectTypeOf(snapshot.arcSource(0)).toBeNumber();
expectTypeOf(snapshot.edgeSource(0)).toBeNumber();
expectTypeOf(snapshot.edgeTarget(0)).toBeNumber();
expectTypeOf(snapshot.edgeIndexOf("e0")).toBeNumber();
expectTypeOf(snapshot.edgeIndexOf(7)).toBeNumber();

// ---- views (7.2): 4-byte arrays except the four f64 views and the scalar totalWeight
expectTypeOf(snapshot.reverse()).toEqualTypeOf<ReverseView>();
expectTypeOf(snapshot.reverse().fwdArc).toEqualTypeOf<U32>();
expectTypeOf(snapshot.coo()).toEqualTypeOf<CooView>();
expectTypeOf(snapshot.edgeList()).toEqualTypeOf<EdgeListView>();
expectTypeOf(snapshot.edgeList().arc).toEqualTypeOf<U32>();
expectTypeOf(snapshot.outDegree()).toEqualTypeOf<U32>();
expectTypeOf(snapshot.inDegree()).toEqualTypeOf<U32>();
expectTypeOf(snapshot.degree()).toEqualTypeOf<U32>();
expectTypeOf(snapshot.weightedOutDegree()).toEqualTypeOf<F64>();
expectTypeOf(snapshot.weightedInDegree()).toEqualTypeOf<F64>();
expectTypeOf(snapshot.weightedDegree()).toEqualTypeOf<F64>();
expectTypeOf(snapshot.selfLoopWeight()).toEqualTypeOf<F64>();
expectTypeOf(snapshot.totalWeight()).toBeNumber();
expectTypeOf(snapshot.selfLoopArcs()).toEqualTypeOf<U32>();
expectTypeOf(snapshot.selfLoopsPerNode()).toEqualTypeOf<U32>();
expectTypeOf(snapshot.selfLoopsAt(0)).toBeNumber();
expectTypeOf(snapshot.mate()).toEqualTypeOf<U32>();
expectTypeOf(snapshot.degreeOrder()).toEqualTypeOf<DegreeOrderView>();
expectTypeOf(snapshot.degreeOrder({ of: "reverse" })).toEqualTypeOf<DegreeOrderView>();
expectTypeOf(snapshot.degreeOrder({ of: undefined })).toEqualTypeOf<DegreeOrderView>();
expectTypeOf(snapshot.isSymmetric()).toBeBoolean();
expectTypeOf(snapshot.prepare(["reverse", "coo"])).toEqualTypeOf<GraphSnapshot>();
expectTypeOf(snapshot.cachedViews()).toEqualTypeOf<readonly ViewName[]>();
expectTypeOf(snapshot.dropCaches()).toBeVoid();

// ---- derived graphs (7.3)
expectTypeOf(snapshot.toUndirected()).toEqualTypeOf<DerivedGraph>();
expectTypeOf(snapshot.toUndirected({ reciprocal: true, weights: "sum" })).toEqualTypeOf<DerivedGraph>();
expectTypeOf(snapshot.transpose()).toEqualTypeOf<DerivedGraph>();
expectTypeOf(snapshot.simplified({ selfLoops: "drop", edgeReducers: { kind: "first" } })).toEqualTypeOf<DerivedGraph>();
expectTypeOf(snapshot.withoutSelfLoops()).toEqualTypeOf<DerivedGraph>();
expectTypeOf(snapshot.filterEdges(new Uint32Array(1))).toEqualTypeOf<DerivedGraph>();
expectTypeOf(snapshot.inducedSubgraph(new Uint32Array([0, 2]))).toEqualTypeOf<DerivedGraph>();
expectTypeOf(snapshot.inducedSubgraph({ mask: new Uint32Array(1) })).toEqualTypeOf<DerivedGraph>();
expectTypeOf(snapshot.contract(new Uint32Array(3), { weights: "sum", parallel: "keep" })).toEqualTypeOf<DerivedGraph>();
expectTypeOf(snapshot.relabel(new Uint32Array(3))).toEqualTypeOf<DerivedGraph>();
expectTypeOf(snapshot.withColumns({ score: new Float32Array(3) })).toEqualTypeOf<GraphSnapshot>();
expectTypeOf(
    snapshot.withColumns(undefined, { flag: { data: new Uint32Array(1), decl: { dtype: "bool" } } }),
).toEqualTypeOf<GraphSnapshot>();

declare const derived: DerivedGraph;
expectTypeOf(derived.snapshot).toEqualTypeOf<GraphSnapshot>();
expectTypeOf(derived.nodeOrigin).toEqualTypeOf<U32 | null>();
expectTypeOf(derived.edgeOrigin).toEqualTypeOf<U32 | null>();
expectTypeOf(derived.nodeRemap).toEqualTypeOf<U32 | null>();
expectTypeOf(derived.edgeRemap).toEqualTypeOf<U32 | null>();
expectTypeOf(derived.blockSizes).toEqualTypeOf<U32 | null>();
expectTypeOf(derived.report).toEqualTypeOf<{ readonly droppedEdges: number; readonly mergedEdges: number }>();
// a derived snapshot is usable wherever a snapshot is
expectTypeOf(derived.snapshot.toUndirected().snapshot).toEqualTypeOf<GraphSnapshot>();

// ---- memory, transfer, checks (9, 11)
expectTypeOf(snapshot.byteLength()).toBeNumber();
expectTypeOf(snapshot.byteLength({ views: true, columns: undefined })).toBeNumber();
expectTypeOf(snapshot.contentHash()).toBeString();
expectTypeOf(snapshot.transferables()).toEqualTypeOf<ArrayBuffer[]>();
expectTypeOf(snapshot.toWire()).toEqualTypeOf<WireSnapshot>();
expectTypeOf(snapshot.toWire({ transfer: true, includeViews: ["reverse"] })).toEqualTypeOf<WireSnapshot>();
expectTypeOf(snapshot.toBytes()).toEqualTypeOf<U8>();
expectTypeOf(snapshot.toByteChunks({ includeViews: undefined })).toEqualTypeOf<Iterable<U8>>();
expectTypeOf(snapshot.validate()).toBeVoid();
expectTypeOf(snapshot.validate({ level: "full", checksum: true })).toBeVoid();

// ---- id map (4.3)
declare const ids: NodeIdMap;
expectTypeOf(ids.kind).toEqualTypeOf<"identity" | "dense" | "numeric" | "string" | "mixed">();
expectTypeOf(ids.idOf(0)).toEqualTypeOf<NodeId>();
expectTypeOf(ids.indexOf("a")).toBeNumber();
expectTypeOf(ids.requireIndex(1)).toBeNumber();
expectTypeOf(ids.indicesOf(["a", 1], "throw")).toEqualTypeOf<U32>();
expectTypeOf(ids.idsSlice()).toEqualTypeOf<NodeId[]>();
expectTypeOf(ids.toMap(new Float64Array(2))).toEqualTypeOf<Map<NodeId, number>>();
expectTypeOf(ids.toStringMap([true, false])).toEqualTypeOf<Map<string, boolean>>();
expectTypeOf(ids.toRecord(new Uint32Array(2))).toEqualTypeOf<Record<string, number>>();
expectTypeOf(ids.entries(["x", "y"])).toEqualTypeOf<IterableIterator<[NodeId, string]>>();
expectTypeOf(ids.stringIndex()).toEqualTypeOf<ReadonlyMap<string, number>>();
expectTypeOf([...ids]).toEqualTypeOf<NodeId[]>();

// ---- a flag-neutral BFS over any AdjacencyView (design section 16.6): compiles with
// noUncheckedIndexedAccess on and off, no `as number`, no non-null assertion
function bfs(view: AdjacencyView, root: number): U32 {
    const { rowPtr, colIdx, nodeCount } = view;
    const level = new Uint32Array(nodeCount).fill(INVALID_INDEX);
    const queue = new Uint32Array(nodeCount);
    let head = 0;
    let tail = 0;
    level[root] = 0;
    queue[tail] = root;
    tail += 1;
    while (head < tail) {
        const u = queue[head];
        head += 1;
        if (u === undefined) {
            break;
        }
        const rowStart = rowPtr[u];
        const rowEnd = rowPtr[u + 1];
        const levelU = level[u];
        if (rowStart === undefined || rowEnd === undefined || levelU === undefined) {
            break;
        }
        for (let a = rowStart; a < rowEnd; a += 1) {
            const v = colIdx[a];
            if (v !== undefined && level[v] === INVALID_INDEX) {
                level[v] = levelU + 1;
                queue[tail] = v;
                tail += 1;
            }
        }
    }
    return level;
}

expectTypeOf(bfs(snapshot, source)).toEqualTypeOf<U32>();
expectTypeOf(bfs(snapshot.reverse(), source)).toEqualTypeOf<U32>();
