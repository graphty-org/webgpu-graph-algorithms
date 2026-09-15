/**
 * `fromEdgeArrays()` (design section 8.1, entry point 1): COO typed arrays with dense node indices,
 * the 20 ms path. The input goes through the builder's index-space fast path -- `addAnonymousNodes`
 * (no id Map unless `ids` is given) plus one `addEdges` call -- and then the freeze pipeline of design
 * section 6.3, so every builder policy (duplicate edges, self-loops, weight precision) and every
 * freeze option (label, prepare, arena, checksum) applies exactly as it does to a hand-built
 * builder. Attribute columns are staged in the builder too, so a merging duplicate policy or a
 * dropped self-loop gathers them through the same remaps as any other column (design section 5.11).
 *
 * The options object is the intersection of BuilderOptionsPatch and FreezeOptions; `splitOptions`
 * routes each field to the builder or to the freeze and is shared with `fromRecords`.
 */

import { GraphBuilder } from "../builder/graph-builder.js";
import { columnFromValues, metaToDecl } from "../columns/column.js";
import { GraphFormatError } from "../errors.js";
import { type GraphSnapshot } from "../snapshot/graph-snapshot.js";
import {
    type BuilderOptionsPatch,
    type ColumnDecl,
    type ColumnInput,
    type EdgeArraysInput,
    type FreezeOptions,
    type GraphBuilderOptions,
    type NodeId,
    type TypedArrayData,
} from "../types/index.js";

// ============================================================ options

/** The builder half and the freeze half of a `BuilderOptionsPatch & FreezeOptions` object. */
interface SplitOptions {
    /** The builder options; `directed` is the caller's input value. */
    readonly builder: GraphBuilderOptions;
    /** The freeze options. */
    readonly freeze: FreezeOptions;
}

/**
 * Route the fields of a factory options object to the builder constructor and to `freeze()`.
 * `duplicateEdges` (a field of both) becomes the builder's default policy, which the freeze then
 * applies; an `options.directed` must agree with the input's own `directed` (E_DIRECTED otherwise; the input is
 * authoritative). Undefined fields are left to the builder's and the freeze's own defaults.
 * @param directed - the input's direction
 * @param options - the combined options
 * @returns the two halves
 */
export function splitOptions(directed: boolean, options: BuilderOptionsPatch & FreezeOptions): SplitOptions {
    if (options.directed !== undefined && options.directed !== directed) {
        throw new GraphFormatError(
            "E_DIRECTED",
            `options.directed (${String(options.directed)}) disagrees with the input's directed (${String(directed)})`,
            { reason: "direction mismatch", directed, found: options.directed },
        );
    }
    return {
        builder: {
            directed,
            weighted: options.weighted,
            weightDtype: options.weightDtype,
            duplicateEdges: options.duplicateEdges,
            selfLoops: options.selfLoops,
            addMissingNodes: options.addMissingNodes,
            expectedNodes: options.expectedNodes,
            expectedEdges: options.expectedEdges,
        },
        freeze: {
            label: options.label,
            prepare: options.prepare,
            arena: options.arena,
            release: options.release,
            profile: options.profile,
            checksum: options.checksum,
        },
    };
}

// ============================================================ columns

/**
 * Whether a column value is a typed array (as opposed to a ColumnInput).
 * @param value - the column value
 * @returns true for a typed array
 */
function isTypedArrayData(value: TypedArrayData | ColumnInput): value is TypedArrayData {
    return ArrayBuffer.isView(value);
}

/**
 * Stage one caller-supplied column in the builder (design section 8.1). A typed array (bare or
 * inside a ColumnInput) is bulk-set with its declaration; a JS array of values is resolved with the
 * same inference `AttributeTable.set()` applies (design section 5.1 widening, list detection) and
 * written row by row into a declared column, so a string / list / json column reaches the snapshot
 * through the same freeze step as every other column.
 * @param builder - the builder
 * @param domain - "node" or "edge"
 * @param rows - the table's row count (nodeCount or edge count); a JS array must have this many entries
 * @param name - the column name (the record key; a `decl.name` is ignored)
 * @param value - the typed array or ColumnInput
 */
function stageColumn(
    builder: GraphBuilder,
    domain: "node" | "edge",
    rows: number,
    name: string,
    value: TypedArrayData | ColumnInput,
): void {
    const data = isTypedArrayData(value) ? value : value.data;
    const decl = isTypedArrayData(value) ? {} : value.decl;
    if (ArrayBuffer.isView(data)) {
        if (domain === "node") {
            builder.setNodeColumn(name, data, decl);
        } else {
            builder.setEdgeColumn(name, data, decl);
        }
        return;
    }
    const column = columnFromValues(domain, rows, name, data, decl);
    const declaration: ColumnDecl = { ...metaToDecl(column.meta), name, dtype: column.dtype };
    const handle = domain === "node" ? builder.declareNodeColumn(declaration) : builder.declareEdgeColumn(declaration);
    for (let row = 0; row < rows; row++) {
        if (!column.isSet(row)) {
            continue;
        }
        const cell = column.value(row);
        if (domain === "node") {
            builder.setNodeValue(handle, row, cell);
        } else {
            builder.setEdgeValue(handle, row, cell);
        }
    }
}

// ============================================================ the entry point

/**
 * The E_COLUMN_LENGTH error for two input arrays whose lengths disagree.
 * @param field - the array that is wrong
 * @param expected - the length it should have
 * @param found - the length it has
 * @returns the error
 */
function lengthError(field: string, expected: number, found: number): GraphFormatError {
    return new GraphFormatError("E_COLUMN_LENGTH", `${field} has ${found} entries, expected ${expected}`, {
        field,
        expected,
        found,
    });
}

/**
 * Resolve the node count of an EdgeArraysInput: `ids.length` when ids are given (and `nodeCount`,
 * when also given, must agree), else `nodeCount`, which is then required.
 * @param input - the input
 * @returns the node count
 */
function resolveNodeCount(input: EdgeArraysInput): number {
    const { nodeCount, ids } = input;
    if (nodeCount !== undefined && (!Number.isInteger(nodeCount) || nodeCount < 0)) {
        throw new GraphFormatError("E_INDEX_RANGE", `nodeCount ${nodeCount} is not a non-negative integer`, {
            field: "nodeCount",
            found: nodeCount,
        });
    }
    if (ids !== undefined) {
        if (nodeCount !== undefined && nodeCount !== ids.length) {
            throw lengthError("ids", nodeCount, ids.length);
        }
        return ids.length;
    }
    if (nodeCount === undefined) {
        throw new GraphFormatError("E_INDEX_RANGE", "nodeCount is required when no ids are given", {
            field: "nodeCount",
        });
    }
    return nodeCount;
}

/**
 * Add the caller's ids to the builder in index order; every id must be distinct so that index i holds
 * ids[i] (E_DUPLICATE_ID otherwise, naming the id and both positions).
 * @param builder - the builder
 * @param ids - the ids in index order
 */
function addIds(builder: GraphBuilder, ids: readonly NodeId[] | Float64Array): void {
    const indices = builder.addNodes(ids);
    for (let i = 0; i < indices.length; i++) {
        if (indices[i] !== i) {
            const id = ids[i];
            throw new GraphFormatError("E_DUPLICATE_ID", `ids[${i}] repeats the id of index ${indices[i]}`, {
                id,
                index: indices[i],
                position: i,
            });
        }
    }
}

/**
 * Build a snapshot from COO typed arrays with dense node indices (design section 8.1): the builder's
 * index-space fast path (`addAnonymousNodes` + `addEdges`, no id Map unless `ids` is given) followed
 * by the freeze pipeline, so the result is exactly what a builder fed the same nodes, edges, columns
 * and meta would freeze to. Without `ids` the id map is `identity` with offset 0 and no storage;
 * with `ids` the kind is detected at freeze (design section 4.2). Float64 weights select the f64
 * staging precision unless `options.weightDtype` says otherwise, so the values are downcast to f32 in
 * the arc array and an f64 shadow column is kept only when some value is not f32-exact (design
 * section 3.7).
 * @param input - the arrays; `nodeCount` is required unless `ids` is given (E_INDEX_RANGE); `src`,
 *   `dst` and `weights` must have one entry per edge (E_COLUMN_LENGTH); every index must be below the
 *   node count (E_UNKNOWN_NODE); no weight may be NaN (E_INVALID_WEIGHT); ids must be distinct legal
 *   ids (E_DUPLICATE_ID / E_INVALID_ID); columns must have the table's row count (E_COLUMN_LENGTH)
 * @param options - builder policies and freeze options; `duplicateEdges` and `selfLoops` apply at the
 *   freeze (E_DUPLICATE_EDGE / E_SELF_LOOP under the "error" policies)
 * @returns the frozen snapshot
 */
export function fromEdgeArrays(
    input: EdgeArraysInput,
    options: BuilderOptionsPatch & FreezeOptions = {},
): GraphSnapshot {
    const { src, dst, weights } = input;
    if (dst.length !== src.length) {
        throw lengthError("dst", src.length, dst.length);
    }
    if (weights !== undefined && weights.length !== src.length) {
        throw lengthError("weights", src.length, weights.length);
    }
    const nodeCount = resolveNodeCount(input);
    const edgeCount = src.length;
    const split = splitOptions(input.directed, options);
    const builderOptions: GraphBuilderOptions = {
        ...split.builder,
        weightDtype: options.weightDtype ?? (weights instanceof Float64Array ? "f64" : undefined),
        expectedNodes: options.expectedNodes ?? nodeCount,
        expectedEdges: options.expectedEdges ?? edgeCount,
    };
    const builder = new GraphBuilder(builderOptions);
    if (input.ids === undefined) {
        builder.addAnonymousNodes(nodeCount);
    } else {
        addIds(builder, input.ids);
    }
    builder.addEdges(src, dst, weights);
    if (input.nodeColumns !== undefined) {
        for (const name of Object.keys(input.nodeColumns)) {
            stageColumn(builder, "node", nodeCount, name, input.nodeColumns[name]);
        }
    }
    if (input.edgeColumns !== undefined) {
        for (const name of Object.keys(input.edgeColumns)) {
            stageColumn(builder, "edge", edgeCount, name, input.edgeColumns[name]);
        }
    }
    if (input.meta !== undefined) {
        builder.setMeta(input.meta);
    }
    return builder.freeze(split.freeze);
}
