/**
 * nodeMass and weight resolution from graph-format primitives only (spec 7.14, 7.5; D28, Q-30): the role-"mass"
 * node column first, then outDegree() + 1; weights from the arc array, from a named edge column expanded to arcs
 * with expandEdges, or none. The Record form of nodeMass is rejected with the hint of spec 7.14 (graphty-element
 * writes its nodeMass config as the role column once at engine creation, so both simulations find it the same way
 * and this package parses no ids).
 */

import { type Column, expandEdges, type F32, type GraphSnapshot, type NodeId } from "@graphty/graph-format";

import { WebGpuGraphError } from "../errors.js";

/** The message of spec 7.14 for the Record form of nodeMass. */
const NODE_MASS_HINT =
    "write a role 'mass' node column (nodes.set(name, vec, { role: 'mass', replaceRole: true })) or pass a " +
    "Float32Array; @graphty/layout's resolveNodeVector does this on the CPU path";

/**
 * An E_INVALID_ARGUMENT with the documented details shape.
 * @param argument - the option name
 * @param value - the offending value
 * @param expected - what was expected
 * @param message - the message
 * @returns the error
 */
function invalid(argument: string, value: unknown, expected: unknown, message: string): WebGpuGraphError {
    return new WebGpuGraphError("E_INVALID_ARGUMENT", message, { argument, value, expected });
}

/**
 * Asserts every mass is finite and > 0 (spec 7.14: a mass of 0 would divide the distributed action by zero).
 * @param mass - the masses
 * @param source - what the masses came from, for the message
 */
function checkMass(mass: ArrayLike<number>, source: string): void {
    for (let i = 0; i < mass.length; i++) {
        const v = mass[i];
        if (!Number.isFinite(v) || v <= 0) {
            throw new WebGpuGraphError(
                "E_INVALID_ARGUMENT",
                `${source}: mass[${i}] = ${v} is not a finite number > 0`,
                {
                    argument: "nodeMass",
                    value: v,
                    index: i,
                    expected: "finite masses > 0",
                },
            );
        }
    }
}

/**
 * outDegree()[i] + 1 for every node (spec 7.2 "Mass" row; one O(n) loop over the cached view).
 * @param s - the snapshot
 * @returns a fresh F32 of nodeCount masses
 */
function degreePlusOne(s: GraphSnapshot): F32 {
    const n = s.nodeCount;
    const degree = s.outDegree();
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        out[i] = degree[i] + 1;
    }
    return out;
}

/**
 * The masses of a node column: f32 as is, f64 / u32 / i32 through gpuView (f64 = the cached f32 copy; u32 / i32
 * converted to a fresh F32), u8 through column.data; every other dtype is E_INVALID_ARGUMENT (PLAN DECISION 14).
 * @param s - the snapshot (the column belongs to s.nodes)
 * @param column - the node column
 * @returns nodeCount masses
 */
function massFromColumn(s: GraphSnapshot, column: Column): F32 {
    const { name } = column.meta;
    if (column.meta.components !== 1) {
        throw invalid(
            "nodeMass",
            name,
            "a one-component numeric node column",
            `node column "${name}" has ${column.meta.components} components; a mass column has one`,
        );
    }
    let values: ArrayLike<number>;
    switch (column.dtype) {
        case "f32":
        case "f64": {
            const view = s.nodes.gpuView(name);
            const out = view instanceof Float32Array ? view : new Float32Array(view);
            checkMass(out, `node column "${name}"`);
            return out;
        }
        case "u32":
        case "i32":
            values = s.nodes.gpuView(name);
            break;
        case "u8":
            values = column.data;
            break;
        default:
            throw invalid(
                "nodeMass",
                name,
                "a numeric node column (f32, f64, u32, i32, u8)",
                `node column "${name}" is ${column.dtype}, not numeric`,
            );
    }
    const out = new Float32Array(s.nodeCount);
    for (let i = 0; i < out.length; i++) {
        out[i] = values[i];
    }
    checkMass(out, `node column "${name}"`);
    return out;
}

/**
 * nodeMass resolution by ROLE (spec 7.14): null -> the role-"mass" node column when present (any numeric dtype
 * through gpuView, converted to a fresh F32 when not f32), else outDegree()[i] + 1; an F32 of length n -> as is;
 * a column NAME -> nodes.get(name) (numeric); a Record -> E_UNSUPPORTED with the spec's message. Always returns n
 * values, every one finite and > 0.
 * @param s - the snapshot
 * @param spec - the ForceAtlas2Options.nodeMass value
 * @returns nodeCount masses (the caller's array when an F32 was given)
 */
export function resolveNodeMass(
    s: GraphSnapshot,
    spec: F32 | string | Readonly<Record<NodeId, number>> | null | undefined,
): F32 {
    const n = s.nodeCount;
    if (spec === null || spec === undefined) {
        const column = s.nodes.byRole("mass");
        if (column === null) {
            return degreePlusOne(s);
        }
        return massFromColumn(s, column);
    }
    if (spec instanceof Float32Array) {
        if (spec.length !== n) {
            throw invalid("nodeMass", spec.length, `${n} values`, `nodeMass has ${spec.length} values, expected ${n}`);
        }
        checkMass(spec, "nodeMass");
        return spec;
    }
    if (typeof spec === "string") {
        const column = s.nodes.get(spec);
        if (column === null) {
            throw invalid(
                "nodeMass",
                spec,
                "an existing numeric node column",
                `nodeMass names node column "${spec}", which the snapshot does not hold`,
            );
        }
        return massFromColumn(s, column);
    }
    if (ArrayBuffer.isView(spec)) {
        throw invalid(
            "nodeMass",
            "typed array",
            "a Float32Array",
            "nodeMass must be a Float32Array, a column name or null",
        );
    }
    throw new WebGpuGraphError(
        "E_UNSUPPORTED",
        `nodeMass as a Record is not supported on the GPU path: ${NODE_MASS_HINT}`,
        {
            option: "nodeMass",
            hint: NODE_MASS_HINT,
        },
    );
}

/** What resolveWeights found. */
export interface ResolvedWeights {
    readonly data: F32 | null;
    readonly source: "arcs" | "column" | "none";
    readonly column: Column | null;
}

/**
 * Weight resolution (spec 7.5, 7.14): true -> { data: s.weights, source: "arcs" } (data null when unweighted ->
 * ones); a string -> a name `s.edges` does not hold is E_INVALID_ARGUMENT (spec 7.14's nodeMass rule applied to
 * weight too), else `column = s.edges.get(name)`, then by `column.dtype`: f32 / f64 / u32 / i32 ->
 * expandEdges(s, s.edges.gpuView(name)) (f64 arrives as gpuView's cached f32 copy; u32 / i32 are converted to a
 * fresh F32; `Column.data` is NOT used because it is U8 for u8 / bool and absent for string / list / json), string /
 * list / json -> the gpuView E_GPU_INELIGIBLE pass-through, u8 / bool / dict -> E_INVALID_ARGUMENT (gpuView returns
 * packed words, not per-edge values); source "column", `column` kept for the version check; false / null /
 * undefined -> none. A named column must have one component (PLAN DECISION 15); the weight values are never
 * inspected.
 * @param s - the snapshot
 * @param spec - the ForceAtlas2Options.weight value
 * @returns the resolved weights
 */
export function resolveWeights(s: GraphSnapshot, spec: boolean | string | null | undefined): ResolvedWeights {
    if (spec === true) {
        return { data: s.weights, source: "arcs", column: null };
    }
    if (spec === false || spec === null || spec === undefined) {
        return { data: null, source: "none", column: null };
    }
    if (typeof spec !== "string") {
        throw invalid(
            "weight",
            spec,
            "true, false, null or an edge column name",
            "weight must be true, false, null or an edge column name",
        );
    }
    const column = s.edges.get(spec);
    if (column === null) {
        throw invalid(
            "weight",
            spec,
            "an existing numeric edge column",
            `weight names edge column "${spec}", which the snapshot does not hold`,
        );
    }
    if (column.meta.components !== 1) {
        throw invalid(
            "weight",
            spec,
            "a one-component numeric edge column",
            `edge column "${spec}" has ${column.meta.components} components; a weight column has one`,
        );
    }
    switch (column.dtype) {
        case "f32":
        case "f64":
        case "u32":
        case "i32": {
            const view = s.edges.gpuView(spec);
            const expanded = expandEdges(s, view);
            const data = expanded instanceof Float32Array ? expanded : new Float32Array(expanded);
            return { data, source: "column", column };
        }
        case "string":
        case "list":
        case "json":
            // gpuView throws the GraphFormatError E_GPU_INELIGIBLE, which passes through unchanged (D12)
            s.edges.gpuView(spec);
            throw invalid("weight", spec, "a numeric edge column", `edge column "${spec}" is ${column.dtype}`);
        default:
            throw invalid(
                "weight",
                spec,
                "an f32 / f64 / u32 / i32 edge column",
                `edge column "${spec}" is ${column.dtype}: its gpuView is packed words, not per-edge values`,
            );
    }
}
