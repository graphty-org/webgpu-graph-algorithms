/**
 * fast-check arbitraries for builder operation sequences (design section 16.1): random ids of both
 * types, weights including 0 / negative / Infinity / non-f32-exact values, duplicates, self-loops,
 * removals, weight updates, column writes and freezes with random options. Every policy enum is a
 * dimension of the configuration arbitrary so shrinking yields a minimal configuration and sequence.
 */

import fc from "fast-check";

import { type DuplicatePolicy, type NodeId } from "../../src/types/index.js";

/** The configuration dimensions of a scenario. */
export interface OpsConfig {
    readonly directed: boolean;
    readonly weighted: boolean | "auto";
    readonly weightDtype: "f32" | "f64";
    readonly duplicateEdges: DuplicatePolicy;
    readonly selfLoops: "keep" | "drop" | "error";
    readonly addMissingNodes: boolean;
    /** Which id pool the sequence draws from. */
    readonly idKind: "numbers" | "strings" | "mixed";
}

/** One builder operation. Edge indices are taken modulo the builder's edgeBound + 1 at run time. */
export type Op =
    | { readonly type: "addNode"; readonly id: NodeId }
    | {
          readonly type: "addEdge";
          readonly source: NodeId;
          readonly target: NodeId;
          readonly weight: number | undefined;
      }
    | { readonly type: "removeNode"; readonly id: NodeId }
    | { readonly type: "removeEdge"; readonly edge: number }
    | { readonly type: "setEdgeWeight"; readonly edge: number; readonly weight: number }
    | { readonly type: "setNodeValue"; readonly id: NodeId; readonly column: string; readonly value: unknown }
    | { readonly type: "setEdgeValue"; readonly edge: number; readonly column: string; readonly value: unknown }
    | { readonly type: "freeze"; readonly duplicateEdges: DuplicatePolicy | undefined; readonly arena: boolean };

const NUMBER_IDS: readonly NodeId[] = [0, 1, 2, 3, 4, 5, 6, 7];
const STRING_IDS: readonly NodeId[] = ["a", "b", "c", "d", "e", "f", "g", "h"];
const MIXED_IDS: readonly NodeId[] = [0, 1, 2, "1", "a", "b", 2.5, -0];

/** The id pool of a configuration. */
function idPool(kind: OpsConfig["idKind"]): readonly NodeId[] {
    switch (kind) {
        case "numbers":
            return NUMBER_IDS;
        case "strings":
            return STRING_IDS;
        default:
            return MIXED_IDS;
    }
}

const DUPLICATE_POLICIES: readonly DuplicatePolicy[] = ["keep", "error", "first", "last", "sum", "min", "max"];

/** Weights that exercise every flag and the f64 shadow rule. */
const WEIGHTS: readonly number[] = [1, 0, -1, 2.5, 0.1, 3, Infinity, -Infinity, 16777217, 1e-9];

/** Attribute values that exercise inference and widening (null and undefined are unset). */
const ATTRIBUTE_VALUES: readonly unknown[] = [true, false, 1, -7, 2.5, "x", "y", null, undefined, [1, 2], { k: 1 }];

/** The scenario configuration arbitrary: every policy enum is a dimension. */
const arbConfig: fc.Arbitrary<OpsConfig> = fc.record({
    directed: fc.boolean(),
    weighted: fc.constantFrom<boolean | "auto">("auto", true, false),
    weightDtype: fc.constantFrom<"f32" | "f64">("f32", "f64"),
    duplicateEdges: fc.constantFrom(...DUPLICATE_POLICIES),
    selfLoops: fc.constantFrom<"keep" | "drop" | "error">("keep", "drop", "error"),
    addMissingNodes: fc.boolean(),
    idKind: fc.constantFrom<OpsConfig["idKind"]>("numbers", "strings", "mixed"),
});

/**
 * An operation arbitrary for a configuration (ids from its pool).
 */
function arbOp(config: OpsConfig): fc.Arbitrary<Op> {
    const ids = idPool(config.idKind);
    const arbId = fc.constantFrom(...ids);
    const arbWeight = fc.constantFrom(...WEIGHTS);
    const arbEdge = fc.nat({ max: 24 });
    const arbColumn = fc.constantFrom("p", "q");
    const arbValue = fc.constantFrom(...ATTRIBUTE_VALUES);
    return fc.oneof(
        { weight: 3, arbitrary: fc.record({ type: fc.constant("addNode" as const), id: arbId }) },
        {
            weight: 8,
            arbitrary: fc.record({
                type: fc.constant("addEdge" as const),
                source: arbId,
                target: arbId,
                weight: fc.option(arbWeight, { nil: undefined }),
            }),
        },
        { weight: 1, arbitrary: fc.record({ type: fc.constant("removeNode" as const), id: arbId }) },
        { weight: 2, arbitrary: fc.record({ type: fc.constant("removeEdge" as const), edge: arbEdge }) },
        {
            weight: 1,
            arbitrary: fc.record({ type: fc.constant("setEdgeWeight" as const), edge: arbEdge, weight: arbWeight }),
        },
        {
            weight: 2,
            arbitrary: fc.record({
                type: fc.constant("setNodeValue" as const),
                id: arbId,
                column: arbColumn,
                value: arbValue,
            }),
        },
        {
            weight: 2,
            arbitrary: fc.record({
                type: fc.constant("setEdgeValue" as const),
                edge: arbEdge,
                column: arbColumn,
                value: arbValue,
            }),
        },
        {
            weight: 2,
            arbitrary: fc.record({
                type: fc.constant("freeze" as const),
                duplicateEdges: fc.option(fc.constantFrom(...DUPLICATE_POLICIES), { nil: undefined }),
                arena: fc.boolean(),
            }),
        },
    );
}

/** A scenario: a configuration plus a sequence of operations that ends with a freeze. */
export interface Scenario {
    readonly config: OpsConfig;
    readonly ops: readonly Op[];
}

/** The scenario arbitrary. */
export const arbScenario: fc.Arbitrary<Scenario> = arbConfig.chain((config) =>
    fc
        .array(arbOp(config), { minLength: 1, maxLength: 40 })
        .map((ops) => ({ config, ops: [...ops, { type: "freeze" as const, duplicateEdges: undefined, arena: true }] })),
);
