/**
 * Model-based property tests of the builder (design section 16.1, P1-P3, P5-P7): random operation
 * sequences are applied to a GraphBuilder and to the Map-of-Maps model of test/helpers/model-graph.ts;
 * after every freeze the snapshot, the report and the builder's renumbered state must equal the
 * model's, and the snapshot must pass I1-I18 (test/helpers/invariants.ts) and validate("full").
 */

import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { INVALID_INDEX } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import { type FreezeReport, type GraphSnapshot } from "../../src/types/index.js";
import { assertInvariants } from "../helpers/invariants.js";
import { ModelError, ModelGraph, type ModelSnapshot } from "../helpers/model-graph.js";
import { arbScenario, type Op, type OpsConfig, type Scenario } from "../helpers/random-ops.js";

/** Property runs per test; FC_RUNS raises it for a nightly-style soak. */
const RUNS = Number(process.env.FC_RUNS ?? "200");

/** Run a builder call and a model call; both must throw the same code or neither may throw. */
function both<A, B>(actual: () => A, expected: () => B): { actual: A; expected: B } | null {
    let actualError: GraphFormatError | null = null;
    let actualValue: A | undefined;
    try {
        actualValue = actual();
    } catch (err) {
        if (!(err instanceof GraphFormatError)) {
            throw err;
        }
        actualError = err;
    }
    let expectedError: ModelError | null = null;
    let expectedValue: B | undefined;
    try {
        expectedValue = expected();
    } catch (err) {
        if (!(err instanceof ModelError)) {
            throw err;
        }
        expectedError = err;
    }
    if (expectedError !== null || actualError !== null) {
        expect(actualError?.code ?? "no error").toBe(expectedError?.code ?? "no error");
        return null;
    }
    return { actual: actualValue as A, expected: expectedValue as B };
}

function compareSnapshot(
    snapshot: GraphSnapshot,
    report: FreezeReport,
    expected: ModelSnapshot,
    builder: GraphBuilder,
    model: ModelGraph,
): void {
    assertInvariants(snapshot);
    expect(snapshot.directed).toBe(expected.directed);
    expect(snapshot.nodeCount).toBe(expected.nodeIds.length);
    expect(snapshot.edgeCount).toBe(expected.edges.length);
    expect(snapshot.ids.toArray()).toEqual(expected.nodeIds);
    expect(snapshot.selfLoopCount).toBe(expected.selfLoopCount);
    expect(snapshot.flags.multigraph).toBe(expected.multigraph);
    expect(snapshot.flags.hasSelfLoops).toBe(expected.selfLoopCount > 0);
    expect(snapshot.flags.weighted).toBe(expected.weighted);
    expect(snapshot.flags.arcToEdgeIsIdentity).toBe(expected.arcToEdgeIsIdentity);
    expect(snapshot.weights === null).toBe(!expected.weighted);
    // P2 / P5 / P6: edge order, orientation and weights
    expected.edges.forEach((edge, e) => {
        expect(snapshot.edgeSource(e)).toBe(edge.source);
        expect(snapshot.edgeTarget(e)).toBe(edge.target);
        if (snapshot.weights !== null) {
            expect(snapshot.weights[snapshot.edgeToArc[e]]).toBe(edge.weight);
        }
    });
    // I4 and the pairing: every row equals the comparator-sorted model row
    for (let u = 0; u < snapshot.nodeCount; u++) {
        const row: [number, number][] = [];
        for (let a = snapshot.rowPtr[u]; a < snapshot.rowPtr[u + 1]; a++) {
            row.push([snapshot.colIdx[a], snapshot.arcToEdge[a]]);
        }
        expect(row).toEqual(expected.rows[u]);
    }
    // P3: the report
    expect(report.nodeRemap === null ? null : Array.from(report.nodeRemap)).toEqual(expected.nodeRemap);
    expect(report.edgeRemap === null ? null : Array.from(report.edgeRemap)).toEqual(expected.edgeRemap);
    expect(report.compacted).toBe(expected.compacted);
    expect(report.droppedSelfLoops).toBe(expected.droppedSelfLoops);
    expect(report.mergedEdges).toBe(expected.mergedEdges);
    expect(report.droppedEdges).toBe(expected.droppedEdges);
    // the weight shadow column (design section 3.7)
    const shadow = snapshot.edges.byRole("weight");
    if (expected.shadow === null) {
        expect(shadow).toBeNull();
    } else {
        expect(shadow).not.toBeNull();
        const column = shadow as NonNullable<typeof shadow>;
        expect(column.meta.name).toBe("graphty.weight");
        expect(column.dtype).toBe(expected.shadow.dtype);
        if (column.dtype === "f32" || column.dtype === "f64") {
            expect(Array.from(column.data)).toEqual(expected.shadow.values);
        }
        if (expected.shadow.validity === null) {
            expect(column.validity).toBeNull();
        } else {
            expected.shadow.validity.forEach((set, e) => {
                expect(column.isSet(e)).toBe(set);
            });
        }
    }
    // attribute columns: names in first-seen order, dtype, values coerced once to the final dtype
    const nodeNames = snapshot.nodes.names();
    expect(nodeNames).toEqual([...expected.nodeColumns.keys()]);
    for (const [name, column] of expected.nodeColumns) {
        const actual = snapshot.nodes.require(name);
        expect(actual.dtype).toBe(column.dtype);
        column.values.forEach((value, row) => {
            expect(actual.isSet(row)).toBe(value !== undefined);
            expect(actual.value(row)).toEqual(value);
        });
    }
    const edgeNames = snapshot.edges.names().filter((name) => name !== "graphty.weight");
    expect(edgeNames).toEqual([...expected.edgeColumns.keys()]);
    for (const [name, column] of expected.edgeColumns) {
        const actual = snapshot.edges.require(name);
        expect(actual.dtype).toBe(column.dtype);
        column.values.forEach((value, row) => {
            expect(actual.isSet(row)).toBe(value !== undefined);
            expect(actual.value(row)).toEqual(value);
        });
    }
    // C7: the builder's indices equal the snapshot's after the freeze
    expect(builder.dirty).toBe(false);
    expect(builder.nodeBound).toBe(snapshot.nodeCount);
    expect(builder.edgeBound).toBe(snapshot.edgeCount);
    expect(builder.nodeCount).toBe(snapshot.nodeCount);
    expect(builder.edgeCount).toBe(snapshot.edgeCount);
    for (let i = 0; i < snapshot.nodeCount; i++) {
        expect(builder.idOf(i)).toBe(snapshot.ids.idOf(i));
        expect(builder.indexOf(snapshot.ids.idOf(i))).toBe(i);
    }
    for (let e = 0; e < snapshot.edgeCount; e++) {
        expect(builder.edgeEndpoints(e)).toEqual([expected.edges[e].source, expected.edges[e].target]);
        expect(builder.edgeWeight(e)).toBe(model.edges[e].weight);
    }
}

function builderOptions(config: OpsConfig): ConstructorParameters<typeof GraphBuilder>[0] {
    return {
        directed: config.directed,
        weighted: config.weighted,
        weightDtype: config.weightDtype,
        duplicateEdges: config.duplicateEdges,
        selfLoops: config.selfLoops,
        addMissingNodes: config.addMissingNodes,
    };
}

/** Apply one operation to both sides. */
function applyOp(op: Op, builder: GraphBuilder, model: ModelGraph, snapshots: GraphSnapshot[]): void {
    switch (op.type) {
        case "addNode": {
            expect(builder.addNode(op.id)).toBe(model.addNode(op.id));
            return;
        }
        case "addEdge": {
            const result = both(
                () => builder.addEdge(op.source, op.target, op.weight),
                () => model.addEdge(op.source, op.target, op.weight),
            );
            if (result !== null) {
                expect(result.actual).toBe(result.expected);
            }
            return;
        }
        case "removeNode": {
            const result = both(
                () => Array.from(builder.removeNode(op.id)),
                () => model.removeNode(op.id).sort((a, b) => a - b),
            );
            if (result !== null) {
                expect(result.actual).toEqual(result.expected);
            }
            return;
        }
        case "removeEdge": {
            const edge = op.edge % (builder.edgeBound + 1);
            expect(builder.removeEdge(edge)).toBe(model.removeEdge(edge));
            return;
        }
        case "setEdgeWeight": {
            const edge = op.edge % (builder.edgeBound + 1);
            both(
                () => {
                    builder.setEdgeWeight(edge, op.weight);
                },
                () => {
                    model.setEdgeWeight(edge, op.weight);
                },
            );
            return;
        }
        case "setNodeValue": {
            const index = builder.indexOf(op.id);
            expect(index).toBe(model.lookup(op.id));
            if (index === INVALID_INDEX) {
                return;
            }
            builder.setNodeValue(op.column, index, op.value);
            model.setNodeValue(index, op.column, op.value);
            return;
        }
        case "setEdgeValue": {
            const edge = op.edge % (builder.edgeBound + 1);
            if (edge >= builder.edgeBound) {
                expect(() => {
                    builder.setEdgeValue(op.column, edge, op.value);
                }).toThrow(expect.objectContaining({ code: "E_INDEX_RANGE" }));
                return;
            }
            builder.setEdgeValue(op.column, edge, op.value);
            model.setEdgeValue(edge, op.column, op.value);
            return;
        }
        case "freeze": {
            const mutationsBefore = builder.mutationCount;
            const nodeBoundBefore = builder.nodeBound;
            const edgeBoundBefore = builder.edgeBound;
            const result = both(
                () => builder.freezeWithReport({ duplicateEdges: op.duplicateEdges, arena: op.arena }),
                () => model.freeze(op.duplicateEdges),
            );
            if (result === null) {
                // a refused freeze leaves the builder untouched (design section 11.1)
                expect(builder.mutationCount).toBe(mutationsBefore);
                expect(builder.nodeBound).toBe(nodeBoundBefore);
                expect(builder.edgeBound).toBe(edgeBoundBefore);
                return;
            }
            expect(builder.mutationCount).toBe(mutationsBefore);
            const { snapshot, report } = result.actual;
            compareSnapshot(snapshot, report, result.expected, builder, model);
            expect(snapshot.arena === null).toBe(!op.arena);
            snapshots.push(snapshot);
            return;
        }
        default: {
            const unknown: never = op;
            throw new Error(`unknown op ${JSON.stringify(unknown)}`);
        }
    }
}

/** Run a whole scenario; returns every snapshot frozen along the way. */
export function runScenario(scenario: Scenario): GraphSnapshot[] {
    const builder = new GraphBuilder(builderOptions(scenario.config));
    const model = new ModelGraph(scenario.config);
    const snapshots: GraphSnapshot[] = [];
    for (const op of scenario.ops) {
        applyOp(op, builder, model, snapshots);
    }
    // I17 / I18: earlier snapshots are unchanged by everything that happened after them
    for (const snapshot of snapshots) {
        snapshot.validate({ level: "full" });
    }
    return snapshots;
}

describe("GraphBuilder against the Map-of-Maps model", () => {
    it("agrees with the model on random operation sequences (P1-P3, P5-P7)", () => {
        fc.assert(
            fc.property(arbScenario, (scenario) => {
                runScenario(scenario);
            }),
            { numRuns: RUNS },
        );
    });

    it("keeps earlier snapshots intact while the builder keeps mutating (I17, I18, shared id map guard)", () => {
        fc.assert(
            fc.property(arbScenario, (scenario) => {
                const builder = new GraphBuilder(builderOptions(scenario.config));
                const model = new ModelGraph(scenario.config);
                const snapshots: GraphSnapshot[] = [];
                const digests: string[] = [];
                for (const op of scenario.ops) {
                    applyOp(op, builder, model, snapshots);
                    if (op.type === "freeze" && snapshots.length > digests.length) {
                        const latest = snapshots[snapshots.length - 1];
                        digests.push(latest.contentHash());
                    }
                }
                snapshots.forEach((snapshot, i) => {
                    expect(snapshot.contentHash()).toBe(digests[i]);
                    // an id added to the builder after the freeze is invisible to the snapshot
                    builder.addNode("added-later");
                    expect(snapshot.ids.indexOf("added-later")).toBe(INVALID_INDEX);
                    expect(snapshot.ids.size).toBe(snapshot.nodeCount);
                });
            }),
            { numRuns: Math.ceil(RUNS / 4) },
        );
    });

    it("is deterministic: the same scenario twice yields byte-identical cores (I15, P11)", () => {
        fc.assert(
            fc.property(arbScenario, (scenario) => {
                const first = runScenario(scenario);
                const second = runScenario(scenario);
                expect(second.length).toBe(first.length);
                first.forEach((snapshot, i) => {
                    expect(second[i].contentHash()).toBe(snapshot.contentHash());
                    expect(Array.from(second[i].rowPtr)).toEqual(Array.from(snapshot.rowPtr));
                    expect(Array.from(second[i].colIdx)).toEqual(Array.from(snapshot.colIdx));
                    expect(second[i].flags).toEqual(snapshot.flags);
                    expect(second[i].ids.toArray()).toEqual(snapshot.ids.toArray());
                });
            }),
            { numRuns: Math.ceil(RUNS / 5) },
        );
    });

    it("a refused freeze leaves the builder and the model untouched (11.1; fast-check counterexample)", () => {
        // A self-loop dropped by the first freeze must reappear in the count of the second one when
        // the first freeze was refused for a duplicate edge: the builder keeps its state (11.1) and
        // the model must do the same, or the two disagree on droppedSelfLoops.
        const scenario: Scenario = {
            config: {
                directed: true,
                weighted: true,
                weightDtype: "f32",
                duplicateEdges: "sum",
                selfLoops: "drop",
                addMissingNodes: true,
                idKind: "strings",
            },
            ops: [
                { type: "addEdge", source: "h", target: "a", weight: undefined },
                { type: "addEdge", source: "h", target: "a", weight: undefined },
                { type: "addEdge", source: "e", target: "e", weight: undefined },
                { type: "freeze", duplicateEdges: "error", arena: false },
                { type: "setEdgeValue", edge: 0, column: "p", value: true },
                { type: "freeze", duplicateEdges: undefined, arena: true },
            ],
        };
        const snapshots = runScenario(scenario);
        expect(snapshots.length).toBe(1);
        expect(snapshots[0].edgeCount).toBe(1);
        expect(snapshots[0].selfLoopCount).toBe(0);
    });
});
