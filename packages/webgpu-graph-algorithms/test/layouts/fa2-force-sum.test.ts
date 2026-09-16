/**
 * The force-sum invariant (spec 11.4; contract 5.5 fa2-force-sum.test.ts): with gravity 0 and distributedAction
 * false, after one iteration |sum_i F_i| <= tol x sum_i |F_i| on every fixture including the coincident one (Newton's
 * third law over the doubled arcs and the antisymmetric pair force, with the antisymmetric unit kick of coincident
 * pairs). The tolerance fa2-force-sum is traced to the force-parity basis row (PLAN DECISION 5). A centroid-drift test
 * is NOT used (per-node speed factors make displacements non-antisymmetric even when forces are).
 */

import type { F32, GraphSnapshot } from "@graphty/graph-format";

import type { GpuContext } from "../../src/context.js";
import type { GpuLayoutTuning } from "../../src/types/layout.js";
import type { ForceAtlas2Options } from "../../src/types/options.js";
import {
    asF32,
    BASE_OPTIONS,
    debugStages,
    NETWORKX,
    PAPER,
    type ParityGraph,
    paritySnapshot,
    startPositions,
    toleranceOf,
    withSim,
} from "../helpers/fa2-parity.js";
import { fixture } from "../helpers/graphs.js";
import { expectBitwiseEqual } from "../helpers/matchers.js";
import { assertCheckPasses, type CheckReport, mergeReports, ratioOf } from "../helpers/sabotage.js";
import { acquire, gpuScale, requireGpu } from "../setup/gpu.js";

const CASE_TIMEOUT = 300_000;
const GRAVITY_FREE: ForceAtlas2Options = { ...BASE_OPTIONS, gravity: 0, distributedAction: false };

interface SumCase {
    readonly name: string;
    readonly snapshot: () => GraphSnapshot;
    /** Supplied positions (the coincident fixture) or null for the seeded start. */
    readonly positions: () => F32 | null;
    readonly options: ForceAtlas2Options;
    readonly tuning: GpuLayoutTuning;
}

function parity(
    graph: ParityGraph,
    weighted: boolean,
    options: ForceAtlas2Options = GRAVITY_FREE,
    tuning: GpuLayoutTuning = PAPER,
): SumCase {
    return {
        name: `${graph}${weighted ? "-w" : ""}${options.linlog === true ? "-linlog" : ""}/${tuning.compat ?? "paper"}/${options.dim ?? 2}d`,
        snapshot: () => paritySnapshot(graph, gpuScale(), weighted),
        positions: () => null,
        options,
        tuning,
    };
}

function named(name: string, options: ForceAtlas2Options = GRAVITY_FREE): SumCase {
    return {
        name: `fixture:${name}`,
        snapshot: () => fixture(name, gpuScale()).snapshot,
        positions: () => fixture(name, gpuScale()).positions,
        options,
        tuning: PAPER,
    };
}

const CASES: readonly SumCase[] = [
    parity("karate", false),
    parity("karate", true),
    parity("grid10", false),
    parity("star200", false),
    parity("random1k", false),
    parity("random1k", true),
    parity("path10", false),
    parity("complete6", false),
    parity("karate", false, { ...GRAVITY_FREE, dim: 3 }),
    parity("karate", false, { ...GRAVITY_FREE, linlog: true }),
    parity("karate", false, GRAVITY_FREE, NETWORKX),
    named("coincident"),
    named("self-loop"),
    named("parallel", { ...GRAVITY_FREE, weight: true }),
    // spec 11.4 "on every fixture": the remaining named fixtures of contract 5.2 with at least two nodes, all sized
    // by gpuScale() through fixture(); "empty" and "one" are excluded because their total force is exactly 0
    // (netForceRatio would report Infinity / "some force exists" would fail, and the invariant is vacuous there)
    named("isolated"),
    named("path1k"),
    named("hub10k"),
];

/**
 * |sum_i F_i| / sum_i |F_i| of a stride-3 force array (Infinity when no node has a force).
 */
function netForceRatio(force: F32, n: number): { readonly ratio: number; readonly total: number } {
    let sx = 0;
    let sy = 0;
    let sz = 0;
    let total = 0;
    for (let i = 0; i < n; i++) {
        sx += force[3 * i];
        sy += force[3 * i + 1];
        sz += force[3 * i + 2];
        total += Math.hypot(force[3 * i], force[3 * i + 1], force[3 * i + 2]);
    }
    return { ratio: total === 0 ? Number.POSITIVE_INFINITY : Math.hypot(sx, sy, sz) / total, total };
}

describe("FA2 force-sum invariant: gravity 0, one iteration, |sum F| <= tol x sum |F| (spec 11.4)", () => {
    let ctx: GpuContext;
    const reports: CheckReport[] = [];

    beforeAll(async () => {
        ctx = await acquire({ label: "fa2-force-sum" });
    });

    for (const c of CASES) {
        it(
            `${c.name}: the net force vanishes within the traced tolerance, twice bitwise`,
            async (t) => {
                requireGpu(t);
                const s = c.snapshot();
                try {
                    const supplied = c.positions();
                    const start = supplied ?? startPositions(s, c.options, false);
                    const runOnce = (): Promise<F32> =>
                        withSim(ctx, c.options, c.tuning, async (sim) => {
                            sim.load(s, Float32Array.from(start));
                            const st = debugStages(sim);
                            await st.run("K3");
                            return asF32(await st.read("force"));
                        });
                    const a = await runOnce();
                    const b = await runOnce();
                    expectBitwiseEqual(a, b, `${c.name}: run 1 vs run 2`);
                    const { ratio, total } = netForceRatio(a, s.nodeCount);
                    expect(total, `${c.name}: some force exists`).toBeGreaterThan(0);
                    const report: CheckReport = {
                        worst: ratioOf(ratio, toleranceOf("fa2-force-sum")),
                        worstLabel: c.name,
                        samples: s.nodeCount,
                    };
                    reports.push(report);
                    console.warn(
                        `[fa2-force-sum] ${c.name}: |sum F| / sum |F| = ${ratio.toExponential(3)} (ratio ${report.worst.toExponential(3)})`,
                    );
                    assertCheckPasses(report);
                } finally {
                    ctx.release(s);
                }
            },
            CASE_TIMEOUT,
        );
    }

    it("prints the worst ratio of the fixture set", () => {
        const worst = mergeReports(reports);
        console.warn(`[fa2-force-sum] worst ${worst.worst.toExponential(3)} at ${worst.worstLabel}`);
    });
});
