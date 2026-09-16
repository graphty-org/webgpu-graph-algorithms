/**
 * Oracle independence (spec 11.4 bullet 2, 7.2 "cross-checks the oracle"; contract 5.4 / 5.5 P3-T4): the f64
 * ForceAtlas2 oracle in compat "networkx" (origin gravity, position-mixed swing, accumulated sums) reproduces the
 * committed NetworkX 3.4.2 trajectories of test/fixtures/networkx/ -- karate, a 10 x 10 grid, a star of 200, a
 * seeded G(200, 600) (and a 3D karate) x base / linlog / distributed / strong / weighted / gravity0 x the iteration
 * counts 1, 5 and one long count -- within 1e-9 at 1 and 5 iterations and 1e-6 on the long leg, raw positions
 * (NetworkX does not rescale; contract 5.4). Because NetworkX shares every FORCE law with compat "paper", the
 * paper-mode forces at iteration 0 are asserted equal to the networkx run's on every fixture, and the fixtures'
 * preconditions (no pair under the 0.01 floor, no node under 0.01 of the origin while regular gravity acts, i.e.
 * gravity != 0 and not strongGravity -- K3's d > FA2_DIST_FLOOR guard exists only on that branch) are re-verified
 * along the oracle's own trajectory. The generator never runs here; its output format is pinned by the
 * hand-computed path3 fixture embedded below (PATH3_HAND), which the committed path3-base-iter1.json must equal.
 *
 * The long leg's count is per fixture (50, 40, 30, 20 or 10): NetworkX's adaptive-speed controller is chaotic on
 * some graph / variant pairs (its own run from a 1-ulp perturbed start diverges by O(1) at 50 iterations on
 * karate-linlog, star200-distributed, gnm200-distributed, ...), so the generator records NetworkX's 1-ulp
 * sensitivity in every file and lowers the long count until it is <= 1e-8; the test asserts that recorded margin
 * (PLAN DECISION P3-T4 re-fixing spec 11.4's "1e-6 at 50", recorded for the owner in docs/decisions/G3.md).
 */

import { readdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { FA2_DISTANCE_FLOOR } from "../../src/constants.js";
import { type EdgeSpec, snapshotOf } from "../helpers/graphs.js";
import { maxRelError } from "../helpers/matchers.js";
import { ForceAtlas2Oracle, forceAtlas2Oracle, type OracleOptions, seededScenePositions } from "./forceatlas2.js";

const FIXTURE_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "../fixtures/networkx");
const GRAPHS = ["karate", "grid10", "star200", "gnm200"] as const;
const VARIANTS = ["base", "linlog", "distributed", "strong", "weighted", "gravity0"] as const;
const LONG_LADDER = [50, 40, 30, 20, 10] as const;
const SHORT_TOLERANCE = 1e-9;
const LONG_TOLERANCE = 1e-6;
const SENSITIVITY_MAX = 1e-8;
const HAND_TOLERANCE = 1e-12;

/** The 5.4 fixture document (plus the generator's recorded 1-ulp sensitivity). The snake_case keys are NetworkX's option names. */
interface NetworkxFixtureGraph {
    readonly name: string;
    readonly directed: boolean;
    readonly nodeCount: number;
    readonly src: readonly number[];
    readonly dst: readonly number[];
    readonly weights: readonly number[] | null;
}

interface NetworkxFixtureOptions {
    readonly max_iter: number;
    readonly jitter_tolerance: number;
    readonly scaling_ratio: number;
    readonly gravity: number;
    readonly distributed_action: boolean;
    readonly strong_gravity: boolean;
    readonly linlog: boolean;
    readonly dissuade_hubs: boolean;
    readonly weight: string | null;
    readonly dim: 2 | 3;
}

interface NetworkxFixture {
    readonly generator: string;
    readonly networkxVersion: string;
    readonly pythonVersion: string;
    readonly command: string;
    readonly sensitivity: number;
    readonly graph: NetworkxFixtureGraph;
    readonly options: NetworkxFixtureOptions;
    readonly initialPositions: readonly (readonly number[])[];
    readonly positions: readonly (readonly number[])[];
    readonly rescaled: boolean;
}

const TOP_KEYS = [
    "command",
    "generator",
    "graph",
    "initialPositions",
    "networkxVersion",
    "options",
    "positions",
    "pythonVersion",
    "rescaled",
    "sensitivity",
];
const GRAPH_KEYS = ["directed", "dst", "name", "nodeCount", "src", "weights"];
const OPTION_KEYS = [
    "dim",
    "dissuade_hubs",
    "distributed_action",
    "gravity",
    "jitter_tolerance",
    "linlog",
    "max_iter",
    "scaling_ratio",
    "strong_gravity",
    "weight",
];

/**
 * The hand-computed fixture: the three-node path 0-1-2 at (-1, 0), (0, 1), (1, 0), one networkx-mode iteration
 * (masses 2, 3, 2; attraction (1, 1) (0, -2) (-1, 1); repulsion (-10, -6) (0, 12) (10, -6); origin gravity (2, 0)
 * (0, -3) (-2, 0); F = (-7, -5) (0, 7) (7, -5); swing = 19 + 4 sqrt(61), traction = 13 + 2 sqrt(89); speed
 * 0.19450843827825515; factors 0.06874716307020648 / 0.06438420254527687 / 0.06874716307020648). The derivation is
 * test/oracle/swing-mode.test.ts's networkx-mode case; the three environment strings are the ones this repository
 * recorded and are compared by pattern, every other field exactly (numbers within 1e-12).
 */
const PATH3_HAND = `{
    "generator": "test/fixtures/networkx/generate.py",
    "networkxVersion": "3.4.2",
    "pythonVersion": "3.10.12",
    "command": "python generate.py --out test/fixtures/networkx/ --iters 1 5 50 --seed 7",
    "sensitivity": 1.5e-16,
    "graph": {"name": "path3", "directed": false, "nodeCount": 3, "src": [0, 1], "dst": [1, 2], "weights": null},
    "options": {"max_iter": 1, "jitter_tolerance": 1.0, "scaling_ratio": 2.0, "gravity": 1.0, "distributed_action": false, "strong_gravity": false, "linlog": false, "dissuade_hubs": false, "weight": null, "dim": 2},
    "initialPositions": [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
    "positions": [[-1.4812301414914453, -0.34373581535103237], [0.0, 1.450689417816938], [1.4812301414914453, -0.34373581535103237]],
    "rescaled": false
}
`;

function isRecord(x: unknown): x is Record<string, unknown> {
    return typeof x === "object" && x !== null && !Array.isArray(x);
}

function isFiniteRows(x: unknown, rows: number, dim: number): x is readonly (readonly number[])[] {
    return (
        Array.isArray(x) &&
        x.length === rows &&
        x.every(
            (row: unknown) =>
                Array.isArray(row) &&
                row.length === dim &&
                row.every((v: unknown) => typeof v === "number" && Number.isFinite(v)),
        )
    );
}

/** The variant a fixture's options encode (exactly one deviation from the base options, or none). */
function variantOf(o: NetworkxFixtureOptions): string {
    const flags = [o.linlog, o.distributed_action, o.strong_gravity, o.weight !== null, o.gravity === 0];
    expect(flags.filter((f) => f).length, "at most one variant flag").toBeLessThanOrEqual(1);
    if (o.linlog) {
        return "linlog";
    }
    if (o.distributed_action) {
        return "distributed";
    }
    if (o.strong_gravity) {
        return "strong";
    }
    if (o.weight !== null) {
        return "weighted";
    }
    if (o.gravity === 0) {
        return "gravity0";
    }
    return "base";
}

/** Validates the 5.4 schema of one parsed document and returns it typed; `label` names it in failures. */
function validateFixture(doc: unknown, label: string): NetworkxFixture {
    expect(isRecord(doc), `${label}: an object`).toBe(true);
    const d = doc as Record<string, unknown>;
    expect(Object.keys(d).sort(), `${label}: top-level keys`).toEqual(TOP_KEYS);
    expect(d.generator, `${label}: generator`).toBe("test/fixtures/networkx/generate.py");
    const version = /^(\d+)\.(\d+)/.exec(String(d.networkxVersion));
    expect(version, `${label}: networkxVersion`).not.toBeNull();
    const major = Number(version?.[1]);
    const minor = Number(version?.[2]);
    expect(
        major > 3 || (major === 3 && minor >= 4),
        `${label}: networkx >= 3.4 (found ${String(d.networkxVersion)})`,
    ).toBe(true);
    expect(String(d.pythonVersion), `${label}: pythonVersion`).toMatch(/^3\.\d+/);
    expect(String(d.command), `${label}: command`).toMatch(/^python generate\.py --out \S+/);
    expect(
        typeof d.sensitivity === "number" && Number.isFinite(d.sensitivity) && d.sensitivity >= 0,
        `${label}: sensitivity`,
    ).toBe(true);
    expect(d.rescaled, `${label}: rescaled`).toBe(false);
    expect(isRecord(d.graph), `${label}: graph`).toBe(true);
    const g = d.graph as Record<string, unknown>;
    expect(Object.keys(g).sort(), `${label}: graph keys`).toEqual(GRAPH_KEYS);
    expect(typeof g.name, `${label}: graph.name`).toBe("string");
    expect(g.directed, `${label}: graph.directed`).toBe(false);
    const n = g.nodeCount;
    expect(Number.isInteger(n) && (n as number) >= 1, `${label}: graph.nodeCount`).toBe(true);
    const { src, dst } = g;
    expect(Array.isArray(src) && Array.isArray(dst) && src.length === dst.length, `${label}: src / dst`).toBe(true);
    const pairs = new Set<string>();
    for (let e = 0; e < (src as unknown[]).length; e++) {
        const u = (src as unknown[])[e];
        const v = (dst as unknown[])[e];
        expect(Number.isInteger(u) && Number.isInteger(v), `${label}: edge ${e} integer`).toBe(true);
        expect(
            (u as number) >= 0 && (u as number) < (n as number) && (v as number) >= 0 && (v as number) < (n as number),
            `${label}: edge ${e} in range`,
        ).toBe(true);
        expect(u, `${label}: edge ${e} is a self-loop`).not.toBe(v);
        const key = `${Math.min(u as number, v as number)}-${Math.max(u as number, v as number)}`;
        expect(pairs.has(key), `${label}: edge ${e} is a parallel edge`).toBe(false);
        pairs.add(key);
    }
    if (g.weights !== null) {
        expect(
            Array.isArray(g.weights) && g.weights.length === (src as unknown[]).length,
            `${label}: weights length`,
        ).toBe(true);
        for (const w of g.weights as unknown[]) {
            expect(
                typeof w === "number" && w > 0 && Number.isFinite(w) && Math.fround(w) === w,
                `${label}: weight ${String(w)} positive and f32-exact`,
            ).toBe(true);
        }
    }
    expect(isRecord(d.options), `${label}: options`).toBe(true);
    const o = d.options as Record<string, unknown>;
    expect(Object.keys(o).sort(), `${label}: option keys`).toEqual(OPTION_KEYS);
    expect(Number.isInteger(o.max_iter) && (o.max_iter as number) >= 1, `${label}: max_iter`).toBe(true);
    expect(o.jitter_tolerance, `${label}: jitter_tolerance`).toBe(1);
    expect(o.scaling_ratio, `${label}: scaling_ratio`).toBe(2);
    expect(o.gravity === 1 || o.gravity === 0, `${label}: gravity`).toBe(true);
    for (const key of ["distributed_action", "strong_gravity", "linlog"]) {
        expect(typeof o[key], `${label}: ${key}`).toBe("boolean");
    }
    expect(o.dissuade_hubs, `${label}: dissuade_hubs`).toBe(false);
    expect(o.weight === null || o.weight === "weight", `${label}: weight`).toBe(true);
    expect(o.weight !== null, `${label}: weight iff graph.weights`).toBe(g.weights !== null);
    expect(o.dim === 2 || o.dim === 3, `${label}: dim`).toBe(true);
    expect(isFiniteRows(d.initialPositions, n as number, o.dim as number), `${label}: initialPositions`).toBe(true);
    expect(isFiniteRows(d.positions, n as number, o.dim as number), `${label}: positions`).toBe(true);
    return d as unknown as NetworkxFixture;
}

function readFixture(file: string): NetworkxFixture {
    const text = readFileSync(resolve(FIXTURE_DIR, file), "utf8");
    const fixture = validateFixture(JSON.parse(text), file);
    expect(file, `${file}: name encodes graph, variant and count`).toBe(
        `${fixture.graph.name}-${variantOf(fixture.options)}-iter${fixture.options.max_iter}.json`,
    );
    return fixture;
}

function snapshotFrom(f: NetworkxFixture) {
    const edges: EdgeSpec[] = [];
    for (let e = 0; e < f.graph.src.length; e++) {
        edges.push(
            f.graph.weights === null
                ? [f.graph.src[e], f.graph.dst[e]]
                : [f.graph.src[e], f.graph.dst[e], f.graph.weights[e]],
        );
    }
    return snapshotOf(edges, { nodeCount: f.graph.nodeCount });
}

/** Stride-3 layout-unit positions from the fixture's rows (z = 0 in 2D). */
function layoutOf(rows: readonly (readonly number[])[], dim: number): Float64Array {
    const out = new Float64Array(3 * rows.length);
    for (let i = 0; i < rows.length; i++) {
        for (let a = 0; a < dim; a++) {
            out[3 * i + a] = rows[i][a];
        }
    }
    return out;
}

/** The first `dim` components of every node of a stride-3 array (what the fixture stores). */
function componentsOf(positions: ArrayLike<number>, n: number, dim: number): Float64Array {
    const out = new Float64Array(n * dim);
    for (let i = 0; i < n; i++) {
        for (let a = 0; a < dim; a++) {
            out[i * dim + a] = positions[3 * i + a];
        }
    }
    return out;
}

function oracleOptionsOf(f: NetworkxFixture, compat: "paper" | "networkx", gravityCenter?: 0 | 1): OracleOptions {
    return {
        compat,
        precision: "f64",
        gravityCenter,
        dim: f.options.dim,
        scalingRatio: f.options.scaling_ratio,
        gravity: f.options.gravity,
        jitterTolerance: f.options.jitter_tolerance,
        linlog: f.options.linlog,
        distributedAction: f.options.distributed_action,
        strongGravity: f.options.strong_gravity,
        weight: f.options.weight !== null,
    };
}

function minPairDistance(p: ArrayLike<number>, n: number): number {
    let best = Number.POSITIVE_INFINITY;
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            best = Math.min(
                best,
                Math.hypot(p[3 * i] - p[3 * j], p[3 * i + 1] - p[3 * j + 1], p[3 * i + 2] - p[3 * j + 2]),
            );
        }
    }
    return best;
}

function minOriginDistance(p: ArrayLike<number>, n: number): number {
    let best = Number.POSITIVE_INFINITY;
    for (let i = 0; i < n; i++) {
        best = Math.min(best, Math.hypot(p[3 * i], p[3 * i + 1], p[3 * i + 2]));
    }
    return best;
}

function toleranceOf(maxIter: number): number {
    return maxIter <= 5 ? SHORT_TOLERANCE : LONG_TOLERANCE;
}

const FILES = readdirSync(FIXTURE_DIR)
    .filter((name) => name.endsWith(".json"))
    .sort();

describe("the committed fixture set", () => {
    it("covers every graph x variant at 1, 5 and exactly one long count, plus path3 and karate3d", () => {
        const byPair = new Map<string, number[]>();
        for (const file of FILES) {
            const m = /^([a-z0-9]+)-([a-z0-9]+)-iter(\d+)\.json$/.exec(file);
            expect(m, `${file}: file name pattern`).not.toBeNull();
            const key = `${String(m?.[1])}-${String(m?.[2])}`;
            byPair.set(key, [...(byPair.get(key) ?? []), Number(m?.[3])]);
        }
        const expectedPairs = new Set<string>(["path3-base", "karate3d-base"]);
        for (const graph of GRAPHS) {
            for (const variant of VARIANTS) {
                expectedPairs.add(`${graph}-${variant}`);
            }
        }
        expect([...byPair.keys()].sort()).toEqual([...expectedPairs].sort());
        for (const [pair, counts] of byPair) {
            const sorted = [...counts].sort((a, b) => a - b);
            if (pair === "path3-base") {
                expect(sorted, pair).toEqual([1]);
                continue;
            }
            expect(sorted.length, `${pair}: three counts`).toBe(3);
            expect(sorted.slice(0, 2), `${pair}: the short counts`).toEqual([1, 5]);
            expect(
                LONG_LADDER.includes(sorted[2] as (typeof LONG_LADDER)[number]),
                `${pair}: long count ${sorted[2]} on the ladder`,
            ).toBe(true);
        }
        expect(FILES.length).toBe(76);
    });

    it("every file validates, names itself consistently and shares its graph's initial positions across variants", () => {
        const startByGraph = new Map<string, string>();
        for (const file of FILES) {
            const f = readFixture(file);
            const start = JSON.stringify(f.initialPositions);
            const previous = startByGraph.get(f.graph.name);
            if (previous === undefined) {
                startByGraph.set(f.graph.name, start);
            } else {
                expect(start, `${file}: the same initial positions as its graph's other fixtures`).toBe(previous);
            }
            if (f.options.max_iter > 5) {
                expect(f.options.max_iter, `${file}: long leg at least 10`).toBeGreaterThanOrEqual(10);
                expect(
                    f.sensitivity,
                    `${file}: NetworkX's own 1-ulp sensitivity under ${SENSITIVITY_MAX}`,
                ).toBeLessThanOrEqual(SENSITIVITY_MAX);
            }
        }
    });
});

describe("the hand-computed path3 fixture (the generator's format, self-contained)", () => {
    const hand = validateFixture(JSON.parse(PATH3_HAND), "PATH3_HAND");

    it("is what the generator wrote for path3-base-iter1.json", () => {
        const committed = readFixture("path3-base-iter1.json");
        expect(committed.graph).toEqual(hand.graph);
        expect(committed.options).toEqual(hand.options);
        expect(committed.initialPositions).toEqual(hand.initialPositions);
        expect(committed.rescaled).toBe(false);
        expect(
            maxRelError(committed.positions.flat(), hand.positions.flat(), 1),
            "positions vs the hand derivation",
        ).toBeLessThanOrEqual(HAND_TOLERANCE);
        expect(committed.sensitivity).toBeLessThanOrEqual(1e-14);
        expect(committed.networkxVersion).toMatch(/^3\.(?:[4-9]|\d{2,})|^[4-9]\./);
        expect(committed.pythonVersion).toMatch(/^3\.(?:1\d|[89])/);
        expect(committed.command).toBe("python generate.py --out test/fixtures/networkx/ --iters 1 5 50 --seed 7");
    });

    it("is reproduced by the oracle to 1e-12", () => {
        const s = snapshotFrom(hand);
        const oracle = new ForceAtlas2Oracle(s, layoutOf(hand.initialPositions, 2), oracleOptionsOf(hand, "networkx"));
        const record = oracle.step();
        expect(maxRelError(componentsOf(oracle.positions, 3, 2), hand.positions.flat(), 1)).toBeLessThanOrEqual(
            HAND_TOLERANCE,
        );
        expect(Math.abs(record.swing - (19 + 4 * Math.sqrt(61)))).toBeLessThanOrEqual(HAND_TOLERANCE * 100);
        expect(Math.abs(record.traction - (13 + 2 * Math.sqrt(89)))).toBeLessThanOrEqual(HAND_TOLERANCE * 100);
        expect(Math.abs(record.speed - 0.19450843827825515)).toBeLessThanOrEqual(HAND_TOLERANCE);
        expect(record.speedEfficiency).toBe(0.7);
    });
});

describe("oracle vs NetworkX (compat networkx, f64, raw positions)", () => {
    for (const file of FILES) {
        describe(file, () => {
            const f = readFixture(file);
            const { nodeCount: n } = f.graph;
            const { dim } = f.options;
            const tolerance = toleranceOf(f.options.max_iter);

            it(`reproduces NetworkX within ${tolerance} at ${f.options.max_iter} iterations`, () => {
                const oracle = new ForceAtlas2Oracle(
                    snapshotFrom(f),
                    layoutOf(f.initialPositions, dim),
                    oracleOptionsOf(f, "networkx"),
                );
                for (let k = 0; k < f.options.max_iter; k++) {
                    oracle.step();
                }
                expect(oracle.iteration).toBe(f.options.max_iter);
                expect(oracle.trace.length).toBe(f.options.max_iter);
                const error = maxRelError(componentsOf(oracle.positions, n, dim), f.positions.flat(), 1);
                expect(error, `${file}: max |p - p_nx| / max(|p_nx|, 1)`).toBeLessThanOrEqual(tolerance);
            });

            it("keeps every pair above the 0.01 floor and, under regular gravity, every node off the origin guard", () => {
                const oracle = new ForceAtlas2Oracle(
                    snapshotFrom(f),
                    layoutOf(f.initialPositions, dim),
                    oracleOptionsOf(f, "networkx"),
                );
                const guardOrigin = f.options.gravity !== 0 && !f.options.strong_gravity;
                for (let k = 0; k < f.options.max_iter; k++) {
                    expect(
                        minPairDistance(oracle.positions, n),
                        `${file}: min pair distance at iteration ${k}`,
                    ).toBeGreaterThan(FA2_DISTANCE_FLOOR);
                    if (guardOrigin) {
                        expect(
                            minOriginDistance(oracle.positions, n),
                            `${file}: min |p| at iteration ${k}`,
                        ).toBeGreaterThan(FA2_DISTANCE_FLOOR);
                    }
                    oracle.step();
                }
            });

            it("gives compat paper (origin gravity) the same iteration-0 forces as compat networkx", () => {
                const paper = new ForceAtlas2Oracle(
                    snapshotFrom(f),
                    layoutOf(f.initialPositions, dim),
                    oracleOptionsOf(f, "paper", 1),
                );
                const networkx = new ForceAtlas2Oracle(
                    snapshotFrom(f),
                    layoutOf(f.initialPositions, dim),
                    oracleOptionsOf(f, "networkx"),
                );
                paper.step();
                networkx.step();
                for (const stage of ["attraction", "repulsion", "gravity", "force"] as const) {
                    const a = paper.stages[stage];
                    const b = networkx.stages[stage];
                    expect(a.length).toBe(3 * n);
                    for (let i = 0; i < a.length; i++) {
                        expect(Object.is(a[i], b[i]), `${file}: ${stage}[${i}] ${a[i]} vs ${b[i]}`).toBe(true);
                    }
                }
                if (f.options.gravity === 0) {
                    // with no gravity the centre does not matter: the default paper mode (centroid) agrees too
                    const centroidPaper = new ForceAtlas2Oracle(
                        snapshotFrom(f),
                        layoutOf(f.initialPositions, dim),
                        oracleOptionsOf(f, "paper"),
                    );
                    centroidPaper.step();
                    for (let i = 0; i < 3 * n; i++) {
                        expect(
                            Object.is(centroidPaper.stages.force[i], networkx.stages.force[i]),
                            `${file}: gravity-0 force[${i}]`,
                        ).toBe(true);
                    }
                }
            });
        });
    }
});

describe("precision f32 (the tight-leg oracle)", () => {
    const f = readFixture("karate-base-iter5.json");
    const options = oracleOptionsOf(f, "paper");

    it("is deterministic, f32-representable and within 1e-3 of the f64 oracle after 5 iterations", () => {
        const a = new ForceAtlas2Oracle(snapshotFrom(f), layoutOf(f.initialPositions, 2), {
            ...options,
            precision: "f32",
        });
        const b = new ForceAtlas2Oracle(snapshotFrom(f), layoutOf(f.initialPositions, 2), {
            ...options,
            precision: "f32",
        });
        const c = new ForceAtlas2Oracle(snapshotFrom(f), layoutOf(f.initialPositions, 2), options);
        expect(a.positions).toBeInstanceOf(Float32Array);
        expect(c.positions).toBeInstanceOf(Float64Array);
        for (let k = 0; k < 5; k++) {
            a.step();
            b.step();
            c.step();
        }
        for (let i = 0; i < a.positions.length; i++) {
            expect(Object.is(a.positions[i], b.positions[i]), `f32 determinism at ${i}`).toBe(true);
            expect(Math.fround(a.positions[i]), `f32-representable at ${i}`).toBe(a.positions[i]);
        }
        expect(maxRelError(a.positions, c.positions, 1)).toBeLessThanOrEqual(1e-3);
        for (let k = 0; k < 5; k++) {
            for (const key of ["swing", "traction", "speed", "speedEfficiency"] as const) {
                const expected = c.trace[k][key];
                expect(
                    Math.abs(a.trace[k][key] - expected) / Math.abs(expected),
                    `trace ${key} at iteration ${k}`,
                ).toBeLessThanOrEqual(1e-3);
            }
        }
        for (const key of ["swing", "traction", "speed", "speedEfficiency"] as const) {
            expect(Math.fround(a.trace[4][key]), `f32-representable trace ${key}`).toBe(a.trace[4][key]);
        }
    });
});

describe("scene-unit wrappers", () => {
    const f = readFixture("karate-base-iter1.json");
    const s = snapshotFrom(f);

    it("seededScenePositions fills every row in scene units and forceAtlas2Oracle round-trips the units", () => {
        const scene = seededScenePositions(s, 42, 2, 100, [5, -5, 0]);
        expect(scene.length).toBe(3 * s.nodeCount);
        for (let i = 0; i < s.nodeCount; i++) {
            expect(Number.isFinite(scene[3 * i]) && Number.isFinite(scene[3 * i + 1])).toBe(true);
            expect(scene[3 * i]).toBeGreaterThanOrEqual(-95);
            expect(scene[3 * i]).toBeLessThan(105);
            expect(scene[3 * i + 2]).toBe(0);
        }
        const options: OracleOptions = { compat: "paper", precision: "f64", dim: 2, scale: 100, center: [5, -5, 0] };
        const run = forceAtlas2Oracle(s, scene, options, 3);
        const layout = new Float64Array(3 * s.nodeCount);
        for (let i = 0; i < s.nodeCount; i++) {
            layout[3 * i] = Math.fround((scene[3 * i] - 5) / 100);
            layout[3 * i + 1] = Math.fround((scene[3 * i + 1] + 5) / 100);
        }
        const direct = new ForceAtlas2Oracle(s, layout, options);
        direct.step();
        direct.step();
        direct.step();
        expect(run.trace.length).toBe(3);
        expect(run.oracle.iteration).toBe(3);
        expect(run.positions).toBeInstanceOf(Float32Array);
        for (let i = 0; i < s.nodeCount; i++) {
            expect(run.positions[3 * i]).toBe(Math.fround(direct.positions[3 * i] * 100 + 5));
            expect(run.positions[3 * i + 1]).toBe(Math.fround(direct.positions[3 * i + 1] * 100 - 5));
            expect(run.positions[3 * i + 2], "2D writes z = center.z").toBe(0);
        }
    });

    it("rejects a directed snapshot, a wrong length and a bad scale", () => {
        const directed = snapshotOf([[0, 1]], { nodeCount: 2, directed: true });
        expect(
            () => new ForceAtlas2Oracle(directed, new Float64Array(6), { compat: "paper", precision: "f64" }),
        ).toThrow(/undirected/);
        expect(() => new ForceAtlas2Oracle(s, new Float64Array(5), { compat: "paper", precision: "f64" })).toThrow(
            /positions\.length/,
        );
        expect(() =>
            forceAtlas2Oracle(s, new Float32Array(3 * s.nodeCount), { compat: "paper", precision: "f64", scale: 0 }, 1),
        ).toThrow(/scale/);
    });
});
