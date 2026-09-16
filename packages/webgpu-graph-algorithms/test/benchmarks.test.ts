/**
 * Pure tests of the benchmark code (contract 6.1-6.4, 6.8; spec 11.7, 10.4 T-13): the datasets are deterministic, the
 * harness times an async body and appends sessions under the runner class, and scripts/bench-compare.js applies the
 * quiet-GPU / new-result / 3x rules in order with the documented exit codes. No GPU is involved.
 */

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { gridEdges, KARATE_EDGES, randomEdges, rmatEdges, snapshotOf, TIERS } from "../benchmarks/datasets.js";
import {
    appendSession,
    bench,
    type BenchResult,
    type BenchSession,
    type GpuSessionInfo,
    makeRandom,
    runnerClass,
    setBenchRuns,
} from "../benchmarks/harness.js";
import {
    EXACT_BUDGET_MS,
    EXACT_LADDER,
    exactMaxNodesFromLadder,
    floorPow2,
    FRAME_RUNG,
    LADDER_EDGE_FACTOR,
    type LadderRow,
    ladderRowsOf,
    type LadderRung,
    LAYOUT_EXACT_GROUP,
} from "../benchmarks/layout-exact.bench.js";
import { checkLayoutResult, parseLayoutRunArgs } from "../benchmarks/layout-run.js";

/** A device whose queue settles immediately: bench() only awaits onSubmittedWorkDone on it. */
const FAKE_DEVICE = {
    queue: { onSubmittedWorkDone: (): Promise<undefined> => Promise.resolve(undefined) },
} as unknown as GPUDevice;

/** A session gpu record of a fictitious NVIDIA adapter. */
const GPU: GpuSessionInfo = {
    vendor: "nvidia",
    architecture: "ada-lovelace",
    device: "NVIDIA GeForce RTX 4070 SUPER",
    description: "NVIDIA 580.173.02 (test)",
    driver: "580.173.02",
    limits: {
        maxBufferSize: 268435456,
        maxStorageBufferBindingSize: 134217728,
        maxStorageBuffersPerShaderStage: 8,
        maxComputeWorkgroupsPerDimension: 65535,
    },
    software: false,
    runtime: "node",
    subgroupMaxSize: 32,
};

/** A result row. */
function result(name: string, medianMs: number, group = "roundtrip"): BenchResult {
    return {
        group,
        name,
        medianMs,
        minMs: medianMs,
        maxMs: medianMs,
        runs: 5,
        memoryDeltaBytes: 0,
        rate: null,
        rateUnit: null,
    };
}

/** A session holding the given results. */
function session(results: readonly BenchResult[], runner = "nvidia-ada-lovelace-driver580"): BenchSession {
    return {
        date: "2026-09-20T00:00:00.000Z",
        host: "test",
        node: "v22.22.1",
        cpu: "test",
        exposeGc: false,
        gpu: GPU,
        runnerClass: runner,
        results,
    };
}

/** A gpu-report.json document with the given nvidia-smi sample. */
function report(
    samples: readonly { utilizationGpu: number; memoryUsedMiB: number }[],
    runner = "nvidia-ada-lovelace-driver580",
): string {
    const available = samples.length > 0;
    return JSON.stringify({
        ok: true,
        runnerClass: runner,
        nvidiaSmi: {
            available,
            samples,
            maxUtilization: available ? Math.max(...samples.map((s) => s.utilizationGpu)) : 0,
            maxMemoryUsedMiB: available ? Math.max(...samples.map((s) => s.memoryUsedMiB)) : 0,
        },
    });
}

describe("benchmarks/datasets.ts (contract 6.2)", () => {
    it("makeRandom is graph-format's xorshift32: seed 7 gives the pinned first draws", () => {
        const random = makeRandom(7);
        expect(random()).toBeCloseTo(0.00044065131805837154, 12);
        expect(random()).toBeCloseTo(0.10952103300951421, 12);
        expect(random()).toBeCloseTo(0.9038964069914073, 12);
        expect(makeRandom(12345)()).toBeCloseTo(0.776938705239445, 12);
    });

    it("randomEdges is seeded G(n, m) with integer weights 1..10 (self-loops and parallels allowed)", () => {
        const a = randomEdges(10, 20, 1);
        const b = randomEdges(10, 20, 1);
        expect(a.nodeCount).toBe(10);
        expect(a.src.length).toBe(20);
        expect(a.dst.length).toBe(20);
        expect(a.weights.length).toBe(20);
        expect(Array.from(a.src)).toEqual(Array.from(b.src));
        expect(Array.from(a.dst)).toEqual(Array.from(b.dst));
        for (let e = 0; e < 20; e++) {
            expect(a.src[e]).toBeLessThan(10);
            expect(a.dst[e]).toBeLessThan(10);
            expect(a.weights[e]).toBeGreaterThanOrEqual(1);
            expect(a.weights[e]).toBeLessThanOrEqual(10);
            expect(Number.isInteger(a.weights[e])).toBe(true);
        }
        expect(Array.from(randomEdges(10, 20, 2).src)).not.toEqual(Array.from(a.src));
    });

    it("rmatEdges(4, 2, 1) has 16 nodes, 32 edges, no self-loops and is deterministic", () => {
        const g = rmatEdges(4, 2, 1);
        expect(g.nodeCount).toBe(16);
        expect(g.src.length).toBe(32);
        for (let e = 0; e < 32; e++) {
            expect(g.src[e]).toBeLessThan(16);
            expect(g.dst[e]).toBeLessThan(16);
            expect(g.src[e]).not.toBe(g.dst[e]);
        }
        // the first five pairs of the (0.57 / 0.19 / 0.19 / 0.05) walk with the (v + 1) % n self-loop fix
        expect(Array.from(g.src.slice(0, 5))).toEqual([0, 0, 0, 10, 5]);
        expect(Array.from(g.dst.slice(0, 5))).toEqual([2, 1, 10, 0, 0]);
        expect(Array.from(rmatEdges(4, 2, 1).dst)).toEqual(Array.from(g.dst));
    });

    it("gridEdges(3, 2) is the 4-neighbour grid in graph-format's row-major order with unit weights", () => {
        const g = gridEdges(3, 2);
        expect(g.nodeCount).toBe(6);
        const pairs = Array.from(g.src).map((u, e) => [u, g.dst[e]]);
        expect(pairs).toEqual([
            [0, 1],
            [0, 3],
            [1, 2],
            [1, 4],
            [2, 5],
            [3, 4],
            [4, 5],
        ]);
        expect(Array.from(g.weights)).toEqual([1, 1, 1, 1, 1, 1, 1]);
    });

    it("KARATE_EDGES is Zachary's karate club and snapshotOf builds an undirected weighted snapshot", () => {
        expect(KARATE_EDGES.nodeCount).toBe(34);
        expect(KARATE_EDGES.src.length).toBe(78);
        const s = snapshotOf(KARATE_EDGES, { label: "karate" });
        expect(s.directed).toBe(false);
        expect(s.nodeCount).toBe(34);
        expect(s.edgeCount).toBe(78);
        expect(s.arcCount).toBe(156);
        expect(s.label).toBe("karate");
        expect(s.flags.weighted).toBe(true);
        expect(s.outDegree()[0]).toBe(16);
        expect(s.outDegree()[33]).toBe(17);
        expect(snapshotOf(randomEdges(10, 20, 1), { directed: true }).directed).toBe(true);
    });

    it("TIERS are the design-15.3 pairs", () => {
        expect(TIERS.map((t) => [t.name, t.nodes, t.edges])).toEqual([
            ["10k/100k", 10_000, 100_000],
            ["100k/1M", 100_000, 1_000_000],
            ["1M/10M", 1_000_000, 10_000_000],
        ]);
    });
});

describe("benchmarks/harness.ts (contract 6.1)", () => {
    it("bench() runs one warm-up plus `runs` timed runs with setup / teardown per run and reports the median and rate", async () => {
        let setups = 0;
        let teardowns = 0;
        let bodies = 0;
        const r = await bench(
            "unit",
            "fake",
            {
                setup: () => {
                    setups += 1;
                    return setups;
                },
                run: async (input) => {
                    bodies += 1;
                    await new Promise((resolveSleep) => setTimeout(resolveSleep, 2));
                    return input;
                },
                teardown: (input) => {
                    expect(input).toBe(setups);
                    teardowns += 1;
                },
            },
            { device: FAKE_DEVICE, runs: 3, items: 1000, unit: "edges" },
        );
        expect(setups).toBe(4);
        expect(bodies).toBe(4);
        expect(teardowns).toBe(4);
        expect(r.group).toBe("unit");
        expect(r.name).toBe("fake");
        expect(r.runs).toBe(3);
        expect(r.medianMs).toBeGreaterThanOrEqual(1);
        expect(r.minMs).toBeLessThanOrEqual(r.medianMs);
        expect(r.maxMs).toBeGreaterThanOrEqual(r.medianMs);
        expect(r.rateUnit).toBe("edges/s");
        expect(r.rate).toBeCloseTo((1000 / r.medianMs) * 1000, 6);
        expect(typeof r.memoryDeltaBytes).toBe("number");
    });

    it("setBenchRuns() sets the default run count; rate is null without items", async () => {
        setBenchRuns(2);
        try {
            const r = await bench("unit", "runs", { setup: () => 0, run: () => 0 }, { device: FAKE_DEVICE });
            expect(r.runs).toBe(2);
            expect(r.rate).toBeNull();
            expect(r.rateUnit).toBeNull();
        } finally {
            setBenchRuns(5);
        }
        expect(() => {
            setBenchRuns(0);
        }).toThrow(/positive integer/);
    });

    it("a timed body that resolves to undefined is legal (contract 6.1: run returns Promise<unknown> | unknown)", async () => {
        const r = await bench(
            "unit",
            "void",
            { setup: () => 0, run: () => undefined },
            { device: FAKE_DEVICE, runs: 1 },
        );
        expect(r.runs).toBe(1);
        const asyncVoid = await bench(
            "unit",
            "async-void",
            { setup: () => 0, run: () => Promise.resolve(undefined) },
            { device: FAKE_DEVICE, runs: 1 },
        );
        expect(asyncVoid.runs).toBe(1);
    });

    it("appendSession writes <dir>/<runnerClass>.json as an array of sessions carrying the gpu field", () => {
        const dir = mkdtempSync(join(tmpdir(), "wgpu-bench-"));
        try {
            const results = [result("degree + 400 KB readback at 100k", 1.5)];
            const file = appendSession(results, GPU, { dir });
            expect(file).toBe(join(dir, `${runnerClass(GPU)}.json`));
            const sessions = JSON.parse(readFileSync(file, "utf8")) as BenchSession[];
            expect(sessions).toHaveLength(1);
            const first = sessions[0];
            expect(first.gpu).toEqual(GPU);
            expect(first.runnerClass).toBe(runnerClass(GPU));
            expect(first.results).toEqual(results);
            expect(first.node).toBe(process.version);
            expect(typeof first.date).toBe("string");
            expect(typeof first.host).toBe("string");
            expect(typeof first.cpu).toBe("string");
            expect(typeof first.exposeGc).toBe("boolean");
            appendSession(results, GPU, { dir });
            expect((JSON.parse(readFileSync(file, "utf8")) as BenchSession[]).length).toBe(2);
        } finally {
            rmSync(dir, { recursive: true, force: true });
        }
    });

    it("runnerClass honours GRAPHTY_RUNNER_CLASS and derives <vendor>-<architecture>-driver<major> otherwise (6.9)", () => {
        expect(runnerClass(GPU, {})).toBe("nvidia-ada-lovelace-driver580");
        expect(runnerClass(GPU, { GRAPHTY_RUNNER_CLASS: "gpu-linux-t4" })).toBe("gpu-linux-t4");
    });
});

describe("scripts/bench-compare.js (contract 6.8; spec 10.4 T-13)", () => {
    const SCRIPT = resolve("scripts/bench-compare.js");
    const CLASS = "nvidia-ada-lovelace-driver580";

    /** Runs the script in a fresh directory populated by `files` (path -> text); returns status and stdout. */
    function run(
        files: Readonly<Record<string, string>>,
        args: readonly string[] = [],
    ): { status: number | null; out: string } {
        const dir = mkdtempSync(join(tmpdir(), "wgpu-compare-"));
        try {
            for (const [path, text] of Object.entries(files)) {
                const full = join(dir, path);
                mkdirSync(join(full, ".."), { recursive: true });
                writeFileSync(full, text);
            }
            const proc = spawnSync(process.execPath, [SCRIPT, ...args], { cwd: dir, encoding: "utf8" });
            return { status: proc.status, out: `${proc.stdout}${proc.stderr}` };
        } finally {
            rmSync(dir, { recursive: true, force: true });
        }
    }

    const quiet = report([
        { utilizationGpu: 0, memoryUsedMiB: 512 },
        { utilizationGpu: 3, memoryUsedMiB: 512 },
    ]);
    const out = (results: readonly BenchResult[]): string => JSON.stringify([session(results)]);

    it("rule 1: no gpu-report.json or no out file -> nothing to compare, exit 0", () => {
        expect(run({})).toMatchObject({ status: 0 });
        expect(run({}).out).toContain("nothing to compare");
        const noOut = run({ "gpu-report.json": quiet, [`benchmarks/results/${CLASS}.json`]: out([result("a", 1)]) });
        expect(noOut.status).toBe(0);
        expect(noOut.out).toContain("nothing to compare");
    });

    it("rule 2: utilisation above 10% or memory taken by another process during the sample -> SKIPPED, exit 0", () => {
        const busy = report([
            { utilizationGpu: 0, memoryUsedMiB: 512 },
            { utilizationGpu: 40, memoryUsedMiB: 512 },
        ]);
        const grown = report([
            { utilizationGpu: 0, memoryUsedMiB: 512 },
            { utilizationGpu: 0, memoryUsedMiB: 900 },
        ]);
        const files = {
            [`benchmarks/out/${CLASS}.json`]: out([result("a", 10)]),
            [`benchmarks/results/${CLASS}.json`]: out([result("a", 1)]),
        };
        for (const doc of [busy, grown]) {
            const r = run({ ...files, "gpu-report.json": doc });
            expect(r.status).toBe(0);
            expect(r.out).toContain("SKIPPED: GPU not quiet");
        }
        // a flat memory series (a resident compositor) is quiet
        const flat = run({ ...files, "gpu-report.json": quiet });
        expect(flat.out).not.toContain("SKIPPED");
        // no nvidia-smi at all: the sample cannot veto
        const none = run({ ...files, "gpu-report.json": report([]) });
        expect(none.out).not.toContain("SKIPPED");
    });

    it("rule 3: no baseline -> every result is new, exit 0", () => {
        const r = run({
            "gpu-report.json": quiet,
            [`benchmarks/out/${CLASS}.json`]: out([result("alpha", 1), result("beta", 2)]),
        });
        expect(r.status).toBe(0);
        expect(r.out).toContain("new (no baseline)");
        expect(r.out).toContain("roundtrip/alpha");
        expect(r.out).toContain("roundtrip/beta");
    });

    it("rule 4: a median above threshold x baseline is a regression (exit 1); at or below passes; unmatched rows are new", () => {
        const baseline = out([result("a", 1), result("b", 2), result("gone", 1)]);
        const files = { "gpu-report.json": quiet, [`benchmarks/results/${CLASS}.json`]: baseline };
        const red = run({
            ...files,
            [`benchmarks/out/${CLASS}.json`]: out([result("a", 3.5), result("b", 2), result("fresh", 9)]),
        });
        expect(red.status).toBe(1);
        expect(red.out).toContain("REGRESSION");
        expect(red.out).toContain("new (no baseline)");
        const green = run({ ...files, [`benchmarks/out/${CLASS}.json`]: out([result("a", 3), result("b", 0.5)]) });
        expect(green.status).toBe(0);
        expect(green.out).not.toContain("REGRESSION");
        const looser = run({ ...files, [`benchmarks/out/${CLASS}.json`]: out([result("a", 3.5)]) }, [
            "--threshold",
            "4",
        ]);
        expect(looser.status).toBe(0);
    });

    it("compares the LAST session of each file and matches rows by group and name", () => {
        const baseline = JSON.stringify([
            session([result("a", 100)]),
            session([result("a", 1), result("a", 1, "upload")]),
        ]);
        const current = JSON.stringify([
            session([result("a", 1)]),
            session([result("a", 5, "upload"), result("a", 1)]),
        ]);
        const r = run({
            "gpu-report.json": quiet,
            [`benchmarks/results/${CLASS}.json`]: baseline,
            [`benchmarks/out/${CLASS}.json`]: current,
        });
        expect(r.status).toBe(1);
        expect(r.out).toContain("upload");
    });

    it("--class overrides the report's runner class", () => {
        const r = run(
            {
                "gpu-report.json": quiet,
                "benchmarks/out/other.json": out([result("a", 10)]),
                "benchmarks/results/other.json": out([result("a", 1)]),
            },
            ["--class", "other"],
        );
        expect(r.status).toBe(1);
        expect(existsSync(SCRIPT)).toBe(true);
    });
});

/**
 * The spec 7.6 curve (ms per iteration of the exact tile with the FA2 body on the RTX 4070 SUPER) as ladder rows; 7.6
 * has no 1k row, so 0.07 stands in for it (the rule only needs the ordering).
 */
const CURVE_7_6: readonly LadderRow[] = [
    { n: 1024, msPerIteration: 0.07 },
    { n: 4096, msPerIteration: 0.26 },
    { n: 8192, msPerIteration: 0.53 },
    { n: 16384, msPerIteration: 1.13 },
    { n: 32768, msPerIteration: 3.48 },
    { n: 65536, msPerIteration: 8.66 },
];

describe("benchmarks/layout-exact.bench.ts (contract 6.3; spec 7.8, 10.4 T-4)", () => {
    it("the exact ladder is 1k / 4k / 8k / 16k / 32k / 65k in powers of two with E = 10n, plus the 10k frame rung", () => {
        expect(EXACT_LADDER.map((r) => r.nodes)).toEqual([1024, 4096, 8192, 16384, 32768, 65536]);
        expect(EXACT_LADDER.map((r) => r.label)).toEqual(["1k", "4k", "8k", "16k", "32k", "65k"]);
        for (const rung of EXACT_LADDER) {
            expect(floorPow2(rung.nodes)).toBe(rung.nodes);
        }
        expect(LADDER_EDGE_FACTOR).toBe(10);
        expect(FRAME_RUNG).toEqual({ label: "10k", nodes: 10_000 });
        // the seven rungs the group walks share the rung shape (P4's calibrateLayout and layout-grid.bench.ts reuse it)
        const rungs: readonly LadderRung[] = [...EXACT_LADDER, FRAME_RUNG];
        expect(rungs).toHaveLength(7);
        expect(rungs.every((r) => Number.isInteger(r.nodes) && r.nodes >= 1 && r.label.length > 0)).toBe(true);
        expect(EXACT_BUDGET_MS).toBe(4);
        expect(LAYOUT_EXACT_GROUP).toBe("layout-exact");
    });

    it("floorPow2 rounds down to a power of two", () => {
        expect(floorPow2(1)).toBe(1);
        expect(floorPow2(2)).toBe(2);
        expect(floorPow2(3)).toBe(2);
        expect(floorPow2(1024)).toBe(1024);
        expect(floorPow2(10_000)).toBe(8192);
        expect(floorPow2(65_535)).toBe(32768);
        expect(floorPow2(65_536)).toBe(65536);
        expect(() => floorPow2(0)).toThrow(/positive/);
        expect(() => floorPow2(0.5)).toThrow(/positive/);
    });

    it("exactMaxNodesFromLadder is the spec 7.8 rule reduced to its budget clause (the grid clause is re-checked at G4)", () => {
        // 3.48 <= 4 < 8.66: the largest rung within 4 ms is 32k
        expect(exactMaxNodesFromLadder(CURVE_7_6)).toBe(32768);
        // 0.53 <= 1 < 1.13
        expect(exactMaxNodesFromLadder(CURVE_7_6, 1)).toBe(8192);
        // 0.26 <= 0.5 < 0.53
        expect(exactMaxNodesFromLadder(CURVE_7_6, 0.5)).toBe(4096);
        expect(exactMaxNodesFromLadder(CURVE_7_6, 100)).toBe(65536);
        // the budget is inclusive
        expect(exactMaxNodesFromLadder(CURVE_7_6, 3.48)).toBe(32768);
        // a rung that is not a power of two rounds down
        expect(
            exactMaxNodesFromLadder([
                { n: 10_000, msPerIteration: 0.9 },
                { n: 16384, msPerIteration: 5 },
            ]),
        ).toBe(8192);
        // the order of the rows is irrelevant
        expect(exactMaxNodesFromLadder([...CURVE_7_6].reverse())).toBe(32768);
        expect(() => exactMaxNodesFromLadder([])).toThrow(/no ladder rows/);
        expect(() => exactMaxNodesFromLadder(CURVE_7_6, 0.01)).toThrow(/no rung within 0.01 ms/);
        expect(() => exactMaxNodesFromLadder([{ n: 1024, msPerIteration: Number.NaN }])).toThrow(/finite/);
        expect(() => exactMaxNodesFromLadder([{ n: 1024, msPerIteration: -1 }])).toThrow(/finite/);
    });

    it("ladderRowsOf keeps the ms/iteration rows of the six ladder rungs, parses n and sorts by n", () => {
        const rows = ladderRowsOf([
            result("ms/iteration (profiler) n=32768 [32k]", 3.5, LAYOUT_EXACT_GROUP),
            result("step(1) wall n=32768 m=327680 2D [32k]", 4.1, LAYOUT_EXACT_GROUP),
            result("ms/iteration (profiler) n=1024 [1k]", 0.07, LAYOUT_EXACT_GROUP),
            // the frame rung is not a ladder rung
            result("ms/iteration (profiler) n=10000 [10k]", 0.8, LAYOUT_EXACT_GROUP),
            // the wall source is accepted (a device without timestamp-query)
            result("ms/iteration (wall) n=4096 [4k]", 0.3, LAYOUT_EXACT_GROUP),
            // another group's row of the same name is ignored
            result("ms/iteration (profiler) n=8192 [8k]", 0.6, "other-group"),
        ]);
        expect(rows).toEqual([
            { n: 1024, msPerIteration: 0.07 },
            { n: 4096, msPerIteration: 0.3 },
            { n: 32768, msPerIteration: 3.5 },
        ]);
        expect(ladderRowsOf([])).toEqual([]);
    });
});

describe("benchmarks/layout-run.ts (contract 6.3)", () => {
    it("parseLayoutRunArgs: the two required sizes and the five defaults", () => {
        expect(parseLayoutRunArgs(["--nodes", "100000", "--edges", "1000000"])).toEqual({
            nodes: 100_000,
            edges: 1_000_000,
            iterations: 100,
            batch: 8,
            seed: 1,
            dim: 2,
            compat: "paper",
        });
        expect(
            parseLayoutRunArgs([
                "--edges",
                "20",
                "--nodes",
                "10",
                "--iterations",
                "5",
                "--batch",
                "2",
                "--seed",
                "7",
                "--dim",
                "3",
                "--compat",
                "networkx",
            ]),
        ).toEqual({ nodes: 10, edges: 20, iterations: 5, batch: 2, seed: 7, dim: 3, compat: "networkx" });
        expect(() => parseLayoutRunArgs([])).toThrow(/--nodes N is required/);
        expect(() => parseLayoutRunArgs(["--nodes", "10"])).toThrow(/--edges M is required/);
        expect(() => parseLayoutRunArgs(["--nodes", "0", "--edges", "1"])).toThrow(/--nodes expects an integer >= 1/);
        expect(() => parseLayoutRunArgs(["--nodes", "10", "--edges", "-1"])).toThrow(/--edges expects an integer >= 0/);
        expect(() => parseLayoutRunArgs(["--nodes", "10", "--edges", "1", "--dim", "4"])).toThrow(
            /--dim expects 2 or 3/,
        );
        expect(() => parseLayoutRunArgs(["--nodes", "10", "--edges", "1", "--compat", "port"])).toThrow(
            /--compat expects paper or networkx/,
        );
        expect(() => parseLayoutRunArgs(["--nodes", "10", "--edges", "1", "--batch", "0"])).toThrow(
            /--batch expects an integer >= 1/,
        );
        expect(() => parseLayoutRunArgs(["--nodes", "10", "--edges", "1", "--iterations", "x"])).toThrow(
            /--iterations expects an integer >= 1/,
        );
        expect(() => parseLayoutRunArgs(["--nodes", "10", "--edges", "1", "--bogus"])).toThrow(
            /unknown option --bogus/,
        );
        expect(() => parseLayoutRunArgs(["--nodes", "10", "--edges", "1", "extra"])).toThrow(
            /unexpected argument extra/,
        );
        expect(() => parseLayoutRunArgs(["--nodes"])).toThrow(/--nodes expects an integer >= 1, got undefined/);
    });

    it("checkLayoutResult: finite positions and a settled or exhausted run pass; anything else names the problem", () => {
        const good = new Float32Array([0, 1, 0, 2, 3, 0]);
        expect(checkLayoutResult(good, 2, 50, true, 100)).toEqual({ ok: true, problems: [] });
        expect(checkLayoutResult(good, 2, 100, false, 100)).toEqual({ ok: true, problems: [] });
        expect(checkLayoutResult(good, 2, 50, false, 100)).toEqual({
            ok: false,
            problems: ["the run ended after 50 of 100 iterations without settling"],
        });
        expect(checkLayoutResult(good, 2, 0, false, 100).problems).toEqual([
            "the run ended after 0 of 100 iterations without settling",
        ]);
        const bad = new Float32Array([0, 1, 0, Number.NaN, 3, Number.POSITIVE_INFINITY]);
        expect(checkLayoutResult(bad, 2, 100, false, 100)).toEqual({
            ok: false,
            problems: ["2 non-finite position components (first at node 1, component 0)"],
        });
        expect(checkLayoutResult(good, 3, 100, false, 100)).toEqual({
            ok: false,
            problems: ["positions has 6 components, expected 9 (3 x 3 nodes)"],
        });
        // both problems are reported, the length one first
        expect(checkLayoutResult(good, 3, 1, false, 100).problems).toEqual([
            "positions has 6 components, expected 9 (3 x 3 nodes)",
            "the run ended after 1 of 100 iterations without settling",
        ]);
    });
});
