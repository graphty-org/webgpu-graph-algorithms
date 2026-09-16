/**
 * Hand-computed checks of the two CPU references of P1 (spec 11.3 "CPU reference", 5.3): the out-degree oracle over
 * the karate club and small graphs, and the f64 reduce oracle over f32 / u32 / vec4f inputs incl. the identity
 * elements of the empty range; plus the pure reduce-input recipe (test/helpers/reduce-input.ts) that DEFINES the
 * reduce noise fixture, pinned here so P1-T6 / P1-T7 cannot drift from it. No device.
 */

import { type ReduceDtype } from "../../src/primitives/reduce.js";
import { KARATE_EDGES, pathEdges, snapshotOf, starEdges } from "../helpers/graphs.js";
import {
    INPUT_SEED,
    lanesOf,
    RANDOM1K_COUNT,
    RANDOM1K_SEED,
    REDUCE_NOISE_COUNTS,
    reduceInput,
} from "../helpers/reduce-input.js";
import { outDegreeOracle } from "./degree.js";
import { reduceIdentity, reduceOracle } from "./reduce.js";

const F32_MAX = 3.4028234663852886e38;
const U32_MAX = 4294967295;

describe("outDegreeOracle (5.3)", () => {
    it("karate undirected: the classic degree sequence, arcs 156", () => {
        const s = snapshotOf(KARATE_EDGES);
        const expected = [
            16, 9, 10, 6, 3, 4, 4, 4, 5, 2, 3, 1, 2, 5, 2, 2, 2, 2, 2, 3, 2, 2, 2, 5, 3, 3, 2, 4, 3, 4, 4, 6, 12, 17,
        ];
        const oracle = outDegreeOracle(s);
        expect(oracle).toBeInstanceOf(Uint32Array);
        expect(oracle.buffer).toBeInstanceOf(ArrayBuffer);
        expect(Array.from(oracle)).toEqual(expected);
        expect(expected.reduce((a, b) => a + b, 0)).toBe(156);
        expect(s.arcCount).toBe(156);
        expect(Array.from(oracle)).toEqual(Array.from(s.outDegree()));
    });

    it("karate directed (every edge listed as u < v): out-degrees, arcs 78", () => {
        const s = snapshotOf(KARATE_EDGES, { directed: true });
        const expected = [
            16, 8, 8, 3, 2, 3, 1, 0, 3, 1, 0, 0, 0, 1, 2, 2, 0, 0, 2, 1, 2, 0, 2, 5, 3, 1, 2, 1, 2, 2, 2, 2, 1, 0,
        ];
        expect(Array.from(outDegreeOracle(s))).toEqual(expected);
        expect(expected.reduce((a, b) => a + b, 0)).toBe(78);
        expect(s.arcCount).toBe(78);
    });

    it("path of 5: [1, 2, 2, 2, 1]; star with 4 leaves: [4, 1, 1, 1, 1]", () => {
        expect(Array.from(outDegreeOracle(snapshotOf(pathEdges(5))))).toEqual([1, 2, 2, 2, 1]);
        expect(Array.from(outDegreeOracle(snapshotOf(starEdges(4))))).toEqual([4, 1, 1, 1, 1]);
    });

    it("no arcs: zeros; no nodes: an empty array", () => {
        expect(Array.from(outDegreeOracle(snapshotOf([], { nodeCount: 5 })))).toEqual([0, 0, 0, 0, 0]);
        expect(outDegreeOracle(snapshotOf([], { nodeCount: 0 })).length).toBe(0);
    });

    it("a self-loop is one arc; a parallel edge is one arc per copy", () => {
        // undirected: (0,0) once at row 0; (0,1) twice at row 0 and twice at row 1 -> [3, 2], arcCount 2 * 3 - 1
        const s = snapshotOf(
            [
                [0, 0],
                [0, 1],
                [0, 1],
            ],
            { nodeCount: 2 },
        );
        expect(s.arcCount).toBe(5);
        expect(Array.from(outDegreeOracle(s))).toEqual([3, 2]);
    });
});

describe("reduceInput (test/helpers/reduce-input.ts): the shared, pure input recipe", () => {
    it("is deterministic per (dtype, count, seed), in range, and its dtype union is the contract's ReduceDtype", () => {
        const a = reduceInput("f32", RANDOM1K_COUNT, RANDOM1K_SEED);
        const b = reduceInput("f32", RANDOM1K_COUNT, RANDOM1K_SEED);
        expect(a).toBeInstanceOf(Float32Array);
        expect(a.length).toBe(1000);
        expect(Array.from(a)).toEqual(Array.from(b));
        expect(Array.from(a).every((v) => v >= 1 && v < 2)).toBe(true);
        const u = reduceInput("u32", 4097, INPUT_SEED + 4097);
        expect(u).toBeInstanceOf(Uint32Array);
        expect(Array.from(u).every((v) => v >= 1 && v <= 127)).toBe(true);
        expect(reduceInput("vec4f", 257, INPUT_SEED + 257).length).toBe(257 * lanesOf("vec4f"));
        expect(reduceInput("f32", 0, INPUT_SEED).length).toBe(0);
        expect(REDUCE_NOISE_COUNTS).toEqual([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]);
        // the dtype parameter accepts exactly the contract's ReduceDtype values
        const dtypes: readonly ReduceDtype[] = ["f32", "u32", "vec4f"];
        for (const dtype of dtypes) {
            expect(reduceInput(dtype, 1, 1).length).toBe(lanesOf(dtype));
        }
    });
});

describe("reduceOracle (5.3): f64 sequential reduction", () => {
    it("f32 scalars: sum 6.75, min 1.5, max 3", () => {
        const v = Float32Array.from([1.5, 2.25, 3]);
        expect(reduceOracle(v, "sum", "f32")).toBe(6.75);
        expect(reduceOracle(v, "min", "f32")).toBe(1.5);
        expect(reduceOracle(v, "max", "f32")).toBe(3);
    });

    it("sum is the f64 total: 16777216 + 1 + 1 = 16777218 (an f32 accumulator would lose the ones)", () => {
        expect(reduceOracle(Float32Array.from([16777216, 1, 1]), "sum", "f32")).toBe(16777218);
        expect(Math.fround(Math.fround(16777216 + 1) + 1)).toBe(16777216);
    });

    it("u32 scalars: sum 19, min 3, max 9; exact above 2^31", () => {
        const v = Uint32Array.from([7, 3, 9]);
        expect(reduceOracle(v, "sum", "u32")).toBe(19);
        expect(reduceOracle(v, "min", "u32")).toBe(3);
        expect(reduceOracle(v, "max", "u32")).toBe(9);
        expect(reduceOracle(Uint32Array.from([4000000000, 200000000]), "sum", "u32")).toBe(4200000000);
    });

    it("vec4f: lane-wise over groups of four words", () => {
        const v = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
        expect(reduceOracle(v, "sum", "vec4f")).toEqual([6, 8, 10, 12]);
        expect(reduceOracle(v, "min", "vec4f")).toEqual([1, 2, 3, 4]);
        expect(reduceOracle(v, "max", "vec4f")).toEqual([5, 6, 7, 8]);
    });

    it("the empty range yields the GPU's identity element (4.5: 0, F32_MAX, -F32_MAX, U32_MAX)", () => {
        expect(reduceIdentity("sum", "f32")).toBe(0);
        expect(reduceIdentity("sum", "u32")).toBe(0);
        expect(reduceIdentity("min", "f32")).toBe(F32_MAX);
        expect(reduceIdentity("max", "f32")).toBe(-F32_MAX);
        expect(reduceIdentity("min", "u32")).toBe(U32_MAX);
        expect(reduceIdentity("max", "u32")).toBe(0);
        expect(reduceIdentity("min", "vec4f")).toBe(F32_MAX);
        expect(reduceOracle(new Float32Array(0), "sum", "f32")).toBe(0);
        expect(reduceOracle(new Float32Array(0), "min", "f32")).toBe(F32_MAX);
        expect(reduceOracle(new Float32Array(0), "max", "f32")).toBe(-F32_MAX);
        expect(reduceOracle(new Uint32Array(0), "min", "u32")).toBe(U32_MAX);
        expect(reduceOracle(new Uint32Array(0), "max", "u32")).toBe(0);
        expect(reduceOracle(new Float32Array(0), "sum", "vec4f")).toEqual([0, 0, 0, 0]);
        expect(reduceOracle(new Float32Array(0), "min", "vec4f")).toEqual([F32_MAX, F32_MAX, F32_MAX, F32_MAX]);
        // F32_MAX is the largest finite f32, the value of the prelude's 0x1.fffffep+127
        expect(Math.fround(F32_MAX)).toBe(F32_MAX);
        expect(Math.fround(F32_MAX * 2)).toBe(Infinity);
    });

    it("a single element is returned exactly for every op", () => {
        expect(reduceOracle(Float32Array.from([0.30000001192092896]), "sum", "f32")).toBe(0.30000001192092896);
        expect(reduceOracle(Uint32Array.from([4294967295]), "max", "u32")).toBe(4294967295);
    });
});
