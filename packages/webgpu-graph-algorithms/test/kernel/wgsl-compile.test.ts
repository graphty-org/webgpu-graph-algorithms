/**
 * The WGSL compile matrix (spec 5.1, 11.3; contract 5.5 P2): every OVERRIDE_MATRIX case compiles through the
 * PipelineCache on the real device and on Dawn's backend=null adapter, twin cases on both feature settings; the
 * matrix is bounded (its size is pinned per phase); matrixCovers accepts every key these compiles created and
 * canonicalises the ways a factory may spell an override set. The browser twin is test/browser/compile-matrix.test.ts;
 * the completeness half (every key the node suite creates is a case) is test/setup/global.ts's teardown, whose
 * pieces are unit-tested here too.
 */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { type TestContext } from "vitest";

import { type GpuContext } from "../../src/context.js";
import { pipelineKey } from "../../src/kernel/pipeline-cache.js";
import { type KernelId, KERNELS, kernelSpec } from "../../src/kernels.js";
import { CAPS_LAVAPIPE, CAPS_SPEC_DEFAULT } from "../helpers/caps-tables.js";
import {
    canonicalKey,
    entryOf,
    EXPECTED_CASES_BY_PHASE,
    matrixCovers,
    OVERRIDE_MATRIX,
    type OverrideCase,
    SEGMENTED_REDUCE_SNIPPETS,
} from "../helpers/override-matrix.js";
import { assertMatrixCoverage, readKeyLogs } from "../setup/global.js";
import { acquire, acquireNullBackend, requireGpu } from "../setup/gpu.js";

const STANDARD = new Set(["USE_PERM", "HAS_WEIGHTS"]);

/** overrides[name] with the `| undefined` the record type hides. */
function lookup(c: OverrideCase, name: string): number | boolean | undefined {
    const { overrides } = c;
    const record: Partial<Record<string, number | boolean>> = overrides;
    return record[name];
}

describe("OVERRIDE_MATRIX (pure)", () => {
    it("is bounded: the case count is the sum of the pins of the phases present in KERNELS", () => {
        const phases = new Set(Object.values(KERNELS).map((entry) => entry.phase));
        let expected = 0;
        for (const phase of phases) {
            expected += EXPECTED_CASES_BY_PHASE[phase];
        }
        expect(OVERRIDE_MATRIX.length).toBe(expected);
        expect(OVERRIDE_MATRIX.length).toBeGreaterThanOrEqual(89); // P1 (37) + P2 (52)
    });

    it("pins the per-kernel counts of the P1 and P2 entries", () => {
        const counts = new Map<string, number>();
        for (const c of OVERRIDE_MATRIX) {
            counts.set(c.id, (counts.get(c.id) ?? 0) + 1);
        }
        expect(counts.get("degree")).toBe(5);
        expect(counts.get("reduce")).toBe(19);
        expect(counts.get("fill")).toBe(1);
        expect(counts.get("segmented-reduce")).toBe(52);
        expect(counts.get("fa2-repulsion-exact")).toBe(9);
        expect(counts.get("fa2-speed-finalize")).toBe(3);
    });

    it("every case names a registry entry, uses only its declared overrides plus the standard pair (graph kernels only), carries snippets iff the entry has slots, matches the twin axis, and is unique", () => {
        const seen = new Set<string>();
        for (const c of OVERRIDE_MATRIX) {
            const entry = entryOf(c.id);
            const declared = new Set(entry.overrideDecls.map((d) => d.name));
            const names = Object.keys(c.overrides);
            for (const name of names) {
                expect(declared.has(name) || STANDARD.has(name), `${c.id}: ${name}`).toBe(true);
            }
            if (names.some((name) => STANDARD.has(name))) {
                expect(
                    entry.bindings.some((b) => b.group === 0),
                    `${c.id}: standard pair on a non-graph kernel`,
                ).toBe(true);
            }
            expect(c.snippets !== undefined, `${c.id}: snippets`).toBe(entry.snippetSlots.length > 0);
            if (c.snippets !== undefined) {
                expect(Object.keys(c.snippets)).toEqual([...entry.snippetSlots]);
            }
            expect(c.twin).toBe(entry.needs.includes("subgroups"));
            const signature = `${c.id}|${JSON.stringify(c.overrides)}|${JSON.stringify(c.snippets ?? null)}`;
            expect(seen.has(signature), `duplicate case ${signature}`).toBe(false);
            seen.add(signature);
        }
    });

    it("contains every entry's defaults, all four snippets of segmented-reduce, and each declared override toggled alone", () => {
        for (const entry of Object.values(KERNELS)) {
            const ofEntry = OVERRIDE_MATRIX.filter((c) => c.id === entry.id);
            expect(
                ofEntry.some((c) => Object.keys(c.overrides).length === 0),
                `${entry.id} defaults`,
            ).toBe(true);
            for (const decl of entry.overrideDecls) {
                const values = new Set<number | boolean>();
                for (const c of ofEntry) {
                    const v = lookup(c, decl.name);
                    if (v !== undefined) {
                        values.add(v);
                    }
                }
                for (const value of values) {
                    if (value === decl.default) {
                        continue;
                    }
                    const alone = ofEntry.some(
                        (c) =>
                            lookup(c, decl.name) === value &&
                            entry.overrideDecls.every(
                                (other) => other.name === decl.name || lookup(c, other.name) === other.default,
                            ),
                    );
                    expect(alone, `${entry.id}: ${decl.name} = ${String(value)} toggled alone`).toBe(true);
                }
            }
        }
        const snippets = OVERRIDE_MATRIX.filter((c) => c.id === "segmented-reduce").map((c) => c.snippets?.VALUE);
        for (const snippet of Object.values(SEGMENTED_REDUCE_SNIPPETS)) {
            expect(snippets).toContain(snippet);
        }
        expect(new Set(snippets).size).toBe(4);
    });

    it("matrixCovers: canonicalises omitted defaults and the standard pair, covers both twins in one call, ignores ad hoc ids, reports foreign combinations and snippets", () => {
        expect(CAPS_LAVAPIPE.features.has("subgroups")).toBe(true);
        expect(CAPS_SPEC_DEFAULT.features.has("subgroups")).toBe(false);

        const explicit = pipelineKey(kernelSpec("reduce", { OP: 1, DTYPE: 0, FINAL: false }), CAPS_LAVAPIPE);
        const omitted = pipelineKey(kernelSpec("reduce", { OP: 1 }), CAPS_LAVAPIPE);
        expect(canonicalKey(explicit)).toBe(canonicalKey(omitted));
        expect(matrixCovers([explicit, omitted], CAPS_SPEC_DEFAULT)).toEqual({ ok: true, missing: [] });

        const noPair = pipelineKey(kernelSpec("fa2-speed-finalize", { SWING_MODE: 1 }), CAPS_LAVAPIPE);
        const withPair = pipelineKey(
            kernelSpec("fa2-speed-finalize", { SWING_MODE: 1, USE_PERM: false, HAS_WEIGHTS: false }),
            CAPS_LAVAPIPE,
        );
        expect(canonicalKey(noPair)).toBe(canonicalKey(withPair));
        expect(matrixCovers([noPair, withPair], CAPS_SPEC_DEFAULT).ok).toBe(true);

        const twinAbsent = pipelineKey(kernelSpec("reduce", { OP: 1 }), CAPS_SPEC_DEFAULT);
        expect(twinAbsent).not.toBe(omitted);
        expect(matrixCovers([twinAbsent, omitted], CAPS_SPEC_DEFAULT).ok).toBe(true);

        const adHoc = pipelineKey(
            {
                id: "sg-slot-order",
                body: "",
                bindings: [],
                overrideDecls: [],
                overrides: {},
                needs: [],
                uniforms: [],
            },
            CAPS_LAVAPIPE,
        );
        expect(matrixCovers([adHoc], CAPS_SPEC_DEFAULT)).toEqual({ ok: true, missing: [] });

        const foreign = pipelineKey(kernelSpec("reduce", { OP: 7 }), CAPS_LAVAPIPE);
        expect(matrixCovers([foreign], CAPS_SPEC_DEFAULT)).toEqual({ ok: false, missing: [foreign] });

        const otherSnippet = pipelineKey(
            kernelSpec("segmented-reduce", {}, { VALUE: "v = weight * 2.0;" }),
            CAPS_LAVAPIPE,
        );
        expect(matrixCovers([otherSnippet], CAPS_SPEC_DEFAULT).ok).toBe(false);
        const knownSnippet = pipelineKey(
            kernelSpec(
                "segmented-reduce",
                { OP: 2, TIER: 0, USE_PERM: false, HAS_WEIGHTS: true },
                { VALUE: SEGMENTED_REDUCE_SNIPPETS.weight },
            ),
            CAPS_LAVAPIPE,
        );
        expect(matrixCovers([knownSnippet], CAPS_SPEC_DEFAULT).ok).toBe(true);
        const conditional = pipelineKey(
            kernelSpec(
                "segmented-reduce",
                { OP: 0, TIER: 0, USE_PERM: false, HAS_WEIGHTS: false },
                { VALUE: SEGMENTED_REDUCE_SNIPPETS.conditional },
            ),
            CAPS_SPEC_DEFAULT,
        );
        expect(matrixCovers([conditional], CAPS_SPEC_DEFAULT).ok).toBe(true);
    });

    it("the teardown's pieces: readKeyLogs accepts a JSON array, JSON lines and plain lines; assertMatrixCoverage names the uncovered key", () => {
        const dir = mkdtempSync(join(tmpdir(), "pipeline-keys-"));
        try {
            const a = pipelineKey(kernelSpec("degree", { USE_PERM: false, HAS_WEIGHTS: true }), CAPS_SPEC_DEFAULT);
            const b = pipelineKey(kernelSpec("fill"), CAPS_SPEC_DEFAULT);
            const foreign = pipelineKey(kernelSpec("reduce", { DTYPE: 5 }), CAPS_SPEC_DEFAULT);
            writeFileSync(join(dir, "worker-1.json"), `${JSON.stringify([a, b])}\n`);
            writeFileSync(join(dir, "worker-2.log"), `${JSON.stringify(a)}\n${b}\n\n`);
            writeFileSync(join(dir, "worker-3.txt"), `${foreign}\n`);
            const keys = readKeyLogs(dir);
            expect([...keys].sort()).toEqual([a, b, foreign].sort());
            expect(() => assertMatrixCoverage(new Set([a, b]))).not.toThrow();
            expect(() => assertMatrixCoverage(keys)).toThrow(foreign);
        } finally {
            rmSync(dir, { recursive: true, force: true });
        }
    });
});

describe("compile matrix on Dawn (real device and backend=null)", () => {
    let real: GpuContext | null = null;
    let realTwin: GpuContext | null = null;
    let nul: GpuContext | null = null;
    let nulTwin: GpuContext | null = null;

    async function contexts(
        t: TestContext,
    ): Promise<{ real: GpuContext; realTwin: GpuContext; nul: GpuContext; nulTwin: GpuContext }> {
        requireGpu(t);
        const a = real ?? (await acquire({ label: "wgsl-compile" }));
        real = a;
        const b = realTwin ?? (await acquire({ subgroups: false, label: "wgsl-compile-twin" }));
        realTwin = b;
        const c = nul ?? (await acquireNullBackend({ label: "wgsl-compile-null" }));
        nul = c;
        const d = nulTwin ?? (await acquireNullBackend({ subgroups: false, label: "wgsl-compile-null-twin" }));
        nulTwin = d;
        return { real: a, realTwin: b, nul: c, nulTwin: d };
    }

    async function compileAll(ctx: GpuContext, cases: readonly OverrideCase[]): Promise<void> {
        for (const c of cases) {
            const kernel = await ctx.pipelines.kernel(kernelSpec(c.id, c.overrides, c.snippets));
            expect(kernel.entryPoint).toBe(entryOf(c.id).entryPoint);
        }
    }

    for (const id of Object.keys(KERNELS) as KernelId[]) {
        it(`compiles every case of ${id} on the real device and on backend=null (both twins where the entry has one)`, async (t) => {
            const { real: r, realTwin: rt, nul: n, nulTwin: nt } = await contexts(t);
            const cases = OVERRIDE_MATRIX.filter((c) => c.id === id);
            expect(cases.length).toBeGreaterThan(0);
            await compileAll(r, cases);
            await compileAll(n, cases);
            const twins = cases.filter((c) => c.twin);
            if (twins.length > 0) {
                expect(rt.caps.features.has("subgroups")).toBe(false);
                await compileAll(rt, twins);
                await compileAll(nt, twins);
            }
        }, 120_000);
    }

    it("the keys these compiles created are all covered by the matrix (self-consistency of the coverage rule)", async (t) => {
        const { real: r, realTwin: rt, nul: n } = await contexts(t);
        const keys = [...r.pipelines.keys(), ...rt.pipelines.keys(), ...n.pipelines.keys()];
        expect(keys.length).toBeGreaterThanOrEqual(OVERRIDE_MATRIX.length);
        expect(matrixCovers(keys, CAPS_SPEC_DEFAULT)).toEqual({ ok: true, missing: [] });
        expect(r.pipelines.size).toBe(OVERRIDE_MATRIX.length);
    });
});
