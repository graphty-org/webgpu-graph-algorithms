/**
 * The sabotage table is itself checked (spec 13 rule f, 11.9 item 1; contract 5.5): every KERNELS entry of a phase
 * listed in SABOTAGE_PHASES that is not exempt has at least three rows, every `find` string occurs exactly once in the
 * entry's NORMATIVE body (so a mutation is a well-defined textual edit of 4.5), every `test` names an existing test
 * file, and the splice helper rejects an absent or duplicated `find`. P2-T2 and P3-T5 extend SABOTAGE_PHASES when
 * their rows land; this file does not change.
 */

import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { type KernelId, KERNELS } from "../../src/kernels.js";
import { type Mutation, SABOTAGE, SABOTAGE_EXEMPT, SABOTAGE_PHASES, sabotagedBody } from "../helpers/sabotage.js";

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");

function rowsOf(id: KernelId): readonly Mutation[] {
    return SABOTAGE[id] ?? [];
}

function tabledIds(): KernelId[] {
    return Object.keys(SABOTAGE) as KernelId[];
}

describe("sabotage coverage (spec 11.9 item 1, 13 rule f)", () => {
    it("lists P1 and exempts exactly fill and fa2-to-scene", () => {
        expect(SABOTAGE_PHASES.includes("P1")).toBe(true);
        expect([...SABOTAGE_EXEMPT].sort()).toEqual(["fa2-to-scene", "fill"]);
    });

    it("every non-exempt kernel of a listed phase has >= 3 rows", () => {
        const listed = Object.values(KERNELS).filter(
            (entry) => SABOTAGE_PHASES.includes(entry.phase) && !SABOTAGE_EXEMPT.includes(entry.id),
        );
        expect(listed.length).toBeGreaterThan(0);
        for (const entry of listed) {
            expect(
                rowsOf(entry.id).length,
                `${entry.id} (${entry.phase}) needs >= 3 sabotage rows`,
            ).toBeGreaterThanOrEqual(3);
        }
    });

    it("the P1 rows are the four degree, four reduce, four K3 and three K4 mutations, by name", () => {
        expect(rowsOf("degree").map((m) => m.name)).toEqual([
            "last-row-skipped",
            "use-perm-select-swapped",
            "rebase-ignored",
            "target-counted-twice",
        ]);
        expect(rowsOf("reduce").map((m) => m.name)).toEqual([
            "final-writes-out-zero",
            "u32-min-identity-zero",
            "f32-min-identity-zero",
            "level-bound-inclusive",
        ]);
        expect(rowsOf("fa2-repulsion-exact").map((m) => m.name)).toEqual([
            "gravity-sign-flipped",
            "inverse-square-law",
            "self-pair-guard-removed",
            "mass-lane-x",
        ]);
        expect(rowsOf("fa2-speed-finalize").map((m) => m.name)).toEqual([
            "efficiency-halving-weakened",
            "efficiency-rise-removed",
            "swing-traction-swapped",
        ]);
        expect(rowsOf("fill")).toEqual([]);
    });

    it("every find string occurs exactly once in the entry's normative body, the replacement differs, minFactor >= 10, names unique", () => {
        for (const id of tabledIds()) {
            const { body } = KERNELS[id];
            const names = new Set<string>();
            for (const m of rowsOf(id)) {
                expect(names.has(m.name), `${id}: duplicate mutation name ${m.name}`).toBe(false);
                names.add(m.name);
                const first = body.indexOf(m.find);
                expect(first, `${id}/${m.name}: find string absent from the body`).toBeGreaterThanOrEqual(0);
                expect(body.indexOf(m.find, first + m.find.length), `${id}/${m.name}: find string not unique`).toBe(-1);
                expect(m.replace, `${id}/${m.name}: replace equals find`).not.toBe(m.find);
                expect(m.minFactor, `${id}/${m.name}: minFactor`).toBeGreaterThanOrEqual(10);
                const mutated = sabotagedBody(id, m);
                expect(mutated).not.toBe(body);
                expect(mutated.includes(m.replace)).toBe(true);
                expect(mutated.length).toBe(body.length - m.find.length + m.replace.length);
            }
        }
    });

    it("every test names an existing test file under test/", () => {
        for (const id of tabledIds()) {
            for (const m of rowsOf(id)) {
                expect(m.test).toMatch(/^test\/.+\.test\.ts$/);
                expect(existsSync(resolve(PACKAGE_ROOT, m.test)), `${id}/${m.name}: ${m.test} does not exist`).toBe(
                    true,
                );
            }
        }
    });

    it("sabotagedBody throws on an absent find and on a duplicated find", () => {
        const absent: Mutation = {
            name: "x",
            find: "no such text in any body",
            replace: "y",
            minFactor: 10,
            test: "test/algorithms/degree.test.ts",
        };
        expect(() => sabotagedBody("degree", absent)).toThrow(/absent/);
        // `rowPtr[` occurs twice in the degree body: rowPtr[i] and rowPtr[i + 1u]
        const duplicated: Mutation = {
            name: "x",
            find: "rowPtr[",
            replace: "y",
            minFactor: 10,
            test: "test/algorithms/degree.test.ts",
        };
        expect(() => sabotagedBody("degree", duplicated)).toThrow(/unique/);
    });
});
