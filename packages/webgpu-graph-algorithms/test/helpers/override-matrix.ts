/**
 * The bounded WGSL compile matrix (spec 5.1, 11.3; contract 5.2): for every KERNELS entry its defaults, the cartesian
 * product of its declared overrides (which contains every override toggled alone), the USE_PERM / HAS_WEIGHTS pair
 * when the kernel binds the graph group, and the exact VALUE snippets the suite uses. The table is generated from the
 * registry by this ONE rule so a kernel cannot be registered without being enumerated (spec 3.5), and `matrixCovers`
 * lets test/setup/global.ts prove that every pipeline key the node suite created came from a case of it.
 *
 * PLAN DECISION (P2-T2): a factory passes kernelSpec exactly the kernel's declared overrides plus USE_PERM /
 * HAS_WEIGHTS iff the kernel binds group 0 (3.10 graphOverrides); `matrixCovers` canonicalises keys -- an absent
 * declared override takes its default, an absent standard pair is false, the device-derived WG / SUBGROUP_MIN /
 * SUBGROUP_MAX are dropped -- so "omit the default" and "spell every override" give the same verdict, and keys of
 * ids outside the registry (ad hoc specs built by tests) are not package variants and are ignored.
 *
 * PLAN DECISION (P2-T2, pending owner confirmation at G2): contract 5.2 / spec 5.1 describe OVERRIDE_MATRIX as an
 * explicit table of "the defaults, each override toggled alone, and the exact combinations the factories emit";
 * this file GENERATES the full product instead, which is a bounded superset of that table (P1 37 cases, P2 52, P3
 * 22 -- EXPECTED_CASES_BY_PHASE, pinned by test/kernel/wgsl-compile.test.ts) and compiles in full on every
 * compile-matrix context (four under Dawn, two under Chromium, SwiftShader included). The superset buys one
 * property the literal table lacks: a combination a factory starts emitting later is a case already, so the
 * teardown never fails for a legitimate variant. docs/decisions/G2.md records the case counts and the measured
 * SwiftShader compile time of test/browser/compile-matrix.test.ts so the owner can accept the superset knowingly.
 */

import { pipelineKey } from "../../src/kernel/pipeline-cache.js";
import { type KernelEntry, type KernelId, KERNELS, kernelSpec } from "../../src/kernels.js";
import { type PlanCaps } from "../../src/types/context.js";

/** One compile case: an id, an override set and whether the subgroup axis applies. */
export interface OverrideCase {
    readonly id: KernelId;
    readonly overrides: Readonly<Record<string, number | boolean>>;
    readonly snippets?: Readonly<Record<string, string>> | undefined;
    readonly twin: boolean;
}

/**
 * Every VALUE snippet a node test compiles for segmented-reduce (the snippet text is part of the pipeline key, spec
 * 5.1, so a snippet outside this record is an uncovered key and fails the teardown): the weighted sum, the degree,
 * a multi-statement conditional (the vocabulary check admits WGSL keywords) and a snippet carrying a comment that
 * names forbidden words (the check strips comments first).
 */
export const SEGMENTED_REDUCE_SNIPPETS: Readonly<{
    weight: string;
    one: string;
    conditional: string;
    commented: string;
}> = Object.freeze({
    weight: "v = weight;",
    one: "v = 1.0;",
    conditional: "if (nbr < row) { v = weight; } else { v = 0.0; }",
    commented: "v = weight; // target, weights[arc], f32(nbr): a comment may say anything",
});

/** The value set of every u32 override the registry declares; a new u32 override name is an explicit edit here (the builder throws otherwise). TIER is 0 only until P4 adds the mid / high tiers. */
const U32_OVERRIDE_VALUES: Readonly<Record<string, readonly number[] | undefined>> = Object.freeze({
    OP: [0, 1, 2],
    DTYPE: [0, 1, 2],
    TIER: [0],
    SWING_MODE: [0, 1],
    GRAVITY_CENTER: [0, 1],
});

/**
 * The case count of each phase's kernels under the rule above, pinned by test/kernel/wgsl-compile.test.ts:
 * P1 = degree 5 + reduce 19 + fill 1 + K3 9 + K4 3; P2 = segmented-reduce 52 (4 snippets x (1 + 3 OP x 1 TIER x 4
 * pairs)); P3 = K1 1 + K2 17 + K5 3 + toScene 1.
 */
export const EXPECTED_CASES_BY_PHASE: Readonly<Record<"P1" | "P2" | "P3", number>> = Object.freeze({
    P1: 37,
    P2: 52,
    P3: 22,
});

/** The standard overrides the composer fills from the device; never part of a variant's identity. */
const DEVICE_DERIVED: ReadonlySet<string> = new Set(["WG", "SUBGROUP_MIN", "SUBGROUP_MAX"]);

/**
 * The registry entry of an id (a throw instead of an undefined read: the registry may be typed Partial while P3
 * entries are pending).
 * @param id - the kernel id
 * @returns the entry
 */
export function entryOf(id: KernelId): KernelEntry {
    const entry: KernelEntry | undefined = KERNELS[id];
    if (entry === undefined) {
        throw new Error(`override-matrix: no KERNELS entry for ${id}`);
    }
    return entry;
}

interface Axis {
    readonly name: string;
    readonly values: readonly (number | boolean)[];
}

/**
 * The override axes of an entry: the standard pair for graph kernels, then every declared override with its value
 * set.
 * @param entry - the registry entry
 * @returns the axes in declaration order
 */
function axesOf(entry: KernelEntry): Axis[] {
    const axes: Axis[] = [];
    if (entry.bindings.some((b) => b.group === 0)) {
        axes.push({ name: "USE_PERM", values: [false, true] }, { name: "HAS_WEIGHTS", values: [false, true] });
    }
    for (const decl of entry.overrideDecls) {
        if (decl.type === "bool") {
            axes.push({ name: decl.name, values: [false, true] });
        } else if (decl.type === "u32") {
            const values = U32_OVERRIDE_VALUES[decl.name];
            if (values === undefined) {
                throw new Error(
                    `override-matrix: no value set for the u32 override ${decl.name} of ${entry.id}; add it to U32_OVERRIDE_VALUES`,
                );
            }
            axes.push({ name: decl.name, values });
        } else {
            axes.push({ name: decl.name, values: [decl.default] });
        }
    }
    return axes;
}

/**
 * The cartesian product of the axes (one empty record for no axes).
 * @param axes - the axes
 * @returns every combination
 */
function product(axes: readonly Axis[]): Record<string, number | boolean>[] {
    let combos: Record<string, number | boolean>[] = [{}];
    for (const axis of axes) {
        const next: Record<string, number | boolean>[] = [];
        for (const combo of combos) {
            for (const value of axis.values) {
                next.push({ ...combo, [axis.name]: value });
            }
        }
        combos = next;
    }
    return combos;
}

/**
 * The snippet records an entry is compiled with: none for entries without slots; every VALUE snippet of
 * SEGMENTED_REDUCE_SNIPPETS for segmented-reduce.
 * @param entry - the registry entry
 * @returns the snippet records (undefined = no snippets)
 */
function snippetSetsOf(entry: KernelEntry): (Readonly<Record<string, string>> | undefined)[] {
    if (entry.snippetSlots.length === 0) {
        return [undefined];
    }
    if (entry.id === "segmented-reduce") {
        return Object.values(SEGMENTED_REDUCE_SNIPPETS).map((snippet) => ({ VALUE: snippet }));
    }
    throw new Error(`override-matrix: no snippet set for ${entry.id} (slots ${entry.snippetSlots.join(", ")})`);
}

/**
 * JSON with sorted keys ("" for undefined).
 * @param record - the record
 * @returns the stable text
 */
function stableJson(record: Readonly<Record<string, number | boolean | string>> | undefined): string {
    if (record === undefined) {
        return "";
    }
    const names = Object.keys(record).sort();
    return JSON.stringify(Object.fromEntries(names.map((name) => [name, record[name]])));
}

/**
 * The cases of one entry: per snippet record, the defaults then the product, duplicates dropped.
 * @param entry - the registry entry
 * @returns the cases
 */
function casesOf(entry: KernelEntry): OverrideCase[] {
    const twin = entry.needs.includes("subgroups");
    const seen = new Set<string>();
    const cases: OverrideCase[] = [];
    for (const snippets of snippetSetsOf(entry)) {
        for (const overrides of [{}, ...product(axesOf(entry))]) {
            const signature = `${stableJson(overrides)}|${stableJson(snippets)}`;
            if (seen.has(signature)) {
                continue;
            }
            seen.add(signature);
            cases.push(Object.freeze({ id: entry.id, overrides: Object.freeze(overrides), snippets, twin }));
        }
    }
    return cases;
}

/** The bounded, explicit table: every entry's defaults, each override toggled alone, and the exact combinations the factories emit. */
export const OVERRIDE_MATRIX: readonly OverrideCase[] = Object.freeze(
    Object.values(KERNELS).flatMap((entry) => casesOf(entry)),
);

/**
 * Whether an id names a registry entry.
 * @param id - the first segment of a pipeline key
 * @returns true for a KernelId
 */
function isKernelId(id: string): id is KernelId {
    return Object.prototype.hasOwnProperty.call(KERNELS, id);
}

/**
 * The caps with and without the subgroups feature (the twin axis of the key's `needs` segment).
 * @param caps - the base caps
 * @returns the two variants
 */
function twinCapsOf(caps: PlanCaps): readonly PlanCaps[] {
    const present = new Set<string>(caps.features);
    present.add("subgroups");
    const absent = new Set<string>([...caps.features].filter((feature) => feature !== "subgroups"));
    return [
        { ...caps, features: present },
        { ...caps, features: absent },
    ];
}

/**
 * The key with its override record canonicalised: a declared override absent -> the entry default, USE_PERM /
 * HAS_WEIGHTS absent -> false, WG / SUBGROUP_MIN / SUBGROUP_MAX dropped, every other name kept verbatim, names
 * sorted. A key of an id outside the registry is returned unchanged.
 * @param key - a PipelineCache.key string (`id|overrides|needs|hash`)
 * @returns the canonical key
 */
export function canonicalKey(key: string): string {
    const parts = key.split("|");
    if (parts.length !== 4) {
        throw new Error(`override-matrix: a pipeline key is not id|overrides|needs|hash: ${key}`);
    }
    const [id, overridesJson, needs, hash] = parts;
    if (!isKernelId(id)) {
        return key;
    }
    const given = JSON.parse(overridesJson) as Partial<Record<string, number | boolean>>;
    const canonical: Record<string, number | boolean> = {};
    for (const [name, value] of Object.entries(given)) {
        if (value !== undefined && !DEVICE_DERIVED.has(name)) {
            canonical[name] = value;
        }
    }
    for (const decl of entryOf(id).overrideDecls) {
        canonical[decl.name] = given[decl.name] ?? decl.default;
    }
    canonical.USE_PERM = given.USE_PERM ?? false;
    canonical.HAS_WEIGHTS = given.HAS_WEIGHTS ?? false;
    return `${id}|${stableJson(canonical)}|${needs}|${hash}`;
}

/**
 * True when every key in `keys` (PipelineCache.key strings) is produced by some case of the matrix on `caps` --
 * for either twin (the caps are evaluated with and without the subgroups feature); keys of ids outside the registry
 * are ignored.
 * @param keys - the keys a suite created
 * @param caps - the caps the keys are computed against
 * @returns ok and the uncovered keys
 */
export function matrixCovers(
    keys: Iterable<string>,
    caps: PlanCaps,
): { readonly ok: boolean; readonly missing: readonly string[] } {
    const expected = new Set<string>();
    for (const variant of twinCapsOf(caps)) {
        for (const c of OVERRIDE_MATRIX) {
            expected.add(canonicalKey(pipelineKey(kernelSpec(c.id, c.overrides, c.snippets), variant)));
        }
    }
    const missing: string[] = [];
    for (const key of keys) {
        if (!isKernelId(key.split("|")[0])) {
            continue;
        }
        if (!expected.has(canonicalKey(key))) {
            missing.push(key);
        }
    }
    return { ok: missing.length === 0, missing };
}
