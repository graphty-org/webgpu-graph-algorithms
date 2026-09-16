/**
 * The kernel registry (contract 3.10, 3.11, 5.5; spec 3.5): every entry equals its 3.10.1 row (bindings, override
 * declarations, blocks, needs, phase, storage count), every body carries exactly one @compute entry point and no
 * `@group(` / `override ` line (asserted on the strings, independent of the composer), every ALL_CAPS identifier
 * a body reads is a prelude constant, a standard override or a declared override, kernelSpec /
 * setKernelBodyOverride behave, the generated blocks have the byte layout of 3.10.2, and graphBindings /
 * graphOverrides apply the dummy rules. No device is touched (the setup file still loads Dawn once per worker).
 * The table below holds all ten ids of P1-P3: entries present in KERNELS are checked against it, and the five P1
 * ids must be present, so P2-T2 and P3-T2 extend the registry without editing this file.
 */

import { PARTIAL_BYTES, STATE_HEADER_BYTES, TRACE_RECORD_BYTES } from "../../src/constants.js";
import { hasErrorCode, WebGpuGraphError } from "../../src/errors.js";
import { PRELUDE_WGSL } from "../../src/kernel/prelude.js";
import { type UniformBlock } from "../../src/kernel/struct-block.js";
import { composeWgsl, entryPointOf, STANDARD_OVERRIDES } from "../../src/kernel/wgsl.js";
import {
    FA2_PARAMS,
    FA2_PARTIAL,
    FA2_STATE,
    FA2_TRACE,
    FILL_PARAMS,
    graphBindings,
    graphOverrides,
    type KernelEntry,
    type KernelId,
    KERNELS,
    kernelSpec,
    RANGE_PARAMS,
    REDUCE_PARAMS,
    setKernelBodyOverride,
} from "../../src/kernels.js";
import { type CoreBinding } from "../../src/memory/residency.js";
import { type Binding } from "../../src/types/memory.js";
import { CAPS_SPEC_DEFAULT, CAPS_TABLES } from "../helpers/caps-tables.js";

// ============================================================ the 3.10.1 table

type Kind = "storage" | "storage-ro" | "uniform";
type BindingRow = readonly [group: number, binding: number, name: string, kind: Kind, wgslType: string];
type OverrideRow = readonly [name: string, type: "u32" | "bool" | "f32", value: number | boolean];

interface ExpectedEntry {
    readonly entryPoint: string;
    readonly bindings: readonly BindingRow[];
    readonly overrideDecls: readonly OverrideRow[];
    readonly uniforms: readonly UniformBlock[];
    readonly needs: readonly "subgroups"[];
    readonly snippetSlots: readonly string[];
    readonly phase: "P1" | "P2" | "P3";
    /** Storage-buffer count per stage (3.10.1: "degree 5, reduce 2, fill 1, segmented-reduce 5, K1 3, K2 6, K3 6, K4 3, K5 6, toScene 2"). */
    readonly storageCount: number;
}

const GRAPH: readonly BindingRow[] = [
    [0, 0, "rowPtr", "storage-ro", "array<u32>"],
    [0, 1, "colIdx", "storage-ro", "array<u32>"],
    [0, 2, "weights", "storage-ro", "array<f32>"],
    [0, 3, "perm", "storage-ro", "array<u32>"],
];

/** The four graph slots followed by a kernel's own rows (the parameter type keeps the literal rows tuples). */
function withGraph(rows: readonly BindingRow[]): readonly BindingRow[] {
    return GRAPH.concat(rows);
}

const TABLE: Readonly<Record<KernelId, ExpectedEntry>> = {
    degree: {
        entryPoint: "degree",
        bindings: withGraph([
            [1, 0, "out", "storage", "array<u32>"],
            [2, 0, "P", "uniform", "RangeParams"],
        ]),
        overrideDecls: [],
        uniforms: [RANGE_PARAMS],
        needs: [],
        snippetSlots: [],
        phase: "P1",
        storageCount: 5,
    },
    reduce: {
        entryPoint: "reduce",
        bindings: [
            [1, 0, "src", "storage-ro", "array<u32>"],
            [1, 1, "out", "storage", "array<u32>"],
            [2, 0, "P", "uniform", "ReduceParams"],
        ],
        overrideDecls: [
            ["OP", "u32", 0],
            ["DTYPE", "u32", 0],
            ["FINAL", "bool", false],
        ],
        uniforms: [REDUCE_PARAMS],
        needs: ["subgroups"],
        snippetSlots: [],
        phase: "P1",
        storageCount: 2,
    },
    fill: {
        entryPoint: "fill",
        bindings: [
            [1, 0, "dst", "storage", "array<u32>"],
            [2, 0, "P", "uniform", "FillParams"],
        ],
        overrideDecls: [],
        uniforms: [FILL_PARAMS],
        needs: [],
        snippetSlots: [],
        phase: "P1",
        storageCount: 1,
    },
    "segmented-reduce": {
        entryPoint: "segmented_reduce",
        bindings: withGraph([
            [1, 0, "out", "storage", "array<f32>"],
            [2, 0, "P", "uniform", "RangeParams"],
        ]),
        overrideDecls: [
            ["OP", "u32", 0],
            ["TIER", "u32", 0],
        ],
        uniforms: [RANGE_PARAMS],
        needs: [],
        snippetSlots: ["VALUE"],
        phase: "P2",
        storageCount: 5,
    },
    "fa2-stats-finalize": {
        entryPoint: "stats_finalize",
        bindings: [
            [1, 0, "partials", "storage-ro", "array<Fa2Partial>"],
            [1, 1, "S", "storage", "Fa2State"],
            [1, 2, "T", "storage", "array<Fa2Trace>"],
            [2, 0, "P", "uniform", "Fa2Params"],
        ],
        overrideDecls: [],
        uniforms: [FA2_PARAMS, FA2_STATE, FA2_TRACE, FA2_PARTIAL],
        needs: ["subgroups"],
        snippetSlots: [],
        phase: "P3",
        storageCount: 3,
    },
    "fa2-attraction": {
        entryPoint: "attraction",
        bindings: withGraph([
            [1, 0, "pos", "storage-ro", "array<vec4f>"],
            [1, 1, "force", "storage", "array<f32>"],
            [2, 0, "P", "uniform", "Fa2Params"],
        ]),
        overrideDecls: [
            ["LINLOG", "bool", false],
            ["DISTRIBUTED", "bool", false],
            ["TIER", "u32", 0],
        ],
        uniforms: [FA2_PARAMS],
        needs: [],
        snippetSlots: [],
        phase: "P3",
        storageCount: 6,
    },
    "fa2-repulsion-exact": {
        entryPoint: "repulsion",
        bindings: [
            [1, 0, "pos", "storage-ro", "array<vec4f>"],
            [1, 1, "S", "storage", "Fa2State"],
            [1, 2, "force", "storage", "array<f32>"],
            [1, 3, "oldForce", "storage-ro", "array<f32>"],
            [1, 4, "fixedMask", "storage-ro", "array<u32>"],
            [1, 5, "partials", "storage", "array<Fa2Partial>"],
            [2, 0, "P", "uniform", "Fa2Params"],
        ],
        overrideDecls: [
            ["SWING_MODE", "u32", 0],
            ["STRONG_GRAVITY", "bool", false],
            ["GRAVITY_CENTER", "u32", 0],
        ],
        uniforms: [FA2_PARAMS, FA2_STATE, FA2_PARTIAL],
        needs: ["subgroups"],
        snippetSlots: [],
        phase: "P1",
        storageCount: 6,
    },
    "fa2-speed-finalize": {
        entryPoint: "speed_finalize",
        bindings: [
            [1, 0, "partials", "storage-ro", "array<Fa2Partial>"],
            [1, 1, "S", "storage", "Fa2State"],
            [1, 2, "T", "storage", "array<Fa2Trace>"],
            [2, 0, "P", "uniform", "Fa2Params"],
        ],
        overrideDecls: [["SWING_MODE", "u32", 0]],
        uniforms: [FA2_PARAMS, FA2_STATE, FA2_TRACE, FA2_PARTIAL],
        needs: ["subgroups"],
        snippetSlots: [],
        phase: "P1",
        storageCount: 3,
    },
    "fa2-integrate": {
        entryPoint: "integrate",
        bindings: [
            [1, 0, "force", "storage-ro", "array<f32>"],
            [1, 1, "oldForce", "storage", "array<f32>"],
            [1, 2, "fixedMask", "storage-ro", "array<u32>"],
            [1, 3, "S", "storage", "Fa2State"],
            [1, 4, "pos", "storage", "array<vec4f>"],
            [1, 5, "partials", "storage", "array<Fa2Partial>"],
            [2, 0, "P", "uniform", "Fa2Params"],
        ],
        overrideDecls: [["SWING_MODE", "u32", 0]],
        uniforms: [FA2_PARAMS, FA2_STATE, FA2_PARTIAL],
        needs: ["subgroups"],
        snippetSlots: [],
        phase: "P3",
        storageCount: 6,
    },
    "fa2-to-scene": {
        entryPoint: "to_scene",
        bindings: [
            [1, 0, "pos", "storage-ro", "array<vec4f>"],
            [1, 1, "scene", "storage", "array<f32>"],
            [2, 0, "P", "uniform", "Fa2Params"],
        ],
        overrideDecls: [],
        uniforms: [FA2_PARAMS],
        needs: [],
        snippetSlots: [],
        phase: "P3",
        storageCount: 2,
    },
};

const P1_IDS: readonly KernelId[] = ["degree", "reduce", "fill", "fa2-repulsion-exact", "fa2-speed-finalize"];

/** The access text the composer emits per kind (contract 4.2). */
const ACCESS: Readonly<Record<Kind, string>> = {
    uniform: "var<uniform>",
    storage: "var<storage, read_write>",
    "storage-ro": "var<storage, read>",
};

/** The ALL_CAPS names the prelude declares (constants and the standard overrides), read from its text. */
const PRELUDE_NAMES: ReadonlySet<string> = new Set(
    Array.from(PRELUDE_WGSL.matchAll(/^(?:const|override) ([A-Z][A-Z0-9_]*):/gm), (m) => m[1]),
);

/** Snippet texts that let a body with marker slots compose with its defaults (only VALUE exists in P0-P3). */
const DEFAULT_SNIPPETS: Readonly<Record<string, string>> = { VALUE: "v = 1.0;" };

function defaultSnippets(entry: KernelEntry): Readonly<Record<string, string>> | undefined {
    if (entry.snippetSlots.length === 0) {
        return undefined;
    }
    const snippets: Record<string, string> = {};
    for (const slot of entry.snippetSlots) {
        const text = DEFAULT_SNIPPETS[slot];
        if (text === undefined) {
            throw new Error(`no default snippet for slot ${slot} of ${entry.id}; extend DEFAULT_SNIPPETS`);
        }
        snippets[slot] = text;
    }
    return snippets;
}

/** Every ALL_CAPS identifier (two or more characters) a body reads, comments stripped. */
function capsIdentifiers(body: string): ReadonlySet<string> {
    const code = body.replace(/\/\/.*$/gm, "");
    return new Set(Array.from(code.matchAll(/\b[A-Z][A-Z0-9_]+\b/g), (m) => m[0]));
}

/** The struct name a wgslType refers to (`Fa2State`, `array<Fa2Partial>` -> `Fa2Partial`), or null for scalars / vectors. */
function structOf(wgslType: string): string | null {
    const inner = wgslType.startsWith("array<") ? wgslType.slice("array<".length, -1) : wgslType;
    return /^[A-Z]/.test(inner) ? inner : null;
}

function caught(fn: () => unknown): unknown {
    try {
        fn();
    } catch (err) {
        return err;
    }
    return undefined;
}

function fakeBinding(label: string): Binding {
    return { buffer: { label } as unknown as GPUBuffer, offset: 0, size: 256, window: null };
}

function fakeCore(options: { readonly colIdx: boolean; readonly weights: boolean }): CoreBinding {
    return {
        serial: 1,
        plan: "arena",
        rowPtr: fakeBinding("rowPtr"),
        colIdx: options.colIdx ? fakeBinding("colIdx") : null,
        weights: options.weights ? fakeBinding("weights") : null,
        arcToEdge: null,
        edgeToArc: null,
        windows: null,
        hasWeights: options.weights,
    };
}

const present = Object.keys(KERNELS) as KernelId[];

// ============================================================ the registry

describe("KERNELS (contract 3.10)", () => {
    it("holds every P1 entry under its own id, frozen, and nothing the 3.10.1 table does not name", () => {
        for (const id of P1_IDS) {
            expect(present, `P1 entry ${id}`).toContain(id);
        }
        for (const id of present) {
            expect(KERNELS[id].id).toBe(id);
            expect(TABLE[id], `3.10.1 row for ${id}`).toBeDefined();
        }
        expect(new Set(present.map((id) => KERNELS[id].id)).size).toBe(present.length);
        expect(Object.isFrozen(KERNELS)).toBe(true);
    });

    it("the standard overrides of wgsl.ts and the prelude constants the bodies rely on are declared by the prelude", () => {
        for (const name of STANDARD_OVERRIDES) {
            expect(PRELUDE_NAMES.has(name), name).toBe(true);
        }
        expect(PRELUDE_NAMES.has("INVALID_INDEX")).toBe(true);
        expect(PRELUDE_NAMES.has("U32_MAX")).toBe(true);
        expect(PRELUDE_NAMES.has("F32_MAX")).toBe(true);
    });

    for (const id of present) {
        const entry: KernelEntry = KERNELS[id];
        const expected = TABLE[id];

        describe(id, () => {
            it("matches its 3.10.1 row: bindings, override declarations, blocks, needs, snippet slots, phase, entry point", () => {
                expect(entry.bindings.map((b) => [b.group, b.binding, b.name, b.kind, b.wgslType])).toEqual(
                    expected.bindings.map((r) => Array.from(r)),
                );
                expect(entry.overrideDecls.map((o) => [o.name, o.type, o.default])).toEqual(
                    expected.overrideDecls.map((r) => Array.from(r)),
                );
                expect(entry.uniforms.length).toBe(expected.uniforms.length);
                entry.uniforms.forEach((block, k) => {
                    expect(block, `uniforms[${k}]`).toBe(expected.uniforms[k]);
                });
                expect(entry.needs).toEqual(expected.needs);
                expect(entry.snippetSlots).toEqual(expected.snippetSlots);
                expect(entry.phase).toBe(expected.phase);
                expect(entry.entryPoint).toBe(expected.entryPoint);
            });

            it("binds at most 8 storage buffers per stage, unique names and slots, one uniform P whose struct is a listed block", () => {
                const storage = entry.bindings.filter((b) => b.kind !== "uniform");
                expect(storage.length).toBe(expected.storageCount);
                expect(storage.length).toBeLessThanOrEqual(8);
                expect(new Set(entry.bindings.map((b) => b.name)).size).toBe(entry.bindings.length);
                expect(new Set(entry.bindings.map((b) => `${b.group}.${b.binding}`)).size).toBe(entry.bindings.length);
                const uniforms = entry.bindings.filter((b) => b.kind === "uniform");
                expect(uniforms.length).toBe(1);
                expect(uniforms[0].group).toBe(2);
                expect(uniforms[0].binding).toBe(0);
                expect(uniforms[0].name).toBe("P");
                const blockNames = entry.uniforms.map((block) => block.name);
                expect(blockNames).toContain(uniforms[0].wgslType);
                for (const b of entry.bindings) {
                    const struct = structOf(b.wgslType);
                    if (struct !== null) {
                        expect(blockNames, `${b.name}: ${b.wgslType}`).toContain(struct);
                    }
                }
                for (const b of entry.bindings.filter((x) => x.group === 0)) {
                    expect(b.kind, `group-0 slot ${b.name} is read-only (3.10.1 dummy rules)`).toBe("storage-ro");
                }
            });

            it("body: one @compute entry point named entryPoint, no @group( / @binding( / override line, plain ASCII, no forbidden literal", () => {
                expect(entry.body.split("@compute").length - 1).toBe(1);
                expect(entryPointOf(entry.body)).toBe(entry.entryPoint);
                expect(entry.body).toContain(`fn ${entry.entryPoint}(`);
                expect(entry.body).not.toContain("@group(");
                expect(entry.body).not.toContain("@binding(");
                expect(entry.body).not.toContain("override ");
                for (let k = 0; k < entry.body.length; k++) {
                    expect(entry.body.charCodeAt(k), `byte ${k} of the ${id} body`).toBeLessThan(128);
                }
                expect(entry.body).not.toMatch(/65535u|256u|0xFFFFFFFFu/);
            });

            it("every ALL_CAPS identifier the body reads is a prelude constant, a standard override or a declared override", () => {
                const declared = new Set(entry.overrideDecls.map((o) => o.name));
                const standard = STANDARD_OVERRIDES as readonly string[];
                for (const name of capsIdentifiers(entry.body)) {
                    const known = PRELUDE_NAMES.has(name) || standard.includes(name) || declared.has(name);
                    expect(known, `${id} reads ${name}, which is neither a prelude name nor a declared override`).toBe(
                        true,
                    );
                }
                for (const decl of entry.overrideDecls) {
                    expect(typeof decl.default).toBe(decl.type === "bool" ? "boolean" : "number");
                    expect(standard.includes(decl.name), `${decl.name} re-declares a standard override`).toBe(false);
                }
            });

            it("composes on every caps table with the emitted bind lines of 4.2 (the subgroup block iff needs and features agree)", () => {
                const twin = entry.needs.includes("subgroups");
                for (const { name, caps } of CAPS_TABLES) {
                    const composed = composeWgsl(kernelSpec(id, undefined, defaultSnippets(entry)), caps);
                    expect(composed.id, name).toBe(id);
                    expect(composed.entryPoint, name).toBe(entry.entryPoint);
                    expect(composed.subgroups, name).toBe(twin && caps.features.has("subgroups"));
                    expect(composed.code.includes("enable subgroups;"), name).toBe(composed.subgroups);
                    expect(composed.code, name).toContain(`fn ${entry.entryPoint}(`);
                    for (const b of entry.bindings) {
                        expect(composed.code, `${name}: ${b.name}`).toContain(
                            `@group(${b.group}) @binding(${b.binding}) ${ACCESS[b.kind]} ${b.name}: ${b.wgslType};`,
                        );
                    }
                    for (const block of entry.uniforms) {
                        expect(composed.code, `${name}: ${block.name}`).toContain(`struct ${block.name}`);
                    }
                    expect(composed.code).not.toContain("//@@");
                }
            });
        });
    }
});

// ============================================================ kernelSpec / setKernelBodyOverride

describe("kernelSpec / setKernelBodyOverride (contract 3.10)", () => {
    it("returns the entry's parts with the overrides and snippets given", () => {
        const spec = kernelSpec("reduce", { OP: 1, DTYPE: 2, FINAL: true });
        expect(spec.id).toBe("reduce");
        expect(spec.body).toBe(KERNELS.reduce.body);
        expect(spec.bindings).toBe(KERNELS.reduce.bindings);
        expect(spec.overrideDecls).toBe(KERNELS.reduce.overrideDecls);
        expect(spec.uniforms).toBe(KERNELS.reduce.uniforms);
        expect(spec.needs).toBe(KERNELS.reduce.needs);
        expect(spec.overrides).toEqual({ OP: 1, DTYPE: 2, FINAL: true });
        expect(spec.snippets).toBeUndefined();
        const plain = kernelSpec("degree");
        expect(plain.overrides).toEqual({});
        expect(plain.snippets).toBeUndefined();
        const withSnippet = kernelSpec("fill", undefined, { VALUE: "v = 1.0;" });
        expect(withSnippet.snippets).toEqual({ VALUE: "v = 1.0;" });
    });

    it("leaves override and snippet validation to the composer, which rejects unknown names at compose time", () => {
        const bad = caught(() => composeWgsl(kernelSpec("reduce", { NOPE: 1 }), CAPS_SPEC_DEFAULT));
        expect(bad).toBeInstanceOf(WebGpuGraphError);
        expect(hasErrorCode(bad, "E_SHADER_COMPILE")).toBe(true);
        expect((bad as WebGpuGraphError).details).toMatchObject({ id: "reduce", stage: "compose" });
        const orphan = caught(() =>
            composeWgsl(kernelSpec("fill", undefined, { VALUE: "v = 1.0;" }), CAPS_SPEC_DEFAULT),
        );
        expect(hasErrorCode(orphan, "E_SHADER_COMPILE")).toBe(true);
        expect((orphan as WebGpuGraphError).details).toMatchObject({ id: "fill", stage: "compose" });
        const { code } = composeWgsl(kernelSpec("degree", { USE_PERM: true, HAS_WEIGHTS: true }), CAPS_SPEC_DEFAULT);
        expect(code).toContain("fn degree(");
        const withDecls = composeWgsl(kernelSpec("reduce", { OP: 2, DTYPE: 1, FINAL: true }), CAPS_SPEC_DEFAULT);
        expect(withDecls.overrides).toMatchObject({ OP: 2, DTYPE: 1, FINAL: true });
    });

    it("rejects an id that is not registered with E_INVALID_ARGUMENT naming argument id", () => {
        const unknown = "nope" as unknown as KernelId;
        const err = caught(() => kernelSpec(unknown));
        expect(err).toBeInstanceOf(WebGpuGraphError);
        expect(hasErrorCode(err, "E_INVALID_ARGUMENT")).toBe(true);
        expect((err as WebGpuGraphError).details).toMatchObject({ argument: "id", value: "nope" });
        expect((err as WebGpuGraphError).details.expected).toEqual(present);
        expect(
            hasErrorCode(
                caught(() => setKernelBodyOverride(unknown, "x")),
                "E_INVALID_ARGUMENT",
            ),
        ).toBe(true);
    });

    it("setKernelBodyOverride changes the body kernelSpec sees, leaves the entry untouched, and null restores", () => {
        const normative = KERNELS.degree.body;
        try {
            setKernelBodyOverride("degree", "// sabotaged");
            expect(kernelSpec("degree").body).toBe("// sabotaged");
            expect(KERNELS.degree.body).toBe(normative);
            expect(kernelSpec("fill").body).toBe(KERNELS.fill.body);
        } finally {
            setKernelBodyOverride("degree", null);
        }
        expect(kernelSpec("degree").body).toBe(normative);
        setKernelBodyOverride("degree", null);
        expect(kernelSpec("degree").body).toBe(normative);
    });
});

// ============================================================ the generated blocks (3.10.2)

interface BlockRow {
    readonly block: UniformBlock;
    readonly name: string;
    readonly layout: "uniform" | "storage";
    readonly byteLength: number;
    readonly offsets: readonly (readonly [field: string, offset: number])[];
}

const BLOCKS: readonly BlockRow[] = [
    {
        block: RANGE_PARAMS,
        name: "RangeParams",
        layout: "uniform",
        byteLength: 32,
        offsets: [
            ["start", 0],
            ["end", 4],
            ["arcBase", 8],
            ["arcEnd", 12],
            ["accumulate", 16],
            ["n", 20],
            ["pad0", 24],
            ["pad1", 28],
        ],
    },
    {
        block: REDUCE_PARAMS,
        name: "ReduceParams",
        layout: "uniform",
        byteLength: 16,
        offsets: [
            ["count", 0],
            ["outOffset", 4],
            ["level", 8],
            ["pad0", 12],
        ],
    },
    {
        block: FILL_PARAMS,
        name: "FillParams",
        layout: "uniform",
        byteLength: 16,
        offsets: [
            ["count", 0],
            ["value", 4],
            ["mode", 8],
            ["pad0", 12],
        ],
    },
    {
        block: FA2_PARAMS,
        name: "Fa2Params",
        layout: "uniform",
        byteLength: 96,
        offsets: [
            ["n", 0],
            ["dim", 4],
            ["flags", 8],
            ["tierStart", 12],
            ["tierEnd", 16],
            ["iterationIndex", 20],
            ["seed", 24],
            ["nearMax", 28],
            ["scalingRatio", 32],
            ["gravity", 36],
            ["jitterTolerance", 40],
            ["scale", 44],
            ["center", 48],
            ["settleThreshold", 64],
            ["extentFactor", 68],
            ["gridMax", 72],
            ["levels", 76],
            ["pad", 80],
        ],
    },
    {
        block: FA2_STATE,
        name: "Fa2State",
        layout: "storage",
        byteLength: 256,
        offsets: [
            ["speed", 0],
            ["speedEfficiency", 4],
            ["swing", 8],
            ["traction", 12],
            ["centroid", 16],
            ["rmsRadius", 32],
            ["radius", 36],
            ["meanDisplacement", 40],
            ["iteration", 44],
            ["min", 48],
            ["max", 64],
            ["gridMin", 80],
            ["eps", 96],
            ["settledCount", 100],
            ["outsideGrid", 104],
            ["maxCellOccupancy", 108],
            ["reserved0", 112],
            ["reserved1", 128],
            ["reserved2", 144],
            ["reserved3", 160],
            ["reserved4", 176],
            ["reserved5", 192],
            ["reserved6", 208],
            ["reserved7", 224],
            ["reserved8", 240],
        ],
    },
    {
        block: FA2_TRACE,
        name: "Fa2Trace",
        layout: "storage",
        byteLength: 32,
        offsets: [
            ["swing", 0],
            ["traction", 4],
            ["speed", 8],
            ["speedEfficiency", 12],
            ["meanDisplacement", 16],
            ["settledCount", 20],
            ["iteration", 24],
            ["pad0", 28],
        ],
    },
    {
        block: FA2_PARTIAL,
        name: "Fa2Partial",
        layout: "storage",
        byteLength: 64,
        offsets: [
            ["sum", 0],
            ["min", 16],
            ["max", 32],
            ["swingTraction", 48],
            ["dispFree", 56],
        ],
    },
];

describe("the generated blocks (contract 3.10.2)", () => {
    for (const row of BLOCKS) {
        it(`${row.name}: ${row.layout}, ${row.byteLength} bytes, every field at its pinned offset`, () => {
            expect(row.block.name).toBe(row.name);
            expect(row.block.layout).toBe(row.layout);
            expect(row.block.byteLength).toBe(row.byteLength);
            expect(row.block.fields.map((f) => f[0])).toEqual(row.offsets.map((o) => o[0]));
            for (const [field, offset] of row.offsets) {
                expect(row.block.offsetOf(field), `${row.name}.${field}`).toBe(offset);
            }
            expect(row.block.wgsl).toContain(`struct ${row.name}`);
        });
    }

    it("the state header, the trace record and the partial record have the sizes constants.ts pins", () => {
        expect(FA2_STATE.byteLength).toBe(STATE_HEADER_BYTES);
        expect(FA2_TRACE.byteLength).toBe(TRACE_RECORD_BYTES);
        expect(FA2_PARTIAL.byteLength).toBe(PARTIAL_BYTES);
    });

    it("Fa2State round-trips through its writer and reader at the pinned offsets", () => {
        const view = new DataView(new ArrayBuffer(FA2_STATE.byteLength));
        FA2_STATE.write(view, {
            speed: 1.5,
            speedEfficiency: 0.25,
            centroid: [1, 2, 3, 0],
            iteration: 7,
            max: [4, 5, 6, 49],
            settledCount: 2,
        });
        expect(view.getFloat32(0, true)).toBe(1.5);
        expect(view.getFloat32(4, true)).toBe(0.25);
        expect(view.getFloat32(16, true)).toBe(1);
        expect(view.getFloat32(24, true)).toBe(3);
        expect(view.getUint32(44, true)).toBe(7);
        expect(view.getFloat32(76, true)).toBe(49);
        expect(view.getUint32(100, true)).toBe(2);
        const back = FA2_STATE.read(view);
        expect(back.speed).toBe(1.5);
        expect(back.centroid).toEqual([1, 2, 3, 0]);
        expect(back.iteration).toBe(7);
        expect(back.swing).toBe(0);
        expect(FA2_STATE.readField(view, "settledCount")).toBe(2);
    });
});

// ============================================================ graphBindings / graphOverrides

describe("graphBindings / graphOverrides (contract 3.10; spec 3.5, 4.1 dummy rules)", () => {
    it("a weighted core with no permutation: real colIdx and weights, rowPtr as the perm dummy", () => {
        const core = fakeCore({ colIdx: true, weights: true });
        const b = graphBindings(core, null);
        expect(b.rowPtr).toBe(core.rowPtr);
        expect(b.colIdx).toBe(core.colIdx);
        expect(b.weights).toBe(core.weights);
        expect(b.perm).toBe(core.rowPtr);
        expect(graphOverrides(core, null)).toEqual({ USE_PERM: false, HAS_WEIGHTS: true });
    });

    it("an unweighted core: colIdx fills the weights slot and HAS_WEIGHTS is false", () => {
        const core = fakeCore({ colIdx: true, weights: false });
        const b = graphBindings(core, null);
        expect(b.colIdx).toBe(core.colIdx);
        expect(b.weights).toBe(core.colIdx);
        expect(b.perm).toBe(core.rowPtr);
        expect(graphOverrides(core, null)).toEqual({ USE_PERM: false, HAS_WEIGHTS: false });
    });

    it("an arcCount === 0 core: rowPtr fills every slot and both overrides are false", () => {
        const core = fakeCore({ colIdx: false, weights: false });
        const b = graphBindings(core, null);
        expect(b.rowPtr).toBe(core.rowPtr);
        expect(b.colIdx).toBe(core.rowPtr);
        expect(b.weights).toBe(core.rowPtr);
        expect(b.perm).toBe(core.rowPtr);
        expect(graphOverrides(core, null)).toEqual({ USE_PERM: false, HAS_WEIGHTS: false });
    });

    it("a permutation binding fills the perm slot and sets USE_PERM", () => {
        const core = fakeCore({ colIdx: true, weights: false });
        const perm = fakeBinding("degreeOrder");
        expect(graphBindings(core, perm).perm).toBe(perm);
        expect(graphOverrides(core, perm)).toEqual({ USE_PERM: true, HAS_WEIGHTS: false });
    });

    it("an explicit weights argument overrides the core's: null on a weighted core -> the colIdx dummy and HAS_WEIGHTS false; a binding on an unweighted core -> HAS_WEIGHTS true", () => {
        const weighted = fakeCore({ colIdx: true, weights: true });
        expect(graphBindings(weighted, null, null).weights).toBe(weighted.colIdx);
        expect(graphOverrides(weighted, null, null)).toEqual({ USE_PERM: false, HAS_WEIGHTS: false });
        const unweighted = fakeCore({ colIdx: true, weights: false });
        const column = fakeBinding("edge-column");
        expect(graphBindings(unweighted, null, column).weights).toBe(column);
        expect(graphOverrides(unweighted, null, column)).toEqual({ USE_PERM: false, HAS_WEIGHTS: true });
        expect(graphBindings(weighted, null, undefined).weights).toBe(weighted.weights);
        expect(graphOverrides(weighted, null, undefined)).toEqual({ USE_PERM: false, HAS_WEIGHTS: true });
    });
});
