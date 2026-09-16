/**
 * The WGSL composer and prelude (spec 3.5; contract 3.9, 4.1-4.3; 5.5 row wgsl.test.ts): the exact emitted format of
 * 4.2 for a two-binding spec, every compose-time E_SHADER_COMPILE { stage: "compose" } (unknown override, snippet
 * without marker, marker without snippet, `@group(` / `override ` in a body, a reserved identifier while a comment
 * passes), the subgroup splice iff needs and caps agree with SUBGROUP_MIN / SUBGROUP_MAX filled, and the literal
 * grep over src/wgsl/** and src/kernel/prelude.ts. Pure: no device.
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { INVALID_INDEX } from "@graphty/graph-format";

import { WORKGROUP_SIZE } from "../../src/constants.js";
import { ShaderStage } from "../../src/device/webgpu-constants.js";
import { isWebGpuGraphError, type WebGpuGraphError } from "../../src/errors.js";
import {
    PRELUDE_LINES,
    PRELUDE_WGSL,
    REDUCE_HELPER_NAMES,
    REDUCE_HELPERS_SUBGROUP_WGSL,
    REDUCE_HELPERS_WORKGROUP_WGSL,
    WGSL_RESERVED_WORDS,
    wgslF32Literal,
} from "../../src/kernel/prelude.js";
import { UniformBlock } from "../../src/kernel/struct-block.js";
import {
    bindGroupLayoutDescriptors,
    bindingNames,
    composeWgsl,
    entryPointOf,
    type OverrideDecl,
    STANDARD_OVERRIDES,
    type WgslModuleSpec,
} from "../../src/kernel/wgsl.js";
import { kernelSpec } from "../../src/kernels.js";
import { type PlanCaps } from "../../src/types/context.js";
import { CAPS_SPEC_DEFAULT, fakeCaps } from "../helpers/caps-tables.js";

const here = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(here, "../../src");

function catchError(fn: () => unknown): WebGpuGraphError {
    try {
        fn();
    } catch (err) {
        if (isWebGpuGraphError(err)) {
            return err;
        }
        throw err;
    }
    throw new Error("expected a WebGpuGraphError");
}

function expectCompose(fn: () => unknown, slot: string, id = "test-add"): WebGpuGraphError {
    const err = catchError(fn);
    expect(err.code).toBe("E_SHADER_COMPILE");
    expect(err.details).toMatchObject({ id, stage: "compose", slot });
    return err;
}

const TEST_PARAMS = UniformBlock.define("TestParams", [
    ["count", "u32"],
    ["value", "u32"],
]);
const BODY = `@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = linear_id(wid, lid.x);
    if (i >= P.count) { return; }
    dst[i] = src[i] + P.value;
}`;
const DECLS: readonly OverrideDecl[] = [
    { name: "SCALE", type: "u32", default: 2 },
    { name: "FLAG", type: "bool", default: false },
    { name: "GAIN", type: "f32", default: 1 },
];
const SPEC: WgslModuleSpec = {
    id: "test-add",
    body: BODY,
    bindings: [
        { group: 1, binding: 0, name: "src", kind: "storage-ro", wgslType: "array<u32>" },
        { group: 1, binding: 1, name: "dst", kind: "storage", wgslType: "array<u32>" },
        { group: 2, binding: 0, name: "P", kind: "uniform", wgslType: "TestParams" },
    ],
    overrideDecls: DECLS,
    overrides: {},
    needs: [],
    uniforms: [TEST_PARAMS],
};
const REDUCING_BODY = `@compute @workgroup_size(WG)
fn total(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = wg_reduce_u32(src[lid.x], lid.x, 0u);
    if (lid.x == 0u) { dst[0] = t; }
}`;
const REDUCING: WgslModuleSpec = { ...SPEC, id: "test-total", body: REDUCING_BODY, needs: ["subgroups"] };
const SUBGROUP_CAPS: PlanCaps = {
    ...fakeCaps(CAPS_SPEC_DEFAULT, {}, { subgroupMinSize: 8, subgroupMaxSize: 32 }),
    features: new Set(["subgroups"]),
};

describe("the prelude", () => {
    it("interpolates every constant from constants.ts and graph-format and declares the five standard overrides", () => {
        expect(PRELUDE_WGSL.startsWith("// ---- prelude:")).toBe(true);
        expect(PRELUDE_WGSL).toContain(`const INVALID_INDEX: u32 = ${INVALID_INDEX}u;`);
        expect(PRELUDE_WGSL).toContain("const INVALID_INDEX: u32 = 4294967295u;");
        expect(PRELUDE_WGSL).toContain("const U32_MAX: u32 = 4294967295u;");
        expect(PRELUDE_WGSL).toContain("const MAX_WORKGROUPS_PER_DIM: u32 = 65535u;");
        expect(PRELUDE_WGSL).toContain("const FA2_DIST_FLOOR: f32 = 0.01;");
        expect(PRELUDE_WGSL).toContain("const FA2_DIST_FLOOR_SQ: f32 = 0.0001;");
        expect(PRELUDE_WGSL).toContain("const FA2_COINCIDENT_SQ: f32 = 1e-8;");
        expect(PRELUDE_WGSL).toContain("const FA2_FLAG_FIRST: u32 = 1u;");
        expect(PRELUDE_WGSL).toContain("const F32_MAX: f32 = 0x1.fffffep+127;");
        expect(PRELUDE_WGSL).toContain(`override WG: u32 = ${WORKGROUP_SIZE}u;`);
        expect(PRELUDE_WGSL).toContain("override USE_PERM: bool = false;");
        expect(PRELUDE_WGSL).toContain("override HAS_WEIGHTS: bool = false;");
        expect(PRELUDE_WGSL).toContain("override SUBGROUP_MIN: u32 = 4u;");
        expect(PRELUDE_WGSL).toContain("override SUBGROUP_MAX: u32 = 0u;");
        for (const helper of [
            "linear_id",
            "group_id",
            "lowbias32",
            "mask_bit",
            "unpack_u8",
            "pair_hash",
            "hash_unit",
            "hash_dir",
            "kick_dir",
        ]) {
            expect(PRELUDE_WGSL).toContain(`fn ${helper}(`);
        }
        expect(STANDARD_OVERRIDES).toEqual(["WG", "USE_PERM", "HAS_WEIGHTS", "SUBGROUP_MIN", "SUBGROUP_MAX"]);
    });

    it("PRELUDE_LINES is the line count of a text with neither a leading nor a trailing newline; the helper blocks likewise", () => {
        expect(PRELUDE_LINES).toBe(PRELUDE_WGSL.split("\n").length);
        for (const text of [PRELUDE_WGSL, REDUCE_HELPERS_WORKGROUP_WGSL, REDUCE_HELPERS_SUBGROUP_WGSL]) {
            expect(text.startsWith("\n")).toBe(false);
            expect(text.endsWith("\n")).toBe(false);
        }
    });

    it("both helper blocks define every REDUCE_HELPER_NAMES function; only the subgroup block sizes SG_SLOTS by SUBGROUP_MIN", () => {
        expect(REDUCE_HELPER_NAMES).toEqual(["wg_reduce_f32", "wg_reduce_u32", "wg_reduce_vec4"]);
        for (const name of REDUCE_HELPER_NAMES) {
            expect(REDUCE_HELPERS_WORKGROUP_WGSL).toContain(`fn ${name}(`);
            expect(REDUCE_HELPERS_SUBGROUP_WGSL).toContain(`fn ${name}(`);
        }
        expect(REDUCE_HELPERS_SUBGROUP_WGSL).toContain(
            "override SG_SLOTS: u32 = (WG + SUBGROUP_MIN - 1u) / SUBGROUP_MIN;",
        );
        expect(REDUCE_HELPERS_SUBGROUP_WGSL).not.toContain("SUBGROUP_MAX");
        // the subgroup identity is the counter slot, never @builtin(subgroup_id) (D16; the word appears only in a comment)
        expect(REDUCE_HELPERS_SUBGROUP_WGSL.replace(/\/\/[^\n]*/g, "")).not.toContain("subgroup_id");
        expect(REDUCE_HELPERS_WORKGROUP_WGSL).not.toContain("subgroup");
    });

    it("formats f32 literals with a decimal point or an exponent", () => {
        expect(wgslF32Literal(0.01)).toBe("0.01");
        expect(wgslF32Literal(0.0001)).toBe("0.0001");
        expect(wgslF32Literal(1e-8)).toBe("1e-8");
        expect(wgslF32Literal(1)).toBe("1.0");
        expect(wgslF32Literal(-3)).toBe("-3.0");
        expect(wgslF32Literal(2.5)).toBe("2.5");
        expect(catchError(() => wgslF32Literal(Number.NaN)).code).toBe("E_INVALID_ARGUMENT");
    });

    it("WGSL_RESERVED_WORDS is the frozen spec 16.2 list: 146 words incl. target, and none of free / valid / tile / slot / key / count", () => {
        expect(Object.isFrozen(WGSL_RESERVED_WORDS)).toBe(true);
        expect(WGSL_RESERVED_WORDS.length).toBe(146);
        expect(new Set(WGSL_RESERVED_WORDS).size).toBe(146);
        for (const word of [
            "target",
            "filter",
            "partition",
            "layout",
            "common",
            "ref",
            "self",
            "type",
            "use",
            "with",
            "NULL",
            "Self",
            "yield",
        ]) {
            expect(WGSL_RESERVED_WORDS).toContain(word);
        }
        for (const word of [
            "free",
            "valid",
            "tile",
            "slot",
            "key",
            "count",
            "row",
            "nbr",
            "weight",
            "pos",
            "force",
            "state",
            "select",
            "fn",
            "let",
            "var",
        ]) {
            expect(WGSL_RESERVED_WORDS).not.toContain(word);
        }
    });

    it("no src/wgsl/** body and no prelude template types the literals 65535u, 256u or 0xFFFFFFFFu (spec 3.5)", () => {
        const wgslDir = resolve(SRC, "wgsl");
        const files = existsSync(wgslDir)
            ? readdirSync(wgslDir)
                  .filter((f) => f.endsWith(".ts"))
                  .map((f) => resolve(wgslDir, f))
            : [];
        files.push(resolve(SRC, "kernel/prelude.ts"));
        for (const file of files) {
            const text = readFileSync(file, "utf8");
            for (const literal of ["65535u", "256u", "0xFFFFFFFFu"]) {
                expect(text.includes(literal), `${file} contains ${literal}`).toBe(false);
            }
        }
    });
});

describe("composeWgsl", () => {
    it("emits the exact 4.2 format for a two-binding spec", () => {
        const composed = composeWgsl(SPEC, CAPS_SPEC_DEFAULT);
        const expected = [
            PRELUDE_WGSL,
            "override SCALE: u32 = 2u;",
            "override FLAG: bool = false;",
            "override GAIN: f32 = 1.0;",
            TEST_PARAMS.wgsl,
            "@group(1) @binding(0) var<storage, read> src: array<u32>;",
            "@group(1) @binding(1) var<storage, read_write> dst: array<u32>;",
            "@group(2) @binding(0) var<uniform> P: TestParams;",
            BODY,
        ].join("\n");
        expect(composed.code).toBe(expected);
        expect(composed.id).toBe("test-add");
        expect(composed.entryPoint).toBe("main");
        expect(composed.subgroups).toBe(false);
        expect(composed.overrides).toEqual({ WG: 256 });
        expect(composed.bodyLine).toBe(PRELUDE_LINES + 3 + TEST_PARAMS.wgsl.split("\n").length + 3 + 1);
        expect(composed.code.split("\n")[composed.bodyLine - 1]).toBe("@compute @workgroup_size(WG)");
        expect(composed.code).not.toContain("enable subgroups;");
        expect(composed.code).not.toContain("wg_reduce_");
        expect(Object.isFrozen(composed)).toBe(true);
        expect(Object.isFrozen(composed.overrides)).toBe(true);
    });

    it("fills WG from the caps unless the spec sets it; a standard override needs no OverrideDecl", () => {
        expect(
            composeWgsl(SPEC, fakeCaps(CAPS_SPEC_DEFAULT, { maxComputeInvocationsPerWorkgroup: 128 })).overrides.WG,
        ).toBe(128);
        expect(
            composeWgsl(SPEC, fakeCaps(CAPS_SPEC_DEFAULT, { maxComputeInvocationsPerWorkgroup: 200 })).overrides.WG,
        ).toBe(128);
        expect(
            composeWgsl(SPEC, fakeCaps(CAPS_SPEC_DEFAULT, { maxComputeInvocationsPerWorkgroup: 1024 })).overrides.WG,
        ).toBe(256);
        const explicit = composeWgsl({ ...SPEC, overrides: { WG: 64, USE_PERM: true, SCALE: 3 } }, CAPS_SPEC_DEFAULT);
        expect(explicit.overrides).toEqual({ WG: 64, USE_PERM: true, SCALE: 3 });
        // the override VALUES never enter the text: they travel as pipeline constants (4.2)
        expect(explicit.code).toContain("override SCALE: u32 = 2u;");
        expect(explicit.code).not.toContain("= 3u");
    });

    it("constants is the subset of overrides the code references outside comments and its own declarations (WebKit rejects a constant for an unread override; spec: not required to be statically used)", () => {
        // SCALE / FLAG / GAIN are declared but the body reads none of them; WG is read by @workgroup_size and linear_id.
        const unused = composeWgsl({ ...SPEC, overrides: { SCALE: 3, FLAG: true, GAIN: 2 } }, CAPS_SPEC_DEFAULT);
        expect(unused.overrides).toEqual({ WG: 256, SCALE: 3, FLAG: true, GAIN: 2 });
        expect(unused.constants).toEqual({ WG: 256 });
        expect(Object.isFrozen(unused.constants)).toBe(true);
        // A read of FLAG inside the body puts it into constants; a mention in a comment does not.
        const reading = composeWgsl(
            {
                ...SPEC,
                body: BODY.replace(
                    "dst[i] = src[i] + P.value;",
                    "dst[i] = select(src[i], src[i] + P.value, FLAG); // GAIN, SCALE",
                ),
                overrides: { SCALE: 3, FLAG: true, GAIN: 2 },
            },
            CAPS_SPEC_DEFAULT,
        );
        expect(reading.constants).toEqual({ WG: 256, FLAG: true });
        // The registry: degree never reads HAS_WEIGHTS, the thread-per-row fa2-attraction never reads TIER.
        const degree = composeWgsl(kernelSpec("degree", { USE_PERM: false, HAS_WEIGHTS: false }), CAPS_SPEC_DEFAULT);
        expect(Object.keys(degree.constants)).not.toContain("HAS_WEIGHTS");
        expect(Object.keys(degree.constants)).toContain("USE_PERM");
        const attraction = composeWgsl(
            kernelSpec("fa2-attraction", {
                USE_PERM: false,
                HAS_WEIGHTS: true,
                LINLOG: true,
                DISTRIBUTED: false,
                TIER: 0,
            }),
            CAPS_SPEC_DEFAULT,
        );
        expect(Object.keys(attraction.constants).sort()).toEqual([
            "DISTRIBUTED",
            "HAS_WEIGHTS",
            "LINLOG",
            "USE_PERM",
            "WG",
        ]);
    });

    it("rejects an unknown or misspelled override key, a wrongly typed value and a redeclared standard override", () => {
        expectCompose(() => composeWgsl({ ...SPEC, overrides: { BOGUS: 1 } }, CAPS_SPEC_DEFAULT), "override:BOGUS");
        expectCompose(() => composeWgsl({ ...SPEC, overrides: { scale: 1 } }, CAPS_SPEC_DEFAULT), "override:scale");
        expectCompose(() => composeWgsl({ ...SPEC, overrides: { FLAG: 1 } }, CAPS_SPEC_DEFAULT), "override:FLAG");
        expectCompose(() => composeWgsl({ ...SPEC, overrides: { SCALE: true } }, CAPS_SPEC_DEFAULT), "override:SCALE");
        expectCompose(() => composeWgsl({ ...SPEC, overrides: { SCALE: -1 } }, CAPS_SPEC_DEFAULT), "override:SCALE");
        expectCompose(() => composeWgsl({ ...SPEC, overrides: { WG: 1.5 } }, CAPS_SPEC_DEFAULT), "override:WG");
        expectCompose(
            () =>
                composeWgsl({ ...SPEC, overrideDecls: [{ name: "WG", type: "u32", default: 64 }] }, CAPS_SPEC_DEFAULT),
            "override:WG",
        );
        expectCompose(
            () =>
                composeWgsl(
                    {
                        ...SPEC,
                        overrideDecls: [
                            { name: "A", type: "u32", default: 1 },
                            { name: "A", type: "u32", default: 2 },
                        ],
                    },
                    CAPS_SPEC_DEFAULT,
                ),
            "override:A",
        );
        expectCompose(
            () =>
                composeWgsl({ ...SPEC, overrideDecls: [{ name: "type", type: "u32", default: 1 }] }, CAPS_SPEC_DEFAULT),
            "override:type",
        );
    });

    it("substitutes snippets at their //@@NAME@@ markers; a snippet without a marker and a marker without a snippet are compose errors", () => {
        const body = `@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = linear_id(wid, lid.x);
    if (i >= P.count) { return; }
    var v = 0u;
    //@@VALUE@@
    dst[i] = v;
}`;
        const spec: WgslModuleSpec = { ...SPEC, body, snippets: { VALUE: "v = src[i] * 2u;" } };
        const composed = composeWgsl(spec, CAPS_SPEC_DEFAULT);
        expect(composed.code).toContain("    v = src[i] * 2u;\n    dst[i] = v;");
        expect(composed.code).not.toContain("//@@");
        expectCompose(
            () => composeWgsl({ ...spec, snippets: { VALUE: "v = 1u;", OTHER: "v = 2u;" } }, CAPS_SPEC_DEFAULT),
            "snippet:OTHER",
        );
        expectCompose(() => composeWgsl({ ...spec, snippets: undefined }, CAPS_SPEC_DEFAULT), "marker:VALUE");
        expectCompose(() => composeWgsl({ ...spec, snippets: {} }, CAPS_SPEC_DEFAULT), "marker:VALUE");
        expectCompose(
            () => composeWgsl({ ...spec, snippets: { VALUE: "v = 1u;\n//@@INNER@@" } }, CAPS_SPEC_DEFAULT),
            "marker:INNER",
        );
    });

    it("rejects a body containing `@group(` or `override ` (the composer emits those lines)", () => {
        const withGroup = `@group(1) @binding(0) var<storage, read> extra: array<u32>;\n${BODY}`;
        expectCompose(() => composeWgsl({ ...SPEC, body: withGroup }, CAPS_SPEC_DEFAULT), "body:@group(");
        const withOverride = `override LOCAL: u32 = 1u;\n${BODY}`;
        expectCompose(() => composeWgsl({ ...SPEC, body: withOverride }, CAPS_SPEC_DEFAULT), "body:override ");
        const snippetWithGroup: WgslModuleSpec = {
            ...SPEC,
            body: BODY.replace("dst[i] = src[i] + P.value;", "//@@X@@"),
            snippets: { X: "@group(3) @binding(0) var<storage, read> q: array<u32>;" },
        };
        expectCompose(() => composeWgsl(snippetWithGroup, CAPS_SPEC_DEFAULT), "body:@group(");
    });

    it("rejects a WGSL reserved word used as an identifier in a body or a snippet, while a comment containing the word passes", () => {
        const reservedBody = BODY.replace(
            "let i = linear_id(wid, lid.x);",
            "let target = 0u;\n    let i = linear_id(wid, lid.x) + target;",
        );
        expectCompose(() => composeWgsl({ ...SPEC, body: reservedBody }, CAPS_SPEC_DEFAULT), "reserved:target");
        const memberBody = BODY.replace("P.value", "P.type");
        expectCompose(() => composeWgsl({ ...SPEC, body: memberBody }, CAPS_SPEC_DEFAULT), "reserved:type");
        const snippetSpec: WgslModuleSpec = {
            ...SPEC,
            body: BODY.replace("dst[i] = src[i] + P.value;", "//@@X@@"),
            snippets: { X: "let filter = 1u;\n    dst[i] = filter;" },
        };
        expectCompose(() => composeWgsl(snippetSpec, CAPS_SPEC_DEFAULT), "reserved:filter");
        const commented = BODY.replace(
            "let i = linear_id(wid, lid.x);",
            "let i = linear_id(wid, lid.x);   // `target` is reserved; /* so is type */",
        );
        expect(composeWgsl({ ...SPEC, body: commented }, CAPS_SPEC_DEFAULT).code).toContain("// `target` is reserved");
        // identifiers that merely contain a reserved word are fine
        const contains = BODY.replace(
            "let i = linear_id(wid, lid.x);",
            "let target_row = 0u;\n    let i = linear_id(wid, lid.x) + target_row;",
        );
        expect(composeWgsl({ ...SPEC, body: contains }, CAPS_SPEC_DEFAULT).entryPoint).toBe("main");
    });

    it("rejects duplicate bindings, a struct no block declares, an unknown need and a helper call without needs", () => {
        const dupSlot: WgslModuleSpec = {
            ...SPEC,
            bindings: [
                ...SPEC.bindings,
                { group: 1, binding: 0, name: "again", kind: "storage-ro", wgslType: "array<u32>" },
            ],
        };
        expectCompose(() => composeWgsl(dupSlot, CAPS_SPEC_DEFAULT), "binding:again");
        const dupName: WgslModuleSpec = {
            ...SPEC,
            bindings: [
                ...SPEC.bindings,
                { group: 3, binding: 0, name: "src", kind: "storage-ro", wgslType: "array<u32>" },
            ],
        };
        expectCompose(() => composeWgsl(dupName, CAPS_SPEC_DEFAULT), "binding:src");
        const undeclared: WgslModuleSpec = {
            ...SPEC,
            bindings: [
                ...SPEC.bindings,
                { group: 3, binding: 0, name: "q", kind: "storage-ro", wgslType: "array<Nope>" },
            ],
        };
        expectCompose(() => composeWgsl(undeclared, CAPS_SPEC_DEFAULT), "type:Nope");
        const builtin: WgslModuleSpec = {
            ...SPEC,
            bindings: [
                ...SPEC.bindings,
                { group: 3, binding: 0, name: "q", kind: "storage-ro", wgslType: "array<vec4<f32>>" },
            ],
        };
        expect(composeWgsl(builtin, CAPS_SPEC_DEFAULT).code).toContain(
            "@group(3) @binding(0) var<storage, read> q: array<vec4<f32>>;",
        );
        const badNeed = { ...SPEC, needs: ["bogus"] as unknown as readonly "subgroups"[] };
        expectCompose(() => composeWgsl(badNeed, CAPS_SPEC_DEFAULT), "needs:bogus");
        expectCompose(
            () => composeWgsl({ ...REDUCING, needs: [] }, CAPS_SPEC_DEFAULT),
            "helper:wg_reduce_u32",
            "test-total",
        );
    });

    it("splices the helper block iff needs lists subgroups: the subgroup form with SUBGROUP_MIN / SUBGROUP_MAX when the caps have the feature, the workgroup twin otherwise", () => {
        const twin = composeWgsl(REDUCING, CAPS_SPEC_DEFAULT);
        expect(twin.subgroups).toBe(false);
        expect(twin.code.startsWith(PRELUDE_WGSL)).toBe(true);
        expect(twin.code).toContain(REDUCE_HELPERS_WORKGROUP_WGSL);
        expect(twin.code).not.toContain("enable subgroups;");
        expect(twin.overrides).toEqual({ WG: 256 });
        const sg = composeWgsl(REDUCING, SUBGROUP_CAPS);
        expect(sg.subgroups).toBe(true);
        expect(sg.code.startsWith(`enable subgroups;\n${PRELUDE_WGSL}`)).toBe(true);
        expect(sg.code).toContain(REDUCE_HELPERS_SUBGROUP_WGSL);
        expect(sg.code).not.toContain("wg_scratch_v");
        expect(sg.overrides).toEqual({ WG: 256, SUBGROUP_MIN: 8, SUBGROUP_MAX: 32 });
        // the scratch is sized by the MINIMUM size (4.3): SG_SLOTS = ceil(WG / SUBGROUP_MIN) >= WG / 8
        const wg = sg.overrides.WG as number;
        const min = sg.overrides.SUBGROUP_MIN as number;
        expect(Math.ceil(wg / min)).toBeGreaterThanOrEqual(wg / 8);
        // the helper block comes after the struct texts and before the bind declarations (4.2 order)
        const structAt = sg.code.indexOf(TEST_PARAMS.wgsl);
        const helpersAt = sg.code.indexOf(REDUCE_HELPERS_SUBGROUP_WGSL);
        const bindAt = sg.code.indexOf("@group(1) @binding(0)");
        expect(structAt).toBeLessThan(helpersAt);
        expect(helpersAt).toBeLessThan(bindAt);
        expect(sg.code.split("\n")[sg.bodyLine - 1]).toBe("@compute @workgroup_size(WG)");
        // a caps table with the feature but a spec without needs splices nothing
        const plain = composeWgsl(SPEC, SUBGROUP_CAPS);
        expect(plain.subgroups).toBe(false);
        expect(plain.code).not.toContain("enable subgroups;");
        expect(plain.code).not.toContain("wg_reduce_");
        expect(plain.overrides).toEqual({ WG: 256 });
    });

    it("floors SUBGROUP_MIN at 4 (a reported minimum of 0 / 1 yields 4) and never below the caps value", () => {
        const zero: PlanCaps = { ...SUBGROUP_CAPS, subgroupMinSize: 0, subgroupMaxSize: 0 };
        expect(composeWgsl(REDUCING, zero).overrides).toEqual({ WG: 256, SUBGROUP_MIN: 4, SUBGROUP_MAX: 0 });
        const one: PlanCaps = { ...SUBGROUP_CAPS, subgroupMinSize: 1, subgroupMaxSize: 64 };
        expect(composeWgsl(REDUCING, one).overrides).toEqual({ WG: 256, SUBGROUP_MIN: 4, SUBGROUP_MAX: 64 });
        const thirtyTwo: PlanCaps = { ...SUBGROUP_CAPS, subgroupMinSize: 32, subgroupMaxSize: 32 };
        expect(composeWgsl(REDUCING, thirtyTwo).overrides).toEqual({ WG: 256, SUBGROUP_MIN: 32, SUBGROUP_MAX: 32 });
    });
});

describe("entryPointOf, bindGroupLayoutDescriptors, bindingNames", () => {
    it("entryPointOf finds the one fn after @compute and rejects zero or two entry points", () => {
        expect(entryPointOf(BODY)).toBe("main");
        expect(
            entryPointOf(
                "fn helper() -> u32 { return 1u; }\n@compute @workgroup_size(WG)\nfn degree(@builtin(workgroup_id) wid: vec3<u32>) {}",
            ),
        ).toBe("degree");
        expect(entryPointOf("@compute @workgroup_size(WG) fn same_line() {}")).toBe("same_line");
        expect(
            entryPointOf("// @compute in a comment does not count\n@compute @workgroup_size(WG)\nfn real() {}"),
        ).toBe("real");
        const none = catchError(() => entryPointOf("fn helper() -> u32 { return 1u; }"));
        expect(none.code).toBe("E_SHADER_COMPILE");
        expect(none.details).toMatchObject({ stage: "compose", slot: "entry" });
        expect(catchError(() => entryPointOf(`${BODY}\n${BODY.replace("fn main", "fn other")}`)).details.slot).toBe(
            "entry",
        );
        expectCompose(() => composeWgsl({ ...SPEC, body: "fn nothing() {}" }, CAPS_SPEC_DEFAULT), "entry");
    });

    it("derives one layout descriptor per group 0..maxGroup with empty groups empty, dynamic uniforms and minBindingSize", () => {
        expect(bindGroupLayoutDescriptors(SPEC)).toEqual([
            { entries: [] },
            {
                entries: [
                    { binding: 0, visibility: ShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                    { binding: 1, visibility: ShaderStage.COMPUTE, buffer: { type: "storage" } },
                ],
            },
            {
                entries: [
                    {
                        binding: 0,
                        visibility: ShaderStage.COMPUTE,
                        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 16 },
                    },
                ],
            },
        ]);
        expect(bindGroupLayoutDescriptors({ ...SPEC, bindings: [] })).toEqual([]);
        // binding order inside a group follows the binding index, not the list order
        const reversed: WgslModuleSpec = { ...SPEC, bindings: [SPEC.bindings[1], SPEC.bindings[0], SPEC.bindings[2]] };
        expect(bindGroupLayoutDescriptors(reversed)[1].entries).toEqual(bindGroupLayoutDescriptors(SPEC)[1].entries);
    });

    it("bindingNames lists a group's names in binding order", () => {
        expect(bindingNames(SPEC, 0)).toEqual([]);
        expect(bindingNames(SPEC, 1)).toEqual(["src", "dst"]);
        expect(bindingNames(SPEC, 2)).toEqual(["P"]);
        expect(bindingNames(SPEC, 3)).toEqual([]);
    });
});
