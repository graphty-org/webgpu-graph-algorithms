/**
 * The WGSL composer (spec 3.5; contract 3.9, 4.2): ONE declaration per module (`WgslModuleSpec`) generates
 * everything the host and the shader must agree on -- the `@group / @binding` block, the `override` lines, the
 * struct texts, the explicit bind-group layouts and the bind() record keys -- by string concatenation in the fixed
 * order of contract 4.2. Every disagreement that would otherwise be a GPUPipelineError on the first device that
 * creates the variant is a compose-time `E_SHADER_COMPILE { stage: "compose" }` on every device.
 *
 * PLAN DECISION: this file imports `workgroupSizeFor` from `../device/caps.js` (one copy of the WG rule) and
 * `ShaderStage` from `../device/webgpu-constants.js` (spec 2.1 rule 1: the core never reads the `GPUShaderStage`
 * global at module top level); `../constants.js` is not imported because every constant the composer emits reaches
 * it through `./prelude.js`. The layer rule (contract 2.4: device < context < memory < kernel) allows both edges.
 */

import { workgroupSizeFor } from "../device/caps.js";
import { ShaderStage } from "../device/webgpu-constants.js";
import { WebGpuGraphError } from "../errors.js";
import { type PlanCaps } from "../types/context.js";
import {
    PRELUDE_WGSL,
    REDUCE_HELPER_NAMES,
    REDUCE_HELPERS_SUBGROUP_WGSL,
    REDUCE_HELPERS_WORKGROUP_WGSL,
    WGSL_RESERVED_WORDS,
    wgslF32Literal,
} from "./prelude.js";
import { type UniformBlock } from "./struct-block.js";

/** One storage or uniform binding of a module (spec 3.5). `wgslType` is the element / struct type text ("array<u32>", "array<vec4f>", "Fa2State"). */
export interface BindingDecl {
    readonly group: 0 | 1 | 2 | 3;
    readonly binding: number;
    readonly name: string;
    readonly kind: "storage" | "storage-ro" | "uniform";
    readonly wgslType: string;
}
/** One module-specific override (spec 3.5); the five standard ones (WG, USE_PERM, HAS_WEIGHTS, SUBGROUP_MIN, SUBGROUP_MAX) come from the prelude and are never listed here. */
export interface OverrideDecl {
    readonly name: string;
    readonly type: "u32" | "bool" | "f32";
    readonly default: number | boolean;
}
/** Spec 3.5 WgslModuleSpec, verbatim. `needs: ["subgroups"]` means "this body calls a wg_reduce_* helper": the composer splices the subgroup helper block when the device has the feature and the workgroup-memory twin otherwise. */
export interface WgslModuleSpec {
    readonly id: string;
    readonly body: string;
    readonly bindings: readonly BindingDecl[];
    readonly overrideDecls: readonly OverrideDecl[];
    readonly overrides: Readonly<Record<string, number | boolean>>;
    readonly needs: readonly "subgroups"[];
    readonly uniforms: readonly UniformBlock[];
    readonly snippets?: Readonly<Record<string, string>> | undefined;
}
/** The names the prelude declares as overrides; spec.overrides may set them without an OverrideDecl. */
export const STANDARD_OVERRIDES = ["WG", "USE_PERM", "HAS_WEIGHTS", "SUBGROUP_MIN", "SUBGROUP_MAX"] as const;
/** What composeWgsl returns: the text, the body's first line in it (for compilation-info formatting), the effective overrides (WG, SUBGROUP_MIN, SUBGROUP_MAX filled) and the features to enable. */
export interface ComposedModule {
    readonly id: string;
    readonly code: string;
    readonly bodyLine: number;
    readonly overrides: Readonly<Record<string, number | boolean>>;
    /**
     * The subset of `overrides` the composed code REFERENCES outside its own declaration -- what the pipeline
     * descriptor's `constants` carries. WebKit (Safari) fails pipeline creation with "Compute library failed
     * creation" when a constant is supplied for an override the entry point never reads (HAS_WEIGHTS in `degree`,
     * TIER in the thread-per-row tiers); Dawn tolerates it. The cache key still uses `overrides` (contract 3.9).
     */
    readonly constants: Readonly<Record<string, number | boolean>>;
    readonly entryPoint: string;
    readonly subgroups: boolean;
}

/** The type of each standard override (the prelude's declarations). */
const STANDARD_OVERRIDE_TYPES: Readonly<Record<(typeof STANDARD_OVERRIDES)[number], OverrideDecl["type"]>> =
    Object.freeze({
        WG: "u32",
        USE_PERM: "bool",
        HAS_WEIGHTS: "bool",
        SUBGROUP_MIN: "u32",
        SUBGROUP_MAX: "u32",
    });
/** The only feature a body may need (spec 3.5). */
const KNOWN_NEEDS: readonly string[] = Object.freeze(["subgroups"]);
/** The smallest legal subgroup size; the scratch of 4.3 is sized by the MINIMUM size. */
const SUBGROUP_FLOOR = 4;
/** Builtin WGSL element types a binding may name without a struct declaration. */
const BUILTIN_TYPES: ReadonlySet<string> = new Set([
    "u32",
    "i32",
    "f32",
    "f16",
    "bool",
    "vec2f",
    "vec3f",
    "vec4f",
    "vec2u",
    "vec3u",
    "vec4u",
    "vec2i",
    "vec3i",
    "vec4i",
    "vec2h",
    "vec3h",
    "vec4h",
]);
const TEMPLATED_BUILTIN = /^(vec[234]|mat[234]x[234]|atomic|array)</;
const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;
/** The `var<...>` form of each binding kind (contract 4.2). */
const ADDRESS_SPACE: Readonly<Record<BindingDecl["kind"], string>> = Object.freeze({
    storage: "var<storage, read_write>",
    "storage-ro": "var<storage, read>",
    uniform: "var<uniform>",
});
/** The reserved words as a set (the frozen list stays the public form). */
const RESERVED: ReadonlySet<string> = new Set(WGSL_RESERVED_WORDS);
const MARKER = /\/\/@@([A-Za-z0-9_]+)@@/;
const IDENTIFIER_TOKENS = /(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*/g;
const COMPUTE_ATTRIBUTE = /@compute\b/g;
const ENTRY_FN = /\bfn\s+([A-Za-z_][A-Za-z0-9_]*)/;

/**
 * The compose-time error of contract 3.9.
 * @param id - the module id
 * @param slot - what failed (`override:<key>`, `snippet:<key>`, `marker:<name>`, `body:<text>`, `binding:<name>`, `type:<name>`, `needs:<feature>`, `helper:<name>`, `reserved:<word>`, `entry`)
 * @param message - the human-readable reason
 * @returns the error (thrown by the caller)
 */
function composeError(id: string, slot: string, message: string): WebGpuGraphError {
    return new WebGpuGraphError("E_SHADER_COMPILE", `${id}: ${message}`, { id, stage: "compose", slot });
}

/**
 * Removes block and line comments so a word inside a comment is never taken for an identifier.
 * @param text - WGSL text
 * @returns the text without comments
 */
function stripComments(text: string): string {
    return text.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/\/\/[^\n]*/g, "");
}

/**
 * The struct a binding's element type names, or null for a builtin type.
 * @param wgslType - the binding's type text ("array<u32>", "array<Fa2Partial>", "Fa2State")
 * @returns the struct name or null
 */
function structNameOf(wgslType: string): string | null {
    const trimmed = wgslType.trim();
    const inner = trimmed.startsWith("array<")
        ? trimmed.slice("array<".length).replace(/\s*(,[^>]*)?>\s*$/, "")
        : trimmed;
    if (BUILTIN_TYPES.has(inner) || TEMPLATED_BUILTIN.test(inner)) {
        return null;
    }
    return inner;
}

/**
 * Checks that an override value matches its declared type.
 * PLAN DECISION: a value of the wrong type (a number for a bool, a bool / negative / fractional for a u32) is a
 * compose-time `E_SHADER_COMPILE { slot: "override:<name>" }` instead of a GPUPipelineError on the first device.
 * @param id - the module id
 * @param name - the override name
 * @param type - the declared type
 * @param value - the value
 */
function assertOverrideValue(id: string, name: string, type: OverrideDecl["type"], value: number | boolean): void {
    const ok =
        type === "bool"
            ? typeof value === "boolean"
            : typeof value === "number" &&
              Number.isFinite(value) &&
              (type === "f32" || (Number.isInteger(value) && value >= 0));
    if (!ok) {
        throw composeError(id, `override:${name}`, `override ${name} (${type}) cannot take the value ${String(value)}`);
    }
}

/**
 * The WGSL literal of an override default (u32 literals carry the `u` suffix, f32 literals a decimal point, bools true / false).
 * @param decl - the override
 * @returns the literal text
 */
function overrideLiteral(decl: OverrideDecl): string {
    if (decl.type === "bool") {
        return decl.default ? "true" : "false";
    }
    const value = decl.default as number;
    return decl.type === "u32" ? `${value}u` : wgslF32Literal(value);
}

/**
 * The `var<...>` declaration of one binding (contract 4.2).
 * @param decl - the binding
 * @returns the declaration line
 */
function bindingLine(decl: BindingDecl): string {
    const space = ADDRESS_SPACE[decl.kind];
    return `@group(${decl.group}) @binding(${decl.binding}) ${space} ${decl.name}: ${decl.wgslType};`;
}

/**
 * The entry-point name of a body, or null when the body does not carry exactly one `@compute` attribute followed by `fn <name>`.
 * @param body - the kernel body
 * @returns the name or null
 */
function findEntryPoint(body: string): string | null {
    const stripped = stripComments(body);
    const matches = [...stripped.matchAll(COMPUTE_ATTRIBUTE)];
    if (matches.length !== 1 || matches[0].index === undefined) {
        return null;
    }
    const fn = ENTRY_FN.exec(stripped.slice(matches[0].index));
    return fn === null ? null : fn[1];
}

/**
 * The entry-point name of a body: the identifier after `fn` on the line following `@compute` (every body has exactly one entry point).
 * PLAN DECISION: a body without exactly one `@compute` entry point is `E_SHADER_COMPILE { stage: "compose", slot: "entry" }`.
 * @param body - the kernel body
 * @returns the entry-point function name
 */
export function entryPointOf(body: string): string {
    const name = findEntryPoint(body);
    if (name === null) {
        throw new WebGpuGraphError(
            "E_SHADER_COMPILE",
            "a body must carry exactly one @compute attribute followed by `fn <name>`",
            {
                id: null,
                stage: "compose",
                slot: "entry",
            },
        );
    }
    return name;
}

/**
 * The bind-group-layout descriptors derived from spec.bindings: one per group 0..maxGroup (empty groups get an empty entry list); uniform decls carry hasDynamicOffset true (spec 5.1).
 * PLAN DECISION: a uniform decl whose struct is one of `spec.uniforms` also carries `minBindingSize = block.byteLength`,
 * so a wrong-size uniform binding fails at createBindGroup (synchronously, with the bind group's label in the message)
 * instead of at the first dispatch (verified on Dawn 0.4.0: without it the error names only the pipeline).
 * @param spec - the module
 * @returns the descriptors in group order
 */
export function bindGroupLayoutDescriptors(spec: WgslModuleSpec): GPUBindGroupLayoutDescriptor[] {
    const maxGroup = spec.bindings.reduce((acc, b) => Math.max(acc, b.group), -1);
    const blocks = new Map(spec.uniforms.map((block) => [block.name, block] as const));
    const descriptors: GPUBindGroupLayoutDescriptor[] = [];
    for (let group = 0; group <= maxGroup; group++) {
        const entries: GPUBindGroupLayoutEntry[] = spec.bindings
            .filter((b) => b.group === group)
            .sort((a, b) => a.binding - b.binding)
            .map((b) => {
                if (b.kind === "uniform") {
                    const block = blocks.get(b.wgslType);
                    const buffer: GPUBufferBindingLayout = { type: "uniform", hasDynamicOffset: true };
                    if (block !== undefined) {
                        buffer.minBindingSize = block.byteLength;
                    }
                    return { binding: b.binding, visibility: ShaderStage.COMPUTE, buffer };
                }
                return {
                    binding: b.binding,
                    visibility: ShaderStage.COMPUTE,
                    buffer: { type: b.kind === "storage" ? "storage" : "read-only-storage" },
                };
            });
        descriptors.push({ entries });
    }
    return descriptors;
}

/**
 * The names of the bindings of one group in binding order.
 * @param spec - the module
 * @param group - the group index
 * @returns the names
 */
export function bindingNames(spec: WgslModuleSpec, group: number): readonly string[] {
    return spec.bindings
        .filter((b) => b.group === group)
        .sort((a, b) => a.binding - b.binding)
        .map((b) => b.name);
}

/**
 * Validates `spec.needs`, `spec.overrideDecls`, `spec.overrides` and `spec.bindings` (contract 3.9's compose-time checks).
 * PLAN DECISION: an `overrideDecls` entry that redeclares a standard override, repeats a name or names a reserved
 * word is `E_SHADER_COMPILE { slot: "override:<name>" }` (otherwise a duplicate-declaration error on the first device).
 * @param spec - the module
 */
function validateDeclarations(spec: WgslModuleSpec): void {
    const { id } = spec;
    for (const need of spec.needs) {
        if (!KNOWN_NEEDS.includes(need)) {
            throw composeError(id, `needs:${need}`, `unknown feature "${need}" in needs`);
        }
    }
    const declared = new Map<string, OverrideDecl["type"]>();
    for (const decl of spec.overrideDecls) {
        if (!IDENTIFIER.test(decl.name) || RESERVED.has(decl.name)) {
            throw composeError(id, `override:${decl.name}`, `override name "${decl.name}" is not a usable identifier`);
        }
        if ((STANDARD_OVERRIDES as readonly string[]).includes(decl.name) || declared.has(decl.name)) {
            throw composeError(
                id,
                `override:${decl.name}`,
                `override "${decl.name}" is declared twice (the prelude declares the standard ones)`,
            );
        }
        assertOverrideValue(id, decl.name, decl.type, decl.default);
        declared.set(decl.name, decl.type);
    }
    for (const [key, value] of Object.entries(spec.overrides)) {
        const standard = (STANDARD_OVERRIDE_TYPES as Readonly<Record<string, OverrideDecl["type"] | undefined>>)[key];
        const type = standard ?? declared.get(key);
        if (type === undefined) {
            throw composeError(
                id,
                `override:${key}`,
                `override "${key}" is neither a standard override nor declared in overrideDecls`,
            );
        }
        assertOverrideValue(id, key, type, value);
    }
    const slots = new Set<string>();
    const names = new Set<string>();
    const blocks = new Set(spec.uniforms.map((block) => block.name));
    for (const b of spec.bindings) {
        if (!IDENTIFIER.test(b.name) || RESERVED.has(b.name)) {
            throw composeError(id, `binding:${b.name}`, `binding name "${b.name}" is not a usable identifier`);
        }
        const slot = `${b.group}.${b.binding}`;
        if (slots.has(slot)) {
            throw composeError(id, `binding:${b.name}`, `two bindings share group ${b.group} binding ${b.binding}`);
        }
        if (names.has(b.name)) {
            throw composeError(id, `binding:${b.name}`, `two bindings share the name "${b.name}"`);
        }
        slots.add(slot);
        names.add(b.name);
        const struct = structNameOf(b.wgslType);
        if (struct !== null && !blocks.has(struct)) {
            throw composeError(
                id,
                `type:${struct}`,
                `binding "${b.name}" names struct "${struct}" but no block in spec.uniforms declares it`,
            );
        }
    }
}

/**
 * Substitutes the snippets into the body (every `//@@NAME@@` marker replaced by `spec.snippets[NAME]`) and checks the markers.
 * @param spec - the module
 * @returns the substituted body
 */
function substituteSnippets(spec: WgslModuleSpec): string {
    const { id } = spec;
    let { body } = spec;
    for (const [key, snippet] of Object.entries(spec.snippets ?? {})) {
        const marker = `//@@${key}@@`;
        if (!body.includes(marker)) {
            throw composeError(id, `snippet:${key}`, `snippet "${key}" has no ${marker} marker in the body`);
        }
        body = body.split(marker).join(snippet);
    }
    const left = MARKER.exec(body);
    if (left !== null) {
        throw composeError(id, `marker:${left[1]}`, `marker //@@${left[1]}@@ is not filled by any snippet`);
    }
    return body;
}

/**
 * The textual checks on the substituted body: no `@group(`, no `override `, no reserved identifier, no helper call without `needs`.
 * PLAN DECISION: a body calling a `wg_reduce_*` helper without `needs: ["subgroups"]` is
 * `E_SHADER_COMPILE { slot: "helper:<name>" }` (otherwise an unresolved identifier on the first device).
 * @param spec - the module
 * @param body - the substituted body
 */
function validateBody(spec: WgslModuleSpec, body: string): void {
    const { id } = spec;
    for (const forbidden of ["@group(", "override "]) {
        if (body.includes(forbidden)) {
            throw composeError(
                id,
                `body:${forbidden}`,
                `a body must not contain "${forbidden}" (the composer emits those lines)`,
            );
        }
    }
    const stripped = stripComments(body);
    const helpers = REDUCE_HELPER_NAMES as readonly string[];
    for (const match of stripped.matchAll(IDENTIFIER_TOKENS)) {
        const word = match[0];
        if (RESERVED.has(word)) {
            throw composeError(
                id,
                `reserved:${word}`,
                `"${word}" is a WGSL reserved word (spec 16.2) and cannot be an identifier`,
            );
        }
        if (helpers.includes(word) && !spec.needs.includes("subgroups")) {
            throw composeError(id, `helper:${word}`, `the body calls ${word} but spec.needs does not list "subgroups"`);
        }
    }
}

/**
 * String concatenation of spec 3.5: prelude (constants, standard overrides, helpers), module overrides, struct texts, the bind declarations, then the body with snippets substituted (4.2 gives the exact emitted format).
 * @param spec - the module
 * @param caps - the device capabilities (WG, the subgroup sizes and the feature set)
 * @returns the composed module
 */
export function composeWgsl(spec: WgslModuleSpec, caps: PlanCaps): ComposedModule {
    validateDeclarations(spec);
    const entryPoint = findEntryPoint(spec.body);
    if (entryPoint === null) {
        throw composeError(
            spec.id,
            "entry",
            "a body must carry exactly one @compute attribute followed by `fn <name>`",
        );
    }
    const body = substituteSnippets(spec);
    validateBody(spec, body);
    const wantsHelpers = spec.needs.includes("subgroups");
    const subgroups = wantsHelpers && caps.features.has("subgroups");
    const overrides: Record<string, number | boolean> = { ...spec.overrides };
    if (overrides.WG === undefined) {
        overrides.WG = workgroupSizeFor(caps);
    }
    if (subgroups) {
        overrides.SUBGROUP_MAX = caps.subgroupMaxSize;
        overrides.SUBGROUP_MIN = Math.max(SUBGROUP_FLOOR, caps.subgroupMinSize);
    }
    const parts: string[] = [];
    if (subgroups) {
        parts.push("enable subgroups;");
    }
    parts.push(PRELUDE_WGSL);
    for (const decl of spec.overrideDecls) {
        parts.push(`override ${decl.name}: ${decl.type} = ${overrideLiteral(decl)};`);
    }
    for (const block of spec.uniforms) {
        parts.push(block.wgsl);
    }
    if (wantsHelpers) {
        parts.push(subgroups ? REDUCE_HELPERS_SUBGROUP_WGSL : REDUCE_HELPERS_WORKGROUP_WGSL);
    }
    for (const decl of spec.bindings) {
        parts.push(bindingLine(decl));
    }
    const head = parts.join("\n");
    const code = `${head}\n${body}`;
    return Object.freeze({
        id: spec.id,
        code,
        bodyLine: head.split("\n").length + 1,
        overrides: Object.freeze(overrides),
        constants: Object.freeze(referencedOverrides(code, overrides)),
        entryPoint,
        subgroups,
    });
}

/**
 * The overrides whose names occur in the code outside comments and outside their own `override` declaration lines
 * (the constants a pipeline may carry on every backend, see ComposedModule.constants).
 * @param code - the composed module text
 * @param overrides - the effective override values
 * @returns the referenced subset, in the same order
 */
function referencedOverrides(
    code: string,
    overrides: Readonly<Record<string, number | boolean>>,
): Record<string, number | boolean> {
    const stripped = code
        .replace(/\/\*[\s\S]*?\*\//g, " ")
        .replace(/\/\/[^\n]*/g, " ")
        .replace(/^\s*override\s+\w+\s*:[^\n]*$/gm, " ");
    const referenced: Record<string, number | boolean> = {};
    for (const [name, value] of Object.entries(overrides)) {
        if (new RegExp(`\\b${name}\\b`).test(stripped)) {
            referenced[name] = value;
        }
    }
    return referenced;
}
