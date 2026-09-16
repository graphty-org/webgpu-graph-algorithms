/**
 * The faked PlanCaps tables of the pure planner tests (contract 5.2, lead b; spec 11.3 "hand-computed expectations
 * under faked caps: spec defaults, SwiftShader-like, lavapipe-like, NVIDIA-like"). Every limit is what a DEVICE created
 * with `limits: "raise"` reports on that stack (note 05 sections 2.4 and 4), never the adapter's raw value: the two
 * alignment limits stay at the spec default 256 because RAISABLE_LIMITS never requests the adapter's smaller value
 * (spec 2.6). CAPS_INTEL_XE (subgroups 8 / 32, P2-T2) is the min != max table of spec 6.
 */

import { type PlanCaps, type PlanLimits } from "../../src/types/context.js";

/** The WebGPU core spec default limits (a device created with no requiredLimits, on every runtime). */
const SPEC_DEFAULT_LIMITS: PlanLimits = Object.freeze({
    maxBufferSize: 268_435_456,
    maxStorageBufferBindingSize: 134_217_728,
    maxStorageBuffersPerShaderStage: 8,
    minStorageBufferOffsetAlignment: 256,
    minUniformBufferOffsetAlignment: 256,
    maxComputeWorkgroupsPerDimension: 65_535,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupStorageSize: 16_384,
    maxUniformBufferBindingSize: 65_536,
});

/** Spec defaults; no subgroups; hardware (the portable baseline every planner must satisfy). */
export const CAPS_SPEC_DEFAULT: PlanCaps = Object.freeze({
    limits: SPEC_DEFAULT_LIMITS,
    features: new Set<string>(),
    subgroupMinSize: 0,
    subgroupMaxSize: 0,
    software: false,
});

/** Chromium SwiftShader: spec-default limits, subgroups of size 4, a software adapter (isFallbackAdapter true). */
export const CAPS_SWIFTSHADER: PlanCaps = Object.freeze({
    limits: SPEC_DEFAULT_LIMITS,
    features: new Set<string>(["subgroups", "timestamp-query"]),
    subgroupMinSize: 4,
    subgroupMaxSize: 4,
    software: true,
});

/**
 * Dawn llvmpipe (the default CI lane): the 128 MiB binding limit is the adapter's own and cannot be raised; the
 * table keeps the 256 MiB spec-default maxBufferSize of the contract (PLAN DECISION 18: a `raise` device reports
 * 4,294,967,295 -- model it with fakeCaps(CAPS_LAVAPIPE, { maxBufferSize: 4_294_967_295 })); the other raisable
 * limits are the adapter's (16 storage buffers, 1024 invocations, 32 KiB workgroup storage); subgroups of size 8.
 */
export const CAPS_LAVAPIPE: PlanCaps = Object.freeze({
    limits: Object.freeze({
        maxBufferSize: 268_435_456,
        maxStorageBufferBindingSize: 134_217_728,
        maxStorageBuffersPerShaderStage: 16,
        minStorageBufferOffsetAlignment: 256,
        minUniformBufferOffsetAlignment: 256,
        maxComputeWorkgroupsPerDimension: 65_535,
        maxComputeInvocationsPerWorkgroup: 1024,
        maxComputeWorkgroupSizeX: 1024,
        maxComputeWorkgroupStorageSize: 32_768,
        maxUniformBufferBindingSize: 65_536,
    }),
    features: new Set<string>(["subgroups", "timestamp-query", "shader-f16"]),
    subgroupMinSize: 8,
    subgroupMaxSize: 8,
    software: true,
});

/**
 * Dawn-node on the RTX 4070 SUPER with `limits: "raise"`: 2 GiB - 4 binding, the adapter's 1 TiB maxBufferSize
 * (a driver figure, not memory: the OOM scope catches exhaustion, spec 4.4), 16 storage buffers, 48 KiB workgroup
 * storage, subgroups of size 32. maxBufferSize is re-fixed to the DEVICE value printed by
 * test/memory/residency.test.ts on the NVIDIA lane (P1-T2 Step 19) and recorded in docs/decisions/G1.md.
 */
export const CAPS_NVIDIA_4070: PlanCaps = Object.freeze({
    limits: Object.freeze({
        maxBufferSize: 1_099_511_627_776,
        maxStorageBufferBindingSize: 2_147_483_644,
        maxStorageBuffersPerShaderStage: 16,
        minStorageBufferOffsetAlignment: 256,
        minUniformBufferOffsetAlignment: 256,
        maxComputeWorkgroupsPerDimension: 65_535,
        maxComputeInvocationsPerWorkgroup: 1024,
        maxComputeWorkgroupSizeX: 1024,
        maxComputeWorkgroupStorageSize: 49_152,
        maxUniformBufferBindingSize: 65_536,
    }),
    features: new Set<string>(["subgroups", "timestamp-query"]),
    subgroupMinSize: 32,
    subgroupMaxSize: 32,
    software: false,
});

/**
 * Intel Xe / Arc: subgroups 8-32 (min != max), otherwise the spec defaults. No planner output may depend on the pair
 * (spec 6, contract 4.3: the subgroup scratch is sized by the MINIMUM size, 8 slots per 64 invocations here).
 */
export const CAPS_INTEL_XE: PlanCaps = Object.freeze({
    limits: CAPS_SPEC_DEFAULT.limits,
    features: new Set<string>([...CAPS_SPEC_DEFAULT.features, "subgroups"]),
    subgroupMinSize: 8,
    subgroupMaxSize: 32,
    software: false,
});

/** Every table, named, for `describe.each` loops (CAPS_INTEL_XE appended at P2-T2). */
export const CAPS_TABLES: readonly { readonly name: string; readonly caps: PlanCaps }[] = Object.freeze([
    { name: "spec-default", caps: CAPS_SPEC_DEFAULT },
    { name: "swiftshader", caps: CAPS_SWIFTSHADER },
    { name: "lavapipe", caps: CAPS_LAVAPIPE },
    { name: "nvidia-4070", caps: CAPS_NVIDIA_4070 },
    { name: "intel-xe", caps: CAPS_INTEL_XE },
]);

/** The ten PlanLimits keys, copied one by one: a real GPUSupportedLimits exposes them as prototype getters, which an object spread would miss. */
const PLAN_LIMIT_KEYS: readonly (keyof PlanLimits)[] = [
    "maxBufferSize",
    "maxStorageBufferBindingSize",
    "maxStorageBuffersPerShaderStage",
    "minStorageBufferOffsetAlignment",
    "minUniformBufferOffsetAlignment",
    "maxComputeWorkgroupsPerDimension",
    "maxComputeInvocationsPerWorkgroup",
    "maxComputeWorkgroupSizeX",
    "maxComputeWorkgroupStorageSize",
    "maxUniformBufferBindingSize",
];

/**
 * A table with some limits and flags overridden (explicit `undefined` values are ignored so a Partial built from
 * optional inputs never clobbers a base value). Works on a real GpuCaps too (ctx.caps): the limits are read key by
 * key, never spread.
 * @param base - the table to start from
 * @param overrides - the limits to replace
 * @param flags - the scalar facts to replace (software, subgroupMinSize, subgroupMaxSize)
 * @returns a new frozen table; `features` is the base's set
 */
export function fakeCaps(
    base: PlanCaps,
    overrides: Partial<PlanLimits>,
    flags?: Partial<Pick<PlanCaps, "software" | "subgroupMinSize" | "subgroupMaxSize">>,
): PlanCaps {
    const limits: Record<string, number> = {};
    for (const key of PLAN_LIMIT_KEYS) {
        limits[key] = base.limits[key];
    }
    for (const key of PLAN_LIMIT_KEYS) {
        const value = overrides[key];
        if (value !== undefined) {
            limits[key] = value;
        }
    }
    return Object.freeze({
        limits: Object.freeze(limits as unknown as PlanLimits),
        features: base.features,
        subgroupMinSize: flags?.subgroupMinSize ?? base.subgroupMinSize,
        subgroupMaxSize: flags?.subgroupMaxSize ?? base.subgroupMaxSize,
        software: flags?.software ?? base.software,
    });
}
