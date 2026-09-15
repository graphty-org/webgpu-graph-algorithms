// scripts/gpu-policy.d.ts (the .js implements exactly this)
/** The parsed GRAPHTY_GPU_REQUIRE. */
export interface GpuPolicy {
    readonly level: "skip" | "any" | "hardware" | "vendor";
    readonly vendor: string | null;
    readonly raw: string;
}
/** unset / "" -> skip; "any"; "hardware"; any other string -> vendor (lower-cased). */
export function parseGpuRequire(value: string | undefined): GpuPolicy;
/**
 * architecture "software" | "swiftshader" or isFallbackAdapter === true (a copy of src isSoftwareAdapter;
 * test/device/policy.test.ts keeps them equal).
 */
export function isSoftwareInfo(info: {
    readonly vendor: string;
    readonly architecture: string;
    readonly isFallbackAdapter?: boolean | undefined;
}): boolean;
/**
 * The policy verdict for an adapter (null info = no adapter): skip -> { ok: false, reason, skip: true } when absent;
 * any -> ok iff present; hardware -> ok iff !software; vendor -> ok iff vendor === policy.vendor and
 * (browser ? isFallbackAdapter === false : true).
 */
export function checkAdapter(
    info: {
        readonly vendor: string;
        readonly architecture: string;
        readonly isFallbackAdapter?: boolean | undefined;
    } | null,
    policy: GpuPolicy,
    options?: { readonly browser?: boolean | undefined } | undefined,
): { readonly ok: boolean; readonly skip: boolean; readonly reason: string | null };
