// scripts/runner-class.d.ts (the .js implements exactly this)
/** The adapter facts the rule reads. */
export interface RunnerClassInfo {
    readonly vendor: string;
    readonly architecture: string;
    readonly description: string;
}
/**
 * `env.GRAPHTY_RUNNER_CLASS` when set and non-empty (the T4 lane fixes `gpu-linux-t4`, 6.4), else
 * `<vendor>-<architecture>-driver<major>` (spec 10.4). CONTRACT DECISION: `major` = the first run of digits in
 * `description`, "0" when none (Dawn's NVIDIA description carries the driver version; SwiftShader / lavapipe carry
 * Mesa's); every character outside [A-Za-z0-9_.-] becomes "_"; lower-cased.
 */
export function runnerClass(
    info: RunnerClassInfo,
    env?: Readonly<Record<string, string | undefined>> | undefined,
): string;
