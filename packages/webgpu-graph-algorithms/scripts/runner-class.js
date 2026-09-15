/**
 * The ONE copy of the benchmark runner-class rule (spec 10.4, 11.7; contract 6.4, 6.9): the name of the
 * benchmarks/results/<runner-class>.json baseline and of the benchmarks/out/<runner-class>.json run output.
 * scripts/gpu-report.js imports it, benchmarks/harness.ts re-exports it and test/device/policy.test.ts pins
 * it, so the three file names cannot drift. Plain ESM JavaScript; the declarations live in runner-class.d.ts.
 *
 * Rule: env.GRAPHTY_RUNNER_CLASS when set and non-empty (the T4 lane fixes "gpu-linux-t4" in gpu.yml so the
 * nightly file name never follows the partner image's driver), else "<vendor>-<architecture>-driver<major>"
 * where major is the FIRST run of digits in info.description ("0" when it has none; Dawn's NVIDIA description
 * is "NVIDIA: 580.173.02 580.173.2.0", lavapipe's "llvmpipe: Mesa 23.2.1-... (LLVM 15.0.7) 0.0.1"), every
 * character outside [A-Za-z0-9_.-] replaced by "_", lower-cased.
 */

/**
 * The runner class of an adapter.
 * @param {{ vendor: string, architecture: string, description: string }} info - the adapter facts
 * @param {Readonly<Record<string, string | undefined>> | undefined} [env] - the environment (default process.env;
 * an empty record where `process` does not exist, so the browser bundle of P3-T6 can import the module)
 * @returns {string} the runner class
 */
export function runnerClass(info, env) {
    const environment = env ?? (typeof process === "undefined" ? {} : process.env);
    const override = environment.GRAPHTY_RUNNER_CLASS;
    if (override !== undefined && override !== "") {
        return override;
    }
    const digits = /\d+/.exec(info.description);
    const major = digits === null ? "0" : digits[0];
    return `${info.vendor}-${info.architecture}-driver${major}`.replace(/[^A-Za-z0-9_.-]/g, "_").toLowerCase();
}
