/**
 * Browser-side declarations for the three Vitest commands of vitest.config.ts (contract 2.5 -- appendBenchRecord,
 * writeNoiseFixture, recordNoiseRow) and the import.meta.env keys the browser project receives through `define`.
 *
 * This file is a MODULE (it imports two types), so the `declare module` block below AUGMENTS
 * @vitest/browser/context's BrowserCommands instead of shadowing the module, and the ImportMetaEnv keys sit
 * inside `declare global` so they keep merging into vite/client's global interface. Both imports are type-only:
 * benchmarks/harness.ts and test/helpers/noise-floor.ts read node:fs and never enter the browser bundle.
 */
/// <reference types="vite/client" />
import type { BrowserBenchPayload } from "../../benchmarks/harness.js";
import type { NoiseRow } from "../helpers/noise-floor.js";

declare module "@vitest/browser/context" {
    interface BrowserCommands {
        /** Appends one BenchSession to benchmarks/out/<runnerClass>.json (contract 2.5, 6.4); resolves the path written. */
        appendBenchRecord(payload: BrowserBenchPayload): Promise<string>;
        /** Writes test/fixtures/noise/<kernel>-<fixture>-<adapterClass>.json when GRAPHTY_NOISE_FLOOR_WRITE=1; resolves the path or "". */
        writeNoiseFixture(payload: {
            readonly kernel: string;
            readonly fixture: string;
            readonly adapterClass: string;
            readonly values: readonly number[];
            readonly dtype: "f32" | "u32";
        }): Promise<string>;
        /** Appends (or replaces by id) one 5.6 row of benchmarks/results/noise-floor.json when GRAPHTY_NOISE_FLOOR_WRITE=1. */
        recordNoiseRow(row: NoiseRow): Promise<string>;
    }
}

declare global {
    interface ImportMetaEnv {
        readonly GRAPHTY_GPU_REQUIRE: string;
        readonly GRAPHTY_BROWSER_GPU: "nvidia" | "swiftshader" | "metal" | "warp";
        readonly GRAPHTY_NOISE_FLOOR_WRITE: string;
    }
}
