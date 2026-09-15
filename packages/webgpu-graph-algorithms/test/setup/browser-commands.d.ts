/// <reference types="vite/client" />
/**
 * Type declarations for the browser project (contract 5.1): the three server-side commands of vitest.config.ts
 * (a browser test cannot write files, spec 11.7) as members of @vitest/browser's BrowserCommands, and the three
 * variables vitest.config.ts forwards into import.meta.env. The `export {}` makes this file a module, so the
 * `declare module` below AUGMENTS "@vitest/browser/context" instead of shadowing it (a script-file `declare
 * module` is an ambient declaration that replaces the real module). P1-T7 narrows recordNoiseRow's row to
 * NoiseRow (a type import), P3-T6 narrows appendBenchRecord's payload to BrowserBenchPayload.
 */
export {};

declare module "@vitest/browser/context" {
    interface BrowserCommands {
        appendBenchRecord(payload: { readonly runnerClass: string; readonly session: unknown }): Promise<string>;
        writeNoiseFixture(payload: {
            readonly kernel: string;
            readonly fixture: string;
            readonly adapterClass: string;
            readonly values: readonly number[];
            readonly dtype: "f32" | "u32";
        }): Promise<string>;
        recordNoiseRow(row: Record<string, unknown>): Promise<string>;
    }
}

declare global {
    interface ImportMetaEnv {
        readonly GRAPHTY_GPU_REQUIRE: string;
        readonly GRAPHTY_BROWSER_GPU: "nvidia" | "swiftshader";
        readonly GRAPHTY_NOISE_FLOOR_WRITE: string;
    }
}
