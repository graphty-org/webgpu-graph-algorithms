/**
 * The browser twin of test/helpers/noise-floor.ts's two writers (contract 5.2; spec 11.5, 11.9 item 3): a browser test
 * cannot write files, so this adapter's raw outputs and noise rows cross the Vitest commands bridge to the server-side
 * `writeNoiseFixture` / `recordNoiseRow` commands of vitest.config.ts (2.5), which write the same JSON shapes as the Node
 * writers. Imports nothing from node:*; the row type is a type-only import.
 */

import { commands } from "@vitest/browser/context";

import type { GpuCaps } from "../../src/types/context.js";
import type { NoiseRow } from "./noise-floor.js";

/**
 * True when the browser project was started with GRAPHTY_NOISE_FLOOR_WRITE=1 (inlined through vite `define`, 2.5).
 * @returns whether the writers should write
 */
function writingEnabled(): boolean {
    return import.meta.env.GRAPHTY_NOISE_FLOOR_WRITE === "1";
}

/**
 * commands.writeNoiseFixture(...) through the Vitest bridge (2.5); resolves the path written, or "" when
 * GRAPHTY_NOISE_FLOOR_WRITE is not "1".
 * @param kernel - the kernel id ("degree", "reduce", "fill", "fa2-repulsion-exact", "fa2-speed-finalize")
 * @param fixture - the fixture name ("karate", "random1k", "linear-id")
 * @param adapterClass - this adapter's class string (adapterClassBrowser)
 * @param values - the raw output (copied into a plain array for the bridge)
 * @param dtype - how the values are compared by test/noise-floor.test.ts: "u32" bitwise, "f32" within the derived floor
 * @returns the path written on the server side, or ""
 */
export async function writeNoiseFixtureBrowser(
    kernel: string,
    fixture: string,
    adapterClass: string,
    values: ArrayLike<number>,
    dtype: "f32" | "u32",
): Promise<string> {
    if (!writingEnabled()) {
        return "";
    }
    return commands.writeNoiseFixture({ kernel, fixture, adapterClass, values: Array.from(values), dtype });
}

/**
 * commands.recordNoiseRow(row) through the bridge; resolves the path written, or "" when writing is off.
 * @param row - the 5.6 row
 * @returns the path of benchmarks/results/noise-floor.json on the server side, or ""
 */
export async function recordNoiseRowBrowser(row: NoiseRow): Promise<string> {
    if (!writingEnabled()) {
        return "";
    }
    return commands.recordNoiseRow(row);
}

/**
 * The same <vendor>-<architecture>-<runtime> string as test/helpers/noise-floor.ts adapterClass() (a duplicated 3-line
 * function: the Node helper imports node:fs and cannot be bundled for Chromium; test/browser/skeleton.test.ts pins the
 * literal the Node form pins).
 * @param caps - the context's caps
 * @returns the adapter class string
 */
export function adapterClassBrowser(caps: GpuCaps): string {
    return `${caps.vendor}-${caps.architecture}-${caps.runtime}`;
}
