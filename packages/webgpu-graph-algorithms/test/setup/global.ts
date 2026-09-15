/**
 * The node project's vitest globalSetup (contract 5.1). P0 form: no-ops, so vitest.config.ts resolves its
 * `globalSetup` entry. P2-T2 fills setup() (clears tmp/pipeline-keys/ and exports GRAPHTY_PIPELINE_KEY_LOG) and
 * teardown() (asserts test/helpers/override-matrix.ts covers every pipeline key the workers logged).
 */

/** Clears the pipeline-key log directory and exports its path (P2-T2); nothing to prepare at P0. */
export function setup(): void {
    // P0: no global state to prepare
}

/** Reads every worker's key log and asserts the override matrix covers each key (P2-T2); nothing to check at P0. */
export function teardown(): void {
    // P0: no global state to check
}
