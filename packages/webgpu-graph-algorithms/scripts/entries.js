/**
 * The bundle entries of @graphty/webgpu-graph-algorithms (spec 2.5): the root barrel and the two acquisition
 * subpaths. Shared by scripts/build-bundle.js (dist/<name>.js) and scripts/bundle-types.js (dist/<name>.d.ts);
 * package.json "exports" lists the same names, which test/build-output.test.ts checks. There is NO root
 * webgpu-graph-algorithms.ts shim (graph-io's convention, not graph-format's).
 *
 * Keys are the output names under dist/, values the source entry relative to the package root.
 */

export const ENTRIES = Object.freeze({
    "webgpu-graph-algorithms": "src/index.ts",
    browser: "src/browser/index.ts",
    node: "src/node/index.ts",
});

/**
 * The declaration file a bundle entry re-exports: the tsc output of its source entry under dist/src/.
 * @param source - the source entry relative to the package root (`src/node/index.ts`)
 * @returns the relative import specifier for the shim (`./src/node/index.js`)
 */
export function declarationSpecifier(source) {
    return `./${source.replace(/\.ts$/, ".js")}`;
}
