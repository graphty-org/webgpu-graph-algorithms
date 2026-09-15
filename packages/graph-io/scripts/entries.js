/**
 * The bundle entries of @graphty/graph-io: the root barrel and one entry per format subpath
 * (design section 8.2: `@graphty/graph-io/gexf`, `/graphml`, `/gml`, `/dot`, `/pajek`, `/csv`,
 * `/json`, `/neo4j`). Shared by scripts/build-bundle.js (which emits dist/<name>.js for each) and
 * scripts/bundle-types.js (which writes the matching dist/<name>.d.ts shim); package.json
 * "exports" must list the same names, which test/build-output.test.ts checks.
 *
 * Keys are the output names under dist/, values the source entry relative to the package root.
 */

export const ENTRIES = Object.freeze({
    "graph-io": "src/index.ts",
    gexf: "src/formats/gexf/index.ts",
    graphml: "src/formats/graphml/index.ts",
    gml: "src/formats/gml/index.ts",
    dot: "src/formats/dot/index.ts",
    pajek: "src/formats/pajek/index.ts",
    csv: "src/formats/csv/index.ts",
    json: "src/formats/json/index.ts",
    neo4j: "src/formats/neo4j/index.ts",
});

/**
 * The declaration file a bundle entry re-exports: the tsc output of its source entry under dist/src/.
 * @param source - the source entry relative to the package root (`src/formats/gexf/index.ts`)
 * @returns the relative import specifier for the shim (`./src/formats/gexf/index.js`)
 */
export function declarationSpecifier(source) {
    return `./${source.replace(/\.ts$/, ".js")}`;
}
