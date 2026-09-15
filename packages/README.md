# graphty staging workspace

This directory is a STAGING AREA for packages that are destined for the graphty pnpm/Nx monorepo at
`/home/apowers/Projects/graphty-monorepo`. It is laid out exactly like the monorepo root so that each
package directory can later be moved with a plain `mv`:

```
packages/                      # mirrors the monorepo root
+-- package.json               # private "graphty-staging": the subset of root devDependencies needed here
+-- pnpm-workspace.yaml        # graph-format, graph-io
+-- knip.config.ts             # the two workspace entries, in the monorepo's shape
+-- tsconfig.base.json         # VERBATIM copy of the monorepo file
+-- eslint.config.js           # VERBATIM copy of the monorepo file (root flat config)
+-- vitest.shared.config.ts    # VERBATIM copy (unused by the packages, kept so paths match)
+-- vite.shared.config.ts      # VERBATIM copy (unused by the packages, kept so paths match)
+-- .prettierrc .prettierignore .npmrc   # VERBATIM copies
+-- move/                      # the root-file edits of the move, ready to apply (see below)
+-- graph-format/              # @graphty/graph-format (design/graph-format/graph-format-design.md)
+-- graph-io/                  # @graphty/graph-io (phase IO1: eight formats, registry, subpath exports)
```

Lint and type RULES are therefore the monorepo's. Tool VERSIONS are not: the staging root resolves
its caret ranges fresh while the monorepo's `pnpm-lock.yaml` pins older ones (knip 5.77.4 here vs
5.88.1 in staging, prettier 3.7.4 vs 3.9.6, typescript-eslint 8.50.1 vs 8.70.0, @vitest/coverage-v8
3.2.4 vs 3.2.7, papaparse 5.5.3 vs 5.7.0, fast-check 4.5.2 vs 4.10.0, tsx 4.21.0 vs 4.23.13). A
"knip clean" or "prettier clean" result here is therefore not proof of the same in the monorepo;
the rehearsal below ran the monorepo's versions. Do not edit the copied shared configs here; if a
rule change is needed, change it in the monorepo and re-copy.

## Working here

```bash
cd /home/apowers/Projects/webgpu-graph-algorithms/packages
pnpm install
pnpm -r run build:all
pnpm -r run lint
pnpm -r run typecheck
pnpm -r run test:run
```

Or per package, from inside `graph-format/` or `graph-io/`: `pnpm run lint`, `pnpm run typecheck`,
`pnpm run test:run`, `pnpm run build:all`, `pnpm run coverage` (previews on ports 9056 / 9057).

## Moving the packages into the monorepo

This checklist is the sequence that was REHEARSED on 2026-09-14 in a scratch copy of the monorepo
(`rsync` of `/home/apowers/Projects/graphty-monorepo` with the two package directories moved in and
every root touch point of design section 13.3 applied). Every command below was run there with the
monorepo's own toolchain (pnpm 10.0.0, Node 22.22.1, the monorepo's `pnpm-lock.yaml`) and passed;
the deviations found on the way were fixed in the packages themselves (section 5 lists them). Move
both packages in one go: `graph-io` depends on `graph-format` (`workspace:*` + peer) and the root
edits below cover both.

### 1. Move the directories

```bash
STAGE=/home/apowers/Projects/webgpu-graph-algorithms/packages
MONO=/home/apowers/Projects/graphty-monorepo
cd "$MONO" && git status --short   # start from a clean tree on a branch

# drop install / build artifacts (the monorepo re-creates them; the pnpm symlinks would be stale)
for p in graph-format graph-io; do
    rm -rf "$STAGE/$p/node_modules" "$STAGE/$p/dist" "$STAGE/$p/coverage" "$STAGE/$p/tmp" "$STAGE/$p"/*.tsbuildinfo
done
mv "$STAGE/graph-format" "$MONO/graph-format"
mv "$STAGE/graph-io" "$MONO/graph-io"
```

Nothing inside either package duplicates a monorepo root file (each carries only `package.json`,
`project.json`, `tsconfig*.json`, `vitest.config.ts`, `README.md`, `CLAUDE.md`, `LICENSE` plus
`src/ test/ scripts/ benchmarks/`, the same set as `algorithms/`), so nothing has to be deleted
after the `mv`. `graph-io/tmp/bundle-smoke.mjs` is a staging-only smoke script and is removed by
the `rm -rf` above. The root `.gitignore` ignores no file of either package (checked with
`git ls-files --others --ignored --exclude-standard graph-format graph-io`: 387 files would have been
added at the rehearsal, 416 after audit round 1, none ignored).

### 2. Root touch points

Two files under `move/` carry every root edit, generated from the rehearsed copy:

```bash
cd "$MONO"
git apply --check "$STAGE/move/root-touch-points.diff"    # dry run; passes against the tree of 2026-09-14
git apply "$STAGE/move/root-touch-points.diff"
python3 "$STAGE/move/apply-root-claude-md.py" CLAUDE.md   # idempotent; refuses to write if an anchor moved
```

What the diff does (design section 13.3, mirroring how `algorithms` is handled), file by file:

- `pnpm-workspace.yaml`: `graph-format`, `graph-io` first (before `algorithms`; bottom of the chain).
- `commitlint.config.js`: `"graph-format"`, `"graph-io"` added to `scope-enum`.
- `knip.config.ts`: the two workspace entries from `$STAGE/knip.config.ts`, with the monorepo's
  per-workspace `ignore: ["dist/**", "coverage/**", "node_modules/**"]` line
  (`entry: ["src/index.ts", "test/**/*.test.ts", "test/types/**/*.test-d.ts", "scripts/**/*.{ts,js}"]`,
  `project: ["src/**/*.ts", "test/**/*.ts", "benchmarks/**/*.ts", "scripts/**/*.{ts,js}"]`).
  `src/index.ts` of graph-io re-exports every subpath barrel, so no extra entries are needed.
- `package.json` (root): `coverage:preview:graph-format` (port 9056) and `coverage:preview:graph-io`
  (port 9057) scripts.
- `README.md` (root): a `### @graphty/graph-format` and a `### @graphty/graph-io` block with npm
  badges, placed before `@graphty/remote-logger`.
- `tools/prepush.sh`: two lines at the top of the fast-test block,
  `(cd graph-format && npm run test:run) || { FAILED=1; TESTS_FAILED=1; }` and the same for
  `graph-io`. Its Build step is `pnpm -r run build` (tsc only, not `build:all`); see section 5 for
  why that works now.
- `tools/merge-coverage.sh`: `PACKAGES=("graph-format" "graph-io" "algorithms" "layout" "graphty" "graphty-element")`
  and the two help / error hints.
- `.github/workflows/ci.yml`: (a) `Upload graph-format build` / `Upload graph-io build` steps
  (`build-graph-format` -> `graph-format/dist/`, `build-graph-io` -> `graph-io/dist/`); (b) shards
  `graph-format` and `graph-io` in the test matrix (`test-command: pnpm exec nx run
graph-format:coverage` / `graph-io:coverage`, `needs-browser: false`, `needs-storybook: false`);
  (c) `Download graph-format build` / `Download graph-io build` steps in the test job (every shard
  downloads every build, so graph-io's shard sees `graph-format/dist`); (d) the two shard names added
  to the `if:` of the `Upload coverage (...)` step -- this step lists shards explicitly, and without
  it no `coverage-graph-format` / `coverage-graph-io` artifact exists and `coverage.yml`'s
  `merge-coverage.sh --ci` fails with "Missing coverage for packages: graph-format graph-io"; (e) a
  `Build graph-format and graph-io (PR)` step after `Build affected (PR)`: on a PR that touches only
  the older packages, `nx affected -t build` builds `graphty`'s dependency closure, which does not
  contain the two new packages (`nx show projects --affected --files=graphty/src/App.tsx` is
  `["graphty"]`), so their `dist/` would be missing, the upload would produce no artifact and every
  test shard's download step would fail with "Artifact not found". The explicit build is a no-op
  through the Nx cache when the affected build already produced it.
- `.github/workflows/release.yml`: `Download graph-format build` / `Download graph-io build` steps
  with `run-id: ${{ github.event.workflow_run.id }}` and the GitHub token, before the algorithms one.
- `CLAUDE.md` (root, via the script): naming-table rows, package-directory rows, the two tree lines
  under Monorepo Structure, coverage-preview commands and ports 9056 / 9057, test-project entries,
  "18 parallel jobs" and the two shard names, the two package CLAUDE.md links, and the build order
  "graph-format -> graph-io -> algorithms -> layout -> graphty-element -> graphty". The script exists
  because those tree and build-order lines carry non-ASCII characters that a plain-ASCII diff cannot
  hold; it edits nothing else.

Not touched, on purpose: `nx.json` (`release.projects: ["*"]` already picks the two projects up:
`nx show projects --with-target=nx-release-publish` lists them), `vite.shared.config.ts`
(no dev server), `coverage.yml` (its `coverage-*` artifact glob already matches), `deploy-pages.yml`
(no docs site for either package), `design/graph-format/` (already in the monorepo).

### 3. Install

```bash
cd "$MONO"
pnpm install            # NOT --frozen-lockfile: this is the step that adds the two importers to pnpm-lock.yaml
pnpm install --frozen-lockfile   # then passes, which is what CI runs
```

Expected lockfile change (`move/pnpm-lock.yaml.diff` is the rehearsed result, for reference only --
regenerate, do not apply): two new `importers:` blocks, `webgpu@0.4.0` (+ `debug`) added, and
`@types/three`'s transitive `@webgpu/types` re-resolved from 0.1.68 to 0.1.72 (its range accepts
both; graph-format's `^0.1.72` makes pnpm dedupe to the newer one). pnpm prints "dependencies have
build scripts that were ignored: webgpu": expected, the postinstall only strips a macOS quarantine
attribute (a macOS developer who wants the GPU audit to run adds
`"pnpm": { "onlyBuiltDependencies": ["webgpu"] }` to the root package.json and runs `pnpm rebuild`;
on Linux and in CI nothing is needed). `pnpm audit --audit-level=high` is unchanged by the two
packages (38 low / moderate findings before and after, exit 0).

### 4. Verify

Run from the monorepo root; the rehearsed results are in brackets.

```bash
pnpm exec nx run-many --target=lint  --projects=graph-format,graph-io   # [pass; eslint 119 + 114 files, tsc clean]
pnpm exec nx run-many --target=build --projects=graph-format,graph-io   # [pass; graph-format first (^build)]
pnpm exec nx run-many --target=test  --projects=graph-format,graph-io   # [59 files / 1304 tests; 49 files / 1498 tests at the rehearsal; 59 / 1309 and 75 / 4271 (34 skipped without IO_BENCH=1) after audit round 1]
pnpm exec knip                                                          # [only the two pre-existing graphty findings, see below]
pnpm exec nx run graph-format:coverage && pnpm exec nx run graph-io:coverage && ./tools/merge-coverage.sh
                                                                        # [96.0 percent / 97.1 percent lines, merged]
(cd graph-format && pnpm exec tsc --noEmit -p tsconfig.strict-consumer.json)   # [pass]
(cd graph-io     && pnpm exec tsc --noEmit -p tsconfig.strict-consumer.json)   # [pass]
# the pre-push hook's own path (tools/prepush.sh builds with plain tsc, then runs the fast tests):
rm -rf graph-format/dist graph-io/dist && pnpm -r --filter graph-format --filter graph-io run build \
    && (cd graph-io && npm run test:run)                                # [49 files passed at the rehearsal (75 after audit round 1); 3 build-output checks skip without the bundle]
```

`pnpm exec knip` exits 1 in the monorepo TODAY, before the move, on two unused exports in
`graphty/src/components/shell/readings/nodeMetricReading.ts` (`DISPLAY_DECIMAL_PLACES`,
`NEAR_ZERO_TIE_SHARE`); the move adds nothing to that list. Since `tools/prepush.sh` runs knip, the
pre-push hook is red until those two are fixed or marked `@public`.

Rehearsal-only gotcha, in case the rehearsal is repeated: knip's git-ignore check matches the
ABSOLUTE path, so a scratch copy under a directory named `tmp/` (which the monorepo's `.gitignore`
lists) makes knip drop every script-derived entry (`tsx benchmarks/run.ts`, `tsx
src/benchmark-all-algorithms.ts`, `bin/*.js`) and report 15 bogus "unused files" across
graph-format, graph-io, algorithms, remote-logger and tools. Put the copy somewhere not named `tmp`,
or neutralise the `tmp/` line of the copy's `.gitignore` for the knip run.

Not rehearsed (needs the real repository): the GitHub Actions run itself, `nx affected` against
real commits (reasoned with `nx show projects --affected --files=...` instead), `nx release` (needs
the tags), the full `tools/prepush.sh` (the other packages' builds; the two new packages' lines of it
were run verbatim), and macOS.

### 5. What changed in the packages because of the rehearsal (already applied here)

Each of these was a failure in the scratch monorepo; the fix lives in the package so the move needs
no further hand edits.

- `graph-format/package.json`: `"webgpu": "^0.4.0"` devDependency. `test/audit/gpu-upload.test.ts`
  does `import("webgpu")` (Google Dawn for Node); in staging the module came from the ROOT
  package.json through `shamefully-hoist`, in the monorepo nothing provides it and `tsc --noEmit`
  (the `lint` script) failed with TS2307. Design 13.2's devDependency list did not foresee the GPU
  audit; this is a recorded deviation (the alternative, an untyped dynamic specifier, would make the
  audit skip silently wherever Dawn is absent). The staging root's own `webgpu` devDependency was
  removed as redundant. `test/build-output.test.ts` now pins both `webgpu` and `@webgpu/types`.
- `graph-format/graph-format.ts` (new, one line: `export * from "./src/index.js";`), included in
  `tsconfig.json` and `tsconfig.build.json`, pinned by `test/build-output.test.ts`. Exactly what
  `algorithms/algorithms.ts` does: a tsc-only build (`tools/prepush.sh` runs `pnpm -r run build`,
  which is `tsc -p tsconfig.build.json`, NOT `build:all`) now also yields `dist/graph-format.js` and
  `dist/graph-format.d.ts`, the files package.json `exports` point at; `build:bundle` then overwrites
  both. Without it the pre-push hook's own sequence failed twice: graph-io's `build` (TS2307 on
  `@graphty/graph-format`, which aborts the whole recursive build) and 45 of graph-io's 49 test files
  ("Failed to resolve entry for package @graphty/graph-format").
- knip 5.77.4 (the monorepo's pinned version; staging had 5.88.1, which does not report exports used
  only in their own file): `DerivedReport` (`graph-format/src/snapshot/derived.ts`), `ColumnSpec`
  (`graph-format/test/helpers/parts.ts`), `ValueKind` (`graph-io/src/common/declared-types.ts`),
  `DotTokenKind` (`graph-io/src/formats/dot/tokenizer.ts`) and `SectionKind`
  (`graph-io/src/formats/pajek/syntax.ts`) are no longer exported (none is public: design 12.2 inlines
  the derived report shape, the others are file-local token / section / value kinds), and
  `fast-check` was removed from `graph-io/package.json` at the time (imported nowhere in graph-io
  then; the fuzz audit suite `test/audit/fuzz-mutations.test.ts` imports it now, so it is a
  devDependency again and knip is clean with it). The built d.ts files still carry the un-exported
  aliases, so `dist/*.d.ts` and the strict-consumer compile are unchanged.
- Not applied, optional: the monorepo's prettier 3.7.4 formats six files differently from staging's
  3.9.6 (long union types broken one member per line: `graph-format/test/audit/differential-structure.test.ts`,
  `graph-format/test/helpers/parts.ts`, `graph-io/src/common/declared-types.ts`, `graph-io/src/types.ts`,
  `graph-io/test/registry.test.ts`, and `for (...;)` spacing in `graph-io/src/formats/pajek/importer.ts`).
  Prettier is not enforced by the monorepo's CI or hooks (the root `format:check` is already red on
  other packages), so run `pnpm exec prettier --write graph-format graph-io` after the move only if
  you want them formatted by the monorepo's version; the two deliberately malformed JSON corpus
  fixtures (`graph-io/test/corpus/malformed/json/{invalid-json,not-json}.json`) make prettier print
  a parse error for those two files, exactly as their graphty-element copies do today.

### 6. After the move

- npmjs.com: `@graphty/graph-format` and `@graphty/graph-io` each need a trusted publisher (OIDC
  provenance, `publishConfig.provenance: true`) before the first release; `nx release` versions them
  independently (`0.1.0` on disk, the first `feat(graph-format):` / `feat(graph-io):` on master
  publishes `0.2.0`, design 13.2 / 13.5).
- Commit scopes `graph-format` and `graph-io` are accepted by commitlint after the diff.
- `graphty-element` consumer migration (design 8.2 landing order, 14.4): its `DataSource` subclasses
  become wrappers over the graph-io importers and `src/data/format-detection.ts` is replaced by
  `sniff()`; the corpus under `graphty-element/test/helpers/corpus/` was copied verbatim into
  `graph-io/test/corpus/` and can be deleted there once the element's own corpus tests move. Once
  graphty-element depends on graph-io, the `Build graph-format and graph-io (PR)` step of ci.yml
  becomes redundant (the affected closure then contains both) and can be dropped.
- The Neo4j corpus (`graph-io/test/corpus/neo4j`, `.../malformed/neo4j`) was written for graph-io;
  its generator is `webgpu-graph-algorithms/tmp/make-neo4j-corpus.py` and need not move.
- `webgpu-graph-algorithms` repeats this checklist at W1 (design 13.3).

### 7. What to delete here afterwards

Delete NOTHING in this staging area except the copied shared configs (`tsconfig.base.json`,
`eslint.config.js`, `vitest.shared.config.ts`, `vite.shared.config.ts`, `.prettierrc`,
`.prettierignore`, `.npmrc`) once both packages have moved; the moved package directories are gone
by then, and the staging `package.json`, `pnpm-workspace.yaml`, `knip.config.ts`, `.gitignore`,
`move/` and this README remain as the record of how the staging and the move were done until the
owner removes them.

## Notes

- Package manager is pnpm 10 (`packageManager: pnpm@10.0.0`), Node >= 18.19.0 (CI uses Node 22).
- The staging root deliberately has no Nx, husky, commitlint or release tooling; the `project.json`
  files are shipped anyway and Nx picks them up on move-in (`nx show projects` lists both; the
  `test` target inherits the root `targetDefaults`, cache and `^production` inputs, as for layout).
- `pnpm.overrides` from the monorepo root are not replicated here; on move-in they rewrite the
  lockfile specifiers of `vite`, `vitest` and `fast-xml-parser` (e.g. `vite: ^7.3.5`) without
  changing the resolved versions the packages were tested with.
- Plain ASCII everywhere (`--` for dashes, straight quotes).
