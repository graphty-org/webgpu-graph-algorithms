# CLAUDE.md

This file provides guidance to Claude Code when working with the @graphty/graph-io package.

## Project Overview

@graphty/graph-io is the importer / exporter companion of @graphty/graph-format. It parses GEXF,
GraphML, GML, DOT, Pajek, CSV, JSON and Neo4j into a `GraphSink` (`GraphBuilder` or a caller's sink)
and writes the same formats back out from a `GraphSnapshot`. The core format stays zero-dependency
and so does this package at runtime: every format is parsed by a hand-written streaming tokeniser
(one shared XML tokenizer for GEXF and GraphML, one shared RFC 4180 record reader for CSV and
Neo4j, line and token readers for Pajek, GML and DOT). It also owns the io contract types that need
the DOM lib (`ReadableStream`, `AbortSignal`).

The normative design is `design/graph-format/graph-format-design.md` in the monorepo: section 8
(populating the format: 8.2 package split, 8.3 sink contract, 8.4 importer shape and direction
rules, 8.5 exporter shape and loss notes, 8.6 report and error aggregation), section 5.1 (declared
types to dtypes, temporal values), section 4 (ids and the canonical coercion rule), section 12.4
(io contract types, normative) and section 13 (package layout). The per-format requirements are in
research note 07 (`tmp/graph-format-design/07-file-format-requirements.md`).

## Package Structure

```
graph-io/
+-- package.json                  # @graphty/graph-io, ESM only, sideEffects false, "." + one subpath per format
+-- project.json                  # Nx project "graph-io"
+-- tsconfig.json                 # lint/typecheck: src/ test/ benchmarks/ + ../graph-format/src (paths), noEmit, lib DOM
+-- tsconfig.build.json           # emit: src/ only, rootDir ".", outDir dist (-> dist/src/), format resolved from its dist
+-- tsconfig.strict-consumer.json # test/types/*.test-d.ts against dist/*.d.ts under the strict flags
+-- vitest.config.ts              # single project, environment node, pool forks, thresholds 80/80/75/80
+-- scripts/entries.js            # the bundle entries: graph-io + one per format (shared by both scripts below)
+-- scripts/build-bundle.js       # one multi-entry vite lib build -> dist/graph-io.js, dist/<format>.js, dist/chunks/*
+-- scripts/bundle-types.js       # dist/graph-io.d.ts and dist/<format>.d.ts, one-line re-exports of dist/src/**
+-- src/
|   +-- index.ts                  # the only root barrel; named exports only
|   +-- types.ts                  # section 12.4 contract types and ImportError
|   +-- registry.ts               # FormatRegistry, importGraph / exportGraph / checkExport / sniff, the default registry
|   +-- sniff.ts                  # rankFormats / sniffFormat (extension + MIME + content), the JSON dialect head sniff
|   +-- children.ts               # the children CSR over a parent / parents column (design 7.1)
|   +-- common/                   # shared by every format (see the module map in STATUS.md)
|   |   +-- codes.ts              # the one definition of every shared issue / loss code (E_MISSING_ID, W_ROLE_DROPPED, ...)
|   |   +-- report.ts             # ImportReportBuilder: issues, error limit, warnOnce, fail() -> ImportError
|   |   +-- input.ts              # textChunks / readText / LineReader: streaming fatal UTF-8, BOM, abort, progress
|   |   +-- options.ts            # resolveImportOptions / resolveExportOptions / reportSinkOptions / reportUnusedOptions
|   |   +-- direction.ts          # DirectionResolver (8.4 rules), pairFolding() for exporters (3.6 pairs, mutual marks)
|   |   +-- ids.ts                # canonical / string / number coercion, IdCoercer (W_ID_MERGED)
|   |   +-- text.ts               # the 5.1 lexical grammar for untyped cells, TextCellWriter (per-column dtype, lexical forms kept)
|   |   +-- declared-types.ts  temporal.ts  lists.ts  attributes.ts   # declared attributes (5.1, 5.5), declareResolved (5.6 rename rule)
|   |   +-- weights.ts            # weightFrom, parseWeightText, explicitWeights() (3.7: the weight role column's validity)
|   |   +-- export.ts             # LOSS codes, capabilities(), checkCapabilities() + CheckExtras (roles, roleNames), sanitizeIds()
|   |   +-- xml.ts                # the streaming XML tokenizer (GEXF, GraphML), entity decoding, xmlIllegalTextNotes()
|   |   +-- escape.ts  format.ts  writer.ts   # quoting per format, formatDecimal / formatGmlReal, encodeChunks / joinText
|   +-- formats/<format>/         # gexf graphml gml dot pajek csv json neo4j
|       +-- index.ts              # the subpath barrel: <fmt>Importer, <fmt>Exporter, option types, code tables
|       +-- importer.ts           # GraphImporter<Opts>
|       +-- exporter.ts           # GraphExporter<Opts>
|       +-- ...                   # private helpers (tokenizer, schema, syntax, records, ...)
+-- test/
|   +-- corpus/<format>/          # fixtures + manifest.json (expected counts); corpus/malformed/<format>/
|   +-- helpers/corpus.ts         # manifest loaders, input shapes (bytes, chunks, streams)
|   +-- helpers/roundtrip.ts      # compareSnapshots / expectSameSnapshot / roundTrip
|   +-- common/*.test.ts          # one per common module
|   +-- formats/<format>/*.test.ts
|   +-- registry.test.ts  sniff.test.ts  children.test.ts  index.test.ts  build-output.test.ts
|   +-- types/*.test-d.ts         # compile-only, run under tsconfig.json and tsconfig.strict-consumer.json
+-- benchmarks/                   # run.ts, run with tsx; ignored by eslint and coverage
```

## Essential Commands

```bash
pnpm run build:all       # tsc -p tsconfig.build.json, then the multi-entry vite bundle and the d.ts shims
pnpm run test:run        # run all tests once (pnpm exec vitest run test/formats/<fmt> for one format)
pnpm run coverage        # v8 coverage, thresholds 80/80/75/80
pnpm run coverage:preview # serve coverage report on port 9057
pnpm run lint            # eslint (root flat config) + tsc --noEmit
pnpm run typecheck       # tsc --noEmit only
pnpm run typecheck:strict-consumer  # compile test/types/*.test-d.ts against dist/*.d.ts (build first)
pnpm run benchmark       # tsx benchmarks/run.ts
pnpm run ready:commit    # build, lint, strict-consumer compile, tests
cd .. && pnpm exec knip  # unused files / exports / dependencies, both workspaces
```

@graphty/graph-format must be built before this package's tests or build run: vitest resolves it
through pnpm's workspace symlink and the format package's `exports` -> `dist/`; tsc for lint and
tests resolves its sources through `paths`. From the workspace root, `pnpm -r run build:all` orders
the two correctly.

## Key Design Principles

- Importers push scalars into the sink in one pass and never freeze; direction is resolved by the
  importer through `DirectionResolver` (design 8.4 rules 1 and 2); the registry's `importGraph()`
  is the only place that creates a builder and freezes.
- Nothing is silently dropped: an unsupported construct is an `ImportIssue` (report) or a `LossNote`
  (`check()`), never a fallback. The builder throws on the first hard error; the importer catches
  it per element, records the issue, skips the element and continues until `errorLimit`, then
  throws `ImportError` with the partial report. Fatal conditions (invalid UTF-8, malformed
  syntax, unknown format) are `report.fail()` -> `ImportError` at once.
- Every exporter's `capabilities` table is its fidelity matrix; `check()` is `checkCapabilities()`
  (with the format's `CheckExtras`: the roles it has a slot for and the fixed names its importer
  gives them) plus the format's own notes, and `export()` throws the `E_*`-coded conditions before
  writing. A note predicts every difference the format's own importer produces on re-import,
  including the importer's own rules (a plain `weight` column read as THE weight, a role-less
  `label` column gaining the role, an f64 column of integers read back as i32, a dict column read
  back as string), not only what the file cannot hold.
- Issue and loss codes: one code per concept, defined once. A concept shared by several formats
  has an unprefixed code in `src/common/codes.ts` (`E_MISSING_ID`, `W_DUPLICATE_NODE`,
  `W_ROLE_DROPPED`, ...); a format-specific one carries the format prefix (`E_GML_SECOND_GRAPH`,
  `W_PAJEK_KEY_DROPPED`). Every subpath exports `<FMT>_ISSUE` and `<FMT>_LOSS`, whose keys are the
  code without its `E_` / `W_` and `<FMT>_` prefixes, and every code an importer records is a
  member of its table.
- Under `onMixedDirection: "directed"` every exporter without mixed direction folds an expanded
  pair back to one directed edge (8.5 "folding pairs back"); `"undirected"` writes the whole graph
  undirected. A mutual pair is written as two directed edges (`W_MUTUAL_EXPANDED`) or, where the
  format has an undirected slot but no mutual one (GraphML, Pajek), as one undirected edge
  (`W_MUTUAL_AS_UNDIRECTED`).
- Every importer checks the cancellation signal at least every 64 elements and once more before
  `report.finish()`, so an abort raised from the sink during a single in-memory chunk rejects.
- Id coercion is an importer option (`ids`); the core never coerces. `sanitizeIds` defaults to
  `"error"`: an exporter never renames a node silently.
- Declares `@graphty/graph-format` in BOTH `dependencies` (`workspace:*`) and `peerDependencies`;
  no consumer PR that depends on the format may merge to master until the format is >= 1.0.0.
- Invalid UTF-8 is a `parse-error`, never a silent U+FFFD (`new TextDecoder("utf-8", { fatal: true })`).

## Adding a format

1. Create `src/formats/<fmt>/importer.ts` exporting `<fmt>Importer: GraphImporter<FmtImportOptions>`
   with `format`, `extensions`, `mimeTypes`, `sniff(head)` (0..1) and `import()`:
   `resolveImportOptions(options, { ids, defaultDirected, weightFrom })`, then
   `new ImportReportBuilder(format, errorLimit)`, `reportSinkOptions(sink, options, report)` and
   `reportUnusedOptions(options, report, USED_OPTIONS)` (W_OPTION_IGNORED for every common option
   the format has no use for), read the input through `textChunks` / `LineReader` (streaming) or
   `readText` (whole document), declare attributes through `common/attributes.ts`
   (`declareAttribute`, `declareResolved`: the `<name>#<id>` rename rule and W_ROLE_TAKEN), push
   edges through `DirectionResolver.setHeader()` / `addEdge()`, coerce ids through `IdCoercer`,
   write untyped cells through `TextCellWriter` (the 5.1 grammar per column), weights with
   `parseWeightText`, check the signal every few dozen elements (`throwIfAborted`),
   record issues with `report.error()` / `report.warning()` / `report.warnOnce()` and the counts in
   `report.counts`, and return `report.finish()`.
2. Create `src/formats/<fmt>/exporter.ts` exporting `<fmt>Exporter: GraphExporter<FmtExportOptions>`
   with a `capabilities` table, `check()` = `checkCapabilities(snapshot, capabilities, resolved)`
   plus the format's own notes, `export()` = `encodeChunks(write())` and `exportToString()` =
   `joinText(write())`, where `write()` is a generator of string parts that iterates nodes and
   logical edges in index order, folds expanded pairs (`pair` / `directed` role columns) and writes
   explicit weights only (the `weight` role column's validity). Use `childrenCsr()` for containment.
3. Create `src/formats/<fmt>/index.ts` (the subpath barrel: importer, exporter, option types, a
   frozen `<FMT>_ISSUE` / `<FMT>_LOSS` code table) and re-export it from `src/index.ts`.
4. Add the entry to `scripts/entries.js` and the `"./<fmt>"` export (types first) to
   `package.json`; `test/build-output.test.ts` checks the three agree.
5. Register it in `createRegistry()` (`src/registry.ts`) and add the name to `GRAPH_FORMATS`
   (`src/sniff.ts`, also the tie-break order of sniffing).
6. Add `test/corpus/<fmt>/` with a `manifest.json` (`format`, `description`, `files[]` with
   `path`, `source`, `license`, `expectedNodes`, `expectedEdges`, `features`), the malformed cases
   under `test/corpus/malformed/<fmt>/`, the name in `CORPUS_FORMATS` (`test/helpers/corpus.ts`), and
   `test/formats/<fmt>/*.test.ts` covering every corpus file (manifest counts, every input shape),
   export -> re-import equality (`expectSameSnapshot`), every LossNote path and every malformed
   file (`ImportError` with a report).
7. Document the format in README.md (the matrix and the known losses) and STATUS.md.

## House Style

Same as @graphty/graph-format (design section 13.4): JSDoc on every exported function, class and
method (`@param name - description`, `@returns`); explicit return types; `import { type X, y }`
inline qualifiers; `.js` suffixes on relative imports; camelCase fields; `curly`; `default-case`;
no default exports; no `console.log` in `src/`; no `eslint-disable` / `ts-expect-error`; plain
ASCII in every source and test file (non-ASCII fixture data is data; non-ASCII test constants are
built with `String.fromCharCode`).

## Distribution

- Root entry: `dist/graph-io.js` (ES module; `@graphty/graph-format` external, no other runtime
  dependency); types `dist/graph-io.d.ts` (a one-line re-export of `dist/src/index.d.ts`).
- Per-format subpaths `@graphty/graph-io/<fmt>`: `dist/<fmt>.js` and `dist/<fmt>.d.ts`, built by the
  same vite invocation so the code shared with the root lives once under `dist/chunks/` (one
  `ImportError` class, one `LOSS` table, one importer object whichever entry a consumer loads).
