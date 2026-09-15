#!/usr/bin/env python3
"""Apply the graph-format / graph-io edits to the monorepo root CLAUDE.md.

The root CLAUDE.md is the one touch point whose edited lines contain non-ASCII
characters (the box-drawing tree under "Monorepo Structure" and the arrows of the
build-order line), so it is edited by this script instead of a unified diff; the
script itself is plain ASCII (the two non-ASCII characters are written as chr() code points).

Usage (from the monorepo root, after moving the two package directories in):

    python3 /home/apowers/Projects/webgpu-graph-algorithms/packages/move/apply-root-claude-md.py CLAUDE.md

Every edit anchors on a line that must occur exactly once; the script refuses to write
anything if an anchor is missing or ambiguous, and it is idempotent (a second run finds
the new lines already present and changes nothing).
"""

import sys

BOX_BRANCH = chr(0x251C) + chr(0x2500) + chr(0x2500)  # the tree prefix of the existing Monorepo Structure block
ARROW = chr(0x2192)  # the arrow of the existing build-order line

# (anchor line, lines to insert BEFORE the anchor, lines to insert AFTER the anchor)
INSERTIONS = [
    (
        "| `@graphty/layout` | **layout** | - |",
        [],
        [
            '| `@graphty/graph-format` | **graph-format** | "format", "snapshot package" |',
            "| `@graphty/graph-io` (and `@graphty/graph-io/<format>` subpaths: gexf, graphml, gml, dot, pajek, csv, json, neo4j)"
            ' | **graph-io** | "io", "importers" |',
        ],
    ),
    (
        "| `@graphty/algorithms` | `algorithms/` | 1.4.0 | 98+ graph algorithms (traversal, pathfinding, centrality, "
        "clustering, flow, link prediction) |",
        [
            "| `@graphty/graph-format` | `graph-format/` | 0.1.0 | Frozen CSR graph snapshot over typed arrays (builder, "
            "id map, attribute columns, views, wire form); zero dependencies |",
            "| `@graphty/graph-io` | `graph-io/` | 0.1.0 | Importers and exporters (GEXF, GraphML, GML, DOT, Pajek, CSV, "
            "JSON, Neo4j) for the graph-format snapshot; subpath exports per format |",
        ],
        [],
    ),
    (
        BOX_BRANCH + " algorithms/           # @graphty/algorithms package",
        [
            BOX_BRANCH + " graph-format/         # @graphty/graph-format package (bottom of the dependency chain)",
            BOX_BRANCH + " graph-io/             # @graphty/graph-io package (depends on graph-format)",
        ],
        [],
    ),
    (
        "pnpm run coverage:preview:graphty          # Port 9054",
        [],
        [
            "pnpm run coverage:preview:graph-format     # Port 9056",
            "pnpm run coverage:preview:graph-io         # Port 9057",
        ],
    ),
    (
        "**graphty:**",
        [
            "**graph-format:**",
            "- Single test project (Node.js); `test/types/*.test-d.ts` are compile-only "
            "(`npm run typecheck:strict-consumer` after a build)",
            "",
            "**graph-io:**",
            "- Single test project (Node.js); resolves `@graphty/graph-format` through `graph-format/dist`, "
            "so build graph-format first",
            "",
        ],
        [],
    ),
    (
        "The CI runs 14 parallel test jobs:",
        [],
        ["- `graph-format`", "- `graph-io`"],
    ),
    (
        "- `algorithms/CLAUDE.md` - Algorithm-specific notes (e.g., floyd-warshall hang)",
        [
            "- `graph-format/CLAUDE.md` - Snapshot invariants, freeze pipeline, adding a view / a dtype",
            "- `graph-io/CLAUDE.md` - Importer / exporter contract, adding a format",
        ],
        [],
    ),
]

# (anchor line, replacement line)
REPLACEMENTS = [
    (
        "- Coverage previews: 9051-9054",
        "- Coverage previews: 9051-9054, graph-format 9056, graph-io 9057",
    ),
    (
        "| `ci.yml` | Push/PR | Build, lint, sharded tests (14 parallel jobs) |",
        "| `ci.yml` | Push/PR | Build, lint, sharded tests (18 parallel jobs) |",
    ),
    (
        "The CI runs 14 parallel test jobs:",
        "The CI runs 18 parallel test jobs:",
    ),
    (
        "  - Build order enforced by TypeScript: `algorithms` " + ARROW + " `layout` " + ARROW + " `graphty-element` "
        + ARROW + " `graphty`",
        "  - Build order enforced by TypeScript: `graph-format` " + ARROW + " `graph-io` " + ARROW + " `algorithms` "
        + ARROW + " `layout` " + ARROW + " `graphty-element` " + ARROW + " `graphty`",
    ),
]


def index_of(lines, anchor):
    """The index of the one line equal to `anchor`, or an error naming the problem."""
    hits = [i for i, line in enumerate(lines) if line == anchor]
    if len(hits) != 1:
        raise SystemExit(f"anchor found {len(hits)} times (need exactly 1): {anchor!r}")
    return hits[0]


def apply(text):
    lines = text.split("\n")
    for anchor, before, after in INSERTIONS:
        new_lines = before + after
        if new_lines and all(line in lines for line in new_lines if line != ""):
            continue  # already applied
        i = index_of(lines, anchor)
        lines[i:i + 1] = before + [anchor] + after
    for anchor, replacement in REPLACEMENTS:
        if replacement in lines and anchor not in lines:
            continue  # already applied
        i = index_of(lines, anchor)
        lines[i] = replacement
    return "\n".join(lines)


def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: apply-root-claude-md.py <path to the monorepo root CLAUDE.md>")
    path = argv[1]
    with open(path, encoding="utf-8") as handle:
        original = handle.read()
    updated = apply(original)
    if updated == original:
        print("CLAUDE.md already up to date")
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(updated)
    print(f"CLAUDE.md updated ({updated.count(chr(10)) - original.count(chr(10))} lines added)")


if __name__ == "__main__":
    main(sys.argv)
