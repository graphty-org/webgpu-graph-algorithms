#!/usr/bin/env python3
"""NetworkX ForceAtlas2 trajectory fixtures (spec 11.4 "oracle independence"; contract 5.4).

The committed JSON files next to this script are the reference the f64 oracle of
test/oracle/forceatlas2.ts is checked against in compat "networkx"
(test/oracle/forceatlas2-networkx.test.ts). The suite never runs Python: this
script runs ONCE, by hand, from packages/webgpu-graph-algorithms with the
interpreter of the gitignored venv at the repository root:

    python3 -m venv --without-pip ../../tmp/nx-venv
    python3 -m pip --python ../../tmp/nx-venv/bin/python install "networkx>=3.4" numpy scipy
    ../../tmp/nx-venv/bin/python test/fixtures/networkx/generate.py \
        --out test/fixtures/networkx/ --iters 1 5 50 --seed 7

(--without-pip because the system Python 3.10 has no ensurepip; pip 25.3 in
~/.local installs into the venv through --python.)

For every graph x variant x iteration count it writes <graph>-<variant>-iter<k>.json
with the SAME initial positions passed through networkx's `pos` argument (so the
oracle and NetworkX start from identical f64 coordinates), the per-edge weights
as an edge attribute for the weighted variant, and NetworkX's RAW output
positions (NetworkX 3.4.2 does not rescale forceatlas2_layout's result;
"rescaled": false). Every file carries the networkx and python versions and
the command line in its header.

Guarantees asserted before a file is written (contract 5.4):
- no self-loops and no parallel edges (NetworkX's degree must equal the
  snapshot's outDegree(), the default mass being degree + 1);
- the initial positions keep every pair >= 0.02 apart and every node >= 0.02
  from the origin, and along the whole trajectory (the positions at the start
  of every iteration up to the largest count) every pair stays > 0.01 apart and,
  when regular gravity is active, every node stays > 0.01 from the origin --
  the `max(d, 0.01)` floor and the `|q| > 0.01` gravity guard of the GPU law
  never engage, which is what makes NetworkX (no floor, no guard) an oracle;
- NetworkX's early exit (`abs(sum(update * factor)) < 1e-10`) did not fire:
  the k+1 run differs from the k run for every recorded k;
- the recorded networkx version is >= 3.4 (forceatlas2_layout exists);
- the LONG leg is meaningful: NetworkX's adaptive-speed controller is chaotic
  on some graph / variant pairs (a 1-ulp perturbation of the start grows to
  O(1) by iteration 50 on karate-linlog, star200-distributed, ...), so for
  every graph x variant the largest requested count is replaced by the
  largest count of the ladder max(iters), 40, 30, 20, 10 at which NetworkX's
  own 1-ulp sensitivity (max |p - p'| / max(|p|, 1) between the run from the
  fixture's start and the run from the start multiplied by 1 + 2^-52) is at
  most SENSITIVITY_MAX = 1e-8, two orders under the 1e-6 tolerance of the
  long leg; the measured value is recorded as "sensitivity" in every file
  (PLAN DECISION P3-T4, recorded for the owner's re-fix of spec 11.4's
  "1e-6 at 50" in docs/decisions/G3.md, spec 10.4 rule).

Positions that violate the separation rule are redrawn (the RandomState is
consumed deterministically, so the same seed always yields the same files).

After writing, the script runs the workspace's prettier (`pnpm exec prettier
--write`, from packages/) on the files it wrote, so the committed fixtures pass
`pnpm run format:check` exactly like the noise fixtures; dump() itself keeps the
arrays compact and prettier does the 120-column wrapping.
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import warnings

import networkx as nx
import numpy as np

GRAPHS = ("karate", "grid10", "star200", "gnm200", "path3", "karate3d")
DEFAULT_ITERS = (1, 5, 50)
VARIANTS = {
    "base": {},
    "linlog": {"linlog": True},
    "distributed": {"distributed_action": True},
    "strong": {"strong_gravity": True},
    "weighted": {"weight": "weight"},
    "gravity0": {"gravity": 0.0},
}
BASE_OPTIONS = {
    "jitter_tolerance": 1.0,
    "scaling_ratio": 2.0,
    "gravity": 1.0,
    "distributed_action": False,
    "strong_gravity": False,
    "linlog": False,
    "dissuade_hubs": False,
    "weight": None,
}
INITIAL_SEPARATION = 0.02
TRAJECTORY_SEPARATION = 0.01
MAX_ATTEMPTS = 50
LONG_LADDER = (40, 30, 20, 10)
SENSITIVITY_MAX = 1e-8
ULP = 2.0 ** -52
# The hand-computed fixture of test/oracle/forceatlas2-networkx.test.ts: nodes 0-1-2 at these positions.
PATH3_POSITIONS = [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]


def karate_edges():
    """Zachary's karate club as sorted index pairs (78 edges, nodes 0..33; attributes dropped)."""
    return sorted((min(u, v), max(u, v)) for u, v in nx.karate_club_graph().edges())


def grid_edges(w, h):
    """The w x h grid, row-major indices."""
    edges = []
    for r in range(h):
        for c in range(w):
            i = r * w + c
            if c + 1 < w:
                edges.append((i, i + 1))
            if r + 1 < h:
                edges.append((i, i + w))
    return sorted(edges)


def star_edges(leaves):
    """Node 0 joined to nodes 1..leaves."""
    return [(0, i) for i in range(1, leaves + 1)]


def gnm_edges(n, m, seed):
    """A seeded simple G(n, m) (no self-loops, no parallels)."""
    g = nx.gnm_random_graph(n, m, seed=seed)
    return sorted((min(u, v), max(u, v)) for u, v in g.edges())


def graph_spec(name, seed):
    """(edge pairs, node count, dim, variants, fixed initial positions or None) of a named graph."""
    if name == "karate":
        return karate_edges(), 34, 2, tuple(VARIANTS), None
    if name == "karate3d":
        return karate_edges(), 34, 3, ("base",), None
    if name == "grid10":
        return grid_edges(10, 10), 100, 2, tuple(VARIANTS), None
    if name == "star200":
        return star_edges(199), 200, 2, tuple(VARIANTS), None
    if name == "gnm200":
        return gnm_edges(200, 600, seed), 200, 2, tuple(VARIANTS), None
    if name == "path3":
        return [(0, 1), (1, 2)], 3, 2, ("base",), np.array(PATH3_POSITIONS)
    raise ValueError(name)


def build_graph(edges, n, weights):
    """A networkx Graph whose iteration order is 0..n-1, with an optional 'weight' attribute per edge."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    assert g.number_of_nodes() == n
    assert g.number_of_edges() == len(edges), "parallel edges collapsed: the fixture would not match outDegree()"
    assert nx.number_of_selfloops(g) == 0, "self-loops are not allowed in a fixture"
    assert list(g) == list(range(n)), "node iteration order must be the index order"
    if weights is not None:
        nx.set_edge_attributes(g, {edge: float(w) for edge, w in zip(edges, weights)}, "weight")
    return g


def draw_positions(rs, n, dim):
    """n points on a jittered grid inside [-1, 1)^dim: every pair >= 0.4 x spacing apart."""
    cols = max(1, math.ceil(n ** (1.0 / dim)))
    while cols ** dim < n:
        cols += 1
    spacing = 2.0 / cols
    cells = rs.permutation(cols ** dim)[:n]
    pos = np.empty((n, dim))
    for row, cell in enumerate(cells):
        for axis in range(dim):
            index = (int(cell) // (cols ** axis)) % cols
            pos[row, axis] = -1.0 + spacing * (index + 0.5) + rs.uniform(-0.3, 0.3) * spacing
    return pos


def min_pair_distance(p):
    """The smallest distance between two distinct rows of p."""
    if p.shape[0] < 2:
        return math.inf
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def min_origin_distance(p):
    """The smallest |p_i|."""
    return float(np.linalg.norm(p, axis=-1).min())


def run_networkx(g, pos0, options, max_iter):
    """NetworkX's positions after max_iter iterations from pos0 (index order), raw (not rescaled)."""
    n = pos0.shape[0]
    pos = {i: np.array(pos0[i], dtype=float) for i in range(n)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the linlog diagonal 0/0 that fill_diagonal repairs
        result = nx.forceatlas2_layout(g, pos=pos, max_iter=max_iter, seed=0, **options)
    return np.array([result[i] for i in range(n)], dtype=float)


def sensitivity(a, b):
    """max |a - b| / max(|a|, 1): the relative divergence of two runs (positions are O(1) or larger)."""
    return float((np.abs(a - b) / np.maximum(np.abs(a), 1.0)).max())


def check_trajectory(g, pos0, options, iters):
    """The recorded counts with their (positions, sensitivity), or a reason string when the fixture rules fail.

    Runs NetworkX from pos0 and from pos0 x (1 + 2^-52) for 0..max(iters)+1 iterations; checks the separation
    rules and the early exit along the way; replaces the largest count by the largest ladder count whose
    sensitivity is <= SENSITIVITY_MAX.
    """
    upto = max(iters) + 1
    traj = [run_networkx(g, pos0, options, k) for k in range(upto + 1)]
    perturbed = [run_networkx(g, pos0 * (1.0 + ULP), options, k) for k in range(upto + 1)]
    guard_origin = options["gravity"] != 0.0 and not options["strong_gravity"]
    for k in range(max(iters)):
        p = traj[k]
        if min_pair_distance(p) <= TRAJECTORY_SEPARATION:
            return None, f"a pair came within {TRAJECTORY_SEPARATION} at iteration {k}"
        if guard_origin and min_origin_distance(p) <= TRAJECTORY_SEPARATION:
            return None, f"a node came within {TRAJECTORY_SEPARATION} of the origin at iteration {k}"
    short = [k for k in iters if k != max(iters)]
    long_candidates = [max(iters)] + [k for k in LONG_LADDER if k < max(iters)]
    long = next((k for k in long_candidates if sensitivity(traj[k], perturbed[k]) <= SENSITIVITY_MAX), None)
    if long is None:
        return None, f"no ladder count in {long_candidates} has a 1-ulp sensitivity <= {SENSITIVITY_MAX}"
    recorded = {}
    for k in sorted(set(short + [long])):
        if np.array_equal(traj[k], traj[k + 1]):
            return None, f"the early exit fired at or before iteration {k}"
        if not np.all(np.isfinite(traj[k])):
            return None, f"a non-finite position at iteration {k}"
        recorded[k] = (traj[k], sensitivity(traj[k], perturbed[k]))
    return recorded, ""


def fixture_document(command, name, edges, n, weights, options, max_iter, pos0, positions, sens):
    """One 5.4 fixture document (plus the measured 1-ulp sensitivity of NetworkX at max_iter)."""
    return {
        "generator": "test/fixtures/networkx/generate.py",
        "networkxVersion": nx.__version__,
        "pythonVersion": platform.python_version(),
        "command": command,
        "sensitivity": sens,
        "graph": {
            "name": name,
            "directed": False,
            "nodeCount": n,
            "src": [int(u) for u, _ in edges],
            "dst": [int(v) for _, v in edges],
            "weights": None if weights is None else [float(w) for w in weights],
        },
        "options": {
            "max_iter": max_iter,
            "jitter_tolerance": options["jitter_tolerance"],
            "scaling_ratio": options["scaling_ratio"],
            "gravity": options["gravity"],
            "distributed_action": options["distributed_action"],
            "strong_gravity": options["strong_gravity"],
            "linlog": options["linlog"],
            "dissuade_hubs": options["dissuade_hubs"],
            "weight": options["weight"],
            "dim": int(pos0.shape[1]),
        },
        "initialPositions": pos0.tolist(),
        "positions": positions.tolist(),
        "rescaled": False,
    }


def dump(doc):
    """One top-level key per line, arrays compact (a 200-node fixture stays under 30 KB)."""
    items = list(doc.items())
    lines = ["{"]
    for index, (key, value) in enumerate(items):
        comma = "," if index < len(items) - 1 else ""
        lines.append(f"    {json.dumps(key)}: {json.dumps(value, allow_nan=False)}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def generate(out_dir, names, iters, seed):
    """Writes every fixture of the named graphs; returns the list of files written."""
    major, minor = (int(x) for x in nx.__version__.split(".")[:2])
    assert (major, minor) >= (3, 4), f"networkx>=3.4 required, found {nx.__version__}"
    assert hasattr(nx, "forceatlas2_layout")
    command = "python generate.py " + " ".join(sys.argv[1:])
    written = []
    for name in names:
        edges, n, dim, variants, fixed_pos = graph_spec(name, seed)
        rs = np.random.RandomState(seed)
        weights = [1.0 + int(k) * 0.25 for k in rs.randint(0, 8, size=len(edges))]  # f32-exact values
        weights = weights if "weighted" in variants else None
        graph_iters = [1] if name == "path3" else list(iters)
        for attempt in range(MAX_ATTEMPTS):
            pos0 = fixed_pos if fixed_pos is not None else draw_positions(rs, n, dim)
            if min_pair_distance(pos0) < INITIAL_SEPARATION or min_origin_distance(pos0) < INITIAL_SEPARATION:
                print(f"{name}: attempt {attempt}: initial separation violated; redrawing")
                continue
            results = {}
            reason = ""
            for variant in variants:
                options = dict(BASE_OPTIONS, **VARIANTS[variant])
                g = build_graph(edges, n, weights if variant == "weighted" else None)
                recorded, reason = check_trajectory(g, pos0, options, graph_iters)
                if recorded is None:
                    reason = f"{variant}: {reason}"
                    break
                results[variant] = (options, recorded)
            if reason:
                print(f"{name}: attempt {attempt}: {reason}; redrawing")
                continue
            break
        else:
            sys.exit(f"{name}: no initial positions satisfied the fixture rules in {MAX_ATTEMPTS} attempts")
        for variant, (options, recorded) in results.items():
            for k, (positions, sens) in recorded.items():
                doc = fixture_document(
                    command, name, edges, n,
                    weights if variant == "weighted" else None,
                    options, k, pos0, positions, sens,
                )
                path = f"{out_dir.rstrip('/')}/{name}-{variant}-iter{k}.json"
                with open(path, "w", encoding="ascii") as f:
                    f.write(dump(doc))
                written.append(path)
                print(f"wrote {path} (sensitivity {sens:.1e})")
    return written


def format_written(written):
    """Runs the workspace's prettier on the written files (the format:check gate of packages/)."""
    workspace = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
    files = [os.path.abspath(path) for path in written]
    result = subprocess.run(
        ["pnpm", "exec", "prettier", "--write", "--log-level", "warn", *files],
        cwd=workspace,
        check=False,
    )
    if result.returncode != 0:
        sys.exit(f"prettier --write failed (exit {result.returncode}) in {workspace}; the fixtures are unformatted")


def main():
    """CLI: --out DIR [--graph NAME]... [--iters K...] [--seed S]."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="directory the JSON files are written into")
    parser.add_argument("--graph", action="append", choices=GRAPHS, help="a graph to generate (default: all)")
    parser.add_argument("--iters", type=int, nargs="+", default=list(DEFAULT_ITERS), help="iteration counts")
    parser.add_argument("--seed", type=int, default=7, help="RandomState seed for positions, weights and G(n, m)")
    args = parser.parse_args()
    names = args.graph if args.graph else list(GRAPHS)
    written = generate(args.out, names, sorted(set(args.iters)), args.seed)
    format_written(written)
    print(f"{len(written)} files; networkx {nx.__version__}, python {platform.python_version()}")


if __name__ == "__main__":
    main()
