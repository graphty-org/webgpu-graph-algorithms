# 01 - Layout needs: what a GPU force-directed layout must replicate and how it plugs in

Research note for the WebGPU graph-algorithms plan. Everything in sections
2-7 was read from the files listed in "Sources" (monorepo state as of
2026-09-14, layout package `src/layouts/force-directed/*`, graphty-element
`src/layout/*`, `src/managers/*`). External facts are cited by URL or by the
path of a cloned repository under `tmp/webgpu-plan/repos/`.

Reference contract (decided in the accepted design, not relitigated here):
`@graphty/graph-format` frozen CSR snapshot; `LayoutSimulation` interface of
design section 14.3; graphty-element owns a stride-3 scene-unit
`Float32Array` of positions (design section 14.4); the GPU package never
falls back to CPU.

Vocabulary used below: n = node count, m = logical edge count, A = arc count
of the undirected snapshot (`2m - selfLoops`), dim = 2 or 3.

---------------------------------------------------------------------------

## 1. Summary

- Four force layouts exist on the CPU: `forceatlas2Layout`,
  `fruchtermanReingoldLayout` (alias `springLayout`), `arfLayout`,
  `kamadaKawaiLayout`. All are ONE-SHOT functions: they run `maxIter`
  iterations synchronously and return a `PositionMap`; none is steppable.
- graphty-element wraps them as `SimpleLayoutEngine` subclasses whose
  `isSettled` is a `readonly true` and whose `step()` is a no-op
  (`LayoutEngine.ts` lines 302-304, 341). The whole layout is computed
  inside the first `getNodePosition()` call of the first frame. They never
  animate, never honour pins or drags, and re-run from scratch when nodes
  are added.
- The two ANIMATED engines the element really uses are third-party:
  `ngraph.forcelayout` (the default, `config/GraphBehavior.ts` line 13) and
  `d3-force-3d`. Both are stepped once per render frame (times
  `stepMultiplier`) by `UpdateManager.updateLayout()`; both support
  pin/unpin and `setNodePosition` for drag; both use Barnes-Hut on the CPU.
- A GPU force layout therefore replaces BOTH shapes: it must be a
  `LayoutSimulation` (steppable, settle-reporting, pinnable, draggable,
  the way ngraph is used today) AND reproduce the formulas / parameters of
  the CPU `forceatlas2Layout` so the "forceatlas2" layout type gives the
  same family of pictures whether or not a GPU is present.
- The CPU FA2 port is a transcription of NetworkX's `forceatlas2_layout`
  with three measurable deviations (section 2.1.9): repulsion falls off as
  1/d^2 instead of 1/d, swing/traction are computed from the force
  magnitude and reset every iteration instead of accumulated, and the
  `adjustSizes` distance correction has a sign difference. The plan must
  pick ONE reference (this note recommends: the CPU port as it will exist
  after the section-14.3 rewrite, with these three points settled there
  first so CPU and GPU agree).
- Every CPU force layout is O(n^2) per iteration and allocates O(n^2)
  (FA2: O(n^2 * dim) per ITERATION). The practical ceiling is a few
  thousand nodes. The scale the format targets is 100k / 1M (design
  section 15.3); the GPU layout is what makes force layouts reach it.

---------------------------------------------------------------------------

## 2. The CPU force layouts, one by one

### 2.1 ForceAtlas2 -- `layout/src/layouts/force-directed/forceatlas2.ts`

#### 2.1.1 Signature, parameters, defaults (lines 26-42)

```ts
forceatlas2Layout(G, pos = null, maxIter = 100, jitterTolerance = 1.0,
    scalingRatio = 2.0, gravity = 1.0, distributedAction = false,
    strongGravity = false, nodeMass = null, nodeSize = null,
    weight = null, _dissuadeHubs = false, linlog = false, seed = null,
    dim = 2): PositionMap
```

`_dissuadeHubs` is accepted and IGNORED (the underscore and the JSDoc
"(unused)" on line 20). `pos` is `PositionMap | null`; `nodeMass` /
`nodeSize` are `Record<Node, number> | null`; `weight` is an edge attribute
NAME read through `graph.getEdgeData(source, target, weight) || 1` (lines
158-160), so a weight of 0 becomes 1.

#### 2.1.2 Initialisation (lines 52-137)

- RNG: `new RandomNumberGenerator(seed ?? undefined)` (line 53); see 2.5.1.
- Positions (three branches):
  - `pos === null`: every coordinate `rng.rand() * 2 - 1`, i.e. uniform in
    [-1, 1) per axis (lines 57-65).
  - `pos` has an entry for every node: copied; a missing axis (2D `pos`
    with `dim = 3`) is filled with `rng.rand() * 2 - 1` (lines 66-75).
  - partial `pos`: bounding box of the given positions per axis (falling
    back to [-1, 1] on an empty axis), missing nodes uniform inside that
    box (lines 76-116).
- Mass: `nodeMass[node]` when truthy, else `getNodeDegree(graph, node) + 1`
  (lines 126-134; a mass of 0 in the record is treated as "not given").
  `getNodeDegree` is a private O(E) `edges().filter(...)` per node (line
  446-448), i.e. O(n * m) before the first iteration. Design 14.3 replaces
  it with `outDegree()` (self-loop counted once, parallel edges counted per
  arc).
- Size: `nodeSize[node]` when truthy else 1; `adjustSizes = nodeSize !==
  null` (lines 123, 136).
- Dense adjacency `A: number[][]` n x n, `A[i][j] = A[j][i] = w` (lines
  140-164): parallel edges collapse (last write wins), the graph is treated
  as undirected, a self-loop sets `A[i][i]` which every loop then skips via
  `i === j`.

#### 2.1.3 One iteration (lines 233-434), in order

1. Zero `attraction`, `repulsion`, `gravities` (n x dim each).
2. ALLOCATE `diff` (n x n x dim) and `distance` (n x n); for all ordered
   pairs i != j: `diff[i][j] = p_i - p_j`, `distance = max(|diff|, 0.01)`
   (lines 244-268). This allocation is per iteration.
3. Attraction over all pairs with `A[i][j] != 0`:
   - linear (default): `attraction_i += -diff_ij * A_ij` (magnitude `w *
     d`, towards j) (lines 288-297).
   - linlog: `attraction_i += -(log(1 + d) / d) * A_ij * diff_ij`
     (magnitude `w * log(1 + d)`) (lines 271-285).
   - if `distributedAction`: `attraction_i /= mass_i` (lines 301-307).
4. Repulsion over all ordered pairs i != j (lines 310-331):
   `d' = adjustSizes ? max(d - (size_i - size_j), 0.01) : d`;
   `factor = mass_i * mass_j / d'^2 * scalingRatio`;
   `repulsion_i += (diff_ij / d') * factor`.
   Net magnitude: `scalingRatio * m_i * m_j / d'^2` (see 2.1.9 for why this
   is NOT the FA2 paper's / NetworkX's `1/d`).
5. Gravity (lines 335-364): centre of mass `c = mean(p)`; `q = p_i - c`;
   strong: `g_i = -gravity * mass_i * q`; regular: `g_i = -gravity * mass_i
   * q / |q|` when `|q| > 0.01`, else 0.
6. `update_i = attraction_i + repulsion_i + gravities_i` (line 375).
7. Swing / traction (lines 379-390), computed from `oldPos` and `newPos =
   oldPos + update`: `swingVector = oldPos - newPos = -update`,
   `tractionVector = oldPos + newPos = 2 * oldPos + update`;
   `totalSwing = sum_i mass_i * |update_i|`,
   `totalTraction = sum_i 0.5 * mass_i * |2 p_i + update_i|`.
   Both totals are RESET to 0 each iteration (lines 370-371). The `_swing =
   1; _traction = 1` on lines 180-181 are unused.
8. `estimateFactor` (lines 184-230; scalar, identical to NetworkX's
   `estimate_factor`): `optJitter = 0.05 * sqrt(n)`; `minJitter =
   sqrt(optJitter)`; `maxJitter = 10`; `minSpeedEfficiency = 0.05`; `other
   = min(maxJitter, optJitter * traction / n^2)`; `jitter = jitterTolerance
   * max(minJitter, other)`; if `swing / traction > 2`: halve
   `speedEfficiency` (floor 0.05) and `jitter = max(jitter,
   jitterTolerance)`; `targetSpeed = swing === 0 ? +Inf : jitter *
   speedEfficiency * traction / swing`; if `swing > jitter * traction`:
   `speedEfficiency *= 0.7` (floor 0.05) else if `speed < 1000`:
   `speedEfficiency *= 1.3`; `speed += min(targetSpeed - speed, 0.5 *
   speed)`. State carried across iterations: `speed` (init 1),
   `speedEfficiency` (init 1).
9. Apply (lines 403-428): per node `swinging = mass_i * |update_i|`;
   `factor = speed / (1 + sqrt(speed * swinging))`; with `adjustSizes`:
   `factor = 0.1 * speed / (1 + sqrt(speed * swinging))`, then `factor =
   min(factor * |update_i|, 10) / |update_i|` (displacement capped at 10);
   `p_i += update_i * factor`; `totalMovement += sum_k |movement_k|`.
10. Termination: `if (totalMovement < 1e-10) break` (line 431), else run
    to `maxIter`. In practice the 1e-10 threshold is never hit (float64
    jitter) and the layout runs exactly `maxIter` iterations.

No pinned / fixed nodes exist in this layout (FR has `fixed`, FA2 does
not).

#### 2.1.4 Output (lines 437-442)

`rescaleLayout(positions)` with default `scale = 1`, `center = null`:
translate to the centroid and scale so the farthest node is at radius 1
(see 2.5.2). So the FA2 result is always inside the unit ball, regardless
of `scalingRatio`; graphty-element multiplies by `scalingFactor` (default
100) when reading (`LayoutEngine.ts` line 352-365).

#### 2.1.5 Dimensions

Any `dim` >= 1 works arithmetically; graphty-element clamps to 2..3 and
passes `dim` from the view mode (`LayoutManager.ts` lines 118-129,
`ForceAtlas2LayoutEngine.ts` lines 143-145). 3D was the subject of two bug
reports (`layout/test/bug-report-2-forceatlas2-3d.test.ts`,
`forceatlas2-npm-bug.test.ts`: NaN z when `pos` was 2D); the missing-axis
fill on line 73 is the fix and must be preserved.

#### 2.1.6 Complexity and allocations

Setup: O(n * m) degree scan, O(n^2) dense `A`. Per iteration: O(n^2 * dim)
time AND a fresh `diff` (n * n * dim numbers, each an array object) plus
`distance` (n * n) plus `update` (n) plus several small arrays per node
(`oldPos`, `newPos`, `swingVector`, `tractionVector`). At n = 10k, dim = 3
that is 300M numbers per iteration; research note 03 puts the practical
ceiling at a few thousand nodes. The `forceatlas2-layout.test.ts` "large
graph" test is a 10 x 10 grid with 30 iterations under 2 s (lines
504-528).

#### 2.1.7 Determinism

`seed` -> `RandomNumberGenerator` -> initial positions only. The
simulation itself is deterministic given the initial positions and the
node ORDER (`nodes()` order defines index order and the summation order of
every reduction). Tests assert same-seed results agree to 0.1 (line
344-385) and different seeds differ (line 387).

#### 2.1.8 Tests that pin behaviour (`layout/test/forceatlas2-layout.test.ts`)

Empty graph -> `{}`; single node; disconnected components separated by >
0.03 after 100 iterations; `maxIter` respected; gravity / strongGravity /
distributedAction / linlog / masses / sizes / weights all "run and produce
finite positions" type checks; same seed -> same layout (tolerance 0.1);
`completeGraph(6)` spread > 0.3 in width and height. None asserts an exact
coordinate, so a GPU implementation with f32 arithmetic and a different
reduction order can pass the same suite when run through the element.

#### 2.1.9 Divergence from the NetworkX reference (verified against upstream)

The port is a transcription of NetworkX `forceatlas2_layout`
(https://raw.githubusercontent.com/networkx/networkx/main/networkx/drawing/layout.py;
same signature order, same `estimate_factor`, same `1e-10` break, same
`degree + 1` mass, same `[-1, 1)` initial positions via `random_layout`
semantics). Three places differ, and each changes the picture:

| Quantity | NetworkX (upstream) | graphty port | Consequence |
| --- | --- | --- | --- |
| Repulsion | `repulsion = einsum(diff, mass_i mass_j / d^2 * k)` = magnitude `k m_i m_j / d` (the FA2 paper's law) | `direction * factor` with `factor = k m_i m_j / d^2` = magnitude `k m_i m_j / d^2` (lines 322-329) | the port's repulsion decays one power faster; layouts are tighter and hubs less separated |
| Swing / traction | `swing += sum(mass * norm(pos - update))`, `traction += sum(0.5 mass norm(pos + update))`, ACCUMULATED across iterations from `swing = traction = 1` | `sum mass * norm(update)` and `sum 0.5 mass norm(2 pos + update)`, RESET each iteration (lines 370-390) | different adaptive speed trajectory; the port converges differently |
| `adjust_sizes` | `distance += -size_i - size_j` | `dist -= size[i] - size[j]` (line 318) | the port subtracts `size_i` and ADDS `size_j`; sizes are not symmetric |
| Distance floor | none (diagonal only) | `max(d, 0.01)` (line 266) and again after the size correction (line 319) | harmless; keep |
| Weights | `to_numpy_array(G, weight)` | `getEdgeData(...) || 1` | same "0 means 1" semantics through the fallback, but the port also treats a missing attribute as 1 |

Recommendation for the plan: settle these in the section-14.3 CPU rewrite
(which is the parity target for the GPU kernel) BEFORE writing WGSL, and
write the FA2 parity tests against that rewrite. The safest choice is to
adopt the upstream `1/d` law and symmetric size correction (they are the
published algorithm and what Gephi / NetworkX users expect) and to keep the
port's per-iteration swing reset (it is what the existing tests and
Chromatic baselines were tuned against). Either way the GPU kernel must
take the choice as a documented constant, not rediscover it.

### 2.2 Fruchterman-Reingold / spring -- `fruchterman-reingold.ts`, `spring.ts`

`springLayout` is a pure alias (`spring.ts` lines 20-34).

Signature (lines 24-34): `fruchtermanReingoldLayout(G, k = null, pos =
null, fixed = null, iterations = 50, scale = 1, center = null, dim = 2,
seed = null)`.

- Initial positions: `rng.rand(dim)` uniform in [0, 1) per axis when `pos`
  is null (lines 65-69). With a partial `pos`, a NEW `RandomNumberGenerator(
  seed)` is created PER missing node (line 60), so with a seed every
  missing node gets the same coordinates (research note 03 section 4 flags
  this; design 14.3 fixes it with one RNG -- "single-RNG fix").
- `k` defaults to `1 / sqrt(n)` (line 77).
- Temperature `t = 0.1`, `dt = t / (iterations + 1)`, `t -= dt` per
  iteration (lines 81-83, 154): linear cooling, never reaches 0.
- Per iteration: `displacement` Record allocated; repulsion over unordered
  pairs `force = k^2 / d` along `delta / d` with `d = |delta| || 0.1`
  (lines 94-115; note the `||`: an exact 0 becomes 0.1, everything else is
  unclamped); attraction over the EDGE LIST `force = d^2 / k` (lines
  118-134); apply: skip `fixed` nodes, move along the displacement
  direction by `min(|disp|, t)` (lines 137-151).
- Termination: fixed `iterations` (default 50); no convergence test.
- Output: `rescaleLayout(positions, scale, center)` UNLESS `fixed` was
  given (lines 158-160), i.e. fixed nodes really stay where the caller put
  them.
- 2D/3D: any `dim`; graphty-element's `SpringLayout` schema defaults `dim`
  to 3 (`SpringLayoutEngine.ts` line 214-220) but `LayoutManager`
  overrides it from the view mode.
- Complexity: O(iterations * (n^2 + m)); the hot loop goes through
  `Record` property lookups and `Array.map` per pair (line 100), so it is
  the slowest of the four at equal n.
- Parity notes for a GPU version: the only per-node input is the `fixed`
  set (this becomes the `NodeMask` of `setFixed`). Repulsion is edge-free;
  attraction is edge-centric (each edge contributes to both endpoints), so
  the GPU form gathers over the undirected CSR row instead (same sum,
  no atomics). The `|| 0.1` singularity guard must be replicated for
  coincident nodes (common on the first iteration of a seeded layout).

### 2.3 ARF -- `arf.ts`

Signature (lines 13-20): `arfLayout(G, pos = null, scaling = 1, a = 1.1,
maxIter = 1000, seed = null)`; throws when `a <= 1` (line 21).

- 2D ONLY: `change` rows are `[0, 0]` (line 90), `randomLayout(G, null, 2,
  seed)` seeds positions (line 34); graphty-element declares
  `maxDimensions = 2` (`ArfLayoutEngine.ts` line 361).
- Dense `K` n x n: 1 everywhere, 0 on the diagonal, `a` on edges (lines
  58-75). `rho = scaling * sqrt(n)` (line 78).
- Per iteration over all ordered pairs: `change_i += K_ij * diff_ij - (rho
  / d) * diff_ij` with `d = |diff| || 0.01` (lines 92-107); `p_i += change_i
  * 1e-3` (lines 110-114); `error = sum_i |change_i|`.
- Termination: `error <= 1e-6` OR `maxIter` (default 1000) (line 86).
- Output: NOT rescaled (line 128).
- No fixed nodes, no weights, no mass.
- GPU form (research note 03 section 12.3): `K` is `1 + (a - 1) *
  isEdge`, so the pair sum is an all-pairs term plus a CSR-row correction
  of `(a - 1) * sum_{j in N(i)} diff_ij`. Low priority.

### 2.4 Kamada-Kawai -- `kamada-kawai.ts`, `algorithms/optimization/*`

Signature (lines 11-18): `kamadaKawaiLayout(G, dist = null, pos = null,
weight = "weight", scale = 1, center = null, dim = 2)`. NO seed parameter:
initial positions are `circularLayout(G, 1, center, dim)` when `pos` is
null (line 68), so the layout is deterministic by construction.

- `dist`: all-pairs shortest paths via Floyd-Warshall over nested Records
  (`kamada-kawai-solver.ts` lines 17-59, O(n^3)), weights via `getEdgeData(
  ..., weight) || 1`, undirected. Unreachable -> `1e6` (line 48 of the
  layout). Design 14.3 lets `dist` be a `Float32Array(n * n)` supplied by
  `@graphty/algorithms` or the GPU package.
- Solver (`_kamadaKawaiSolve`, lines 68-140 of the solver): `invDist = 1 /
  (d + 1e-3)` (0 on the diagonal), `meanWeight = 1e-3`, L-BFGS with memory
  10, backtracking (Armijo) line search up to 20 evaluations, up to 500
  iterations, stop at `||grad|| < 1e-5`. Each cost evaluation is O(n^2 *
  dim) and reshapes `posVec` into `number[][]` (allocation per call).
- Output: `rescaleLayout(finalPos, scale, center)`.
- Complexity: O(n^3) APSP + O(500 * 21 * n^2 * dim); ceiling ~1-2k nodes.
- GPU relevance: not a force simulation in the steppable sense (a line
  search needs a cost readback per evaluation). Its two GPU-able pieces
  are the APSP (n BFS frontiers, an algorithms-package primitive) and the
  dense cost/gradient kernel (research note 03 section 12.4). Treat as a
  later, separate slice; it does not need `LayoutSimulation`.

### 2.5 Shared utilities

#### 2.5.1 `RandomNumberGenerator` (`layout/src/utils/random.ts`)

LCG: `m = 2^35 - 31`, `a = 185852`, `c = 1`, `state = seed % m`, `next =
(a * state + c) % m`, returns `state / m` (lines 21-36). `seed || random`
means seed 0 is "unseeded" (line 22). `rand(dim)` returns an array of
`dim` consecutive draws (lines 43-69). Products up to `185852 * 2^35`
~ 6.4e15 stay below 2^53, so the sequence is exact in float64 and is
reproducible in TypeScript on either side of the GPU boundary. A GPU layout
that seeds its own initial positions (when the element gives it NaN rows
for unplaced nodes) should generate them on the CPU with this generator in
node-index order so that a seed produces the same start as the CPU path.

#### 2.5.2 `rescaleLayout` (`layout/src/utils/rescale.ts` lines 84-228)

Centroid subtraction (NaN-aware), then scale so `max |p - centroid| =
scale`, then add `center`. FA2, FR (without `fixed`) and KK end with it;
ARF does not. For a steppable GPU layout this normalisation cannot be
applied per step (it would fight the drag / pin positions and change the
scene scale every frame); section 5 discusses what replaces it.

#### 2.5.3 `_processParams` (`utils/params.ts`)

Only validates `center.length === dim` (throws otherwise) and defaults
`center` to zeros.

---------------------------------------------------------------------------

## 3. The animated engines graphty-element actually uses

These are what "an animated force layout" means to the element today, and
what a GPU `LayoutSimulation` will be judged against for feel.

### 3.1 ngraph -- `graphty-element/src/layout/NGraphLayoutEngine.ts`

- The DEFAULT layout (`config/GraphBehavior.ts` line 13: `type: z.string()
  .default("ngraph")`).
- Constructor maps `dim` -> `dimensions` (default 3) and passes
  `springLength`, `springCoefficient`, `gravity`, `theta`, `dragCoefficient`,
  `timeStep` only when defined (lines 124-155); the zod schema on lines
  15-81 advertises defaults (30, 0.0008, -1.2, 0.8, 0.02, 20) but the
  constructor never parses it, so ngraph's OWN defaults apply when the
  caller omits them: `springLength 10`, `springCoefficient 0.8`, `gravity
  -12`, `theta 0.8`, `dragCoefficient 0.9`, `timeStep 0.5`
  (`node_modules/ngraph.forcelayout/lib/createPhysicsSimulator.js` lines
  28-70, version 3.3.1).
- `seed` -> `ngraph.random(seed)` (line 149-151).
- `step()` (lines 169-191): one physics step; `lastMove / nodeCount`
  averaged over the last 10 steps; settled when ngraph says stable
  (`lastMove / bodies <= 0.01`, `ngraph.forcelayout/index.js` lines 62-63)
  OR the 10-step average `<= 0.05` OR `stepCount >= 1000`.
- ngraph's integrator (`lib/codeGenerators/generateIntegrator.js`):
  velocity Verlet-ish with `v += dt / mass * F`, velocity normalised when
  `|v| > 1`, `p += dt * v`, pinned bodies skipped, returns `sum(|dp|)^2 /
  n`. Forces: Barnes-Hut quadtree/octree repulsion with `theta`, Hooke
  springs, drag.
- `addNode` / `addEdge` reset `_settled = false`, `_stepCount = 0`, so any
  topology change restarts settling (lines 205-224).
- `pin` / `unpin` -> `ngraphLayout.pinNode(body, bool)` (lines 291-303).
- `setNodePosition` writes the body's `pos` in place (lines 240-247); no
  reheat.
- `getNodePosition` returns ngraph's internal position object; no
  allocation; positions are already in scene units (no scaling factor).

### 3.2 d3 -- `graphty-element/src/layout/D3GraphLayoutEngine.ts`

- `forceSimulation().numDimensions(3).alpha(1)` with `forceLink()` strength
  0.9, `forceManyBody()`, `forceCenter()` (lines 154-168); `alphaMin`
  default 0.1 (schema line 22; d3's own default is 0.001), `alphaDecay`
  0.0228, `velocityDecay` 0.4.
- `step()` = `refresh()` then `tick()` (lines 233-236); `refresh` re-feeds
  the node and link arrays and reheats `alpha(1)` when nodes / edges were
  added or `reheat` was set (lines 184-228).
- `isSettled` = no pending nodes AND `alpha < alphaMin` (lines 242-249).
- `pin` sets `fx, fy, fz` WITHOUT reheating (comment lines 349-352: "Reheating
  here would cause the layout to never settle"); `unpin` clears them and
  reheats; `setNodePosition` writes `x, y, z` and reheats (lines 309-315).

### 3.3 What the two engines have in common (the behaviour to preserve)

| Behaviour | ngraph | d3 |
| --- | --- | --- |
| step granularity | one physics step per `step()` | one tick per `step()` |
| settle signal | movement-per-node threshold + max steps | alpha decay below `alphaMin` |
| restart on topology change | yes (reset counters) | yes (alpha = 1) |
| pin | body flag, integrator skips it | fixed coordinates, no reheat |
| drag | position overwrite per pointer move | position overwrite + reheat |
| position units | scene units directly | scene units directly |
| 2D | `dimensions: 2` at construction; switching view mode RECREATES the engine (`LayoutManager.updateLayoutDimension`, lines 321-358) | always 3D internally |

---------------------------------------------------------------------------

## 4. How graphty-element drives a layout per frame

### 4.1 Frame loop

`RenderManager.startRenderLoop` calls `engine.runRenderLoop(() =>
updateCallback())` (`managers/RenderManager.ts` lines 165-170); the
callback is `Graph.update()` (`Graph.ts` lines 493-495), which runs
`UpdateManager.update()` then the settlement check (`Graph.ts` lines
530-598).

`UpdateManager.update()` (`managers/UpdateManager.ts` lines 146-198), per
frame, SYNCHRONOUSLY:

1. `camera.update()`.
2. If `!layoutManager.running`: only `updateEdges()` (for manual drags)
   and a pending zoom-to-fit; return.
3. `updateLayout()`: `for (i < stepMultiplier) layoutManager.step()`
   (lines 203-214); `stepMultiplier` comes from
   `styles.config.behavior.layout.stepMultiplier`, default 1
   (`config/GraphBehavior.ts` line 15). `LayoutManager.step()` calls
   `engine.step()` only when `running && !engine.isSettled`
   (`LayoutManager.ts` lines 241-245).
4. `updateNodes()`: `for (node of layoutManager.nodes) node.update()`
   (lines 220-253). `Node.update()` reads `layoutEngine.getNodePosition(
   this)` and copies `x, y, z ?? 0` into `mesh.position` (`Node.ts` lines
   175-183), skipped while `this.dragging` (line 170). A bounding box is
   accumulated from `mesh.getAbsolutePosition()` and `node.size`.
5. `updateEdges()`: `Edge.updateRays(ctx)` then `edge.update()` per edge,
   which reads `layoutEngine.getEdgePosition(this)` (two `{x,y,z}`), then
   compares `srcNode.mesh.position` / `dstNode.mesh.position` against
   cached vectors with `equalsWithEpsilon(.., 0.001)` and only transforms
   the edge mesh when an endpoint moved (`Edge.ts` lines 285-300).
6. Zoom-to-fit, statistics.

Everything is synchronous and on the main thread; there is no
`await` anywhere in the frame. A GPU simulation whose `step()` returns a
Promise therefore cannot be called from this loop as-is (section 5.3).

### 4.2 Layout creation and pre-steps

`LayoutManager._setLayoutInternal(type, opts)` (`LayoutManager.ts` lines
108-225): merges the dimension option from the view mode (`is2D ? 2 : 3`,
via `LayoutEngine.getOptionsForDimensionByType`), instantiates via the
registry, `engine.addNodes(all)`, `engine.addEdges(all)`, `await
engine.init()`, `dataManager.setLayoutEngine(engine)`, then runs `preSteps`
synchronous steps (default 0, `GraphBehavior.ts` line 14), sets `running =
true`, emits `layout-initialized` (which requests zoom-to-fit) and
`layout-changed`, and disposes the previous engine if it has `dispose()`
(lines 178-181; `dispose` is optional, detected by `hasDispose`).

`init()` is the only async hook an engine has today; it is where a GPU
engine can acquire the device and upload the snapshot.

### 4.3 Settlement and "running"

`LayoutManager.isSettled` is `!running || !engine || engine.isSettled`
(lines 278-291). `Graph.update()` (lines 547-560): when `isSettled &&
running`, emit `graph-settled`, set `running = false`, start label
animations, and on the FIRST settlement enable zoom-to-fit and capture the
camera state. `running` goes back to `true` on `data-added` events with
`shouldStartLayout` (`Graph.ts` lines 337-343), on drag end
(`NodeBehavior.ts` line 183 `context.setRunning(true)`), and in
`LayoutManager.updatePositions` (line 449).

`ScreenshotCapture` polls `layoutManager.isSettled` to decide when to
capture (`screenshot/ScreenshotCapture.ts` lines 332-380), and Storybook /
Chromatic stories rely on it, so a GPU engine must report `settled`
truthfully and eventually (a simulation that never settles breaks
screenshots and keeps the label animations from starting).

### 4.4 Topology changes

- `DataManager.addNodes` / `addEdges` call `layoutEngine.addNode(n)` /
  `addEdge(e)` immediately per record (`DataManager.ts` lines 215-217,
  287-289, 392-394) and set `shouldStartLayout` (lines 229, 407), which
  `Graph.ts` turns into `running = true`.
- The `data-add` operation-queue trigger calls
  `layoutManager.updatePositions(allNodes)` (`Graph.ts` lines 214-225),
  which, absent an engine `updatePositions` method, runs up to 10
  synchronous steps (`LayoutManager.ts` lines 443-474).
- Removal never reaches an engine today (neither engine implements
  `removeNode` / `removeEdge`; `DataManager.ts` lines 328-329, 448-449 are
  guarded by type checks).
- Design 14.4 replaces this with: `engine.load(dm.undirected(getSnapshot()
  ).snapshot, positions)` at layout set, and `engine.reload(undirected,
  report, positions)` on `snapshot-replaced` (the M3/M4/M5 row of the
  14.4 table); "`LayoutSimulation` engines keep stepping on the new
  array". `report.nodeRemap` tells the engine which node rows moved.

### 4.5 Pinned nodes and dragging

- `Node.pin()` / `unpin()` -> `layoutEngine.pin(this)` / `unpin(this)`
  (`Node.ts` lines 295-303). `pinOnDrag` (default true, `GraphBehavior.ts`
  line 8) pins on drag end (`NodeBehavior.ts` lines 186-189).
- During a drag, `NodeBehavior.onDragUpdate` writes `mesh.position` AND
  calls `layoutEngine.setNodePosition(node, {x, y, z})` on every pointer
  move (lines 131-170); the XR path does the same through
  `setPositionDirect` (lines 228-243). `Node.update()` skips the mesh copy
  while dragging (line 170), so the drag position wins over the engine
  for that node until drag end.
- `onDragEnd` sets `running = true` so the simulation resumes around the
  moved node (line 183).
- Design 14.3/14.4: drag becomes `simulation.setPosition(i, x, y, z)` "a
  12-byte write (writeBuffer on the GPU)" while stepping; pins become
  `setFixed(mask)` with the same bitmap layout as a bool column with role
  `fixed`.

### 4.6 2D versus 3D

`is2D` = `viewMode === "2d" || twoD` (`LayoutManager.ts` line 118). The
manager passes `{ dim: 2 | 3 }` to engines that expose it and RE-CREATES
the engine when the mode switches (`updateLayoutDimension`, lines
321-358). For 2D the engine is expected to leave `z = 0` (`posToCoords`
uses `pos[2] ?? 0`, `LayoutEngine.ts` line 360). Meshes are recreated with
`is2D` flat shapes but positions stay `{x, y, z}`. A GPU `LayoutSimulation`
gets `dim` as a uniform and must simply not integrate z when `dim === 2`
(the position array is stride 3 regardless, design section 5.2 / C14).

### 4.7 Scene scale

`SimpleLayoutEngine` results are in normalised layout units (unit ball
after `rescaleLayout`) multiplied by `scalingFactor` (default 100, schema
`z.number().min(1).max(1000)`) at read time (`LayoutEngine.ts` lines
185-189, 261-267, 352-365). ngraph / d3 work in scene units directly (no
factor). Design 14.3's `toPositionColumn(result, scale, center, out)`
applies the factor once at store time and `fromPositionColumn` divides it
out when seeding a re-run. A GPU FA2 that steps on the element's array
therefore either (a) simulates in scene units with its force constants
scaled accordingly, or (b) simulates in layout units in its own buffer
and multiplies by `scalingFactor` on every write-back. Option (b) keeps
the CPU and GPU force constants identical (recommended; see section 8.4).

### 4.8 Rendering engine

`RenderManager` builds a Babylon `Engine` (WebGL) by default and a
`WebGPUEngine` only when `config.useWebGPU` (`managers/RenderManager.ts`
lines 63-69, 112-115). Nothing in `src` passes `useWebGPU` (grep shows
only the RenderManager itself), so the element renders on WebGL today.
Consequence: the compute device is NOT Babylon's device; the layout
package's GPU accelerator owns its own `GPUDevice` and the position
readback crosses back to the CPU array every frame (section 8.5). Sharing
a device with a future `WebGPUEngine` (`engine._device`) is an optimisation
for later, not a requirement.

---------------------------------------------------------------------------

## 5. The `LayoutSimulation` contract and what a GPU drop-in must satisfy

Design section 14.3 (design doc lines 3976-3985):

```ts
export interface LayoutSimulation {
    load(snapshot: GraphSnapshot, positions: F32): void;        // positions: the owner's stride-3 scene-unit array, read AND written in place
    step(iterations?: number): void | Promise<void>;           // GPU implementations are async (mapAsync readback); the GPU buffer is authoritative while stepping
    readonly settled: boolean;
    setFixed(mask: NodeMask): void;                             // the same bitmap layout as a bool column with role "fixed"
    setPosition(index: number, x: number, y: number, z: number): void;   // drag during simulation: a 12-byte write (writeBuffer on the GPU)
    dispose(): void;
}
```

Facts the contract fixes (with the format's definitions):

- `positions` is stride 3 and in SCENE units, owned by `DataManager`
  (`positions.subarray(0, 3 * nodeCount)` attached as the role-`position`
  column, design lines 4152-4153); the kernel takes the stride as a
  uniform (design line 3997). `NaN` marks an unplaced node (design line
  4075-4077) -- a GPU `load` must seed those rows (section 8.3).
- `NodeMask` is `Uint32Array(ceil(n / 32))`, bit i = node i, LSB-first
  (`packages/graph-format/src/types/columns.ts` lines 716-722); helpers
  `makeBitmap`, `bitmapGet/Set/Clear`, `bitmapToIndices` exist in
  `src/columns/bitmap.ts`. It uploads as-is and a WGSL kernel tests
  `(mask[i >> 5] >> (i & 31)) & 1`.
- `snapshot` given to a layout is the element's cached UNDIRECTED snapshot
  (`dm.undirected(s).snapshot`, design 14.4 "Layout engines receive the
  same `undirected(s).snapshot` object as the adapters"), so rows already
  hold both arcs and `outDegree()` is the degree (self-loop once).
- `weights` is `F32 | null` on the snapshot; FA2 reads them when `weight
  === true`, or a named edge column expanded with `expandEdges` (design
  14.4, cached per column version) -- the expanded per-arc array is what
  the GPU uploads.
- Step may be async; "the GPU buffer is authoritative while stepping":
  between `step()` calls the CPU array is a COPY the renderer reads; a
  `setPosition` during a step is applied to the GPU buffer, not to the
  array.

### 5.1 What "drop-in" means in practice

A GPU engine registered as the "forceatlas2" `LayoutEngine` type (or
selected by the element as the accelerated implementation behind that
type) must satisfy, in order of how visible a miss would be:

1. Same options object as the CPU FA2 of section 14.3 (the
   `ForceAtlas2Options` shape of research note 03 section 10.2: `pos`,
   `maxIter`, `jitterTolerance`, `scalingRatio`, `gravity`,
   `distributedAction`, `strongGravity`, `nodeMass`, `nodeSize`, `weight`,
   `linlog`, `seed`, `dim`, plus `scale` / `center` from
   `CommonLayoutOptions`) so `ForceAtlas2LayoutEngine`'s zod config maps
   1:1 and the Storybook controls (`stories/Layout.stories.ts` lines
   87-137) keep working.
2. Same forces and same adaptive-speed controller as the CPU FA2 (section
   2.1.3, with the 2.1.9 choices made identically), so that N steps of the
   GPU version and N iterations of the CPU version produce statistically
   the same picture (parity test: run both from the same seeded start on
   the same small graph for the same iteration count; compare per-node
   distance histograms, stress, edge-length distribution, NOT coordinates).
3. `settled` semantics compatible with the element's settlement check
   (section 4.3): must become `true` in bounded time. FA2's own `1e-10`
   movement test is not usable (never true in floating point); use the
   `maxIter` budget as the hard stop and a movement-per-node threshold
   like ngraph's for early settle (section 8.6).
4. Pins honoured every step (`setFixed` mask), drag position honoured
   (`setPosition`), and resume-after-drag (element sets `running = true`;
   the engine must un-settle itself, e.g. `setPosition` resets the
   iteration budget the way `d3` reheats and `ngraph` resets counters).
5. Topology change: `load` again (or `reload` with the freeze report)
   without losing the placed coordinates; new rows seeded near their
   neighbours or randomly (design 14.4 leaves the product choice open).
6. `dim === 2` keeps `z` untouched (zero).
7. Deterministic given `seed` and the snapshot (same initial positions as
   the CPU path via the LCG of 2.5.1); the simulation itself need not be
   bit-reproducible across GPUs (float reductions), but SHOULD be
   reproducible on the same device for the same dispatch shape, which is a
   test convenience.
8. `dispose()` destroys every buffer; `release(snapshot)` on
   `snapshot-replaced` (design 14.4) drops the cached uploads of the old
   snapshot.

### 5.2 Delta between `LayoutEngine` (today) and `LayoutSimulation` (design)

| Today's abstract `LayoutEngine` (`LayoutEngine.ts` lines 36-63) | `LayoutSimulation` | Bridge |
| --- | --- | --- |
| `init(): Promise<void>` | `load(snapshot, positions)` | `init` acquires the device; `load` is called by `LayoutManager` after `getSnapshot()` (design M10) |
| `addNode` / `addEdge` per record | none (snapshot-driven) | the manager stops feeding records to simulation engines |
| `getNodePosition(n): Position` (allocates) | positions array is read directly | `getNodePositionInto(index, out)` (design 14.4) |
| `getEdgePosition(e)` | same array via `edgesByIndex` | `getEdgePositionsInto(e, outSrc, outDst)` |
| `step(): void` | `step(iterations?): void \| Promise<void>` | see 5.3 |
| `pin(n)` / `unpin(n)` | `setFixed(mask)` | the manager keeps the pinned bitmap and re-sends it on change |
| `setNodePosition(n, p)` | `setPosition(i, x, y, z)` | index instead of object |
| `get isSettled` | `readonly settled` | rename |
| optional `dispose()` | `dispose()` | required |
| `static getOptionsForDimension` | `dim` option | unchanged |

### 5.3 The async step problem (must be decided in the plan)

`UpdateManager.updateLayout()` is synchronous and calls `step()` up to
`stepMultiplier` times per frame; the render loop does not await. A GPU
`step()` that returns a Promise needs one of:

- (A) Fire-and-forget with double buffering: `step()` submits the command
  buffers for k iterations and returns immediately; the readback of the
  previous submission is awaited in the background and copied into the
  element's array when it resolves; `settled` reflects the last completed
  readback. The frame loop never blocks; the renderer is at most one
  submission behind. `step()` while a submission is in flight is a no-op
  (or queues at most one more). This matches "the GPU buffer is
  authoritative while stepping" and is what cosmos does with its WebGL
  simulation (positions live in textures; the CPU reads them back only on
  demand).
- (B) `LayoutManager.step()` becomes async and the frame loop awaits it:
  rejected, because `UpdateManager.update()` is called from Babylon's
  render loop and the whole manager chain is synchronous.

Recommendation: (A), with `step(iterations)` meaning "run this many
iterations in one submission"; the element calls it once per frame with
`stepMultiplier` (so the per-frame cost is one submission and one
`mapAsync`), and the per-frame readback is `12 * n` bytes (1.2 MB at 100k
nodes; the 4070 SUPER moves that in well under a millisecond, but the
`mapAsync` round trip is one frame of latency). Section 8.5 details the
readback.

---------------------------------------------------------------------------

## 6. Scale that graphty targets today

Measured facts, not aspirations:

| Source | Number |
| --- | --- |
| `graphty-element/stories/PerformanceTest.stories.ts` lines 55-57, 60 ("Performance/Large Graph") | 150 nodes / 250 edges, ngraph, seed 42 |
| `layout/test/forceatlas2-layout.test.ts` lines 504-528 ("should handle large graphs reasonably") | 100 nodes, 30 iterations, < 2 s |
| `layout/README.md` line 1037 | "Large Graphs (>1000 nodes)": advises circular first then 50 spring iterations |
| `NGraphLayoutEngine.ts` line 176 | forced settle after 1000 steps |
| design section 15.3 (lines 4350-4356) | targets: mobile 100k / 1M, desktop interactive 1M / 10M, batch 10M / 100M |
| design section 15.4 (line 4403) | "a static layout re-run on 100k nodes is a full FA2 / KK run (seconds)" -- the design already assumes 100k-node FA2 is the workload |
| design line 4210 | "per-node Babylon mesh creation, JMESPath style selection and the on-change Proxies still dominate at 100k nodes" |
| README.md line 28 (monorepo) | "support for large datasets through mesh instancing and GPU acceleration" (claim) |

So: the element is exercised at ~10^2 nodes today; the format is
benchmarked at 10^5 / 10^6; the GPU layout should be designed and tested
at 10^4 (interactive parity with ngraph feel), 10^5 (the format's mobile
tier; per-frame readback still trivial) and 10^6 nodes (desktop tier;
where the O(n^2) kernel is no longer viable and an approximation is
mandatory, section 8.2). The renderer will not draw 10^6 nodes today
(per-node `InstancedMesh`, research note 04 section 5), so at that tier the
layout is a batch / Node.js use case, not a 60 fps one.

---------------------------------------------------------------------------

## 7. ForceAtlas2 option parity list

Read from `layout/src/layouts/force-directed/forceatlas2.ts` (function
signature) and `graphty-element/src/layout/ForceAtlas2LayoutEngine.ts`
(zod config lines 99-115 and UI schema lines 10-97).

### 7.1 Must honour in the first GPU version

| Option | CPU default | Element schema | GPU binding | Notes |
| --- | --- | --- | --- | --- |
| `maxIter` | 100 | int > 0, default 100 | iteration budget / settle bound | in a steppable engine this is the total iteration budget across frames, not a single call |
| `jitterTolerance` | 1.0 | > 0, default 1.0, advanced | scalar in the speed controller (CPU side) | |
| `scalingRatio` | 2.0 | > 0, default 2.0 | uniform | repulsion multiplier |
| `gravity` | 1.0 | > 0, default 1.0 | uniform | the element schema forbids 0 (`positive()`), the CPU accepts 0; accept 0 on the GPU |
| `strongGravity` | false | bool, advanced | uniform flag or `override` | linear vs unit-vector gravity |
| `distributedAction` | false | bool, advanced | uniform flag | divides attraction by mass |
| `linlog` | false | bool, advanced | uniform flag | `log(1 + d) / d` attraction |
| `nodeMass` | null -> degree + 1 | `Record<number, number> \| null` | `Float32Array(n)` upload | design 14.3: accept `Float32Array`, column name or Record; default `outDegree()[i] + 1` computed on the CPU (or in a tiny kernel from `rowPtr`) |
| `weight` (`weightPath` in the element) | null (unweighted) | `string \| null` | per-arc `Float32Array(A)` or `snapshot.weights` or none | inert in the element today (research note 03 section 8.1: the literal `{nodes, edges}` has no `getEdgeData`); becomes live through the snapshot -- documented behaviour change |
| `seed` | null (random) | `number \| null`, advanced | CPU-side LCG for the initial positions | seed 0 == unseeded (LCG quirk) |
| `dim` | 2 | int 2..3, default 2 | uniform; element overrides from view mode | 2D must not touch z |
| `pos` | null | `Record<number, number[]> \| null` | initial upload | the element will pass the current position array instead (design 14.4 `fromPositionColumn`); keep the "missing axis / missing node" fill rules of 2.1.2 |
| `scalingFactor` (element-level, `SimpleLayoutConfig`) | 100 | 1..1000 | scene scale applied on write-back | not a layout option; section 4.7 |

### 7.2 Can be deferred (documented as unsupported until a later slice)

| Option | Why deferrable | What to do meanwhile |
| --- | --- | --- |
| `nodeSize` / `adjustSizes` | changes the repulsion distance per pair (needs `size` in the tile) and the apply rule (`0.1 * speed`, cap 10); rarely set; its CPU semantics are the sign-suspect line of 2.1.9 | throw or ignore with a warning in v1; add once the CPU rewrite fixes the formula |
| `dissuadeHubs` | ignored by the CPU port and absent from NetworkX; the element still exposes it (schema line 66-73) | accept and ignore, exactly like the CPU |
| `scale` / `center` (`CommonLayoutOptions`) | FA2 today has neither (always `rescaleLayout` to the unit ball); in a steppable engine per-step rescaling is wrong (section 8.4) | honour at `load` time for the initial placement only |
| `weight` as a NAMED edge column | needs the element's `expandEdges` cache (design 14.4) | v1 supports `weight === true` (snapshot weights) and `null` |

### 7.3 Options that have no CPU counterpart but the engine needs

| Option | Source | Purpose |
| --- | --- | --- |
| `fixed` mask (`setFixed`) | FR has `fixed`; FA2 does not; the element needs pins (section 4.5) | integrate step skips masked nodes (they still exert forces) |
| `settle` threshold | ngraph's `0.01` per-node movement, element's `0.05` average (section 3.1) | the `settled` flag (section 8.6) |
| `iterationsPerStep` | `stepMultiplier` (section 4.1) | how many kernel iterations one submission runs |
| approximation control (`theta`, or grid resolution) | ngraph `theta 0.8`, GraphWaGu `theta 0.8`, cosmos has no theta | only when the approximate repulsion path is added (section 8.2); exact O(n^2) needs none |

---------------------------------------------------------------------------

## 8. GPU implications

### 8.1 Data the FA2 kernel binds (all from the undirected snapshot)

| Buffer | Bytes | Source | Update frequency |
| --- | --- | --- | --- |
| `rowPtr` | 4(n + 1) | `snapshot.rowPtr` (arena hot prefix or per array, design 10.3) | per snapshot |
| `colIdx` | 4A | `snapshot.colIdx` | per snapshot |
| `weights` | 4A or absent | `snapshot.weights` / expanded column | per snapshot / per column version |
| `mass` | 4n | `outDegree()[i] + 1` or `nodeMass` | per snapshot / per option change |
| `size` | 4n | only with `adjustSizes` | deferred |
| `positions` | 12n (stride 3, `array<f32>` NOT `array<vec3f>`, design C14) | element array on `load`; GPU-authoritative after | read back every step (section 8.5) |
| `fixed` mask | 4 ceil(n / 32) | `setFixed` | on pin/unpin |
| `force` | 12n scratch | | per iteration |
| `swing`, `traction` partials | 4 ceil(n / 256) each | workgroup reductions | per iteration |
| `centerOfMass` partials | 12 ceil(n / 256) | reduction over positions | per iteration |
| `movement` partial | 4 ceil(n / 256) | for `settled` | per iteration |
| params uniform | < 256 B | `scalingRatio, gravity, flags, speed, speedEfficiency, jitterTolerance, dim, n, stride` | per iteration (speed changes) |

Note the per-iteration CPU dependency: `estimateFactor` needs the GLOBAL
`swing` and `traction` sums before the apply pass can run. Options:
(1) one readback per iteration (kills the k-iterations-per-submission
plan); (2) a single-workgroup "finalize" kernel that reduces the partials
and updates `speed` / `speedEfficiency` in a small storage buffer that the
apply kernel reads -- `estimateFactor` is ~20 scalar operations, trivially
expressible in WGSL. Choose (2): the whole iteration then stays on the
device and `k` iterations are `k * (5 or 6)` dispatches in one command
buffer with no host round trip.

### 8.2 Repulsion: exact tiles first, an approximation second

- The CPU reference is EXACT all-pairs. An exact tiled n-body kernel
  (positions and masses staged through workgroup memory, 256 per tile) is
  O(n^2) per iteration but memory-free: at n = 10k that is 10^8 pair
  evaluations per iteration, roughly 1-2 ms on the 4070 SUPER class of
  GPU, so 10k nodes at 60 fps with several iterations per frame is
  realistic; at n = 100k it is 10^10 pairs per iteration (hundreds of ms),
  usable in Node.js batch mode but not interactively. cosmos draws the
  exact/approximate line at 4096 points for the same reason ("4096^2 ~ 17M
  pair evaluations per step stays around a millisecond on modest GPUs",
  `repos/cosmos/src/modules/ForceManyBody/index.ts` lines 55-76).
- Every large-scale GPU force layout the owner pointed at approximates:
  GraphWaGu builds a Barnes-Hut tree on the GPU (Morton codes, radix sort,
  `create_tree.wgsl`, then `compute_forcesBH.wgsl` walks it with an
  explicit 64-entry stack per thread and `theta = 0.8`,
  `repos/GraphWaGu/src/wgsl/compute_forcesBH.wgsl` lines 41-115 and
  `src/webgpu/force_directed.ts` line 41); cosmos 3.x uses a grid pyramid
  with a Monte-Carlo near field and no theta at all (`repos/cosmos/src/
  config.ts` lines 390-397, the deprecated `simulationRepulsionTheta`
  comment); ngraph and d3 use Barnes-Hut on the CPU. The plan should
  schedule the exact kernel as the first slice (parity, simplicity,
  correctness oracle) and an approximate repulsion (grid/pyramid or BH)
  as the follow-up slice that unlocks 10^5-10^6 nodes, selected by node
  count with a documented crossover (cosmos: 4096; expect ~8-16k on a
  discrete GPU).
- Approximation changes the picture relative to the exact CPU FA2; parity
  tests for the approximate path compare against the EXACT GPU path on the
  same start (distributional metrics), not against coordinates.

### 8.3 Attraction, gravity, integration

- Attraction is a CSR-row gather per node over the undirected snapshot:
  `for a in rowPtr[i]..rowPtr[i+1]: j = colIdx[a]; w = weights ? weights[a]
  : 1` -- no atomics, both arcs present by construction (design 10.5).
  Degree skew is the load-balance problem: a hub row of 10^5 arcs on one
  thread stalls the workgroup. `degreeOrder().segmentOffsets` (design 10.6)
  gives the high / mid / low tiers for workgroup-per-node / subgroup-per-
  node / thread-per-node scheduling; the first slice may ignore it (the
  element's graphs are small) but the kernel API should take the
  permutation with the `override USE_PERM` pattern from day one.
- Gravity needs the centroid: one reduction over positions per iteration
  (partials + finalize kernel; the same finalize kernel as 8.1).
- Integration: per node, `swinging = mass * |force|`, `factor = speed /
  (1 + sqrt(speed * swinging))`, `p += force * factor` unless `fixed`; z
  untouched when `dim == 2`; accumulate `|dp|` for the settle reduction.
- Initial placement (`load`): rows that are NaN (unplaced) get LCG
  coordinates on the CPU (2.5.1) in index order, scaled to scene units
  (section 8.4); rows that are finite are kept (that is how a re-`load`
  after a topology change preserves the user's layout, design 14.4).

### 8.4 Units and `rescaleLayout`

The CPU FA2 simulates in [-1, 1] units and normalises to the unit ball at
the END; the element multiplies by `scalingFactor` (100). A steppable
engine writes scene units every step and cannot renormalise each time
(it would rescale pinned / dragged nodes and change the camera framing
every frame). Recommendation: simulate in layout units in the GPU
buffer, apply `scalingFactor` (and `center`) in the write-back kernel or
on the CPU copy, and never rescale after `load`. FA2's forces are not
scale-invariant (repulsion `1/d` or `1/d^2` vs linear attraction), so
simulating in scene units with the same constants would produce a
different equilibrium; keeping layout units on the device keeps the CPU
constants and defaults valid. `fromPositionColumn` (design 14.3) already
divides by `scale` when seeding, which is exactly the inverse mapping
`load` needs for finite rows.

### 8.5 Readback per frame

- What the renderer needs each frame is the full `12n`-byte position array
  (every `Node.update()` reads its row; every `Edge.update()` compares two
  rows). A `copyBufferToBuffer` into a `MAP_READ` staging buffer plus
  `mapAsync`, copied into the element's `Float32Array` (`set` from the
  mapped range BEFORE `unmap`, design 10.7), is one submission per frame.
  Ring of 2-3 staging buffers so a slow `mapAsync` never blocks the next
  submission.
- Latency: the mapped result arrives a frame (or more) after submission;
  positions the renderer draws lag the simulation by one step. This is
  invisible for a settling layout. Drag interaction is the exception:
  `setPosition` writes the dragged row into the GPU buffer (`writeBuffer`,
  12 bytes at offset `12 * i`) AND the element already writes
  `mesh.position` directly during a drag (`NodeBehavior.ts` line 162), so
  the readback lag does not affect the dragged node; neighbours lag one
  frame, as they do with ngraph's one-step-per-frame today.
- `column.markDirty()` once per frame on the position column (design 14.4
  M12) after the copy so any OTHER GPU consumer of the position column
  (a future renderer upload) sees the new version.
- Node.js: identical code path with Dawn (`webgpu@0.4.0`), no frame loop:
  `await step(k)` in a loop until `settled` is the batch API; the browser
  path is the same function driven by the frame loop with fire-and-forget.

### 8.6 Settlement

FA2 has no usable convergence test (2.1.3 item 10). Define `settled` as:
`iterationsDone >= maxIter` OR the per-node mean displacement over the last
k iterations (on the device, from the movement reduction) is below a
threshold in LAYOUT units (ngraph uses 0.01 per body in scene units,
`ngraph.forcelayout/index.js` line 63; the element loosens it to a 10-step
average of 0.05, `NGraphLayoutEngine.ts` lines 179-191). `setPosition`,
`setFixed` (unpin) and `load` reset `iterationsDone` (reheat), matching d3
(`D3GraphLayoutEngine.ts` lines 309-315, 358-365) and ngraph (lines
205-224). The threshold and window are options with defaults; the element
must see `settled === true` within `maxIter` steps regardless of the
threshold so screenshots and label animations proceed (section 4.3).

### 8.7 Determinism and testing hooks

- Seeded initial positions come from the CPU LCG (2.5.1), so the GPU and
  CPU runs start identically; the differential test then runs the CPU
  reference (post-14.3 rewrite, `Float64Array` scratch) and the GPU kernel
  for the same iteration count and compares per-iteration `swing`,
  `traction`, `speed` traces (readable from the finalize buffer) within a
  tolerance, plus final distributional metrics. f32 vs f64 and reduction
  order make coordinate equality unattainable; the trace comparison
  catches formula errors early, on graphs of 10-1000 nodes, in Node.js.
- Property tests that need no reference: fixed nodes never move;
  `dim === 2` leaves z exactly 0; disconnected components separate;
  gravity pulls the centroid to 0; `setPosition` is visible in the next
  readback; `dispose` frees every buffer (Dawn's `device.lost` / leak
  checks); `load` on a snapshot with `arcCount === 0` binds no zero-length
  buffer (design 10.5).

### 8.8 Ordering consequence for the plan

1. Settle the CPU FA2 formulas of 2.1.9 in the layout package's section-
   14.3 rewrite (the parity oracle) before writing WGSL; port its tests to
   the indexed API.
2. Walking skeleton -> FA2 exact-tile slice implementing `LayoutSimulation`
   (load / step(k) / settled / setFixed / setPosition / dispose), Node.js
   tests first, one browser story through the element.
3. graphty-element: the `LayoutManager` bridge (5.2, 5.3): sync frame loop
   + fire-and-forget async step + per-frame readback into the element
   array; select the GPU engine behind the "forceatlas2" type when an
   accelerator is injected, else the CPU `LayoutSimulation` (the CPU FA2
   should ALSO become steppable in the rewrite so the element has one
   code path and the GPU is only an implementation swap).
4. Approximate repulsion (grid pyramid or BH) as the scale slice;
   `degreeOrder` load balancing for the attraction gather at the same
   time.
5. FR / spring on the same skeleton (two kernels, `fixed` mask,
   temperature on the CPU side); ARF if ever wanted; KK later via the APSP
   primitive.

---------------------------------------------------------------------------

## Sources

Monorepo (read-only), all under `/home/apowers/Projects/graphty-monorepo/`:

- `layout/src/layouts/force-directed/forceatlas2.ts` (448 lines),
  `fruchterman-reingold.ts` (163), `spring.ts` (34), `arf.ts` (132),
  `kamada-kawai.ts` (109), `index.ts` (9)
- `layout/src/algorithms/optimization/kamada-kawai-solver.ts` (lines 1-140)
- `layout/src/utils/random.ts`, `utils/rescale.ts`, `utils/params.ts`,
  `utils/graph.ts`; `layout/src/layouts/basic/random.ts`
- `layout/src/index.ts`, `layout/src/types/{graph,layout,index}.ts`
- `layout/test/forceatlas2-layout.test.ts` (lines 344-600), test file list
- `layout/README.md` lines 1030-1075
- `graphty-element/src/layout/LayoutEngine.ts` (365 lines),
  `ForceAtlas2LayoutEngine.ts` (173), `SpringLayoutEngine.ts` (120),
  `ArfLayoutEngine.ts` (114), `D3GraphLayoutEngine.ts` (388),
  `NGraphLayoutEngine.ts` (322), `index.ts` (34)
- `graphty-element/src/managers/LayoutManager.ts` (475 lines),
  `UpdateManager.ts` (lines 1-340), `DataManager.ts` (lines 47-60,
  137-145, 180-240, 320-330, 392-410, 448-520), `RenderManager.ts` (lines
  55-130, 165-175), `LifecycleManager.ts` (lines 119-143)
- `graphty-element/src/Graph.ts` (lines 200-260, 332-348, 480-598,
  2052-2054), `Node.ts` (lines 150-303), `NodeBehavior.ts` (lines
  131-243), `Edge.ts` (lines 270-300, 425-470), `config/GraphBehavior.ts`
  (lines 1-26)
- `graphty-element/stories/PerformanceTest.stories.ts` (lines 15-95),
  `stories/Layout.stories.ts` (lines 1-160)
- `README.md` line 28
- `node_modules/ngraph.forcelayout/` version 3.3.1:
  `lib/createPhysicsSimulator.js` lines 1-200, `index.js` lines 45-75,
  `lib/codeGenerators/generateIntegrator.js`
- `design/graph-format/graph-format-design.md`: lines 1000-1030 (position
  column C14), 2443-2553 (sections 10.4-10.8), 3959-4047 (14.3),
  4048-4211 (14.4), 4212-4244 (14.5), 4340-4412 (15.2-15.4)
- `tmp/graph-format-design/03-layout-needs.md` (873 lines, all),
  `04-graphty-element-usage.md` (sections 4.2-6), `09-webgpu-requirements.md`
  (sections 0 and 3)

Staged packages under `/home/apowers/Projects/webgpu-graph-algorithms/`:

- `packages/graph-format/src/types/columns.ts` (lines 700-730: `NodeMask`),
  `src/types/snapshot.ts` (lines 380-640: `GraphSnapshotContract`),
  `src/columns/bitmap.ts`, `src/columns/remap.ts` (export list)
- `src/types/index.ts` (the obsolete `CSRGraph` scaffold)

External (verified by fetch or clone on 2026-09-14):

- NetworkX `forceatlas2_layout` and `estimate_factor`, fetched from
  https://raw.githubusercontent.com/networkx/networkx/main/networkx/drawing/layout.py
  (main branch at fetch time; quoted in 2.1.9)
- GraphWaGu, cloned to `tmp/webgpu-plan/repos/GraphWaGu`
  (https://github.com/harp-lab/GraphWaGu): `README.md`,
  `src/webgpu/force_directed.ts` lines 24-41, 237-260, 572-586, 952-956,
  `src/wgsl/compute_forcesBH.wgsl`, `src/wgsl/compute_attractive_new.wgsl`,
  `src/wgsl/` file list (create_tree, morton_codes, radix_sort)
- cosmos 3.4.1, cloned to `tmp/webgpu-plan/repos/cosmos`
  (https://github.com/cosmosgl/cosmos): `src/config.ts` lines 360-440,
  `src/variables.ts` lines 15-78, `src/modules/ForceManyBody/index.ts`
  lines 14-76, `src/modules/Store/index.ts` lines 329-332
- jaredmcqueen/analytics, cloned to `tmp/webgpu-plan/repos/analytics`
  (https://github.com/jaredmcqueen/analytics): `README.md` lines 1-12
  (WebGL, "fruchterman reingold ... all performed on the GPU", "60 FPS ...
  1 million nodes" -- claim, not measured here)
- `tmp/webgpu-plan/repos/d3-force-webgpu/README.md` lines 1-25 (a
  d3-force API clone on WebGPU compute; it advertises a CPU fallback,
  which this project forbids -- listed for API shape only)

Not verified / not fetched: the GraphWaGu paper PDF (Google Drive link
in its README), cosmograph.app examples and the cosmograph PyPI package
(the owner's links; only the `cosmos` source was read), the NVIDIA /
Buffalo / CACM references (algorithm-side, out of scope for this note).
