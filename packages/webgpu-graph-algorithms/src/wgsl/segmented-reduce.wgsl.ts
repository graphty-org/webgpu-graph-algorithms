/**
 * The `segmented-reduce` kernel body (spec 6 row 3; contract 4.5), TIER 0 = thread-per-row: one invocation per row
 * of [P.start, P.end) folds the VALUE snippet over the row's arcs inside the bound window [P.arcBase, P.arcEnd) and
 * writes out[i] as f32; a row with no arcs gets the identity element (0 for sum, F32_MAX for min, -F32_MAX for
 * max). The mid / high degree tiers (TIER 1 / 2 over degreeOrder()) land at P4. The body is normative: a sabotage
 * mutation (test/helpers/sabotage.ts) is a textual edit of it, so it is not restyled.
 */

/**
 * Entry point `segmented_reduce`; overrides OP (0 sum, 1 min, 2 max) and TIER (0); snippet slot VALUE: statements
 * assigning `v` from `row`, `arc`, `nbr`, `weight` (`target` is a WGSL reserved word, hence `nbr`).
 */
export const segmentedReduceWgsl = /* wgsl */ `
fn identity() -> f32 { if (OP == 1u) { return F32_MAX; } if (OP == 2u) { return -F32_MAX; } return 0.0; }
fn comb(a: f32, b: f32) -> f32 { if (OP == 1u) { return min(a, b); } if (OP == 2u) { return max(a, b); } return a + b; }

@compute @workgroup_size(WG)
fn segmented_reduce(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = linear_id(wid, lid.x) + P.start;
    if (row >= P.end) { return; }
    let i = select(row, perm[row], USE_PERM);
    let a0 = max(rowPtr[i], P.arcBase);
    let a1 = min(rowPtr[i + 1u], P.arcEnd);
    var acc = identity();
    for (var arc = a0; arc < a1; arc = arc + 1u) {
        let nbr = colIdx[arc - P.arcBase];               // the neighbour index (\`target\` is a WGSL reserved word)
        var weight = 1.0;
        if (HAS_WEIGHTS) { weight = weights[arc - P.arcBase]; }
        var v = 0.0;
        //@@VALUE@@
        acc = comb(acc, v);
    }
    out[i] = select(acc, comb(out[i], acc), P.accumulate == 1u);
}
`;
