/**
 * The `degree` kernel body (contract 4.5; spec 6 row 3, 11.5): the out-degree of every row of `[P.start, P.end)`
 * through the row-walking gather with the USE_PERM dummy pattern, counting the arcs of the bound window whose
 * neighbour index is a valid node; `P.accumulate == 1` adds into `out[i]` instead of overwriting (the P4 windowed
 * loop). Bindings, overrides and the `RangeParams` block come from the registry entry (src/kernels.ts); WG,
 * USE_PERM and `linear_id` from the prelude. The text is normative: test/helpers/sabotage.ts mutates it textually,
 * so it is copied from the contract and never re-derived or restyled.
 */
export const degreeWgsl = /* wgsl */ `
@compute @workgroup_size(WG)
fn degree(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = linear_id(wid, lid.x) + P.start;
    if (row >= P.end) { return; }
    let i = select(row, perm[row], USE_PERM);
    let a0 = max(rowPtr[i], P.arcBase);
    let a1 = min(rowPtr[i + 1u], P.arcEnd);
    var d = 0u;
    for (var arc = a0; arc < a1; arc = arc + 1u) {
        let nbr = colIdx[arc - P.arcBase];               // \`target\` is a WGSL reserved word (spec 16.2); never use it as an identifier
        d = d + select(0u, 1u, nbr < P.n);
    }
    out[i] = select(d, out[i] + d, P.accumulate == 1u);
}
`;
