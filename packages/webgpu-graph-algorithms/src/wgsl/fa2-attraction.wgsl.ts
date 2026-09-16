/**
 * K2 of the ForceAtlas2 iteration, `fa2-attraction` (spec 7.5; contract 4.5): the thread-per-row gather of the
 * attraction force over the undirected CSR rows (both arcs present, so the sum is symmetric with no atomics), the
 * linear or linlog law, the optional weights, the distributed-action division by the mass in `pos.w`, written as
 * the FIRST writer of `force` each iteration. P3 ships the thread-per-row tier over `[tierStart, tierEnd)` = `[0, n)`
 * with `USE_PERM = false`; the subgroup / workgroup tiers arrive with P4 (`TIER`).
 *
 * Body only (spec 3.5, D9); normative text (contract 4.5); the K2 sabotage mutations (P3-T5) are textual edits of it.
 */

/** The K2 body: entry point `attraction`; an early return is legal here because no barrier follows (spec 3.5 rule 1). */
export const fa2AttractionWgsl = /* wgsl */ `fn store_force(i: u32, f: vec3f) {
    force[3u * i] = f.x;
    force[3u * i + 1u] = f.y;
    force[3u * i + 2u] = f.z;
}

@compute @workgroup_size(WG)
fn attraction(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = linear_id(wid, lid.x) + P.tierStart;
    if (row >= P.tierEnd) { return; }                          // no barrier follows in this tier (3.5 rule 1)
    let i = select(row, perm[row], USE_PERM);
    let pi = pos[i];                                           // xyz + mass in one load (D23)
    var f = vec3f(0.0);
    for (var a = rowPtr[i]; a < rowPtr[i + 1u]; a = a + 1u) {
        let j = colIdx[a];
        if (j == i) { continue; }                              // a self-loop exerts no force
        var w = 1.0;
        if (HAS_WEIGHTS) { w = weights[a]; }
        let d = pos[j].xyz - pi.xyz;                           // toward j
        let len = max(length(d), FA2_DIST_FLOOR);
        let mag = select(w, w * log(1.0 + len) / len, LINLOG); // linear: |F| = w len; linlog: |F| = w log(1 + len)
        f = f + d * mag;
    }
    if (DISTRIBUTED) { f = f / pi.w; }
    store_force(i, f);                                         // overwrites: attraction is the first writer of force each iteration
}`;
