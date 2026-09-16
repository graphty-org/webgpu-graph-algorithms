/**
 * K1 of the ForceAtlas2 iteration, `fa2-stats-finalize` (spec 7.4; contract 4.5): one workgroup folds the partials
 * the previous integrate (K5) wrote into the state block -- centroid, RMS radius, layout radius (max |p - c|, the
 * exact spec 3.3 value from `partials.max.w`), bounding box, mean displacement over free nodes (0 when every node
 * is fixed), the settle counter -- increments the iteration counter and writes the K1 half of the trace record.
 * On the first iteration after load() (`FA2_FLAG_FIRST`) it folds nothing and keeps the host-written state.
 *
 * This file holds the kernel BODY only (spec 3.5, D9): no bind-group lines and no `override` lines -- the composer
 * emits them from the registry entry in src/kernels.ts (contract 3.10.1). The text is normative (contract 4.5) and
 * is the target of the K1 sabotage mutations (test/helpers/sabotage.ts, P3-T5); amend the contract before editing.
 */

/** The K1 body: entry point `stats_finalize`; calls the reduction helpers (`needs: ["subgroups"]`, contract 4.3). */
export const fa2StatsFinalizeWgsl = /* wgsl */ `// K1: folds the previous integrate's partials into the state block (spec 7.4); one workgroup
@compute @workgroup_size(WG)
fn stats_finalize(@builtin(local_invocation_id) lid: vec3<u32>) {
    let groups = (P.n + WG - 1u) / WG;
    let fold = (P.flags & FA2_FLAG_FIRST) == 0u;    // the first iteration after load() keeps the host-written state
    var sum = vec4f(0.0);
    var lo = vec4f(F32_MAX);
    var hi = vec4f(-F32_MAX);
    var disp = 0.0;
    var free = 0u;
    if (fold) {
        for (var g = lid.x; g < groups; g = g + WG) {   // sequential per lane in index order: deterministic
            let q = partials[g];
            sum = sum + q.sum;
            lo = min(lo, q.min);
            hi = max(hi, q.max);
            disp = disp + q.dispFree.x;
            free = free + u32(q.dispFree.y);
        }
    }
    let tSum = wg_reduce_vec4(sum, lid.x, 0u);
    let tLo = wg_reduce_vec4(lo, lid.x, 1u);
    let tHi = wg_reduce_vec4(hi, lid.x, 2u);
    let tDisp = wg_reduce_f32(disp, lid.x, 0u);
    let tFree = wg_reduce_u32(free, lid.x, 0u);
    if (lid.x == 0u) {
        if (fold) {
            let n = f32(P.n);
            let c = tSum.xyz / n;
            S.centroid = vec4f(c, 0.0);
            S.rmsRadius = sqrt(max(tSum.w, 0.0) / n);                  // RMS radius about the previous centroid (7.17)
            S.min = vec4f(tLo.xyz, 0.0);
            S.max = vec4f(tHi.xyz, 0.0);
            S.radius = sqrt(max(tHi.w, 0.0));                          // max |p - centroid| about the same previous centroid as rmsRadius (K5 puts |q|^2 in max.w)
            let meanDisp = select(tDisp / f32(tFree), 0.0, tFree == 0u);  // all-fixed: 0, never NaN (7.4)
            S.meanDisplacement = meanDisp;
            S.settledCount = select(0u, S.settledCount + 1u, meanDisp <= P.settleThreshold * S.rmsRadius);
        }
        S.iteration = S.iteration + 1u;
        T[P.iterationIndex].meanDisplacement = S.meanDisplacement;
        T[P.iterationIndex].settledCount = S.settledCount;
        T[P.iterationIndex].iteration = S.iteration;
    }
}`;
