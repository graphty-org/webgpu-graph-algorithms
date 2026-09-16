/**
 * The `fa2-speed-finalize` kernel body (K4; contract 4.5; spec 7.10): ONE workgroup sums `partials[*].swingTraction`
 * sequentially per lane (deterministic), reduces in uniform control flow, then lane 0 runs the line-for-line port
 * of the CPU `estimateFactor` (SWING_MODE 1 accumulates NetworkX's sums across iterations from S.swing / S.traction)
 * and writes `S.speed`, `S.speedEfficiency`, `S.swing`, `S.traction` and the four controller fields of
 * `T[P.iterationIndex]`. `target` is a WGSL reserved word, hence `targetSpeed`. Normative text, copied verbatim:
 * the P1-T5 sabotage rows (`* 0.5` -> `* 0.9`, the 1.3 rise, the swing / traction swap) are textual edits of it.
 * The halving predicate is `swing > 2.0 * tr`, the exact form of the port's `swing / traction > 2` (contract 4.5
 * CONTRACT DECISION K4-1, G3 finding G3-F6): every paper-mode first iteration after load() has oldForce = 0, so
 * traction is EXACTLY half the swing and the predicate sits on its knife edge; a multiplication by 2 and the
 * comparison are exact on every device, while WGSL grants f32 division 2.5 ULP and Dawn on NVIDIA returns 2 + 1 ulp
 * for 2.2% of the inputs x / (x / 2) (tmp/p3-fix/div-probe.mjs), which took the branch the CPU reference never takes.
 */
export const fa2SpeedFinalizeWgsl = /* wgsl */ `
@compute @workgroup_size(WG)
fn speed_finalize(@builtin(local_invocation_id) lid: vec3<u32>) {
    let groups = (P.n + WG - 1u) / WG;
    var st = vec2f(0.0);
    for (var g = lid.x; g < groups; g = g + WG) { st = st + partials[g].swingTraction; }   // sequential per lane: deterministic
    let t = wg_reduce_vec4(vec4f(st, 0.0, 0.0), lid.x, 0u);
    if (lid.x == 0u) {
        var swing = t.x;
        var traction = t.y;
        if (SWING_MODE == 1u) { swing = S.swing + t.x; traction = S.traction + t.y; }   // NetworkX accumulates across iterations from 1
        let n = f32(P.n);
        let optJitter = 0.05 * sqrt(n);
        let minJitter = sqrt(optJitter);
        let maxJitter = 10.0;
        let tr = max(traction, 1.0e-30);                                             // guards the division only (7.10)
        let other = min(maxJitter, optJitter * traction / (n * n));
        var jitter = P.jitterTolerance * max(minJitter, other);
        var eff = S.speedEfficiency;
        if (swing > 2.0 * tr) {                                                      // swing / traction > 2 in the exact form (contract 4.5 CONTRACT DECISION K4-1: 2 x is exact, a WGSL f32 division is not)
            if (eff > 0.05) { eff = eff * 0.5; }                                     // the CPU's conditional multiply (7.2)
            jitter = max(jitter, P.jitterTolerance);
        }
        let targetSpeed = select(jitter * eff * traction / swing, 1.0e30, swing == 0.0);  // +Inf in the port; 1e30 gives the same min() below (\`target\` is reserved)
        if (swing > jitter * traction) {
            if (eff > 0.05) { eff = eff * 0.7; }
        } else if (S.speed < 1000.0) {
            eff = eff * 1.3;
        }
        S.speed = S.speed + min(targetSpeed - S.speed, 0.5 * S.speed);
        S.speedEfficiency = eff;
        S.swing = swing;
        S.traction = traction;
        T[P.iterationIndex].swing = swing;
        T[P.iterationIndex].traction = traction;
        T[P.iterationIndex].speed = S.speed;
        T[P.iterationIndex].speedEfficiency = eff;
    }
}
`;
