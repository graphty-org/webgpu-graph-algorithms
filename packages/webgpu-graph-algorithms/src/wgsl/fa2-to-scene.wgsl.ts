/**
 * The per-batch `fa2-to-scene` kernel (spec 7.13, 7.18; contract 4.5): unpacks the vec4f layout-unit positions into
 * the stride-3 scene array the staging copy reads back, applying `scale` and `center`; in 2D the third component is
 * `center.z` on every readback whatever z the device holds.
 *
 * Body only (spec 3.5, D9); normative text (contract 4.5); exempt from the sabotage matrix (SABOTAGE_EXEMPT: a wrong
 * toScene fails the exact-equality position tests directly).
 */

/** The toScene body: entry point `to_scene`; an early return is legal here because no barrier follows (spec 3.5 rule 1). */
export const fa2ToSceneWgsl = /* wgsl */ `@compute @workgroup_size(WG)
fn to_scene(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = linear_id(wid, lid.x);
    if (i >= P.n) { return; }
    let s = pos[i].xyz * P.scale + P.center.xyz;
    scene[3u * i] = s.x;
    scene[3u * i + 1u] = s.y;
    scene[3u * i + 2u] = select(s.z, P.center.z, P.dim == 2u);       // 2D writes z = center.z on every readback (7.13)
}`;
