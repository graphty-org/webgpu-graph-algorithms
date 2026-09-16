/**
 * The `reduce` kernel body (contract 4.5; spec 6 row 1): one level of the 2-3 level reduction. Overrides: OP
 * 0 sum / 1 min / 2 max, DTYPE 0 f32 / 1 u32 / 2 vec4f (four words per element, loaded through bitcast from the
 * u32 storage view), FINAL true for the one-workgroup level that folds the partials sequentially per lane in
 * index order (deterministic) and writes one element at `out[P.outOffset]`; the level-1 form writes
 * `out[P.outOffset + group]`. Calls the prelude's `wg_reduce_*` helpers (needs: ["subgroups"]) in uniform
 * control flow: DTYPE is a pipeline constant, so the branch on it is uniform. Normative text, copied verbatim.
 */
export const reduceWgsl = /* wgsl */ `
// DTYPE 0 = f32, 1 = u32, 2 = vec4f (4 words per element); OP 0 = sum, 1 = min, 2 = max; FINAL = the one-workgroup level
fn identity_f() -> f32 { if (OP == 1u) { return F32_MAX; } if (OP == 2u) { return -F32_MAX; } return 0.0; }
fn identity_u() -> u32 { if (OP == 1u) { return U32_MAX; } return 0u; }        // U32_MAX from the prelude: the literal is forbidden in bodies (4.1)
fn comb_f(a: f32, b: f32) -> f32 { if (OP == 1u) { return min(a, b); } if (OP == 2u) { return max(a, b); } return a + b; }
fn comb_u(a: u32, b: u32) -> u32 { if (OP == 1u) { return min(a, b); } if (OP == 2u) { return max(a, b); } return a + b; }
fn comb_v(a: vec4f, b: vec4f) -> vec4f { if (OP == 1u) { return min(a, b); } if (OP == 2u) { return max(a, b); } return a + b; }
fn load_f(i: u32) -> f32 { return bitcast<f32>(src[i]); }
fn load_v(i: u32) -> vec4f {
    return vec4f(bitcast<f32>(src[4u * i]), bitcast<f32>(src[4u * i + 1u]), bitcast<f32>(src[4u * i + 2u]), bitcast<f32>(src[4u * i + 3u]));
}
fn store_f(i: u32, v: f32) { out[i] = bitcast<u32>(v); }
fn store_v(i: u32, v: vec4f) {
    out[4u * i] = bitcast<u32>(v.x);
    out[4u * i + 1u] = bitcast<u32>(v.y);
    out[4u * i + 2u] = bitcast<u32>(v.z);
    out[4u * i + 3u] = bitcast<u32>(v.w);
}

@compute @workgroup_size(WG)
fn reduce(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    var accF = identity_f();
    var accU = identity_u();
    var accV = vec4f(identity_f());
    if (FINAL) {
        for (var i = lid.x; i < P.count; i = i + WG) {           // sequential per lane in index order: deterministic
            if (DTYPE == 0u) { accF = comb_f(accF, load_f(i)); }
            else if (DTYPE == 1u) { accU = comb_u(accU, src[i]); }
            else { accV = comb_v(accV, load_v(i)); }
        }
    } else {
        let i = linear_id(wid, lid.x);
        if (i < P.count) {
            if (DTYPE == 0u) { accF = load_f(i); }
            else if (DTYPE == 1u) { accU = src[i]; }
            else { accV = load_v(i); }
        }
    }
    // uniform control flow: the workgroup reduction of the selected dtype (DTYPE is a pipeline constant, so the branch is uniform)
    var tF = 0.0;
    var tU = 0u;
    var tV = vec4f(0.0);
    if (DTYPE == 0u) { tF = wg_reduce_f32(accF, lid.x, OP); }
    else if (DTYPE == 1u) { tU = wg_reduce_u32(accU, lid.x, OP); }
    else { tV = wg_reduce_vec4(accV, lid.x, OP); }
    if (lid.x == 0u) {
        let g = select(group_id(wid), 0u, FINAL);
        let o = P.outOffset + g;
        if (DTYPE == 0u) { store_f(o, tF); }
        else if (DTYPE == 1u) { out[o] = tU; }
        else { store_v(o, tV); }
    }
}
`;
