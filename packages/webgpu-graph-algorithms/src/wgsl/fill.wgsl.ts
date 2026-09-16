/**
 * The `fill` kernel body (contract 4.5): `dst[i] = P.value` (mode 0) or `i + P.value` (mode 1, iota) for every
 * `i < P.count`, `i` from the prelude's `linear_id` so a 2D dispatch covers counts above MAX_1D_ITEMS. The 17M-item
 * linear_id test of spec 5.2 / 11.5 is mode 1 over 16,776,961 words. Normative text, copied verbatim.
 */
export const fillWgsl = /* wgsl */ `
@compute @workgroup_size(WG)
fn fill(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = linear_id(wid, lid.x);
    if (i >= P.count) { return; }
    dst[i] = select(P.value, i + P.value, P.mode == 1u);
}
`;
