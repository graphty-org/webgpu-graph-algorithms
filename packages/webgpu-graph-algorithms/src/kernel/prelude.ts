/**
 * The WGSL prelude every composed module receives first (contract 4.1), the two reduction-helper blocks the
 * composer splices for a body whose spec lists `needs: ["subgroups"]` (contract 4.3, D16), and the WGSL
 * reserved-word list the composer rejects identifiers from (contract 3.9).
 *
 * CONTRACT DECISION (3.9): this text lives in the kernel layer, not in `src/wgsl/`, because the composer must own
 * what it splices and `src/wgsl/` then holds kernel BODIES only. Every numeric constant is interpolated from
 * `constants.ts` and graph-format's `INVALID_INDEX`; the literal-grep test of spec 3.5 (`test/kernel/wgsl.test.ts`)
 * covers this file's template, so no u32 sentinel or workgroup-size literal is ever typed here.
 */

import { INVALID_INDEX } from "@graphty/graph-format";

import {
    FA2_COINCIDENT_SQ,
    FA2_DISTANCE_FLOOR,
    FA2_DISTANCE_FLOOR_SQ,
    FA2_FLAG_FIRST,
    MAX_WORKGROUPS_PER_DIM,
    U32_MAX,
    WORKGROUP_SIZE,
} from "../constants.js";
import { WebGpuGraphError } from "../errors.js";

/**
 * Formats a finite number as a WGSL f32 literal that always carries a decimal point or an exponent (`0.01`,
 * `1e-8`, `2.0`), so an integral value never becomes an abstract-int literal (contract 4.1).
 * @param value - the number to format
 * @returns the literal text
 */
export function wgslF32Literal(value: number): string {
    if (!Number.isFinite(value)) {
        throw new WebGpuGraphError("E_INVALID_ARGUMENT", `wgslF32Literal: ${value} is not a finite number`, {
            argument: "value",
            value,
            expected: "a finite number",
        });
    }
    const text = String(value);
    return /[.e]/.test(text) ? text : `${text}.0`;
}

/** The prelude text with the constants interpolated from constants.ts and graph-format's INVALID_INDEX (4.1). */
export const PRELUDE_WGSL: string = /* wgsl */ `// ---- prelude: constants, standard overrides, helpers (every module receives this text first)
const INVALID_INDEX: u32 = ${INVALID_INDEX}u;
const U32_MAX: u32 = ${U32_MAX}u;
const MAX_WORKGROUPS_PER_DIM: u32 = ${MAX_WORKGROUPS_PER_DIM}u;
const FA2_DIST_FLOOR: f32 = ${wgslF32Literal(FA2_DISTANCE_FLOOR)};
const FA2_DIST_FLOOR_SQ: f32 = ${wgslF32Literal(FA2_DISTANCE_FLOOR_SQ)};
const FA2_COINCIDENT_SQ: f32 = ${wgslF32Literal(FA2_COINCIDENT_SQ)};
const FA2_FLAG_FIRST: u32 = ${FA2_FLAG_FIRST}u;
const F32_MAX: f32 = 0x1.fffffep+127;
override WG: u32 = ${WORKGROUP_SIZE}u;
override USE_PERM: bool = false;
override HAS_WEIGHTS: bool = false;
override SUBGROUP_MIN: u32 = 4u;
override SUBGROUP_MAX: u32 = 0u;

fn linear_id(wid: vec3<u32>, lid: u32) -> u32 { return (wid.x + wid.y * MAX_WORKGROUPS_PER_DIM) * WG + lid; }
fn group_id(wid: vec3<u32>) -> u32 { return wid.x + wid.y * MAX_WORKGROUPS_PER_DIM; }
fn lowbias32(x0: u32) -> u32 {
    var x = x0;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}
fn mask_bit(w: u32, i: u32) -> bool { return ((w >> (i & 31u)) & 1u) == 1u; }
fn unpack_u8(w: u32, i: u32) -> u32 { return (w >> (8u * (i & 3u))) & 0xFFu; }
fn pair_hash(i: u32, j: u32) -> u32 { return lowbias32((min(i, j) * 0x9E3779B9u) ^ max(i, j)); }
fn hash_unit(h: u32) -> f32 { return f32(h >> 8u) * (1.0 / 16777216.0); }
fn hash_dir(h: u32, dim: u32) -> vec3f {
    let phi = 6.283185307179586 * hash_unit(h);
    if (dim == 2u) { return vec3f(cos(phi), sin(phi), 0.0); }
    let z = 2.0 * hash_unit(lowbias32(h ^ 0x5bd1e995u)) - 1.0;
    let r = sqrt(max(0.0, 1.0 - z * z));
    return vec3f(r * cos(phi), r * sin(phi), z);
}
fn kick_dir(i: u32, j: u32, dim: u32) -> vec3f {
    let d = hash_dir(pair_hash(i, j), dim);
    return select(d, -d, i > j);
}`;

/** Line count of PRELUDE_WGSL (the compilation-info formatter subtracts it plus the emitted declarations). */
export const PRELUDE_LINES: number = PRELUDE_WGSL.split("\n").length;

/** The workgroup-memory reduction helpers (the twin) (4.3). */
export const REDUCE_HELPERS_WORKGROUP_WGSL: string = /* wgsl */ `var<workgroup> wg_scratch_v: array<vec4f, WG>;
var<workgroup> wg_scratch_u: array<u32, WG>;
fn combine_v(a: vec4f, b: vec4f, op: u32) -> vec4f {
    if (op == 1u) { return min(a, b); }
    if (op == 2u) { return max(a, b); }
    return a + b;
}
fn combine_u(a: u32, b: u32, op: u32) -> u32 {
    if (op == 1u) { return min(a, b); }
    if (op == 2u) { return max(a, b); }
    return a + b;
}
fn wg_reduce_vec4(v: vec4f, lid: u32, op: u32) -> vec4f {
    workgroupBarrier();
    wg_scratch_v[lid] = v;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) { wg_scratch_v[lid] = combine_v(wg_scratch_v[lid], wg_scratch_v[lid + s], op); }
        workgroupBarrier();
    }
    let total = wg_scratch_v[0];
    workgroupBarrier();
    return total;
}
fn wg_reduce_u32(v: u32, lid: u32, op: u32) -> u32 {
    workgroupBarrier();
    wg_scratch_u[lid] = v;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) { wg_scratch_u[lid] = combine_u(wg_scratch_u[lid], wg_scratch_u[lid + s], op); }
        workgroupBarrier();
    }
    let total = wg_scratch_u[0];
    workgroupBarrier();
    return total;
}
fn wg_reduce_f32(v: f32, lid: u32, op: u32) -> f32 { return wg_reduce_vec4(vec4f(v, 0.0, 0.0, 0.0), lid, op).x; }`;

/** The subgroup reduction helpers, spliced only with `enable subgroups;` (4.3). */
export const REDUCE_HELPERS_SUBGROUP_WGSL: string = /* wgsl */ `override SG_SLOTS: u32 = (WG + SUBGROUP_MIN - 1u) / SUBGROUP_MIN;   // one slot per subgroup; the count is largest when the compiler picks the SMALLEST size
var<workgroup> sg_counter: atomic<u32>;
var<workgroup> sg_val_v: array<vec4f, SG_SLOTS>;
var<workgroup> sg_val_u: array<u32, SG_SLOTS>;
var<workgroup> sg_key: array<u32, SG_SLOTS>;
var<workgroup> sg_sorted_v: array<vec4f, SG_SLOTS>;
var<workgroup> sg_sorted_u: array<u32, SG_SLOTS>;
fn combine_v(a: vec4f, b: vec4f, op: u32) -> vec4f {
    if (op == 1u) { return min(a, b); }
    if (op == 2u) { return max(a, b); }
    return a + b;
}
fn combine_u(a: u32, b: u32, op: u32) -> u32 {
    if (op == 1u) { return min(a, b); }
    if (op == 2u) { return max(a, b); }
    return a + b;
}
fn wg_reduce_vec4(v: vec4f, lid: u32, op: u32) -> vec4f {
    workgroupBarrier();
    if (lid == 0u) { atomicStore(&sg_counter, 0u); }
    workgroupBarrier();
    let s_add = subgroupAdd(v);
    let s_min = subgroupMin(v);
    let s_max = subgroupMax(v);
    var partial = s_add;
    if (op == 1u) { partial = s_min; }
    if (op == 2u) { partial = s_max; }
    let key = subgroupMin(lid);                       // the smallest local id of this subgroup: a stable identity without @builtin(subgroup_id)
    var slot = 0u;
    if (subgroupElect()) { slot = atomicAdd(&sg_counter, 1u); }   // D16: the elected lane takes a slot from the counter
    slot = subgroupBroadcast(slot, 0u);                            // and broadcasts it (lane 0 is the elected lane in uniform control flow)
    if (subgroupElect()) { sg_val_v[slot] = partial; sg_key[slot] = key; }
    workgroupBarrier();
    let count = atomicLoad(&sg_counter);
    if (lid < count) {                                             // rank the slots by key so the final sum has a fixed order (11.9 item 4)
        let mine = sg_key[lid];
        var rank = 0u;
        for (var k = 0u; k < count; k = k + 1u) { if (sg_key[k] < mine) { rank = rank + 1u; } }
        sg_sorted_v[rank] = sg_val_v[lid];
    }
    workgroupBarrier();
    var total = sg_sorted_v[0];
    for (var k = 1u; k < count; k = k + 1u) { total = combine_v(total, sg_sorted_v[k], op); }
    workgroupBarrier();
    return total;
}
fn wg_reduce_u32(v: u32, lid: u32, op: u32) -> u32 {
    workgroupBarrier();
    if (lid == 0u) { atomicStore(&sg_counter, 0u); }
    workgroupBarrier();
    let s_add = subgroupAdd(v);
    let s_min = subgroupMin(v);
    let s_max = subgroupMax(v);
    var partial = s_add;
    if (op == 1u) { partial = s_min; }
    if (op == 2u) { partial = s_max; }
    let key = subgroupMin(lid);
    var slot = 0u;
    if (subgroupElect()) { slot = atomicAdd(&sg_counter, 1u); }
    slot = subgroupBroadcast(slot, 0u);
    if (subgroupElect()) { sg_val_u[slot] = partial; sg_key[slot] = key; }
    workgroupBarrier();
    let count = atomicLoad(&sg_counter);
    if (lid < count) {
        let mine = sg_key[lid];
        var rank = 0u;
        for (var k = 0u; k < count; k = k + 1u) { if (sg_key[k] < mine) { rank = rank + 1u; } }
        sg_sorted_u[rank] = sg_val_u[lid];
    }
    workgroupBarrier();
    var total = sg_sorted_u[0];
    for (var k = 1u; k < count; k = k + 1u) { total = combine_u(total, sg_sorted_u[k], op); }
    workgroupBarrier();
    return total;
}
fn wg_reduce_f32(v: f32, lid: u32, op: u32) -> f32 { return wg_reduce_vec4(vec4f(v, 0.0, 0.0, 0.0), lid, op).x; }`;

/** The helper function names a body may call when its spec lists needs: ["subgroups"]. */
export const REDUCE_HELPER_NAMES = ["wg_reduce_f32", "wg_reduce_u32", "wg_reduce_vec4"] as const;

/**
 * The WGSL reserved words of spec section 16.2 (the `_reserved` production), frozen; composeWgsl rejects a body or
 * snippet that uses one (3.9). PLAN DECISION: the list is the 146 words of the W3C Candidate Recommendation Draft
 * of 2026-08-31 (identical to the editor's draft of 2026-09-01); the contract's count of 147 predates the removal
 * of `binding_array`. `free`, `valid`, `tile`, `slot`, `key` and `count` are NOT reserved.
 */
export const WGSL_RESERVED_WORDS: readonly string[] = Object.freeze(
    `NULL Self abstract active alignas alignof as asm asm_fragment async attribute auto await become cast catch class
    co_await co_return co_yield coherent column_major common compile compile_fragment concept const_cast consteval
    constexpr constinit crate debugger decltype delete demote demote_to_helper do dynamic_cast enum explicit export
    extends extern external fallthrough filter final finally friend from fxgroup get goto groupshared highp impl
    implements import inline instanceof interface layout lowp macro macro_rules match mediump meta mod module move
    mut mutable namespace new nil noexcept noinline nointerpolation non_coherent noncoherent noperspective null
    nullptr of operator package packoffset partition pass patch pixelfragment precise precision premerge priv
    protected pub public readonly ref regardless register reinterpret_cast require resource restrict self set shared
    sizeof smooth snorm static static_assert static_cast std subroutine super target template this thread_local
    throw trait try type typedef typeid typename typeof union unless unorm unsafe unsized use using varying virtual
    volatile wgsl where with writeonly yield`
        .trim()
        .split(/\s+/),
);
