/**
 * The CPU ForceAtlas2 reference (spec 11.3, 11.4, 11.9 item 2; contract 5.3): an index-based transcription of the
 * spec 7.2 formula table with the two 4.6 corrections (origin gravity and the m|F| local swing in networkx mode),
 * mirroring the kernel sequence K1 K2 K3 K4 K5 of contract 4.5 stage by stage so inspect() can be compared per
 * kernel. It is the SPEC of the WGSL, not a transcription of it: every formula below cites the 7.2 row or the
 * networkx 3.4.2 layout.py line it comes from, and the oracle is itself checked against NetworkX trajectories
 * (test/oracle/forceatlas2-networkx.test.ts) and hand-computed values (test/oracle/swing-mode.test.ts).
 *
 * f64 by default. `precision: "f32"` rounds every operation with Math.fround and sums in the GPU's tile / lane
 * order (K3 walks j ascending in 256-wide tiles, the workgroup reductions are the 4.3 tree, the K1 / K4 folds are
 * the per-lane grid-stride loop followed by the tree), the tight leg of the trace test of spec 11.4.
 *
 * Units: positions are LAYOUT units, stride 3 (z = 0 in 2D, as load() uploads them). forceAtlas2Oracle() below is
 * the scene-unit wrapper (the toScene inverse on the way in, toScene on the way out).
 *
 * The u32 hash of the coincident kick (kickDir) re-implements the prelude's lowbias32 / pair_hash / hash_dir with
 * JavaScript's 32-bit operators on HASH WORDS and node indices below 2^32; no arc index or byte offset is ever
 * touched by a bitwise operator here (house rule).
 */

import { type F32, type GraphSnapshot, maskTest, type NodeMask } from "@graphty/graph-format";

import {
    FA2_COINCIDENT_SQ,
    FA2_DEFAULTS,
    FA2_DISTANCE_FLOOR,
    FA2_DISTANCE_FLOOR_SQ,
    WORKGROUP_SIZE,
} from "../../src/constants.js";
import { seedPositions } from "../../src/layouts/seed.js";
import type { ForceAtlas2TraceRecord } from "../../src/types/layout.js";
import type { ForceAtlas2Options } from "../../src/types/options.js";

/** Options of the oracle: the FA2 options plus the mode, the precision and the resolved per-node inputs. */
export interface OracleOptions extends ForceAtlas2Options {
    readonly compat: "paper" | "networkx";
    readonly precision: "f64" | "f32";
    /** 0 centroid, 1 origin (defaults from compat: paper -> 0, networkx -> 1). */
    readonly gravityCenter?: 0 | 1 | undefined;
    readonly fixed?: NodeMask | null | undefined;
    readonly mass?: ArrayLike<number> | null | undefined;
    readonly weights?: ArrayLike<number> | null | undefined;
}

/** The per-iteration intermediates the GPU exposes through inspect() (spec 11.9 item 2), stride 3. */
export interface OracleStages {
    readonly attraction: Float64Array;
    readonly repulsion: Float64Array;
    readonly gravity: Float64Array;
    readonly force: Float64Array;
    readonly oldForce: Float64Array;
    readonly swingPerNode: Float64Array;
    readonly tractionPerNode: Float64Array;
    readonly displacement: Float64Array;
    readonly partials: {
        readonly swing: number;
        readonly traction: number;
        readonly sum: readonly [number, number, number];
        readonly sumSq: number;
        readonly min: readonly [number, number, number];
        readonly max: readonly [number, number, number];
        readonly disp: number;
        readonly free: number;
    };
}

/** One trace record plus the K1 statistics of the same iteration. */
export interface OracleTraceRecord extends ForceAtlas2TraceRecord {
    readonly rmsRadius: number;
    /** max |p - c| about the same previous centroid as rmsRadius (what K1 writes into state.radius, 4.5). */
    readonly layoutRadius: number;
    readonly centroid: readonly [number, number, number];
}

/** What K1 writes at the start of an iteration (the fold of the previous integrate's partials, contract 4.5). */
interface OracleFold {
    readonly centroid: readonly [number, number, number];
    readonly rmsRadius: number;
    readonly layoutRadius: number;
    readonly meanDisplacement: number;
    readonly settledCount: number;
}

/**
 * A simulation's iteration-start state, as resync() seeds it (P3-T5 PLAN DECISION 17, the re-synchronised trace
 * legs): the controller as K4 left it after the previous iteration, K1's statistics of the iteration about to run,
 * and F(t-1) per node (stride 3, the paper-mode swing reference; null keeps load()'s zeros).
 */
export interface OracleResyncState extends OracleFold {
    readonly speed: number;
    readonly speedEfficiency: number;
    readonly swing: number;
    readonly traction: number;
    readonly iteration: number;
    readonly oldForce: ArrayLike<number> | null;
}

/** A rounding function: Math.fround in f32 mode, the identity in f64 mode. */
type Round = (x: number) => number;

/** The identity rounding of f64 mode. */
function identity(x: number): number {
    return x;
}

/** The prelude's F32_MAX (0x1.fffffep+127): the min / max identities of the vec4 reductions (contract 4.3, 4.5). */
const F32_MAX = 3.4028234663852886e38;

/** A reduction operator of wg_reduce_* (contract 4.3): 0 = sum, 1 = min, 2 = max. */
type ReduceOp = "sum" | "min" | "max";

/**
 * combine_v / combine_u of contract 4.3 on one lane.
 * @param a - the left operand
 * @param b - the right operand
 * @param op - the operator
 * @param r - the rounding of the precision mode
 * @returns the combined value
 */
function combine(a: number, b: number, op: ReduceOp, r: Round): number {
    if (op === "min") {
        return Math.min(a, b);
    }
    if (op === "max") {
        return Math.max(a, b);
    }
    return r(a + b);
}

/**
 * The workgroup-form wg_reduce of contract 4.3 over WORKGROUP_SIZE lanes, in place: lanes[l] = combine(lanes[l],
 * lanes[l + s]) for s = WG / 2, WG / 4, ..., 1 (the fixed tree order both twins agree with to summation noise).
 * @param lanes - exactly WORKGROUP_SIZE values (destroyed)
 * @param op - the operator
 * @param r - the rounding of the precision mode
 * @returns the workgroup total (lanes[0])
 */
function treeReduce(lanes: Float64Array, op: ReduceOp, r: Round): number {
    for (let s = WORKGROUP_SIZE / 2; s > 0; s = Math.floor(s / 2)) {
        for (let l = 0; l < s; l++) {
            lanes[l] = combine(lanes[l], lanes[l + s], op, r);
        }
    }
    return lanes[0];
}

/**
 * The per-workgroup totals of a per-node value (K3's swing / traction epilogue, K5's partials A and C): node i sits
 * in lane i % WG of group floor(i / WG); lanes beyond n carry the identity, exactly as the invalid lanes of 4.5 do.
 * @param values - one value per node (length >= n)
 * @param n - the node count
 * @param op - the operator
 * @param r - the rounding of the precision mode
 * @param laneIdentity - the value of an invalid lane (0 for sums, F32_MAX for min, -F32_MAX for max)
 * @returns one total per workgroup (ceil(n / WG) entries)
 */
function groupTotals(values: ArrayLike<number>, n: number, op: ReduceOp, r: Round, laneIdentity: number): Float64Array {
    const groups = Math.ceil(n / WORKGROUP_SIZE);
    const totals = new Float64Array(groups);
    const lanes = new Float64Array(WORKGROUP_SIZE);
    for (let g = 0; g < groups; g++) {
        for (let l = 0; l < WORKGROUP_SIZE; l++) {
            const i = g * WORKGROUP_SIZE + l;
            lanes[l] = i < n ? values[i] : laneIdentity;
        }
        totals[g] = treeReduce(lanes, op, r);
    }
    return totals;
}

/**
 * The one-workgroup fold of K1 and K4 over the per-workgroup partials: lane l accumulates partials[l],
 * partials[l + WG], ... sequentially (the `for (g = lid; g < groups; g += WG)` loop of 4.5), then the tree.
 * @param partials - one value per workgroup
 * @param op - the operator
 * @param r - the rounding of the precision mode
 * @param laneIdentity - the lane's initial value (0, F32_MAX or -F32_MAX)
 * @returns the folded total
 */
function foldGroups(partials: ArrayLike<number>, op: ReduceOp, r: Round, laneIdentity: number): number {
    const lanes = new Float64Array(WORKGROUP_SIZE).fill(laneIdentity);
    for (let g = 0; g < partials.length; g++) {
        const l = g % WORKGROUP_SIZE;
        lanes[l] = combine(lanes[l], partials[g], op, r);
    }
    return treeReduce(lanes, op, r);
}

/**
 * WGSL length() of a three-vector in the precision mode (sqrt of the left-to-right sum of squares).
 * @param r - the rounding of the precision mode
 * @param x - component x
 * @param y - component y
 * @param z - component z
 * @returns the length
 */
function len3(r: Round, x: number, y: number, z: number): number {
    return r(Math.sqrt(r(r(r(x * x) + r(y * y)) + r(z * z))));
}

/**
 * Wellons' lowbias32 integer hash of the prelude (contract 4.1), on u32 words.
 * @param x0 - the input word
 * @returns the hashed word in [0, 2^32)
 */
function lowbias32(x0: number): number {
    let x = x0 >>> 0;
    x = (x ^ (x >>> 16)) >>> 0;
    x = Math.imul(x, 0x7feb352d) >>> 0;
    x = (x ^ (x >>> 15)) >>> 0;
    x = Math.imul(x, 0x846ca68b) >>> 0;
    x = (x ^ (x >>> 16)) >>> 0;
    return x;
}

/**
 * pair_hash of the prelude: lowbias32((min(i, j) * 0x9E3779B9) ^ max(i, j)), symmetric in (i, j).
 * @param i - one node index
 * @param j - the other node index
 * @returns the pair's hash word
 */
function pairHash(i: number, j: number): number {
    const lo = Math.min(i, j);
    const hi = Math.max(i, j);
    return lowbias32(((Math.imul(lo, 0x9e3779b9) >>> 0) ^ hi) >>> 0);
}

/**
 * hash_unit of the prelude: the top 24 bits of a word as a value in [0, 1), exact in f32 and f64.
 * @param h - a hash word
 * @returns the unit value
 */
function hashUnit(h: number): number {
    return (h >>> 8) * (1 / 16777216);
}

/**
 * kick_dir of the prelude (contract 4.1): the deterministic unit direction of the coincident kick of spec 7.2,
 * antisymmetric (kickDir(i, j) = -kickDir(j, i)); z = 0 in 2D.
 * @param i - the node receiving the kick
 * @param j - the coincident node
 * @param dim - 2 or 3
 * @param precision - "f32" mirrors the GPU's f32 trigonometry; "f64" (default) is the oracle's own precision
 * @returns the unit direction as [x, y, z]
 */
export function kickDir(
    i: number,
    j: number,
    dim: 2 | 3,
    precision: "f64" | "f32" = "f64",
): readonly [number, number, number] {
    const r: Round = precision === "f32" ? Math.fround : identity;
    const h = pairHash(i, j);
    const phi = r(r(6.283185307179586) * hashUnit(h));
    let d: [number, number, number];
    if (dim === 2) {
        d = [r(Math.cos(phi)), r(Math.sin(phi)), 0];
    } else {
        const z = r(r(2 * hashUnit(lowbias32((h ^ 0x5bd1e995) >>> 0))) - 1);
        const rr = r(Math.sqrt(Math.max(0, r(1 - r(z * z)))));
        d = [r(rr * r(Math.cos(phi))), r(rr * r(Math.sin(phi))), z];
    }
    return i > j ? [-d[0], -d[1], -d[2]] : d;
}

/** What estimateFactor returns: the new controller state plus the two intermediates a unit test pins. */
export interface EstimateFactorResult {
    readonly speed: number;
    readonly speedEfficiency: number;
    readonly jitter: number;
    readonly targetSpeed: number;
}

/**
 * estimateFactor, line for line (spec 7.2 row "estimateFactor", 7.10; networkx 3.4.2 layout.py lines 1387-1420;
 * K4 of contract 4.5): the conditional `eff *= 0.5` / `*= 0.7` form that skips at or below the 0.05 floor, the
 * 1e30 target when swing is 0 (the same min() outcome as NetworkX's +inf), the 0.5 x speed maximum rise; the
 * halving predicate in the exact form `swing > 2 tr` of CONTRACT DECISION K4-1 (the same truth value as the port's
 * `swing / traction > 2`, decided identically on every device).
 * @param n - the node count
 * @param swing - the global swing (accumulated in networkx mode)
 * @param traction - the global traction
 * @param speed - the current speed
 * @param speedEfficiency - the current speed efficiency
 * @param jitterTolerance - the jitterTolerance option
 * @param precision - "f32" rounds every operation as K4 does
 * @returns the new speed and efficiency, with the jitter and target speed of the step
 */
export function estimateFactor(
    n: number,
    swing: number,
    traction: number,
    speed: number,
    speedEfficiency: number,
    jitterTolerance: number,
    precision: "f64" | "f32",
): EstimateFactorResult {
    const r: Round = precision === "f32" ? Math.fround : identity;
    const nf = r(n);
    const jt = r(jitterTolerance);
    const optJitter = r(r(0.05) * r(Math.sqrt(nf)));
    const minJitter = r(Math.sqrt(optJitter));
    const maxJitter = 10;
    const tr = Math.max(traction, r(1e-30)); // guards the division only (7.10)
    const other = Math.min(maxJitter, r(r(optJitter * traction) / r(nf * nf)));
    let jitter = r(jt * Math.max(minJitter, other));
    let eff = speedEfficiency;
    if (swing > r(2 * tr)) {
        // swing / traction > 2 in its exact form (contract 4.5 CONTRACT DECISION K4-1: at iteration 0 the traction
        // is exactly half the swing, and a device's f32 division may round x / (x / 2) above 2; 2 x never does)
        if (eff > r(0.05)) {
            eff = r(eff * r(0.5)); // the CPU's conditional multiply (7.2)
        }
        jitter = Math.max(jitter, jt);
    }
    const targetSpeed = swing === 0 ? r(1e30) : r(r(r(jitter * eff) * traction) / swing);
    if (swing > r(jitter * traction)) {
        if (eff > r(0.05)) {
            eff = r(eff * r(0.7));
        }
    } else if (speed < 1000) {
        eff = r(eff * r(1.3));
    }
    const newSpeed = r(speed + Math.min(r(targetSpeed - speed), r(r(0.5) * speed)));
    return { speed: newSpeed, speedEfficiency: eff, jitter, targetSpeed };
}

/** The per-workgroup partials A and C of one integrate (contract 4.5 K5), kept until the next K1 fold. */
interface GroupPartials {
    readonly sumX: Float64Array;
    readonly sumY: Float64Array;
    readonly sumZ: Float64Array;
    readonly sumW: Float64Array;
    readonly minX: Float64Array;
    readonly minY: Float64Array;
    readonly minZ: Float64Array;
    readonly maxX: Float64Array;
    readonly maxY: Float64Array;
    readonly maxZ: Float64Array;
    readonly maxW: Float64Array;
    readonly disp: Float64Array;
    readonly free: Float64Array;
}

/** The K1 totals of a set of partials (foldTotals): the folded sums, box, max |p - c|^2, displacement and free count. */
interface FoldTotals {
    readonly sum: readonly [number, number, number];
    readonly sumSq: number;
    readonly min: readonly [number, number, number];
    readonly max: readonly [number, number, number];
    readonly maxSq: number;
    readonly disp: number;
    readonly free: number;
}

/**
 * Empty stages (before the first step()).
 * @param n - the node count
 * @returns zeroed stages
 */
function emptyStages(n: number): OracleStages {
    return {
        attraction: new Float64Array(3 * n),
        repulsion: new Float64Array(3 * n),
        gravity: new Float64Array(3 * n),
        force: new Float64Array(3 * n),
        oldForce: new Float64Array(3 * n),
        swingPerNode: new Float64Array(n),
        tractionPerNode: new Float64Array(n),
        displacement: new Float64Array(3 * n),
        partials: { swing: 0, traction: 0, sum: [0, 0, 0], sumSq: 0, min: [0, 0, 0], max: [0, 0, 0], disp: 0, free: 0 },
    };
}

/**
 * The CPU ForceAtlas2 reference: positions in LAYOUT units (stride 3), f64 scratch (or f32 scratch summing in tile
 * order when precision is "f32").
 */
export class ForceAtlas2Oracle {
    readonly n: number;
    readonly dim: 2 | 3;
    /** Layout-unit positions (Float64Array or Float32Array by precision). */
    readonly positions: Float64Array | Float32Array;

    private readonly rowPtr: Uint32Array;
    private readonly colIdx: Uint32Array;
    private readonly arcCount: number;
    private readonly weights: Float64Array | null;
    private readonly mass: Float64Array;
    private readonly precision: "f64" | "f32";
    private readonly round: Round;
    private readonly swingMode: 0 | 1;
    private readonly gravityCenter: 0 | 1;
    private readonly linlog: boolean;
    private readonly distributed: boolean;
    private readonly strongGravity: boolean;
    private readonly scalingRatio: number;
    private readonly gravity: number;
    private readonly jitterTolerance: number;
    private readonly settleThreshold: number;
    private fixed: NodeMask | null;
    private readonly oldForce: Float64Array;

    private speedValue = 1;
    private speedEfficiencyValue = 1;
    private swingValue = 1;
    private tractionValue = 1;
    private settledCountValue = 0;
    private iterationValue = 0;
    private centroid: [number, number, number] = [0, 0, 0];
    private rmsRadius = 0;
    private radius = 0;
    private minValue: [number, number, number] = [0, 0, 0];
    private maxValue: [number, number, number] = [0, 0, 0];
    private meanDisplacement = 0;
    /** The partials of the last integrate; null before the first step (K1 keeps the host-written state then). */
    private pending: GroupPartials | null = null;
    private stagesValue: OracleStages;
    private readonly traceRecords: OracleTraceRecord[] = [];

    /**
     * Builds the reference over a snapshot from layout-unit positions (stride 3; taken as given -- the caller rounds
     * to f32 when it wants the GPU's start; z is forced to 0 in 2D as load() does).
     * @param s - the undirected snapshot (both arcs of every edge present)
     * @param positions - 3 x nodeCount layout-unit coordinates
     * @param options - the mode, precision and FA2 options (FA2_DEFAULTS applied)
     */
    constructor(s: GraphSnapshot, positions: ArrayLike<number>, options: OracleOptions) {
        if (s.directed) {
            throw new Error("ForceAtlas2Oracle: the snapshot must be undirected (pass toUndirected().snapshot)");
        }
        const n = s.nodeCount;
        if (positions.length !== 3 * n) {
            throw new Error(`ForceAtlas2Oracle: positions.length ${positions.length} !== 3 x ${n}`);
        }
        this.n = n;
        this.dim = options.dim ?? FA2_DEFAULTS.dim;
        this.precision = options.precision;
        const r: Round = options.precision === "f32" ? Math.fround : identity;
        this.round = r;
        this.rowPtr = s.rowPtr;
        this.colIdx = s.colIdx;
        this.arcCount = s.arcCount;
        this.positions = options.precision === "f32" ? new Float32Array(3 * n) : new Float64Array(3 * n);
        for (let i = 0; i < n; i++) {
            this.positions[3 * i] = r(positions[3 * i]);
            this.positions[3 * i + 1] = r(positions[3 * i + 1]);
            this.positions[3 * i + 2] = this.dim === 2 ? 0 : r(positions[3 * i + 2]);
        }
        this.mass = ForceAtlas2Oracle.resolveMass(s, options);
        this.weights = ForceAtlas2Oracle.resolveWeights(s, options);
        this.fixed = null;
        this.setFixed(options.fixed ?? null);
        this.oldForce = new Float64Array(3 * n);
        this.swingMode = options.compat === "networkx" ? 1 : 0;
        this.gravityCenter = options.gravityCenter ?? (options.compat === "networkx" ? 1 : 0);
        this.linlog = options.linlog ?? FA2_DEFAULTS.linlog;
        this.distributed = options.distributedAction ?? FA2_DEFAULTS.distributedAction;
        this.strongGravity = options.strongGravity ?? FA2_DEFAULTS.strongGravity;
        this.scalingRatio = r(options.scalingRatio ?? FA2_DEFAULTS.scalingRatio);
        this.gravity = r(options.gravity ?? FA2_DEFAULTS.gravity);
        this.jitterTolerance = options.jitterTolerance ?? FA2_DEFAULTS.jitterTolerance;
        this.settleThreshold = r(options.settleThreshold ?? FA2_DEFAULTS.settleThreshold);
        if (this.scalingRatio <= 0 || this.gravity < 0 || this.jitterTolerance <= 0 || this.settleThreshold < 0) {
            throw new Error(
                "ForceAtlas2Oracle: scalingRatio > 0, gravity >= 0, jitterTolerance > 0, settleThreshold >= 0 required",
            );
        }
        this.stagesValue = emptyStages(n);
        this.writeInitialStatistics();
    }

    /** The controller state. */
    get speed(): number {
        return this.speedValue;
    }

    /** The speed efficiency of the controller. */
    get speedEfficiency(): number {
        return this.speedEfficiencyValue;
    }

    /** The global swing (fresh in paper mode, accumulated from 1 in networkx mode). */
    get swing(): number {
        return this.swingValue;
    }

    /** The global traction (same accumulation rule as swing). */
    get traction(): number {
        return this.tractionValue;
    }

    /** Consecutive iterations whose mean displacement was under settleThreshold x rmsRadius (spec 7.17). */
    get settledCount(): number {
        return this.settledCountValue;
    }

    /** The state's iteration counter (K1 increments it every step). */
    get iteration(): number {
        return this.iterationValue;
    }

    /**
     * The state's bounding box (S.min / S.max as K1 last wrote them; load()'s values before the first fold).
     * PLAN DECISION P3-T4: not in contract 5.3's declaration -- added so fa2-inspect.test.ts can compare the state
     * after K1 (the contract declares no accessor for S.min / S.max).
     */
    get bounds(): { readonly min: readonly [number, number, number]; readonly max: readonly [number, number, number] } {
        return {
            min: [this.minValue[0], this.minValue[1], this.minValue[2]],
            max: [this.maxValue[0], this.maxValue[1], this.maxValue[2]],
        };
    }

    /** The intermediates of the LAST step(). */
    get stages(): OracleStages {
        return this.stagesValue;
    }

    /** The trace of every step() so far. */
    get trace(): readonly OracleTraceRecord[] {
        return this.traceRecords;
    }

    /**
     * Replaces the fixed mask (no implicit reheat: the parity tests call reheat() where the simulation would).
     * @param mask - the NodeMask (ceil(n / 32) words, LSB-first) or null for no fixed node
     */
    setFixed(mask: NodeMask | null): void {
        if (mask !== null && mask.length < Math.ceil(this.n / 32)) {
            throw new Error(`ForceAtlas2Oracle: mask.length ${mask.length} < ceil(${this.n} / 32)`);
        }
        this.fixed = mask === null ? null : new Uint32Array(mask);
    }

    /**
     * Writes one node's layout-unit position (z forced to 0 in 2D, as the simulation's setPosition does); the
     * pending partials are untouched, so the next K1 folds the previous integrate's statistics exactly as the GPU does.
     * @param index - the node index
     * @param x - layout x
     * @param y - layout y
     * @param z - layout z
     */
    setPosition(index: number, x: number, y: number, z: number): void {
        if (!Number.isInteger(index) || index < 0 || index >= this.n) {
            throw new Error(`ForceAtlas2Oracle: index ${index} out of range`);
        }
        const r = this.round;
        this.positions[3 * index] = r(x);
        this.positions[3 * index + 1] = r(y);
        this.positions[3 * index + 2] = this.dim === 2 ? 0 : r(z);
    }

    /** reheat() semantics of D8 (settledCount = 0; mode 1: swing = traction = 1). */
    reheat(): void {
        this.settledCountValue = 0;
        if (this.swingMode === 1) {
            this.swingValue = 1;
            this.tractionValue = 1;
        }
    }

    /**
     * Re-synchronises the reference with a simulation's iteration-start state (P3-T5 PLAN DECISION 17: the
     * re-synchronised legs of fa2-trace-parity.test.ts, fa2-twins.test.ts and fa2-properties.test.ts seed a FRESH
     * oracle from the GPU's positions and this state before every iteration, so one step() is compared with one
     * GPU iteration and the chaotic amplification of the free-running trajectory, spec 7.16, never enters the
     * comparison). The pending partials are cleared: the next step() folds nothing, exactly as FA2_FLAG_FIRST does,
     * and reports the seeded K1 statistics in its record. The positions are the constructor's; the mask and the
     * options are untouched.
     * @param state - the controller, the K1 statistics and F(t-1) (stride 3, or null for zeros)
     */
    resync(state: OracleResyncState): void {
        const r = this.round;
        this.speedValue = r(state.speed);
        this.speedEfficiencyValue = r(state.speedEfficiency);
        this.swingValue = r(state.swing);
        this.tractionValue = r(state.traction);
        this.centroid = [r(state.centroid[0]), r(state.centroid[1]), r(state.centroid[2])];
        this.rmsRadius = r(state.rmsRadius);
        this.radius = r(state.layoutRadius);
        this.meanDisplacement = r(state.meanDisplacement);
        this.settledCountValue = state.settledCount;
        this.iterationValue = state.iteration;
        this.pending = null;
        if (state.oldForce === null) {
            this.oldForce.fill(0);
        } else {
            if (state.oldForce.length !== 3 * this.n) {
                throw new Error(`ForceAtlas2Oracle: oldForce.length ${state.oldForce.length} !== 3 x ${this.n}`);
            }
            for (let k = 0; k < 3 * this.n; k++) {
                this.oldForce[k] = r(state.oldForce[k]);
            }
        }
    }

    /**
     * What the next step()'s K1 would write from the last step()'s partials (the fold of contract 4.5 over the
     * integrate just performed: centroid, rmsRadius and layoutRadius about the centroid of that step, the mean free
     * displacement, the settle counter), without advancing; null before the first step() (K1 folds nothing then).
     * The re-synchronised legs compare it with the K1 statistics the GPU reports for its next iteration.
     * @returns the fold or null
     */
    peekFold(): OracleFold | null {
        return this.pending === null ? null : this.computeFold(this.foldTotals(this.pending));
    }

    /**
     * One iteration: K1 fold (from the previous stages), K2, K3 + epilogue, K4, K5; returns the trace record.
     * @returns the trace record of this iteration
     */
    step(): OracleTraceRecord {
        const { n } = this;
        const r = this.round;
        // ---- K1: fold the previous integrate's partials (skipped on the first iteration after load, FA2_FLAG_FIRST)
        if (this.pending !== null) {
            this.fold(this.pending);
        }
        this.iterationValue += 1;
        const k1 = {
            meanDisplacement: this.meanDisplacement,
            settledCount: this.settledCountValue,
            rmsRadius: this.rmsRadius,
            layoutRadius: this.radius,
            centroid: [this.centroid[0], this.centroid[1], this.centroid[2]] as const,
        };
        if (n === 0) {
            // An empty graph runs no GPU work (spec 11.4 behaviour pins): the state is reported unchanged.
            const empty: OracleTraceRecord = {
                swing: this.swingValue,
                traction: this.tractionValue,
                speed: this.speedValue,
                speedEfficiency: this.speedEfficiencyValue,
                ...k1,
            };
            this.traceRecords.push(empty);
            return empty;
        }
        const stages = emptyStages(n);
        const { positions: pos, mass } = this;
        // ---- K2: attraction over the CSR row, in arc order (spec 7.5; 7.2 rows "Attraction", "Distributed action")
        if (this.arcCount > 0) {
            for (let i = 0; i < n; i++) {
                let fx = 0;
                let fy = 0;
                let fz = 0;
                const px = pos[3 * i];
                const py = pos[3 * i + 1];
                const pz = pos[3 * i + 2];
                for (let a = this.rowPtr[i]; a < this.rowPtr[i + 1]; a++) {
                    const j = this.colIdx[a];
                    if (j === i) {
                        continue; // a self-loop exerts no force
                    }
                    const w = this.weights === null ? 1 : this.weights[a];
                    const dx = r(pos[3 * j] - px);
                    const dy = r(pos[3 * j + 1] - py);
                    const dz = r(pos[3 * j + 2] - pz);
                    let mag = w;
                    if (this.linlog) {
                        const len = Math.max(len3(r, dx, dy, dz), r(FA2_DISTANCE_FLOOR));
                        mag = r(r(w * r(Math.log(r(1 + len)))) / len); // |F| = w log(1 + d) (7.2 row "Attraction (linlog)")
                    }
                    fx = r(fx + r(dx * mag));
                    fy = r(fy + r(dy * mag));
                    fz = r(fz + r(dz * mag));
                }
                if (this.distributed) {
                    fx = r(fx / mass[i]);
                    fy = r(fy / mass[i]);
                    fz = r(fz / mass[i]);
                }
                stages.attraction[3 * i] = fx;
                stages.attraction[3 * i + 1] = fy;
                stages.attraction[3 * i + 2] = fz;
            }
        }
        // ---- K3: all-pairs repulsion in tile order (j ascending), gravity, force +=, swing / traction per node
        const swing = new Float64Array(n);
        const traction = new Float64Array(n);
        const cx = this.centroid[0];
        const cy = this.centroid[1];
        const cz = this.centroid[2];
        for (let i = 0; i < n; i++) {
            const px = pos[3 * i];
            const py = pos[3 * i + 1];
            const pz = pos[3 * i + 2];
            const mi = mass[i];
            let fx = 0;
            let fy = 0;
            let fz = 0;
            for (let j = 0; j < n; j++) {
                if (j === i) {
                    continue;
                }
                const mj = mass[j];
                const dx = r(px - pos[3 * j]);
                const dy = r(py - pos[3 * j + 1]);
                const dz = r(pz - pos[3 * j + 2]);
                let d2 = r(r(r(dx * dx) + r(dy * dy)) + r(dz * dz));
                if (d2 < r(FA2_COINCIDENT_SQ)) {
                    // coincident: antisymmetric unit kick of magnitude k m_i m_j / 0.01 (7.2 row "Coincident nodes")
                    const kick = kickDir(i, j, this.dim, this.precision);
                    const magnitude = r(r(r(this.scalingRatio * mi) * mj) / r(FA2_DISTANCE_FLOOR));
                    fx = r(fx + r(kick[0] * magnitude));
                    fy = r(fy + r(kick[1] * magnitude));
                    fz = r(fz + r(kick[2] * magnitude));
                    continue;
                }
                d2 = Math.max(d2, r(FA2_DISTANCE_FLOOR_SQ)); // d >= 0.01 (7.2 row "Distance floor")
                const k = r(r(this.scalingRatio * mi) * mj);
                const scale = r(k / d2); // |F| = k m_i m_j / d along d / d (7.2 row "Repulsion")
                fx = r(fx + r(dx * scale));
                fy = r(fy + r(dy * scale));
                fz = r(fz + r(dz * scale));
            }
            stages.repulsion[3 * i] = fx;
            stages.repulsion[3 * i + 1] = fy;
            stages.repulsion[3 * i + 2] = fz;
            // gravity (spec 7.9; 7.2 rows "Gravity centre", "Gravity law"; networkx: origin, layout.py 1466-1471)
            let gx = 0;
            let gy = 0;
            let gz = 0;
            const qx = this.gravityCenter === 0 ? r(px - cx) : px;
            const qy = this.gravityCenter === 0 ? r(py - cy) : py;
            const qz = this.gravityCenter === 0 ? r(pz - cz) : pz;
            const gm = r(-this.gravity * mi);
            if (this.strongGravity) {
                gx = r(gm * qx);
                gy = r(gm * qy);
                gz = r(gm * qz);
            } else {
                const d = len3(r, qx, qy, qz);
                if (d > r(FA2_DISTANCE_FLOOR)) {
                    gx = r(r(gm * qx) / d);
                    gy = r(r(gm * qy) / d);
                    gz = r(r(gm * qz) / d);
                }
            }
            stages.gravity[3 * i] = gx;
            stages.gravity[3 * i + 1] = gy;
            stages.gravity[3 * i + 2] = gz;
            // the epilogue: f = repulsion + gravity; fnew = attraction + f (K3 adds to K2's force)
            fx = r(fx + gx);
            fy = r(fy + gy);
            fz = r(fz + gz);
            const Fx = r(stages.attraction[3 * i] + fx);
            const Fy = r(stages.attraction[3 * i + 1] + fy);
            const Fz = r(stages.attraction[3 * i + 2] + fz);
            stages.force[3 * i] = Fx;
            stages.force[3 * i + 1] = Fy;
            stages.force[3 * i + 2] = Fz;
            const ox = this.oldForce[3 * i];
            const oy = this.oldForce[3 * i + 1];
            const oz = this.oldForce[3 * i + 2];
            stages.oldForce[3 * i] = ox;
            stages.oldForce[3 * i + 1] = oy;
            stages.oldForce[3 * i + 2] = oz;
            if (this.swingMode === 1) {
                // NetworkX: positions and forces mixed, every node (layout.py 1479-1480)
                swing[i] = r(mi * len3(r, r(px - Fx), r(py - Fy), r(pz - Fz)));
                traction[i] = r(r(r(0.5) * mi) * len3(r, r(px + Fx), r(py + Fy), r(pz + Fz)));
            } else if (!this.isFixed(i)) {
                // paper: m |F(t) - F(t-1)|, 0.5 m |F(t) + F(t-1)| over FREE nodes (Gephi ForceAtlas2.java 283-293)
                swing[i] = r(mi * len3(r, r(Fx - ox), r(Fy - oy), r(Fz - oz)));
                traction[i] = r(r(r(0.5) * mi) * len3(r, r(Fx + ox), r(Fy + oy), r(Fz + oz)));
            }
        }
        stages.swingPerNode.set(swing);
        stages.tractionPerNode.set(traction);
        // ---- K4: fold partials B, accumulate in networkx mode, estimateFactor
        const freshSwing = foldGroups(groupTotals(swing, n, "sum", r, 0), "sum", r, 0);
        const freshTraction = foldGroups(groupTotals(traction, n, "sum", r, 0), "sum", r, 0);
        let globalSwing = freshSwing;
        let globalTraction = freshTraction;
        if (this.swingMode === 1) {
            globalSwing = r(this.swingValue + freshSwing); // NetworkX accumulates across iterations from 1
            globalTraction = r(this.tractionValue + freshTraction);
        }
        const est = estimateFactor(
            n,
            globalSwing,
            globalTraction,
            this.speedValue,
            this.speedEfficiencyValue,
            this.jitterTolerance,
            this.precision,
        );
        this.speedValue = est.speed;
        this.speedEfficiencyValue = est.speedEfficiency;
        this.swingValue = globalSwing;
        this.tractionValue = globalTraction;
        // ---- K5: integrate (spec 7.11; 7.2 row "Local speed / apply"; networkx layout.py 1497-1501)
        const speed = this.speedValue;
        const sumX = new Float64Array(n);
        const sumY = new Float64Array(n);
        const sumZ = new Float64Array(n);
        const sumW = new Float64Array(n);
        const disp = new Float64Array(n);
        const free = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            const Fx = stages.force[3 * i];
            const Fy = stages.force[3 * i + 1];
            const Fz = stages.force[3 * i + 2];
            const mi = mass[i];
            let swingI = r(mi * len3(r, Fx, Fy, Fz)); // SWING_MODE 1: NetworkX's local swinging m |F| (layout.py 1497)
            if (this.swingMode === 0) {
                swingI = r(
                    mi *
                        len3(
                            r,
                            r(Fx - this.oldForce[3 * i]),
                            r(Fy - this.oldForce[3 * i + 1]),
                            r(Fz - this.oldForce[3 * i + 2]),
                        ),
                );
            }
            const factor = r(speed / r(1 + r(Math.sqrt(r(speed * swingI)))));
            const fixed = this.isFixed(i);
            const dx = fixed ? 0 : r(Fx * factor); // no clamp on dp (D25)
            const dy = fixed ? 0 : r(Fy * factor);
            let dz = fixed ? 0 : r(Fz * factor);
            if (this.dim === 2) {
                dz = 0; // 2D never integrates z (7.13)
            }
            stages.displacement[3 * i] = dx;
            stages.displacement[3 * i + 1] = dy;
            stages.displacement[3 * i + 2] = dz;
            const px = r(pos[3 * i] + dx);
            const py = r(pos[3 * i + 1] + dy);
            const pz = r(pos[3 * i + 2] + dz);
            pos[3 * i] = px;
            pos[3 * i + 1] = py;
            pos[3 * i + 2] = pz;
            if (this.swingMode === 0) {
                this.oldForce[3 * i] = Fx; // fixed nodes too, so a later unpin sees no stale swing (7.11)
                this.oldForce[3 * i + 1] = Fy;
                this.oldForce[3 * i + 2] = Fz;
            }
            const qx = r(px - cx);
            const qy = r(py - cy);
            const qz = r(pz - cz);
            sumX[i] = px;
            sumY[i] = py;
            sumZ[i] = pz;
            sumW[i] = r(r(r(qx * qx) + r(qy * qy)) + r(qz * qz)); // |p - c|^2 about the start-of-iteration centroid
            if (!fixed) {
                disp[i] = len3(r, dx, dy, dz);
                free[i] = 1;
            }
        }
        const pending: GroupPartials = {
            sumX: groupTotals(sumX, n, "sum", r, 0),
            sumY: groupTotals(sumY, n, "sum", r, 0),
            sumZ: groupTotals(sumZ, n, "sum", r, 0),
            sumW: groupTotals(sumW, n, "sum", r, 0),
            minX: groupTotals(sumX, n, "min", r, F32_MAX),
            minY: groupTotals(sumY, n, "min", r, F32_MAX),
            minZ: groupTotals(sumZ, n, "min", r, F32_MAX),
            maxX: groupTotals(sumX, n, "max", r, -F32_MAX),
            maxY: groupTotals(sumY, n, "max", r, -F32_MAX),
            maxZ: groupTotals(sumZ, n, "max", r, -F32_MAX),
            maxW: groupTotals(sumW, n, "max", r, -F32_MAX),
            disp: groupTotals(disp, n, "sum", r, 0),
            free: groupTotals(free, n, "sum", identity, 0),
        };
        this.pending = pending;
        const folded = this.foldTotals(pending);
        this.stagesValue = {
            ...stages,
            partials: {
                swing: freshSwing,
                traction: freshTraction,
                sum: folded.sum,
                sumSq: folded.sumSq,
                min: folded.min,
                max: folded.max,
                disp: folded.disp,
                free: folded.free,
            },
        };
        const record: OracleTraceRecord = {
            swing: globalSwing,
            traction: globalTraction,
            speed: this.speedValue,
            speedEfficiency: this.speedEfficiencyValue,
            ...k1,
        };
        this.traceRecords.push(record);
        return record;
    }

    /**
     * The mass vector: `mass` when given, else a Float32Array `nodeMass`, else outDegree()[i] + 1 (spec 7.14).
     * @param s - the snapshot
     * @param options - the oracle options
     * @returns n masses
     */
    private static resolveMass(s: GraphSnapshot, options: OracleOptions): Float64Array {
        const n = s.nodeCount;
        const mass = new Float64Array(n);
        const source = options.mass ?? (options.nodeMass instanceof Float32Array ? options.nodeMass : null);
        if (source !== null) {
            if (source.length !== n) {
                throw new Error(`ForceAtlas2Oracle: mass.length ${source.length} !== ${n}`);
            }
            for (let i = 0; i < n; i++) {
                mass[i] = source[i];
            }
        } else if (options.nodeMass === undefined || options.nodeMass === null) {
            const degree = s.outDegree();
            for (let i = 0; i < n; i++) {
                mass[i] = degree[i] + 1;
            }
        } else {
            throw new Error(
                "ForceAtlas2Oracle: resolve a column or record nodeMass with resolveNodeMass and pass it as `mass`",
            );
        }
        for (let i = 0; i < n; i++) {
            if (!(mass[i] > 0)) {
                throw new Error(`ForceAtlas2Oracle: mass[${i}] must be positive`);
            }
        }
        return mass;
    }

    /**
     * The per-arc weights: `weights` when given, else the snapshot's arc weights when `weight` is true, else none.
     * @param s - the snapshot
     * @param options - the oracle options
     * @returns arcCount weights or null (every weight 1)
     */
    private static resolveWeights(s: GraphSnapshot, options: OracleOptions): Float64Array | null {
        let source: ArrayLike<number> | null = null;
        if (options.weights !== undefined && options.weights !== null) {
            source = options.weights;
        } else if (options.weight === true) {
            source = s.weights;
        } else if (typeof options.weight === "string") {
            throw new Error("ForceAtlas2Oracle: resolve a weight column with resolveWeights and pass it as `weights`");
        }
        if (source === null) {
            return null;
        }
        if (source.length !== s.arcCount) {
            throw new Error(`ForceAtlas2Oracle: weights.length ${source.length} !== arcCount ${s.arcCount}`);
        }
        const weights = new Float64Array(s.arcCount);
        for (let a = 0; a < s.arcCount; a++) {
            weights[a] = source[a];
        }
        return weights;
    }

    /**
     * mask_bit of the prelude over the fixed mask.
     * @param i - the node index
     * @returns true when the node is fixed
     */
    private isFixed(i: number): boolean {
        return this.fixed !== null && maskTest(this.fixed, i);
    }

    /**
     * load()'s CPU statistics (contract 3.13 ForceSimulation.load): centroid, bounding box, RMS radius and radius
     * (max |p - centroid|) in f64, iteration 0, settledCount 0, meanDisplacement 0.
     */
    private writeInitialStatistics(): void {
        const { n, positions: pos } = this;
        const r = this.round;
        const c: [number, number, number] = [0, 0, 0];
        const lo: [number, number, number] = [
            Number.POSITIVE_INFINITY,
            Number.POSITIVE_INFINITY,
            Number.POSITIVE_INFINITY,
        ];
        const hi: [number, number, number] = [
            Number.NEGATIVE_INFINITY,
            Number.NEGATIVE_INFINITY,
            Number.NEGATIVE_INFINITY,
        ];
        for (let i = 0; i < n; i++) {
            for (let a = 0; a < 3; a++) {
                const v = pos[3 * i + a];
                c[a] += v;
                lo[a] = Math.min(lo[a], v);
                hi[a] = Math.max(hi[a], v);
            }
        }
        if (n === 0) {
            this.centroid = [0, 0, 0];
            this.minValue = [0, 0, 0];
            this.maxValue = [0, 0, 0];
            this.rmsRadius = 0;
            this.radius = 0;
        } else {
            c[0] /= n;
            c[1] /= n;
            c[2] /= n;
            let sumSq = 0;
            let maxSq = 0;
            for (let i = 0; i < n; i++) {
                const qx = pos[3 * i] - c[0];
                const qy = pos[3 * i + 1] - c[1];
                const qz = pos[3 * i + 2] - c[2];
                const q2 = qx * qx + qy * qy + qz * qz;
                sumSq += q2;
                maxSq = Math.max(maxSq, q2);
            }
            this.centroid = [r(c[0]), r(c[1]), r(c[2])];
            this.minValue = [r(lo[0]), r(lo[1]), r(lo[2])];
            this.maxValue = [r(hi[0]), r(hi[1]), r(hi[2])];
            this.rmsRadius = r(Math.sqrt(sumSq / n));
            this.radius = r(Math.sqrt(maxSq));
        }
        this.meanDisplacement = 0;
        this.settledCountValue = 0;
        this.iterationValue = 0;
    }

    /**
     * The K1 totals of a set of partials (the same fold K1 performs; also what stages.partials reports).
     * @param p - the per-workgroup partials
     * @returns the folded totals
     */
    private foldTotals(p: GroupPartials): FoldTotals {
        const r = this.round;
        return {
            sum: [foldGroups(p.sumX, "sum", r, 0), foldGroups(p.sumY, "sum", r, 0), foldGroups(p.sumZ, "sum", r, 0)],
            sumSq: foldGroups(p.sumW, "sum", r, 0),
            min: [
                foldGroups(p.minX, "min", r, F32_MAX),
                foldGroups(p.minY, "min", r, F32_MAX),
                foldGroups(p.minZ, "min", r, F32_MAX),
            ],
            max: [
                foldGroups(p.maxX, "max", r, -F32_MAX),
                foldGroups(p.maxY, "max", r, -F32_MAX),
                foldGroups(p.maxZ, "max", r, -F32_MAX),
            ],
            maxSq: foldGroups(p.maxW, "max", r, -F32_MAX),
            disp: foldGroups(p.disp, "sum", r, 0),
            free: foldGroups(p.free, "sum", identity, 0),
        };
    }

    /**
     * K1's fold (contract 4.5), computed without being applied: centroid = sum / n, rmsRadius = sqrt(sumSq / n)
     * about the PREVIOUS centroid, radius = sqrt(max |p - c|^2), meanDisplacement = disp / free (0 when every node
     * is fixed), the settle counter continued from the current one.
     * @param t - the totals of the previous integrate's partials
     * @returns what K1 writes
     */
    private computeFold(t: FoldTotals): OracleFold {
        const r = this.round;
        const nf = r(this.n);
        const rmsRadius = r(Math.sqrt(r(Math.max(t.sumSq, 0) / nf)));
        const meanDisplacement = t.free === 0 ? 0 : r(t.disp / r(t.free)); // all-fixed: 0, never NaN (7.4)
        return {
            centroid: [r(t.sum[0] / nf), r(t.sum[1] / nf), r(t.sum[2] / nf)],
            rmsRadius,
            layoutRadius: r(Math.sqrt(Math.max(t.maxSq, 0))),
            meanDisplacement,
            settledCount: meanDisplacement <= r(this.settleThreshold * rmsRadius) ? this.settledCountValue + 1 : 0,
        };
    }

    /**
     * K1's fold applied to the state (contract 4.5): the statistics of computeFold and the min / max box.
     * @param p - the previous integrate's partials
     */
    private fold(p: GroupPartials): void {
        const t = this.foldTotals(p);
        const f = this.computeFold(t);
        this.centroid = [f.centroid[0], f.centroid[1], f.centroid[2]];
        this.rmsRadius = f.rmsRadius;
        this.minValue = [t.min[0], t.min[1], t.min[2]];
        this.maxValue = [t.max[0], t.max[1], t.max[2]];
        this.radius = f.layoutRadius;
        this.meanDisplacement = f.meanDisplacement;
        this.settledCountValue = f.settledCount;
    }
}

/**
 * Positions in SCENE units for a snapshot: seedPositions into a fresh F32 (the same LCG start as the GPU, spec 11.4
 * "same start").
 * @param s - the snapshot
 * @param seed - the LCG seed (null / 0: unseeded)
 * @param dim - 2 or 3
 * @param scale - the scene scale
 * @param center - the scene centre or null for the origin
 * @returns 3 x nodeCount scene-unit coordinates
 */
export function seededScenePositions(
    s: GraphSnapshot,
    seed: number | null,
    dim: 2 | 3,
    scale: number,
    center: ArrayLike<number> | null,
): F32 {
    const positions = new Float32Array(3 * s.nodeCount);
    positions.fill(Number.NaN);
    seedPositions(s, positions, seed, dim, scale, center, "fa2");
    return positions;
}

/**
 * The resolved scene centre of an option record.
 * @param center - the option value
 * @returns [x, y, z]
 */
function centerOf(center: ArrayLike<number> | undefined): readonly [number, number, number] {
    if (center === undefined) {
        return [0, 0, 0];
    }
    return [center.length > 0 ? center[0] : 0, center.length > 1 ? center[1] : 0, center.length > 2 ? center[2] : 0];
}

/**
 * Runs the oracle for `iterations` from a scene-unit array and returns the scene-unit result (the toScene inverse
 * applied). The layout-unit start is rounded to f32 in BOTH precisions because the GPU holds positions as vec4f
 * (D23), which keeps "same start" exact for every scale and centre.
 * @param s - the snapshot
 * @param scenePositions - 3 x nodeCount scene-unit coordinates (finite)
 * @param options - the oracle options (scale / center / dim from the FA2 options)
 * @param iterations - how many step() calls
 * @returns the scene-unit positions, the trace and the oracle itself
 */
export function forceAtlas2Oracle(
    s: GraphSnapshot,
    scenePositions: F32,
    options: OracleOptions,
    iterations: number,
): { readonly positions: F32; readonly trace: readonly OracleTraceRecord[]; readonly oracle: ForceAtlas2Oracle } {
    const n = s.nodeCount;
    const dim = options.dim ?? FA2_DEFAULTS.dim;
    const scale = options.scale ?? FA2_DEFAULTS.scale;
    if (!(scale > 0)) {
        throw new Error("forceAtlas2Oracle: scale must be positive");
    }
    const center = centerOf(options.center);
    const layout = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
        for (let a = 0; a < 3; a++) {
            layout[3 * i + a] = a < dim ? Math.fround((scenePositions[3 * i + a] - center[a]) / scale) : 0;
        }
    }
    const oracle = new ForceAtlas2Oracle(s, layout, options);
    for (let k = 0; k < iterations; k++) {
        oracle.step();
    }
    const out = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
        out[3 * i] = oracle.positions[3 * i] * scale + center[0];
        out[3 * i + 1] = oracle.positions[3 * i + 1] * scale + center[1];
        out[3 * i + 2] = dim === 2 ? center[2] : oracle.positions[3 * i + 2] * scale + center[2]; // 2D writes z = center.z (7.13)
    }
    return { positions: out, trace: oracle.trace, oracle };
}
