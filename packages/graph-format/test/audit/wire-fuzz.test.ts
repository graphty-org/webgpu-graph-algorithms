/**
 * Adversarial corruption fuzzer of the GSNP container and the in-memory wire form (design sections
 * 9.1, 9.2, 9.5 and 11.2). Golden snapshots covering every id-map kind and every dtype are
 * serialised with toBytes() and then mutated: single-bit flips of the header, random byte flips and
 * overwrites in the manifest and in every region segment, truncation at every section boundary,
 * lies about lengths, offsets, alignment, counts, flags, versions, the arena and the id map, unknown
 * dtypes / id-map kinds / view names, column metadata lies, a swapped endianness probe, every chunk
 * split, and non-JSON values (NaN, bigint, symbols, class instances, detached and shared buffers)
 * in an in-memory manifest. Every outcome of fromBytes() / fromByteChunks() / fromWire() must be:
 *
 * - a snapshot that passes validate({ level: "full" }), whose views, ids and columns are readable
 *   and in range, and that re-serialises to a container decoding to an equal snapshot; or
 * - a GraphFormatError carrying one of the reader's documented codes (E_BAD_SERIALIZATION,
 *   E_UNSUPPORTED_VERSION, E_UNSUPPORTED, E_INVALID_SNAPSHOT).
 *
 * Anything else (a TypeError / RangeError, an undocumented code, a snapshot failing validation) is
 * an anomaly and fails the test that produced it. Anomalies already pinned by a dedicated failing
 * test in wire-probes.test.ts are recognised by signature (KNOWN_DEFECTS) so that one open defect
 * does not hide a new one.
 */

import { describe, expect, it } from "vitest";

import { GraphBuilder } from "../../src/builder/graph-builder.js";
import { ALIGNMENT, INVALID_INDEX, MAX_COUNT } from "../../src/constants.js";
import { GraphFormatError } from "../../src/errors.js";
import { type GraphSnapshot } from "../../src/snapshot/graph-snapshot.js";
import { type AttributeTable, type ValidationLevel, type WireSnapshot } from "../../src/types/index.js";
import { CONTAINER_HEADER_BYTES, fromByteChunks, fromBytes } from "../../src/wire/bytes.js";
import { fromWire } from "../../src/wire/from-wire.js";
import { at, expectSnapshotsEqual, idsSnapshot, richSnapshot, splitContainer } from "../wire/helpers.js";

// ============================================================ deterministic randomness

/** mulberry32: a small seeded PRNG so every run mutates the same bytes. */
function prng(seed: number): () => number {
    let a = seed >>> 0;
    return (): number => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const LEVELS: readonly ValidationLevel[] = ["full", "structure", "none"];

// ============================================================ known defects (pinned in wire-probes.test.ts)

/** What the oracle found wrong with one mutated input. */
interface Anomaly {
    readonly golden: string;
    readonly mutation: string;
    readonly problem: string;
}

interface KnownDefect {
    /** The wire-probes.test.ts test that pins the defect. */
    readonly pinnedBy: string;
    readonly matches: (anomaly: Anomaly) => boolean;
}

/**
 * Anomaly signatures already pinned by a dedicated (deliberately failing) test, so that the fuzz
 * suites keep reporting NEW anomalies while these are open. Remove an entry once its pin passes.
 */
const KNOWN_DEFECTS: readonly KnownDefect[] = [
    {
        pinnedBy: "validate() reports a violated unique column as E_INVALID_SNAPSHOT",
        matches: (a) => /undocumented code E_DUPLICATE_(EDGE_)?ID:/.test(a.problem),
    },
    {
        pinnedBy: "a bigint or null-prototype manifest value is refused with E_BAD_SERIALIZATION",
        matches: (a) =>
            a.problem.includes("Do not know how to serialize a BigInt") ||
            a.problem.includes("Cannot convert object to primitive value"),
    },
    {
        pinnedBy: "negative zero is not accepted as a count or identity offset",
        matches: (a) => a.mutation.includes(":= -0") && a.problem.includes("does not re-serialise"),
    },
];

function describeError(err: unknown): string {
    if (err instanceof Error) {
        return `${err.constructor.name}: ${err.message}`;
    }
    return String(err);
}

/**
 * Exercise every view, the id map and every column of a decoded snapshot and check the range rules
 * the design promises (I2 for view arrays, I11 for ids, I12 for columns). Throws an Error naming
 * the first problem.
 */
function exerciseSnapshot(s: GraphSnapshot): void {
    const { nodeCount, arcCount, edgeCount } = s;
    const below = (name: string, values: ArrayLike<number>, bound: number): void => {
        for (let i = 0; i < values.length; i++) {
            if (!(values[i] < bound) || values[i] === INVALID_INDEX) {
                throw new Error(`${name}[${i}] = ${values[i]} is not below ${bound}`);
            }
        }
    };
    const rev = s.reverse();
    below("reverse.colIdx", rev.colIdx, nodeCount);
    below("reverse.fwdArc", rev.fwdArc, arcCount);
    below("reverse.arcToEdge", rev.arcToEdge, edgeCount);
    if (rev.rowPtr.length !== nodeCount + 1 || rev.rowPtr[nodeCount] !== arcCount) {
        throw new Error("reverse.rowPtr shape");
    }
    below("coo.src", s.coo().src, nodeCount);
    below("edgeList.src", s.edgeList().src, nodeCount);
    below("edgeList.dst", s.edgeList().dst, nodeCount);
    below("edgeList.arc", s.edgeList().arc, arcCount);
    let total = 0;
    const out = s.outDegree();
    for (let u = 0; u < nodeCount; u++) {
        total += out[u];
    }
    if (total !== arcCount) {
        throw new Error(`outDegree sums to ${total}, arcCount ${arcCount}`);
    }
    s.inDegree();
    s.degree();
    s.weightedOutDegree();
    s.weightedInDegree();
    s.weightedDegree();
    s.selfLoopWeight();
    s.totalWeight();
    below("selfLoopArcs", s.selfLoopArcs(), arcCount);
    if (s.selfLoopArcs().length !== s.selfLoopCount) {
        throw new Error("selfLoopArcs length");
    }
    s.selfLoopsPerNode();
    if (!s.directed) {
        below("mate", s.mate(), arcCount);
    }
    for (const of of ["forward", "reverse"] as const) {
        const order = s.degreeOrder({ of });
        below(`degreeOrder(${of}).perm`, order.perm, nodeCount);
        const seen = new Uint8Array(nodeCount);
        for (let i = 0; i < order.perm.length; i++) {
            if (seen[order.perm[i]] === 1) {
                throw new Error(`degreeOrder(${of}).perm repeats ${order.perm[i]}`);
            }
            seen[order.perm[i]] = 1;
        }
    }
    s.isSymmetric();
    s.contentHash();
    const ids = s.ids.toArray();
    if (ids.length !== nodeCount) {
        throw new Error("ids.toArray length");
    }
    for (let i = 0; i < nodeCount; i++) {
        if (s.ids.indexOf(ids[i]) !== i) {
            throw new Error(`ids.indexOf(ids[${i}]) !== ${i}`);
        }
    }
    s.ids.toMap(new Uint32Array(nodeCount));
    const tables: readonly [string, AttributeTable][] = [
        ["nodes", s.nodes],
        ["edges", s.edges],
        ["graph", s.graph],
        ...[...s.extensions].map(([n, t]): [string, AttributeTable] => [`extensions[${n}]`, t]),
    ];
    for (const [tableName, table] of tables) {
        for (const column of table) {
            for (let row = 0; row < column.length; row++) {
                column.value(row);
                column.isSet(row);
            }
            if (column.dtype === "string") {
                column.decodeAll();
            }
            if (column.dtype === "dict") {
                for (let row = 0; row < column.length; row++) {
                    if (column.isSet(row) && column.codes[row] >= column.dictionary.length) {
                        throw new Error(`${tableName}.${column.meta.name} code out of range at ${row}`);
                    }
                }
            }
            if (column.meta.refersTo !== null) {
                const bound = column.meta.refersTo === "node" ? nodeCount : edgeCount;
                const holder = column.dtype === "list" ? column.child : column;
                if (holder.dtype !== "u32") {
                    throw new Error(`${tableName}.${column.meta.name} refersTo on a ${holder.dtype} column`);
                }
                const values = holder.data;
                for (let i = 0; i < values.length; i++) {
                    const v = values[i];
                    if (v !== INVALID_INDEX && !(v < bound)) {
                        throw new Error(`${tableName}.${column.meta.name} reference ${v} at ${i} not below ${bound}`);
                    }
                }
            }
        }
    }
    if (s.edges.byRole("id") !== null) {
        s.edgeIndexOf("nonexistent");
    }
}

// ============================================================ golden snapshots

interface Golden {
    readonly name: string;
    readonly make: () => GraphSnapshot;
    /** Whether the per-column mutation suites run (the three rich goldens share one column set). */
    readonly columns: boolean;
}

function emptySnapshot(): GraphSnapshot {
    return new GraphBuilder({ directed: true }).freeze();
}

function identityDirected(): GraphSnapshot {
    const b = new GraphBuilder({ directed: true });
    b.addAnonymousNodes(5);
    b.addEdge(0, 1, 2);
    b.addEdge(1, 2, 0.5);
    b.addEdge(2, 3);
    b.addEdge(3, 3);
    b.addEdge(4, 0, -1);
    return b.freeze();
}

function unweightedUndirected(): GraphSnapshot {
    const b = new GraphBuilder({ directed: false, weighted: false });
    b.addNodes(["p", "q", "r"]);
    b.addEdge("p", "q");
    b.addEdge("q", "r");
    b.addEdge("r", "p");
    return b.freeze();
}

const GOLDENS: readonly Golden[] = [
    { name: "rich undirected string ids", make: () => richSnapshot(), columns: true },
    { name: "rich directed string ids", make: () => richSnapshot({ directed: true }), columns: false },
    { name: "rich mixed ids", make: () => richSnapshot({ ids: [1, "a", 2.5, "b"] }), columns: false },
    { name: "identity offset 0", make: () => idsSnapshot([0, 1, 2, 3]), columns: true },
    { name: "identity offset 1 undirected", make: () => idsSnapshot([1, 2, 3, 4], false), columns: true },
    { name: "dense", make: () => idsSnapshot([0, 2, 4, 6]), columns: true },
    { name: "numeric", make: () => idsSnapshot([0.5, 100, -3, 1e10]), columns: true },
    { name: "string", make: () => idsSnapshot(["a", "b", "\u00e9\u4e2d", ""]), columns: true },
    { name: "mixed", make: () => idsSnapshot([1, "a", 2, "b"], false), columns: true },
    { name: "empty", make: emptySnapshot, columns: true },
    { name: "identity permutation directed", make: identityDirected, columns: true },
    { name: "unweighted undirected", make: unweightedUndirected, columns: true },
];

// ============================================================ the oracle

/** Codes a reader may throw for untrusted input (design sections 9.1, 9.5 and 11.3). */
const READER_CODES: ReadonlySet<string> = new Set([
    "E_BAD_SERIALIZATION",
    "E_UNSUPPORTED_VERSION",
    "E_UNSUPPORTED",
    "E_INVALID_SNAPSHOT",
]);

type Outcome =
    | { readonly kind: "ok"; readonly snapshot: GraphSnapshot }
    | { readonly kind: "error"; readonly code: string; readonly error: GraphFormatError }
    | { readonly kind: "foreign"; readonly error: unknown };

function run(fn: () => GraphSnapshot): Outcome {
    try {
        return { kind: "ok", snapshot: fn() };
    } catch (err) {
        if (err instanceof GraphFormatError) {
            return { kind: "error", code: err.code, error: err };
        }
        return { kind: "foreign", error: err };
    }
}

/**
 * Judge one outcome: a foreign error or an undocumented code is always an anomaly; a returned
 * snapshot must pass validation at the level requested, and at "full" it must also be exercisable
 * and re-serialise to an equal snapshot with the same content hash.
 */
function judge(outcome: Outcome, level: ValidationLevel, golden: string, mutation: string, into: Anomaly[]): void {
    if (outcome.kind === "foreign") {
        into.push({ golden, mutation, problem: `non-GraphFormatError: ${describeError(outcome.error)}` });
        return;
    }
    if (outcome.kind === "error") {
        if (!READER_CODES.has(outcome.code)) {
            into.push({ golden, mutation, problem: `undocumented code ${outcome.code}: ${outcome.error.message}` });
        }
        return;
    }
    if (level === "none") {
        return;
    }
    const s = outcome.snapshot;
    try {
        s.validate({ level });
    } catch (err) {
        into.push({ golden, mutation, problem: `returned snapshot fails validate(${level}): ${describeError(err)}` });
        return;
    }
    if (level !== "full") {
        return;
    }
    try {
        exerciseSnapshot(s);
    } catch (err) {
        into.push({ golden, mutation, problem: `returned snapshot is not exercisable: ${describeError(err)}` });
        return;
    }
    try {
        const again = fromBytes(s.toBytes());
        expectSnapshotsEqual(s, again);
        if (again.contentHash() !== s.contentHash()) {
            throw new Error("contentHash differs after re-serialisation");
        }
    } catch (err) {
        into.push({ golden, mutation, problem: `returned snapshot does not re-serialise: ${describeError(err)}` });
    }
}

/** Anomalies not covered by a known, separately pinned defect, grouped by message shape. */
function unexplained(anomalies: readonly Anomaly[]): string {
    const groups = new Map<string, number>();
    for (const a of anomalies) {
        if (KNOWN_DEFECTS.some((known) => known.matches(a))) {
            continue;
        }
        const key = `[${a.golden}] ${a.mutation.replace(/[0-9]+/g, "#")}: ${a.problem.replace(/[0-9]+/g, "#")}`;
        groups.set(key, (groups.get(key) ?? 0) + 1);
    }
    return [...groups]
        .slice(0, 30)
        .map(([key, n]) => `${n} x ${key}`)
        .join("\n");
}

function judgeAll(
    make: (level: ValidationLevel) => GraphSnapshot,
    golden: string,
    mutation: string,
    into: Anomaly[],
    levels: readonly ValidationLevel[] = LEVELS,
): void {
    for (const level of levels) {
        judge(
            run(() => make(level)),
            level,
            golden,
            `${mutation} at ${level}`,
            into,
        );
    }
}

// ============================================================ mutation helpers

function flipBit(bytes: Uint8Array, offset: number, bit: number): Uint8Array {
    const out = bytes.slice();
    out[offset] ^= 1 << bit;
    return out;
}

function setByte(bytes: Uint8Array, offset: number, value: number): Uint8Array {
    const out = bytes.slice();
    out[offset] = value;
    return out;
}

interface FoundRef {
    readonly path: string;
    readonly ref: Record<string, unknown>;
}

/** Every WireBufferRef in a parsed manifest, with its dotted path. */
function collectRefs(node: unknown, path: string, into: FoundRef[]): void {
    if (Array.isArray(node)) {
        node.forEach((item, i) => {
            collectRefs(item, `${path}.${i}`, into);
        });
        return;
    }
    if (typeof node !== "object" || node === null) {
        return;
    }
    const o = node as Record<string, unknown>;
    if (typeof o.dtype === "string" && typeof o.byteOffset === "number" && typeof o.byteLength === "number") {
        into.push({ path, ref: o });
        return;
    }
    for (const key of Object.keys(o)) {
        collectRefs(o[key], path === "" ? key : `${path}.${key}`, into);
    }
}

function alignUp(value: number, multiple: number): number {
    const r = value % multiple;
    return r === 0 ? value : value + multiple - r;
}

/** Rebuild a container from a mutated manifest object and the original region bytes. */
function withManifest(bytes: Uint8Array, edit: (manifest: Record<string, unknown>) => void): Uint8Array {
    const parts = splitContainer(bytes);
    edit(parts.manifest);
    const manifestBytes = new TextEncoder().encode(JSON.stringify(parts.manifest));
    const start = alignUp(CONTAINER_HEADER_BYTES + manifestBytes.byteLength, ALIGNMENT);
    const out = new Uint8Array(start + parts.region.byteLength);
    out.set(bytes.subarray(0, CONTAINER_HEADER_BYTES));
    new DataView(out.buffer).setUint32(12, manifestBytes.byteLength, true);
    out.set(manifestBytes, CONTAINER_HEADER_BYTES);
    out.set(parts.region, start);
    return out;
}

interface Segment {
    readonly path: string;
    readonly start: number;
    readonly end: number;
}

/** The distinct segment byte ranges of a container region, from its manifest. */
function segmentsOf(bytes: Uint8Array): Segment[] {
    const refs: FoundRef[] = [];
    collectRefs(splitContainer(bytes).manifest, "", refs);
    const seen = new Set<string>();
    const out: Segment[] = [];
    for (const { path, ref } of refs) {
        const start = ref.byteOffset as number;
        const end = start + (ref.byteLength as number);
        const key = `${start}:${end}`;
        if (seen.has(key)) {
            continue;
        }
        seen.add(key);
        out.push({ path, start, end });
    }
    return out;
}

/** The object at a dotted path, or undefined when any step is missing. */
function walkTo(manifest: Record<string, unknown>, path: string): Record<string, unknown> | undefined {
    let node: unknown = manifest;
    for (const key of path === "" ? [] : path.split(".")) {
        if (typeof node !== "object" || node === null) {
            return undefined;
        }
        node = (node as Record<string, unknown>)[key];
    }
    return typeof node === "object" && node !== null ? (node as Record<string, unknown>) : undefined;
}

/** Set a dotted path inside a manifest object; a missing parent leaves the manifest untouched. */
function setPath(manifest: Record<string, unknown>, path: string, value: unknown): void {
    const parentPath = path.split(".").slice(0, -1).join(".");
    const key = path.split(".").pop() as string;
    const parent = walkTo(manifest, parentPath);
    if (parent === undefined) {
        return;
    }
    Object.defineProperty(parent, key, { value, enumerable: true, writable: true, configurable: true });
}

/** Dotted paths of every column (list children included) in a parsed manifest. */
function columnPathsOf(manifest: Record<string, unknown>): string[] {
    const out: string[] = [];
    const walk = (list: unknown, base: string): void => {
        (list as Record<string, unknown>[]).forEach((column, i) => {
            out.push(`${base}.${i}`);
            if (column.child !== null && column.child !== undefined) {
                out.push(`${base}.${i}.child`);
            }
        });
    };
    walk(manifest.nodeColumns, "nodeColumns");
    walk(manifest.edgeColumns, "edgeColumns");
    walk(manifest.graphColumns, "graphColumns");
    (manifest.extensions as Record<string, unknown>[]).forEach((ext, i) => {
        walk(ext.columns, `extensions.${i}.columns`);
    });
    return out;
}

// ============================================================ the container tests

describe("wire fuzz: byte-level corruption of the GSNP container", () => {
    for (const golden of GOLDENS) {
        describe(golden.name, () => {
            const original = golden.make();
            const bytes = original.toBytes();
            const regionStart = bytes.byteLength - splitContainer(bytes).region.byteLength;
            const manifestLength = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(12, true);
            const segments = segmentsOf(bytes);
            const regionEnd = regionStart + Math.max(0, ...segments.map((s) => s.end));

            it("round-trips unmutated at every level, through chunks and with the content hash intact", () => {
                for (const level of LEVELS) {
                    const back = fromBytes(bytes, { validate: level });
                    expectSnapshotsEqual(original, back);
                    back.validate({ level: "full" });
                    expect(back.contentHash()).toBe(original.contentHash());
                    const chunked = fromByteChunks([bytes], { validate: level });
                    expectSnapshotsEqual(original, chunked);
                    expect(chunked.contentHash()).toBe(original.contentHash());
                }
            });

            it("every single-bit flip of the 16-byte header is refused with a documented code or ignored", () => {
                const anomalies: Anomaly[] = [];
                for (let offset = 0; offset < CONTAINER_HEADER_BYTES; offset++) {
                    for (let bit = 0; bit < 8; bit++) {
                        const mutated = flipBit(bytes, offset, bit);
                        const label = `header byte ${offset} bit ${bit}`;
                        judgeAll((level) => fromBytes(mutated, { validate: level }), golden.name, label, anomalies);
                        judgeAll(
                            (level) => fromByteChunks([mutated], { validate: level }),
                            golden.name,
                            `${label} (chunks)`,
                            anomalies,
                        );
                    }
                }
                expect(unexplained(anomalies)).toBe("");
            });

            it("random byte flips and overwrites in the manifest never escape as foreign errors", () => {
                const anomalies: Anomaly[] = [];
                const rand = prng(0x1234 + manifestLength);
                const count = Math.min(manifestLength * 3, 600);
                for (let i = 0; i < count; i++) {
                    const offset = CONTAINER_HEADER_BYTES + Math.floor(rand() * manifestLength);
                    const overwrite = i % 3 === 0;
                    const mutated = overwrite
                        ? setByte(bytes, offset, Math.floor(rand() * 256))
                        : flipBit(bytes, offset, Math.floor(rand() * 8));
                    judgeAll(
                        (level) => fromBytes(mutated, { validate: level }),
                        golden.name,
                        `manifest byte ${offset} ${overwrite ? "set" : "flip"}`,
                        anomalies,
                    );
                }
                expect(unexplained(anomalies)).toBe("");
            });

            it("random byte flips in every region segment yield a fully valid snapshot or a documented error", () => {
                const anomalies: Anomaly[] = [];
                const rand = prng(0x5678 + bytes.byteLength);
                for (const segment of segments) {
                    const length = segment.end - segment.start;
                    if (length === 0) {
                        continue;
                    }
                    const trials = Math.min(length * 2, 32);
                    for (let i = 0; i < trials; i++) {
                        const inner = Math.floor(rand() * length);
                        const offset = regionStart + segment.start + inner;
                        const mutated =
                            i % 4 === 0
                                ? setByte(bytes, offset, i % 8 === 0 ? 0xff : 0)
                                : flipBit(bytes, offset, Math.floor(rand() * 8));
                        judgeAll(
                            (level) => fromBytes(mutated, { validate: level }),
                            golden.name,
                            `region ${segment.path} byte ${inner}`,
                            anomalies,
                        );
                    }
                }
                expect(unexplained(anomalies)).toBe("");
            });

            it("truncation at every section boundary is refused with a documented code", () => {
                const anomalies: Anomaly[] = [];
                const cuts = new Set<number>([
                    0,
                    1,
                    4,
                    8,
                    12,
                    15,
                    CONTAINER_HEADER_BYTES,
                    CONTAINER_HEADER_BYTES + manifestLength - 1,
                    CONTAINER_HEADER_BYTES + manifestLength,
                    regionStart - 1,
                    regionStart,
                    bytes.byteLength - 1,
                    bytes.byteLength - ALIGNMENT,
                ]);
                for (const segment of segments) {
                    cuts.add(regionStart + segment.start);
                    cuts.add(regionStart + segment.start + 1);
                    cuts.add(regionStart + segment.end - 1);
                    cuts.add(regionStart + segment.end);
                }
                for (const cut of cuts) {
                    if (cut < 0 || cut >= bytes.byteLength) {
                        continue;
                    }
                    const truncated = bytes.slice(0, cut);
                    for (const level of LEVELS) {
                        const outcome = run(() => fromBytes(truncated, { validate: level }));
                        judge(outcome, level, golden.name, `truncate at ${cut} at ${level}`, anomalies);
                        if (outcome.kind === "ok" && cut < regionEnd) {
                            anomalies.push({
                                golden: golden.name,
                                mutation: `truncate at ${cut} at ${level}`,
                                problem: "accepted a container cut inside its buffer region",
                            });
                        }
                    }
                    judgeAll(
                        (level) => fromByteChunks([truncated], { validate: level }),
                        golden.name,
                        `truncate at ${cut} (chunks)`,
                        anomalies,
                    );
                }
                expect(unexplained(anomalies)).toBe("");
            });

            it("lies in every buffer reference are refused or yield a valid snapshot", () => {
                const anomalies: Anomaly[] = [];
                const refs: FoundRef[] = [];
                collectRefs(splitContainer(bytes).manifest, "", refs);
                const targets = golden.columns
                    ? refs
                    : refs.filter((r) => r.path.startsWith("core") || r.path.startsWith("ids"));
                const numericLies: readonly [string, (v: number) => unknown][] = [
                    ["zero", () => 0],
                    ["one", () => 1],
                    ["minus one", () => -1],
                    ["plus one", (v) => v + 1],
                    ["plus four", (v) => v + 4],
                    ["minus four", (v) => v - 4],
                    ["fraction", (v) => v + 0.5],
                    ["2^32", () => 4294967296],
                    ["2^53", () => 9007199254740992],
                    ["MAX_COUNT", () => MAX_COUNT],
                    ["string", (v) => String(v)],
                    ["null", () => null],
                    ["array", () => []],
                ];
                for (const { path } of targets) {
                    for (const field of ["buffer", "byteOffset", "byteLength", "length"]) {
                        for (const [lieName, lie] of numericLies) {
                            const mutated = withManifest(bytes, (m) => {
                                const target = at(m, path);
                                target[field] = lie(target[field] as number);
                            });
                            judgeAll(
                                (level) => fromBytes(mutated, { validate: level }),
                                golden.name,
                                `${path}.${field} := ${lieName}`,
                                anomalies,
                            );
                        }
                    }
                    for (const dtype of ["u32", "i32", "f32", "f64", "u8", "utf8", "f16", "bool", 7, null]) {
                        const mutated = withManifest(bytes, (m) => {
                            at(m, path).dtype = dtype;
                        });
                        judgeAll(
                            (level) => fromBytes(mutated, { validate: level }),
                            golden.name,
                            `${path}.dtype := ${String(dtype)}`,
                            anomalies,
                        );
                    }
                    const dropped = withManifest(bytes, (m) => {
                        setPath(m, path, null);
                    });
                    judgeAll(
                        (level) => fromBytes(dropped, { validate: level }),
                        golden.name,
                        `${path} := null`,
                        anomalies,
                    );
                }
                expect(unexplained(anomalies)).toBe("");
            });

            it("lies about counts, flags, directedness, versions, arena, ids and top-level members are refused or harmless", () => {
                const anomalies: Anomaly[] = [];
                const scalarLies: readonly [string, unknown][] = [
                    ["zero", 0],
                    ["one", 1],
                    ["minus one", -1],
                    ["fraction", 1.5],
                    ["2^32", 4294967296],
                    ["MAX_COUNT", MAX_COUNT],
                    ["MAX_COUNT + 1", MAX_COUNT + 1],
                    ["string", "3"],
                    ["null", null],
                    ["true", true],
                    ["object", {}],
                ];
                const paths = [
                    "counts.nodes",
                    "counts.edges",
                    "counts.arcs",
                    "counts.selfLoops",
                    "ids.size",
                    "ids.offset",
                    "arena.buffer",
                    "arena.byteOffset",
                    "arena.byteLength",
                    "arena.hotByteLength",
                    "formatVersion",
                    "wire.0",
                    "wire.1",
                    "label",
                    "producer",
                ];
                for (const path of paths) {
                    for (const [lieName, lie] of scalarLies) {
                        const mutated = withManifest(bytes, (m) => {
                            setPath(m, path, lie);
                        });
                        judgeAll(
                            (level) => fromBytes(mutated, { validate: level }),
                            golden.name,
                            `${path} := ${lieName}`,
                            anomalies,
                        );
                    }
                    const plusOne = withManifest(bytes, (m) => {
                        const key = path.split(".").pop() as string;
                        const parent = walkTo(m, path.split(".").slice(0, -1).join("."));
                        const current = parent?.[key];
                        if (parent !== undefined && typeof current === "number") {
                            parent[key] = current + 1;
                        }
                    });
                    judgeAll(
                        (level) => fromBytes(plusOne, { validate: level }),
                        golden.name,
                        `${path} := +1`,
                        anomalies,
                    );
                }
                const flags = [
                    "multigraph",
                    "hasSelfLoops",
                    "arcToEdgeIsIdentity",
                    "weighted",
                    "allWeightsOne",
                    "nonNegativeWeights",
                    "finiteWeights",
                ];
                for (const flag of flags) {
                    for (const value of [true, false, 1, "true", null]) {
                        const mutated = withManifest(bytes, (m) => {
                            at(m, "flags")[flag] = value;
                        });
                        judgeAll(
                            (level) => fromBytes(mutated, { validate: level }),
                            golden.name,
                            `flags.${flag} := ${String(value)}`,
                            anomalies,
                        );
                    }
                }
                for (const value of [true, false, 1, "false", null]) {
                    const mutated = withManifest(bytes, (m) => {
                        m.directed = value;
                    });
                    judgeAll(
                        (level) => fromBytes(mutated, { validate: level }),
                        golden.name,
                        `directed := ${String(value)}`,
                        anomalies,
                    );
                }
                for (const kind of ["identity", "dense", "numeric", "string", "mixed", "sparse", 3, null]) {
                    const mutated = withManifest(bytes, (m) => {
                        at(m, "ids").kind = kind;
                    });
                    judgeAll(
                        (level) => fromBytes(mutated, { validate: level }),
                        golden.name,
                        `ids.kind := ${String(kind)}`,
                        anomalies,
                    );
                }
                const tops = [
                    "core",
                    "ids",
                    "flags",
                    "counts",
                    "arena",
                    "meta",
                    "views",
                    "extensions",
                    "nodeColumns",
                    "edgeColumns",
                    "graphColumns",
                    "copied",
                ];
                for (const top of tops) {
                    for (const value of [null, [], {}, 0, "x", [null], [{}]]) {
                        const mutated = withManifest(bytes, (m) => {
                            m[top] = value;
                        });
                        judgeAll(
                            (level) => fromBytes(mutated, { validate: level }),
                            golden.name,
                            `${top} := ${JSON.stringify(value)}`,
                            anomalies,
                        );
                    }
                    const gone = withManifest(bytes, (m) => {
                        setPath(m, top, undefined);
                    });
                    judgeAll((level) => fromBytes(gone, { validate: level }), golden.name, `${top} deleted`, anomalies);
                }
                const extra = withManifest(bytes, (m) => {
                    m.futureField = { anything: [1, 2, 3] };
                    (m.counts as Record<string, unknown>).futureCount = 9;
                });
                for (const level of LEVELS) {
                    expectSnapshotsEqual(original, fromBytes(extra, { validate: level }));
                }
                expect(unexplained(anomalies)).toBe("");
            });

            it("unknown dtypes, column metadata lies, slot lies and view lies are refused, skipped or harmless", () => {
                const anomalies: Anomaly[] = [];
                const columnPaths = golden.columns ? columnPathsOf(splitContainer(bytes).manifest) : [];
                const metaLies: readonly [string, Record<string, unknown>][] = [
                    ["dtype f16", { dtype: "f16" }],
                    ["dtype list", { dtype: "list" }],
                    ["dtype json", { dtype: "json" }],
                    ["dtype u8", { dtype: "u8" }],
                    ["dtype bool", { dtype: "bool" }],
                    ["dtype string", { dtype: "string" }],
                    ["dtype dict", { dtype: "dict" }],
                    ["dtype f64", { dtype: "f64" }],
                    ["dtype 3", { dtype: 3 }],
                    ["itemDtype i64", { itemDtype: "i64" }],
                    ["itemDtype list", { itemDtype: "list" }],
                    ["itemDtype u32", { itemDtype: "u32" }],
                    ["components 0", { components: 0 }],
                    ["components 17", { components: 17 }],
                    ["components 2", { components: 2 }],
                    ["components 1.5", { components: 1.5 }],
                    ["components string", { components: "3" }],
                    ["itemComponents 3", { itemComponents: 3 }],
                    ["refersTo graph", { refersTo: "graph" }],
                    ["refersTo node", { refersTo: "node" }],
                    ["refersTo edge", { refersTo: "edge" }],
                    ["nullable false", { nullable: false }],
                    ["nullable string", { nullable: "no" }],
                    ["mutable true", { mutable: true }],
                    ["mutable string", { mutable: "yes" }],
                    ["role 3", { role: 3 }],
                    ["role id", { role: "id" }],
                    ["role weight", { role: "weight" }],
                    ["role position", { role: "position" }],
                    ["unique true", { unique: true }],
                    ["name empty", { name: "" }],
                    ["name number", { name: 7 }],
                    ["name duplicate", { name: "pos" }],
                    ["name proto", { name: "__proto__" }],
                    ["default object", { default: { a: 1 } }],
                    ["default tag", { default: { $num: "Infinity" } }],
                    ["default bad tag", { default: { $num: "huge" } }],
                    ["default string", { default: "x" }],
                    ["fill string", { fill: "x" }],
                    ["fill tag NaN", { fill: { $num: "NaN" } }],
                    ["fill array", { fill: [1] }],
                    ["fill null", { fill: null }],
                    ["options non-array", { options: "x" }],
                    ["options numbers", { options: [1, 2] }],
                    ["origin string", { origin: "gexf" }],
                    ["origin bad field", { origin: { format: 7 } }],
                    ["extra array", { extra: [1] }],
                    ["extra proto", { extra: JSON.parse('{"__proto__": {"polluted": true}}') as unknown }],
                    ["extra constructor", { extra: { constructor: { prototype: {} } } }],
                    ["domain edge", { domain: "edge" }],
                    ["dynamic 1", { dynamic: 1 }],
                ];
                for (const path of columnPaths) {
                    for (const [lieName, lie] of metaLies) {
                        const mutated = withManifest(bytes, (m) => {
                            for (const key of Object.keys(lie)) {
                                setPath(m, `${path}.meta.${key}`, lie[key]);
                            }
                        });
                        const policies = lieName.includes("dtype")
                            ? (["error", "skip"] as const)
                            : (["error"] as const);
                        for (const unknownColumns of policies) {
                            judgeAll(
                                (level) => fromBytes(mutated, { validate: level, unknownColumns }),
                                golden.name,
                                `${path}.meta ${lieName} (${unknownColumns})`,
                                anomalies,
                            );
                        }
                    }
                    const countLies: readonly [string, unknown][] = [
                        ["nullCount -1", -1],
                        ["nullCount +1", 1],
                        ["nullCount huge", 1e9],
                        ["nullCount string", "0"],
                        ["nullCount missing", undefined],
                    ];
                    for (const [lieName, lie] of countLies) {
                        const mutated = withManifest(bytes, (m) => {
                            at(m, path).nullCount = lie;
                        });
                        judgeAll(
                            (level) => fromBytes(mutated, { validate: level }),
                            golden.name,
                            `${path} ${lieName}`,
                            anomalies,
                        );
                    }
                    for (const slot of ["data", "validity", "dictionary", "strings", "offsets", "child", "jsonText"]) {
                        const slotLies: readonly [string, (rowPtr: Record<string, unknown>) => unknown][] = [
                            ["null", () => null],
                            ["object", () => ({})],
                            ["rowPtr ref", (rowPtr) => rowPtr],
                            [
                                "utf8 pair of rowPtr",
                                (rowPtr) => ({
                                    offsets: rowPtr,
                                    utf8: { ...rowPtr, dtype: "utf8", length: rowPtr.byteLength },
                                }),
                            ],
                        ];
                        for (const [lieName, lie] of slotLies) {
                            const mutated = withManifest(bytes, (m) => {
                                at(m, path)[slot] = lie({ ...at(m, "core.rowPtr") });
                            });
                            judgeAll(
                                (level) => fromBytes(mutated, { validate: level }),
                                golden.name,
                                `${path}.${slot} := ${lieName}`,
                                anomalies,
                            );
                        }
                    }
                }
                const viewLies: readonly unknown[] = [
                    { unknownView: { data: "x" } },
                    { unknownView: {} },
                    { unknownView: { data: null } },
                    { outDegree: { data: null } },
                    { outDegree: {} },
                    { reverse: {} },
                    { degreeOrder: {} },
                    { mate: {} },
                    { edgeList: {} },
                    { coo: {} },
                    { totalWeight: { data: {} } },
                    "x",
                    [],
                    7,
                ];
                for (const views of viewLies) {
                    const mutated = withManifest(bytes, (m) => {
                        m.views = views;
                    });
                    judgeAll(
                        (level) => fromBytes(mutated, { validate: level }),
                        golden.name,
                        `views := ${JSON.stringify(views)}`,
                        anomalies,
                    );
                }
                const carrying = withManifest(bytes, (m) => {
                    const rowPtr = (): Record<string, unknown> => ({ ...at(m, "core.rowPtr") });
                    m.views = {
                        outDegree: { data: rowPtr() },
                        inDegree: { data: rowPtr() },
                        degreeOrder: { perm: rowPtr(), segmentOffsets: rowPtr() },
                        selfLoopArcs: { data: rowPtr() },
                        coo: { src: rowPtr() },
                        edgeList: { src: rowPtr() },
                        mate: { data: rowPtr() },
                        reverse: { rowPtr: rowPtr(), colIdx: rowPtr(), fwdArc: rowPtr() },
                        weirdView: { data: rowPtr() },
                    };
                });
                judgeAll(
                    (level) => fromBytes(carrying, { validate: level }),
                    golden.name,
                    "views aliasing rowPtr",
                    anomalies,
                );
                expect(unexplained(anomalies)).toBe("");
            });

            it("the endianness probe, magic, major and minor are checked with the documented codes and details", () => {
                const swapped = bytes.slice();
                swapped.set([0x01, 0x02, 0x03, 0x04], 8);
                const probe = run(() => fromBytes(swapped));
                expect(probe.kind === "error" && probe.code).toBe("E_BAD_SERIALIZATION");
                expect(probe.kind === "error" && probe.error.details.ref).toBe("endianness");
                const majorTwo = bytes.slice();
                new DataView(majorTwo.buffer).setUint16(4, 2, true);
                const major = run(() => fromBytes(majorTwo));
                expect(major.kind === "error" && major.code).toBe("E_UNSUPPORTED_VERSION");
                expect(major.kind === "error" && major.error.details.kind).toBe("wire");
                expect(major.kind === "error" && major.error.details.found).toBe(2);
                const minorNine = bytes.slice();
                new DataView(minorNine.buffer).setUint16(6, 9, true);
                const minor = run(() => fromBytes(minorNine));
                expect(minor.kind === "error" && minor.code).toBe("E_BAD_SERIALIZATION");
                const magic = run(() => fromBytes(setByte(bytes, 0, 0x48)));
                expect(magic.kind === "error" && magic.code).toBe("E_BAD_SERIALIZATION");
                expect(magic.kind === "error" && magic.error.details.ref).toBe("magic");
                // a newer minor in both header and manifest is accepted
                const newer = withManifest(bytes, (m) => {
                    m.wire = [1, 7];
                });
                new DataView(newer.buffer).setUint16(6, 7, true);
                expectSnapshotsEqual(original, fromBytes(newer));
                // a formatVersion the reader does not know, with an unchanged layout
                const format = run(() =>
                    fromBytes(
                        withManifest(bytes, (m) => {
                            m.formatVersion = 2;
                        }),
                    ),
                );
                expect(format.kind === "error" && format.code).toBe("E_UNSUPPORTED_VERSION");
                expect(format.kind === "error" && format.error.details.kind).toBe("format");
            });
        });
    }
});

// ============================================================ in-memory wire form

describe("wire fuzz: in-memory wire form (fromWire)", () => {
    for (const golden of GOLDENS) {
        it(`${golden.name}: structuredClone round trip, poisoned manifest values and poisoned buffers`, () => {
            const original = golden.make();
            const wire = original.toWire();
            const cloned = structuredClone(wire);
            for (const level of LEVELS) {
                const back = fromWire(cloned, { validate: level });
                expectSnapshotsEqual(original, back);
                back.validate({ level: "full" });
                expect(back.contentHash()).toBe(original.contentHash());
            }
            const anomalies: Anomaly[] = [];
            const poison: readonly [string, unknown][] = [
                ["NaN", Number.NaN],
                ["Infinity", Infinity],
                ["-0", -0],
                ["undefined", undefined],
                ["function", (): number => 1],
                ["symbol", Symbol("x")],
                ["bigint", 5n],
                ["Date", new Date(0)],
                ["Map", new Map()],
                ["Uint8Array", new Uint8Array(4)],
                ["class instance", new Date(1)],
                ["null-proto", Object.create(null) as unknown],
                ["array", [1]],
                ["nested proto", JSON.parse('{"__proto__": {"x": 1}}') as unknown],
            ];
            const refs: FoundRef[] = [];
            collectRefs(structuredClone(wire.manifest), "", refs);
            const refFields = ["buffer", "byteOffset", "byteLength", "length", "dtype"];
            const scalarPaths = [
                "counts.nodes",
                "counts.edges",
                "counts.arcs",
                "counts.selfLoops",
                "ids.size",
                "ids.offset",
                "ids.kind",
                "directed",
                "label",
                "formatVersion",
                "format",
                "wire",
                "flags.weighted",
                "meta.extra",
                "meta.keywords",
                "meta.name",
                "meta.weightOrigin",
                "meta.idType",
                "arena",
                "views",
                "copied",
                "producer",
                "nodeColumns.0.meta.dtype",
                "nodeColumns.0.meta.extra",
                "nodeColumns.0.meta.default",
                "nodeColumns.0.meta.origin",
                "nodeColumns.0.meta.options",
                "nodeColumns.0.nullCount",
                ...(golden.columns ? refs : refs.filter((r) => !r.path.includes("Columns"))).flatMap((r) =>
                    refFields.map((f) => `${r.path}.${f}`),
                ),
            ];
            for (const path of scalarPaths) {
                for (const [poisonName, value] of poison) {
                    const mutated = structuredClone(wire);
                    setPath(mutated.manifest as unknown as Record<string, unknown>, path, value);
                    judgeAll(
                        (level) => fromWire(mutated, { validate: level }),
                        golden.name,
                        `fromWire ${path} := ${poisonName}`,
                        anomalies,
                    );
                }
            }
            const detached = new ArrayBuffer(64);
            structuredClone(detached, { transfer: [detached] });
            const bufferPoison: readonly [string, unknown][] = [
                ["Uint8Array", new Uint8Array(64)],
                ["SharedArrayBuffer", new SharedArrayBuffer(64)],
                ["null", null],
                ["number", 3],
                ["detached", detached],
                ["short", new ArrayBuffer(3)],
                ["empty", new ArrayBuffer(0)],
            ];
            for (const [poisonName, value] of bufferPoison) {
                for (let i = 0; i < wire.buffers.length; i++) {
                    const mutated = structuredClone(wire);
                    (mutated.buffers as unknown[])[i] = value;
                    judgeAll(
                        (level) => fromWire(mutated, { validate: level }),
                        golden.name,
                        `fromWire buffers[${i}] := ${poisonName}`,
                        anomalies,
                    );
                }
            }
            const shapes: readonly [string, unknown][] = [
                ["null", null],
                ["array", []],
                ["manifest only", { manifest: wire.manifest }],
                ["buffers only", { buffers: wire.buffers }],
                ["buffers object", { manifest: wire.manifest, buffers: {} }],
                ["manifest string", { manifest: "x", buffers: [] }],
                ["class instance", new Map([["manifest", wire.manifest]])],
                ["extra buffers", { manifest: wire.manifest, buffers: [...wire.buffers, new ArrayBuffer(8)] }],
                ["fewer buffers", { manifest: wire.manifest, buffers: wire.buffers.slice(0, -1) }],
                ["reversed buffers", { manifest: wire.manifest, buffers: [...wire.buffers].reverse() }],
            ];
            for (const [shapeName, shape] of shapes) {
                judgeAll(
                    (level) => fromWire(shape as WireSnapshot, { validate: level }),
                    golden.name,
                    `fromWire(${shapeName})`,
                    anomalies,
                );
            }
            expect(unexplained(anomalies)).toBe("");
        });
    }
});

// ============================================================ chunked input

describe("wire fuzz: chunk splits", () => {
    it("every split decodes to the same snapshot; dropped, duplicated, reversed, empty or foreign chunks are refused", () => {
        const anomalies: Anomaly[] = [];
        for (const golden of GOLDENS) {
            const original = golden.make();
            const bytes = original.toBytes();
            const rand = prng(0x99 + bytes.byteLength);
            for (let trial = 0; trial < 12; trial++) {
                const cuts = new Set<number>();
                const pieces = 1 + Math.floor(rand() * 6);
                for (let i = 0; i < pieces; i++) {
                    cuts.add(Math.floor(rand() * bytes.byteLength));
                }
                const sorted = [...cuts].sort((a, b) => a - b);
                const chunks: Uint8Array[] = [];
                let last = 0;
                for (const cut of sorted) {
                    chunks.push(bytes.subarray(last, cut));
                    last = cut;
                }
                chunks.push(bytes.subarray(last));
                const label = `split ${sorted.join(",")}`;
                for (const level of LEVELS) {
                    const outcome = run(() => fromByteChunks(chunks, { validate: level }));
                    judge(outcome, level, golden.name, `${label} at ${level}`, anomalies);
                    if (outcome.kind !== "ok") {
                        anomalies.push({ golden: golden.name, mutation: label, problem: "a valid split was refused" });
                        continue;
                    }
                    try {
                        expectSnapshotsEqual(original, outcome.snapshot);
                        expect(outcome.snapshot.arena).toBeNull();
                        expect(outcome.snapshot.contentHash()).toBe(original.contentHash());
                    } catch (err) {
                        anomalies.push({
                            golden: golden.name,
                            mutation: label,
                            problem: `decoded snapshot differs: ${describeError(err)}`,
                        });
                    }
                }
                if (chunks.length > 1) {
                    const variants: readonly [string, Uint8Array[]][] = [
                        ["dropped last", chunks.slice(0, -1)],
                        ["dropped first", chunks.slice(1)],
                        ["reversed", [...chunks].reverse()],
                        ["duplicated last", [...chunks, chunks[chunks.length - 1]]],
                        ["duplicated first", [chunks[0], ...chunks]],
                    ];
                    for (const [name, list] of variants) {
                        judgeAll(
                            (level) => fromByteChunks(list, { validate: level }),
                            golden.name,
                            `${name} ${sorted.join(",")}`,
                            anomalies,
                        );
                    }
                }
            }
            const zeroedShared = new Uint8Array(new SharedArrayBuffer(bytes.byteLength));
            const oddSubarray = new Uint8Array(bytes.byteLength + 3);
            oddSubarray.set(bytes, 3);
            const oddCases: readonly [string, Uint8Array[]][] = [
                ["no chunks", []],
                ["one empty chunk", [new Uint8Array(0)]],
                ["trailing garbage chunk", [bytes, new Uint8Array(300)]],
                ["leading empty chunks", [new Uint8Array(0), new Uint8Array(0), bytes]],
                ["string chunk", ["x" as unknown as Uint8Array]],
                ["zeroed shared chunk", [zeroedShared]],
                [
                    "shared copy of the container",
                    [
                        ((): Uint8Array => {
                            const shared = new Uint8Array(new SharedArrayBuffer(bytes.byteLength));
                            shared.set(bytes);
                            return shared;
                        })(),
                    ],
                ],
                ["container at byte offset 3 of its buffer", [oddSubarray.subarray(3)]],
                ["bytes then a second copy", [bytes, bytes]],
            ];
            for (const [name, list] of oddCases) {
                judgeAll((level) => fromByteChunks(list, { validate: level }), golden.name, name, anomalies);
            }
        }
        expect(unexplained(anomalies)).toBe("");
    });
});
