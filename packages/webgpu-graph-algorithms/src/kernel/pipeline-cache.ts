/**
 * PipelineCache (spec 5.1; contract 3.9): compile-once compute pipelines keyed by
 * `(id, overrides, needs present on the device, snippets)` with EXPLICIT bind-group layouts derived from
 * `spec.bindings` (never `layout: "auto"`, so one layout serves every dummy-binding variant and bind groups are
 * reused across variants). A WGSL error is `E_SHADER_COMPILE { stage: "compile", messages, lines }` with
 * body-relative line numbers; a pipeline error that is not a compilation message (an oversized workgroup, an
 * invalid layout) is `E_VALIDATION`.
 */

import { formatCompilationInfo } from "../device/error-scope.js";
import { WebGpuGraphError } from "../errors.js";
import { type PlanCaps } from "../types/context.js";
import { Kernel } from "./kernel.js";
import { bindGroupLayoutDescriptors, type ComposedModule, composeWgsl, type WgslModuleSpec } from "./wgsl.js";

/** One cached entry: the pipeline, its layouts and the Kernel built on them. */
interface CacheEntry {
    readonly key: string;
    readonly composed: ComposedModule;
    readonly pipeline: GPUComputePipeline;
    readonly layouts: readonly GPUBindGroupLayout[];
    readonly kernel: Kernel;
}

/**
 * JSON with the keys of every (nested) object sorted, so two override records with the same content give one key.
 * @param value - a JSON-compatible value
 * @returns the canonical text
 */
function stableJson(value: unknown): string {
    if (Array.isArray(value)) {
        return `[${value.map((item: unknown) => stableJson(item)).join(",")}]`;
    }
    if (typeof value === "object" && value !== null) {
        const record = value as Record<string, unknown>;
        const keys = Object.keys(record).sort();
        return `{${keys.map((key) => `${JSON.stringify(key)}:${stableJson(record[key])}`).join(",")}}`;
    }
    return JSON.stringify(value);
}

/**
 * FNV-1a (32-bit) of a text as 8 hex digits; "0" for an absent or empty snippet record. Hashes text, never an arc
 * index or a byte offset, so the bitwise operators are within the house rule.
 * @param snippets - the snippet record
 * @returns the hash text
 */
function hashSnippets(snippets: Readonly<Record<string, string>> | undefined): string {
    if (snippets === undefined || Object.keys(snippets).length === 0) {
        return "0";
    }
    const text = stableJson(snippets);
    let hash = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) {
        hash ^= text.charCodeAt(i);
        hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return hash.toString(16).padStart(8, "0");
}

/**
 * The cache key of a spec on a device (pure; PLAN DECISION: exported so the override-matrix helper of P2-T2 can
 * compute keys without a device): `id + "|" + stableJson(overrides) + "|" + needs.filter(present).join(",") + "|" + hash(snippets)`.
 * @param spec - the module
 * @param caps - the device capabilities (the feature set decides which needs are present)
 * @returns the key
 */
export function pipelineKey(spec: WgslModuleSpec, caps: PlanCaps): string {
    const present = spec.needs.filter((need) => caps.features.has(need)).join(",");
    return `${spec.id}|${stableJson(spec.overrides)}|${present}|${hashSnippets(spec.snippets)}`;
}

/**
 * The message of an unknown error value.
 * @param err - the thrown value
 * @returns its message text
 */
function messageOf(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
}

/** Compile-once pipelines keyed by (id, overrides, needs present on the device, snippets) with explicit bind-group layouts (spec 5.1). */
export class PipelineCache {
    private readonly device: GPUDevice;
    private readonly caps: PlanCaps;
    /** In-flight and finished compiles by key (a promise, so two concurrent get() calls share one compile). */
    private readonly pending = new Map<string, Promise<CacheEntry>>();
    /** Finished compiles by key. */
    private readonly entries = new Map<string, CacheEntry>();
    /** Keys in the order their compiles were started (a failed compile is removed again). */
    private readonly order: string[] = [];
    private disposed = false;

    /**
     * Builds a cache over a device.
     * @param device - the device pipelines are created on
     * @param caps - its capabilities (WG, subgroup sizes, features)
     */
    constructor(device: GPUDevice, caps: PlanCaps) {
        this.device = device;
        this.caps = caps;
    }

    /**
     * The cache key of a spec on this device: `id + "|" + stableJson(overrides) + "|" + needs.filter(present).join(",") + "|" + hash(snippets)`.
     * @param spec - the module
     * @returns the key
     */
    key(spec: WgslModuleSpec): string {
        return pipelineKey(spec, this.caps);
    }

    /**
     * createComputePipelineAsync inside a validation scope with the layouts derived from spec.bindings; compilation messages become E_SHADER_COMPILE { id, stage: "compile", messages } with body-relative lines.
     * @param spec - the module
     * @returns the pipeline (the same object for the same key)
     */
    async get(spec: WgslModuleSpec): Promise<GPUComputePipeline> {
        return (await this.entry(spec)).pipeline;
    }

    /**
     * The Kernel (pipeline + layouts + binding names) for a spec; cached with the pipeline.
     * @param spec - the module
     * @returns the kernel (the same object for the same key)
     */
    async kernel(spec: WgslModuleSpec): Promise<Kernel> {
        return (await this.entry(spec)).kernel;
    }

    /**
     * Compiles every spec not yet cached (load() calls it so the first step() does not compile).
     * @param specs - the modules
     */
    async warm(specs: readonly WgslModuleSpec[]): Promise<void> {
        await Promise.all(specs.map((spec) => this.entry(spec)));
    }

    /**
     * Number of cached pipelines.
     * @returns the count of finished compiles
     */
    get size(): number {
        return this.entries.size;
    }

    /**
     * Every key created so far, in creation order (the override-matrix coverage test reads it).
     * PLAN DECISION: keys are listed in compile-START order and only those whose compile finished (warm() compiles
     * concurrently, so completion order is not deterministic; a failed compile is removed).
     * @returns the keys
     * @internal
     */
    keys(): readonly string[] {
        return this.order.filter((key) => this.entries.has(key));
    }

    /**
     * The bind-group layouts of a cached kernel by key.
     * @param key - a key from keys()
     * @returns the layouts, or null when the key is not cached
     * @internal
     */
    layoutsOf(key: string): readonly GPUBindGroupLayout[] | null {
        return this.entries.get(key)?.layouts ?? null;
    }

    /**
     * PLAN DECISION: forgets every pipeline and refuses further compiles with E_DISPOSED (the context's dispose()
     * may call it; a GPUComputePipeline has no destroy(), so the objects are simply released).
     */
    dispose(): void {
        this.disposed = true;
        this.pending.clear();
        this.entries.clear();
        this.order.length = 0;
    }

    /**
     * The cached entry of a spec, compiling it once per key.
     * @param spec - the module
     * @returns the entry
     */
    private entry(spec: WgslModuleSpec): Promise<CacheEntry> {
        if (this.disposed) {
            return Promise.reject(
                new WebGpuGraphError("E_DISPOSED", "PipelineCache: disposed", { label: "PipelineCache" }),
            );
        }
        const key = this.key(spec);
        const inFlight = this.pending.get(key);
        if (inFlight !== undefined) {
            return inFlight;
        }
        this.order.push(key);
        const compile = this.compile(spec, key).then(
            (entry) => {
                if (!this.disposed) {
                    this.entries.set(key, entry);
                }
                return entry;
            },
            (err: unknown) => {
                this.pending.delete(key);
                const at = this.order.indexOf(key);
                if (at >= 0) {
                    this.order.splice(at, 1);
                }
                throw err;
            },
        );
        this.pending.set(key, compile);
        return compile;
    }

    /**
     * One compile: compose, create the shader module inside a synchronous push / pop validation pair (so concurrent
     * compiles never interleave scopes), check the compilation info, create the layouts and the pipeline.
     * PLAN DECISION: `E_SHADER_COMPILE` at stage "compile" also carries `lines` (the body-relative 1-based line of every
     * error message, from `composed.bodyLine`) next to the contract's `messages`; a `createComputePipelineAsync`
     * rejection (a GPUPipelineError, which pushes nothing into a validation scope on Dawn 0.4.0) is
     * `E_VALIDATION { label: "<id>|pipeline", message }`; bool overrides travel as the pipeline constants 1 / 0
     * because dawn.node's WebIDL layer rejects a boolean constant.
     * @param spec - the module
     * @param key - its key (the pipeline label)
     * @returns the entry
     */
    private async compile(spec: WgslModuleSpec, key: string): Promise<CacheEntry> {
        const composed = composeWgsl(spec, this.caps);
        const { device } = this;
        const label = `${spec.id}|pipeline`;
        device.pushErrorScope("validation");
        const module = device.createShaderModule({ code: composed.code, label: spec.id });
        const layouts = bindGroupLayoutDescriptors(spec).map((descriptor, group) =>
            device.createBindGroupLayout({ ...descriptor, label: `${spec.id}/layout${group}` }),
        );
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: layouts, label: `${spec.id}|layout` });
        const popped = device.popErrorScope();
        const info = await module.getCompilationInfo();
        const scopeError = await popped;
        const errors = info.messages.filter((message) => message.type === "error");
        if (errors.length > 0) {
            const preludeLines = composed.bodyLine - 1;
            throw new WebGpuGraphError(
                "E_SHADER_COMPILE",
                `${spec.id}: WGSL compilation failed (${errors.length} error(s))`,
                {
                    id: spec.id,
                    stage: "compile",
                    messages: formatCompilationInfo(info, preludeLines),
                    lines: errors.map((message) => message.lineNum - preludeLines),
                },
            );
        }
        if (scopeError !== null) {
            throw new WebGpuGraphError("E_VALIDATION", `${spec.id}: ${scopeError.message}`, {
                label,
                message: scopeError.message,
            });
        }
        // dawn.node's WebIDL layer rejects a boolean pipeline constant ("value is not a number"), so bools travel as 1 / 0;
        // only the REFERENCED overrides are supplied (composed.constants): WebKit rejects a constant for an unread override
        const constants: Record<string, number> = {};
        for (const [name, value] of Object.entries(composed.constants)) {
            if (typeof value === "boolean") {
                constants[name] = value ? 1 : 0;
            } else {
                constants[name] = value;
            }
        }
        let pipeline: GPUComputePipeline;
        try {
            pipeline = await device.createComputePipelineAsync({
                label: key,
                layout: pipelineLayout,
                compute: { module, entryPoint: composed.entryPoint, constants },
            });
        } catch (err: unknown) {
            const message = messageOf(err);
            throw new WebGpuGraphError("E_VALIDATION", `${spec.id}: pipeline creation failed: ${message}`, {
                label,
                message,
            });
        }
        const kernel = new Kernel(device, spec, composed, pipeline, layouts);
        return { key, composed, pipeline, layouts, kernel };
    }
}
