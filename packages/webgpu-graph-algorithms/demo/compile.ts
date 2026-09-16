/**
 * Diagnostic page: compiles every registry entry of the bounded override matrix on THIS device through the package's
 * own composer and pipeline cache, then a set of bisecting variants of fa2-attraction, and reports every failure with
 * the GPUPipelineError message. Used to pin backend-specific (Metal / D3D12) rejections that the Vulkan lanes cannot see.
 */

import { installRemoteLog } from "./remote-log.js";
import { requestGpuContext } from "../src/browser/index.js";
import { composeWgsl, type WgslModuleSpec } from "../src/kernel/wgsl.js";
import { type KernelId, KERNELS, kernelSpec } from "../src/kernels.js";
import { OVERRIDE_MATRIX } from "../test/helpers/override-matrix.js";

const out = document.getElementById("out") as HTMLDivElement;
const adapterBox = document.getElementById("adapter") as HTMLDivElement;
const lines: string[] = [];

function log(text: string): void {
    lines.push(text);
    console.log(text);
}

function row(table: HTMLTableElement, cells: string[], ok: boolean | null): void {
    const tr = document.createElement("tr");
    for (const [i, c] of cells.entries()) {
        const td = document.createElement("td");
        td.textContent = c;
        if (i === 1 && ok !== null) {
            td.className = ok ? "ok" : "bad";
        }
        tr.appendChild(td);
    }
    table.appendChild(tr);
}

async function tryPipeline(
    ctx: Awaited<ReturnType<typeof requestGpuContext>>,
    spec: WgslModuleSpec,
): Promise<string | null> {
    try {
        await ctx.pipelines.get(spec);
        return null;
    } catch (error) {
        return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    }
}

function withBody(spec: WgslModuleSpec, edit: (body: string) => string, id: string): WgslModuleSpec {
    return { ...spec, id, body: edit(spec.body) };
}

async function main(): Promise<void> {
    installRemoteLog();
    let ctx: Awaited<ReturnType<typeof requestGpuContext>>;
    try {
        ctx = await requestGpuContext({ powerPreference: "high-performance" });
    } catch (error) {
        adapterBox.textContent = `no device: ${error instanceof Error ? error.message : String(error)}`;
        return;
    }
    const caps = ctx.caps;
    const head = `adapter ${caps.vendor} / ${caps.architecture} / ${caps.device} / ${caps.description}; software=${caps.software}; features=${[...caps.features].join(",")}; wgsl=${[...caps.wgslFeatures].join(",")}; subgroups ${caps.subgroupMinSize}-${caps.subgroupMaxSize}; ua=${navigator.userAgent}`;
    adapterBox.textContent = head;
    log(head);

    // 1. The bounded matrix (the same table the node and browser compile tests use).
    const table = document.createElement("table");
    table.innerHTML = "<tr><th>kernel | overrides</th><th>result</th></tr>";
    out.appendChild(table);
    let failures = 0;
    for (const c of OVERRIDE_MATRIX) {
        const spec = kernelSpec(c.id, c.overrides, c.snippets);
        const error = await tryPipeline(ctx, spec);
        const label = `${c.id} | ${JSON.stringify(c.overrides)}${c.snippets ? " | snippets" : ""}`;
        row(table, [label, error ?? "ok"], error === null);
        log(`${error === null ? "OK  " : "FAIL"} ${label}${error === null ? "" : ` -- ${error}`}`);
        if (error !== null) {
            failures++;
        }
    }
    log(`matrix: ${OVERRIDE_MATRIX.length} cases, ${failures} failures`);

    // 2. Bisecting variants of fa2-attraction (each changes ONE thing; a variant that compiles names the culprit).
    const base = kernelSpec("fa2-attraction", { TIER: 0, LINLOG: false, DISTRIBUTED: false });
    const variants: WgslModuleSpec[] = [
        withBody(
            base,
            (b) =>
                b.replace(
                    "let mag = select(w, w * log(1.0 + len) / len, LINLOG);",
                    "var mag = w; if (LINLOG) { mag = w * log(1.0 + len) / len; }",
                ),
            "bisect:no-select-linlog",
        ),
        withBody(
            base,
            (b) =>
                b.replace("let i = select(row, perm[row], USE_PERM);", "var i = row; if (USE_PERM) { i = perm[row]; }"),
            "bisect:no-select-perm",
        ),
        withBody(
            base,
            (b) => b.replace("@compute @workgroup_size(WG)", "@compute @workgroup_size(256)"),
            "bisect:literal-workgroup-size",
        ),
        withBody(
            base,
            (b) => b.replace("if (row >= P.tierEnd) { return; }", "if (row >= P.n) { return; }"),
            "bisect:tier-bound-n",
        ),
        withBody(
            base,
            (b) =>
                b
                    .replace("if (j == i) { continue; }", "if (j != i) {")
                    .replace("f = f + d * mag;\n    }", "f = f + d * mag; }\n    }"),
            "bisect:no-continue",
        ),
        withBody(
            base,
            (b) => b.replace("let pi = pos[i];", "let pi = vec4f(pos[i].x, pos[i].y, pos[i].z, pos[i].w);"),
            "bisect:pos-component-loads",
        ),
        {
            ...base,
            id: "bisect:no-overrides",
            overrideDecls: [],
            overrides: {},
            body: base.body
                .replace(/\bLINLOG\b/g, "false")
                .replace(/\bDISTRIBUTED\b/g, "false")
                .replace(/\bTIER\b/g, "0u"),
        },
        withBody(
            base,
            () => `fn store_force(i: u32, f: vec3f) {
    force[3u * i] = f.x;
    force[3u * i + 1u] = f.y;
    force[3u * i + 2u] = f.z;
}
@compute @workgroup_size(WG)
fn attraction(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = linear_id(wid, lid.x);
    if (i >= P.n) { return; }
    store_force(i, vec3f(0.0));
}`,
            "bisect:minimal-body",
        ),
    ];
    const table2 = document.createElement("table");
    table2.innerHTML = "<tr><th>fa2-attraction variant</th><th>result</th></tr>";
    out.appendChild(table2);
    for (const v of variants) {
        let error: string | null;
        try {
            composeWgsl(v, caps);
            error = await tryPipeline(ctx, v);
        } catch (e) {
            error = `compose: ${e instanceof Error ? e.message : String(e)}`;
        }
        row(table2, [v.id, error ?? "ok"], error === null);
        log(`${error === null ? "OK  " : "FAIL"} ${v.id}${error === null ? "" : ` -- ${error}`}`);
    }

    // 3. The composed source of the failing kernel, for the report.
    const composed = composeWgsl(base, caps);
    const pre = document.createElement("pre");
    pre.textContent = composed.code;
    out.appendChild(pre);
    log("---- composed fa2-attraction ----");
    log(composed.code);
    ctx.dispose();
}

(document.getElementById("copy") as HTMLButtonElement).addEventListener("click", () => {
    void navigator.clipboard.writeText(lines.join("\n"));
});

void main();
