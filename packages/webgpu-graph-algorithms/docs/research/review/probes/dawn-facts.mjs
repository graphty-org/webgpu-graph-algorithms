// Probe: (1) does webgpu@0.4.0 honour backend=null; (2) default device limits without requiredLimits
// vs adapter limits; (3) timestamp-query granted and whether results look quantised.
import { createRequire } from "node:module";
const require = createRequire("/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/package.json");
const { create, globals } = await import(require.resolve("webgpu"));
Object.assign(globalThis, globals);

async function tryFlags(flags) {
    try {
        const gpu = create(flags);
        const a = await gpu.requestAdapter();
        if (!a) { console.log(`flags=${JSON.stringify(flags)}: requestAdapter() -> null`); return null; }
        console.log(`flags=${JSON.stringify(flags)}: adapter ${a.info.vendor}/${a.info.architecture} "${a.info.description}" fallback=${a.isFallbackAdapter}`);
        return a;
    } catch (e) { console.log(`flags=${JSON.stringify(flags)}: threw ${e.message}`); return null; }
}
await tryFlags(["backend=null"]);
const a = await tryFlags([]);
if (a) {
    const d = await a.requestDevice();   // NO requiredLimits
    const keys = ["maxBufferSize", "maxStorageBufferBindingSize", "maxStorageBuffersPerShaderStage", "maxComputeWorkgroupStorageSize",
        "maxComputeInvocationsPerWorkgroup", "maxComputeWorkgroupSizeX", "maxComputeWorkgroupsPerDimension", "minStorageBufferOffsetAlignment", "minUniformBufferOffsetAlignment"];
    for (const k of keys) console.log(`${k}: adapter=${a.limits[k]} device(default)=${d.limits[k]}`);
    console.log("adapter features:", [...a.features].join(","));
    if (a.features.has("timestamp-query")) {
        const a2 = await create([]).requestAdapter(); const d2 = await a2.requestDevice({ requiredFeatures: ["timestamp-query"] });
        const qs = d2.createQuerySet({ type: "timestamp", count: 2 });
        const res = d2.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        const st = d2.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const code = `@group(0) @binding(0) var<storage, read_write> o: array<u32>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) { var s = 0u; for (var i = 0u; i < 2000u; i++) { s = s * 1664525u + 1013904223u + g.x; } if (g.x < arrayLength(&o)) { o[g.x] = s; } }`;
        const p = d2.createComputePipeline({ layout: "auto", compute: { module: d2.createShaderModule({ code }), entryPoint: "main" } });
        const buf = d2.createBuffer({ size: 4 * 65536, usage: GPUBufferUsage.STORAGE });
        const bg = d2.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
        const samples = [];
        for (let r = 0; r < 5; r++) {
            const e = d2.createCommandEncoder();
            const ps = e.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
            ps.setPipeline(p); ps.setBindGroup(0, bg); ps.dispatchWorkgroups(1024); ps.end();
            e.resolveQuerySet(qs, 0, 2, res, 0); e.copyBufferToBuffer(res, 0, st, 0, 16);
            d2.queue.submit([e.finish()]);
            await st.mapAsync(GPUMapMode.READ);
            const t = new BigUint64Array(st.getMappedRange().slice(0));
            st.unmap();
            samples.push(Number(t[1] - t[0]));
        }
        console.log("timestamp-query deltas (ns):", samples.join(", "), "-- all multiples of 100000?", samples.every((s) => s % 100000 === 0));
        d2.destroy();
    }
    d.destroy();
}
