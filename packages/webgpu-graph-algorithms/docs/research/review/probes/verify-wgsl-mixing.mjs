// PERF verifier probe: does Tint reject the plan's unparenthesised hash expressions (7.2 line 1340, 7.7 line 1571)?
import { createRequire } from "node:module";
const require = createRequire("/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/package.json");
const { create, globals } = await import(require.resolve("webgpu"));
Object.assign(globalThis, globals);
const gpu = create(["adapter=llvmpipe"]);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice();
const cases = {
  "7.2 as written": "fn h(i: u32, j: u32) -> u32 { return lowbias32(i * 0x9E3779B9u ^ j); }",
  "7.7 as written": "fn h(cell: u32, iteration: u32, seed: u32) -> u32 { return lowbias32(cell ^ iteration * 0x9E3779B9u ^ seed); }",
  "7.2 fixed": "fn h(i: u32, j: u32) -> u32 { return lowbias32((i * 0x9E3779B9u) ^ j); }",
  "7.7 fixed": "fn h(cell: u32, iteration: u32, seed: u32) -> u32 { return lowbias32(cell ^ (iteration * 0x9E3779B9u) ^ seed); }",
};
const pre = "fn lowbias32(x0: u32) -> u32 { var x = x0; x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u; return x; }\n";
for (const [name, code] of Object.entries(cases)) {
  const m = device.createShaderModule({ code: pre + code + "\n@compute @workgroup_size(1) fn main() { _ = h(1u, 2u" + (code.includes("seed") ? ", 3u" : "") + "); }" });
  const info = await m.getCompilationInfo();
  const errs = info.messages.filter((x) => x.type === "error").map((x) => `${x.lineNum}:${x.linePos} ${x.message}`);
  console.log(name, "->", errs.length ? "ERROR " + errs.join(" | ") : "ok");
}
device.destroy();
