// MAINT review probe: when is an `uncapturederror` event delivered in Dawn-node 0.4.0 relative to
// the awaits a test would naturally do (onSubmittedWorkDone, mapAsync)? The plan (5.7, 11.2) says
// "an uncapturederror listener fails the current test" -- that only attributes correctly if the
// event arrives before the test's own awaits resolve. Also: is a second requestDevice on the same
// adapter allowed (plan 2.2 `options.adapter` path; test helpers that create a second device)?
import { createRequire } from "node:module";
const require = createRequire("/home/apowers/Projects/webgpu-graph-algorithms/packages/graph-format/package.json");
const dawn = await import(require.resolve("webgpu"));
Object.assign(globalThis, dawn.globals);
const gpu = dawn.create(process.argv.slice(2));
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice();
console.log("adapter:", adapter.info.vendor, adapter.info.architecture);
const events = [];
device.addEventListener("uncapturederror", (ev) => { events.push(`uncaptured@${performance.now().toFixed(2)}: ${ev.error.message.split("\n")[0]}`); });

// A kernel bug: bind a buffer too small for the dispatch's indexing is clamped (no error); instead
// produce a genuine validation error at command-buffer creation: copy out of range.
const a = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const st = device.createBuffer({ size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
const enc = device.createCommandEncoder();
enc.copyBufferToBuffer(a, 0, st, 0, 512);   // 512 > 256: validation error, surfaces at finish()/submit
const cb = enc.finish();
const t0 = performance.now();
device.queue.submit([cb]);
console.log(`submitted@${(performance.now() - t0).toFixed(2)}; events so far: ${events.length}`);
await device.queue.onSubmittedWorkDone();
console.log(`onSubmittedWorkDone resolved@${(performance.now() - t0).toFixed(2)}; events so far: ${events.length}`);
await new Promise((r) => setTimeout(r, 0));
console.log(`after setTimeout(0)@${(performance.now() - t0).toFixed(2)}; events so far: ${events.length}`);
await new Promise((r) => setTimeout(r, 50));
console.log(`after setTimeout(50)@${(performance.now() - t0).toFixed(2)}; events: ${events.length}`);
for (const e of events) console.log("  ", e);

// Second device on the same adapter?
try { await adapter.requestDevice(); console.log("second requestDevice on same adapter: OK"); }
catch (e) { console.log("second requestDevice on same adapter:", String(e).split("\n")[0]); }
device.destroy();
