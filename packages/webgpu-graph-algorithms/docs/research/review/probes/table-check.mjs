// GFM table column-count check: ignores fenced code blocks, treats "\|" as text
// and "|" inside backtick spans as a cell separator (as GitHub does).
import { readFileSync } from "node:fs";
const path = process.argv[2] ?? "/home/apowers/Projects/webgpu-graph-algorithms/design/webgpu-acceleration-plan.md";
const lines = readFileSync(path, "ascii").split("\n");
let fenced = false, bad = 0, tables = 0, expect = null;
function cells(line) {
    const out = []; let cur = "";
    for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (c === "\\" && line[i + 1] === "|") { cur += "|"; i++; continue; }
        if (c === "|") { out.push(cur); cur = ""; continue; }
        cur += c;
    }
    out.push(cur);
    if (out[0].trim() === "") out.shift();
    if (out.length && out[out.length - 1].trim() === "") out.pop();
    return out;
}
for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (l.startsWith("```")) { fenced = !fenced; continue; }
    if (fenced) continue;
    if (l.startsWith("|")) {
        const n = cells(l).length;
        if (expect === null) { expect = n; tables++; }
        else if (n !== expect) { bad++; console.log(`line ${i + 1}: ${n} cells, table has ${expect}`); }
    } else { expect = null; }
}
console.log(`tables=${tables} bad_rows=${bad}`);
process.exit(bad ? 1 : 0);
