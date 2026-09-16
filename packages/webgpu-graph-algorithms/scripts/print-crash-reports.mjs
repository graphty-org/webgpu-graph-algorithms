/**
 * Prints the newest macOS crash reports of `node` for the host lane (hosts.yml): ReportCrash writes
 * ~/Library/Logs/DiagnosticReports/<process>-<date>.ips for every SIGSEGV / SIGABRT / SIGBUS, the vitest worker
 * processes included, while vitest itself reports a dead worker only as tinypool's "Channel closed". The report's
 * faulting thread names the native frames (Dawn's Metal backend, the webgpu binding, node) that the job log
 * otherwise never sees. Informational: never exits non-zero.
 *
 *   node scripts/print-crash-reports.mjs               # the newest 3 reports of a process named node
 *   node scripts/print-crash-reports.mjs 5 chromium    # the newest 5 reports of a process whose name starts with chromium
 */
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const MAX_FRAMES = 64;
const count = Number.parseInt(process.argv[2] ?? "3", 10) || 3;
const prefix = process.argv[3] ?? "node";
const dir = join(homedir(), "Library", "Logs", "DiagnosticReports");

/**
 * The report files of the process, newest first.
 * @returns absolute paths
 */
function reportFiles() {
    if (!existsSync(dir)) {
        return [];
    }
    return readdirSync(dir)
        .filter((name) => name.startsWith(prefix) && name.endsWith(".ips"))
        .map((name) => join(dir, name))
        .sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs)
        .slice(0, count);
}

/**
 * One frame of an .ips backtrace as "image + offset symbol + location".
 * @param frame - the frame record
 * @param images - the report's usedImages
 * @returns the line
 */
function formatFrame(frame, images) {
    const image = images[frame.imageIndex];
    const name = image?.name ?? image?.path ?? `image#${frame.imageIndex}`;
    const symbol = typeof frame.symbol === "string" ? ` ${frame.symbol}` : "";
    const location = typeof frame.symbolLocation === "number" ? ` + ${frame.symbolLocation}` : "";
    return `${name} + ${frame.imageOffset}${symbol}${location}`;
}

/**
 * Prints one report: the header line, the exception, the faulting thread's frames.
 * @param file - the .ips path
 */
function printReport(file) {
    console.log(`==> ${file}`);
    const text = readFileSync(file, "utf8");
    const newline = text.indexOf("\n");
    const headerText = newline < 0 ? text : text.slice(0, newline);
    const bodyText = newline < 0 ? "" : text.slice(newline + 1);
    let header = null;
    let body = null;
    try {
        header = JSON.parse(headerText);
        body = JSON.parse(bodyText);
    } catch {
        console.log("(not the two-JSON .ips layout; the first 4000 characters follow)");
        console.log(text.slice(0, 4000));
        return;
    }
    console.log(
        `process ${body.procName ?? header.app_name} at ${header.timestamp} on ${body.osVersion?.train ?? header.os_version ?? "?"}`,
    );
    if (body.exception) {
        const e = body.exception;
        console.log(
            `exception ${e.type ?? "?"} signal ${e.signal ?? "?"} subtype ${e.subtype ?? "-"} codes ${e.codes ?? "-"}`,
        );
    }
    if (body.termination) {
        const t = body.termination;
        console.log(`termination ${t.indicator ?? "?"} (${t.namespace ?? "?"} ${t.code ?? "?"}) by ${t.byProc ?? "?"}`);
    }
    if (typeof body.exceptionReason === "object" && body.exceptionReason !== null) {
        console.log(`reason ${JSON.stringify(body.exceptionReason)}`);
    }
    for (const [library, lines] of Object.entries(body.asi ?? {})) {
        console.log(`${library}: ${Array.isArray(lines) ? lines.join(" | ") : String(lines)}`);
    }
    const images = Array.isArray(body.usedImages) ? body.usedImages : [];
    const threads = Array.isArray(body.threads) ? body.threads : [];
    const faulting = typeof body.faultingThread === "number" ? body.faultingThread : -1;
    const thread = threads[faulting];
    if (thread === undefined) {
        console.log(`no faulting thread recorded (${threads.length} threads)`);
    } else {
        const label = thread.name ?? thread.queue ?? "";
        console.log(`faulting thread ${faulting}${label ? ` (${label})` : ""}, ${thread.frames?.length ?? 0} frames:`);
        for (const [i, frame] of (thread.frames ?? []).slice(0, MAX_FRAMES).entries()) {
            console.log(`  #${String(i).padStart(2)} ${formatFrame(frame, images)}`);
        }
    }
    if (Array.isArray(body.lastExceptionBacktrace)) {
        console.log("last exception backtrace:");
        for (const [i, frame] of body.lastExceptionBacktrace.slice(0, MAX_FRAMES).entries()) {
            console.log(`  #${String(i).padStart(2)} ${formatFrame(frame, images)}`);
        }
    }
}

const files = reportFiles();
if (files.length === 0) {
    console.log(`no ${prefix}*.ips crash report under ${dir}`);
} else {
    for (const file of files) {
        printReport(file);
    }
}
