/**
 * Forwards this page's console output, uncaught errors and unhandled rejections to the dev box's remote log server
 * (the remote-logger MCP, https://dev.ato.ms:9087/log), so a browser without reachable devtools (an iPad, a headset)
 * can be debugged from the terminal. Batched POSTs, plain fetch, no dependency; never throws into the page.
 */

const ENDPOINT = "https://dev.ato.ms:9087/log";
const SESSION = `${navigator.userAgent.includes("iPad") ? "ipad" : navigator.userAgent.includes("Macintosh") ? "mac" : "browser"}-${Date.now().toString(36)}`;
const PROJECT = "webgpu-fa2-demo";

interface LogEntry {
    readonly time: string;
    readonly level: "DEBUG" | "INFO" | "WARN" | "ERROR";
    readonly message: string;
}

let queue: LogEntry[] = [];
let timer: number | null = null;

function flush(): void {
    timer = null;
    if (queue.length === 0) {
        return;
    }
    const logs = queue;
    queue = [];
    void fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: SESSION, projectMarker: PROJECT, logs }),
        keepalive: true,
    }).catch(() => undefined);
}

function push(level: LogEntry["level"], parts: unknown[]): void {
    const message = parts
        .map((p) => {
            if (p instanceof Error) {
                return `${p.name}: ${p.message}${p.stack ? `\n${p.stack}` : ""}`;
            }
            if (typeof p === "string") {
                return p;
            }
            try {
                return JSON.stringify(p);
            } catch {
                return String(p);
            }
        })
        .join(" ");
    queue.push({ time: new Date().toISOString(), level, message: `[${SESSION}] ${message}` });
    if (timer === null) {
        timer = window.setTimeout(flush, 500);
    }
}

/** Installs the console / error hooks once; returns the session id shown in the terminal's log listing. */
export function installRemoteLog(): string {
    const original = { log: console.log, info: console.info, warn: console.warn, error: console.error };
    console.log = (...a: unknown[]) => {
        original.log(...a);
        push("INFO", a);
    };
    console.info = (...a: unknown[]) => {
        original.info(...a);
        push("INFO", a);
    };
    console.warn = (...a: unknown[]) => {
        original.warn(...a);
        push("WARN", a);
    };
    console.error = (...a: unknown[]) => {
        original.error(...a);
        push("ERROR", a);
    };
    window.addEventListener("error", (e) => push("ERROR", [`uncaught: ${e.message} (${e.filename}:${e.lineno})`]));
    window.addEventListener("unhandledrejection", (e) => push("ERROR", ["unhandled rejection:", e.reason]));
    window.addEventListener("pagehide", flush);
    push("INFO", [`page ${location.pathname} loaded; ${navigator.userAgent}`]);
    return SESSION;
}
