// The demo's own vite config (root = demo/). WebGPU is a secure-context API, so on anything but localhost the
// page must be served over HTTPS: as graphty-monorepo/graphty-element/vite.config.ts does, the server turns on
// HTTPS when HTTPS_KEY_PATH and HTTPS_CERT_PATH point at a key / certificate pair (the dev box's *.ato.ms files),
// accepts any hostname (vite 6+ blocks non-localhost hosts by default), and serves files from the whole workspace.
import { existsSync, readFileSync } from "node:fs";

import { defineConfig, type UserConfig } from "vite";

export default defineConfig(() => {
    const config: UserConfig = {
        server: {
            host: process.env.HOST ?? "0.0.0.0",
            allowedHosts: true,
            fs: { allow: ["../.."] },
        },
    };
    const keyPath = process.env.HTTPS_KEY_PATH;
    const certPath = process.env.HTTPS_CERT_PATH;
    if (
        keyPath !== undefined &&
        certPath !== undefined &&
        existsSync(keyPath) &&
        existsSync(certPath) &&
        config.server
    ) {
        config.server.https = { key: readFileSync(keyPath), cert: readFileSync(certPath) };
    }
    return config;
});
