import { defineConfig } from "vite";
import { resolve } from "path";
import { readFileSync } from "fs";
import { config } from "dotenv";

// Load environment variables from .env file
config();

const host = process.env["HOST"] ?? "0.0.0.0";
const port = process.env["PORT"] ? Number.parseInt(process.env["PORT"], 10) : 9010;

// HTTPS configuration
let https: { cert: string; key: string } | undefined = undefined;
if (process.env["HTTPS_CERT_PATH"] && process.env["HTTPS_KEY_PATH"]) {
    try {
        https = {
            cert: readFileSync(process.env["HTTPS_CERT_PATH"], "utf-8"),
            key: readFileSync(process.env["HTTPS_KEY_PATH"], "utf-8")
        };
    } catch (error) {
        console.warn("Failed to load HTTPS certificates:", error);
        https = undefined;
    }
}

export default defineConfig({
    build: {
        lib: {
            entry: resolve(__dirname, "src/index.ts"),
            name: "WebGPUGraphAlgorithms",
            fileName: "index",
            formats: ["es"]
        },
        rollupOptions: {
            external: [],
            output: {
                globals: {}
            }
        },
        target: "esnext",
        sourcemap: true,
        minify: false
    },
    server: {
        host,
        port,
        ...(https && { https }),
        open: "/examples/",
        cors: true
    },
    preview: {
        host,
        port,
        ...(https && { https }),
        open: "/examples/"
    },
    optimizeDeps: {
        exclude: []
    },
    resolve: {
        alias: {
            "@": resolve(__dirname, "./src")
        }
    }
});