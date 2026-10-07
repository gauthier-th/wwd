import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const page = (name: string) => fileURLToPath(new URL(name, import.meta.url));
// The API (uvicorn server.api:app); the built app is also served by it in production.
const api = { "/api": { target: process.env.WWD_API ?? "http://127.0.0.1:8000", changeOrigin: true } };

export default defineConfig({
  plugins: [react()],
  // Pre-bundling rewrites import.meta.url, which onnxruntime-web uses to find its .wasm files.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  worker: { format: "es" },
  server: { proxy: api },
  preview: { proxy: api },
  build: {
    rollupOptions: { input: { main: page("index.html"), harness: page("harness.html") } },
  },
});
