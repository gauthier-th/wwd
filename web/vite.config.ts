import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const page = (name: string) => fileURLToPath(new URL(name, import.meta.url));

// Browser-only: the build is static files. The base model's files come from `public/base/`
// (scripts/export_web_assets.py); WWD_PUBLIC points elsewhere, e.g. the tiny base for tests.
export default defineConfig({
  plugins: [react()],
  base: process.env.WWD_BASE_PATH ?? "/",
  publicDir: process.env.WWD_PUBLIC ?? "public",
  // Pre-bundling rewrites import.meta.url, which onnxruntime-web uses to find its .wasm files.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  worker: { format: "es" },
  build: {
    rollupOptions: { input: { main: page("index.html"), harness: page("harness.html") } },
  },
});
