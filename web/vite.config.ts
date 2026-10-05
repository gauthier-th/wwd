import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const page = (name: string) => fileURLToPath(new URL(name, import.meta.url));

export default defineConfig({
  plugins: [react()],
  // Pre-bundling rewrites import.meta.url, which onnxruntime-web uses to find its .wasm files.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  worker: { format: "es" },
  build: {
    rollupOptions: { input: { main: page("index.html"), harness: page("harness.html") } },
  },
});
