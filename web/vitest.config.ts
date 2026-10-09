import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    // onnxruntime-web/wasm has no Node build; its Node entry loads the same wasm binary.
    alias: { "onnxruntime-web/wasm": "onnxruntime-web" },
  },
  test: { include: ["src/**/*.test.ts"], environment: "node", testTimeout: 300_000 },
});
