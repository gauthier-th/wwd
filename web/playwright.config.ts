import { fileURLToPath } from "node:url";

import { defineConfig, devices } from "@playwright/test";

// Chromium plays this file as the fake microphone; Firefox's fake device generates a tone.
const speech = fileURLToPath(new URL("../runtime/testdata/audio/speech.wav", import.meta.url));
const port = process.env.WWD_E2E_PORT ?? "4173";

export default defineConfig({
  testDir: "e2e",
  timeout: 180_000,
  reporter: [["list"]],
  // Tests run against the production build, static files only, with the tiny base model
  // exported as web assets (e2e/static-server.sh): asset paths (onnx, wasm, worklet, workers)
  // are covered, and every training runs in the browser under test.
  webServer: {
    command: "bash e2e/static-server.sh",
    url: `http://localhost:${port}/wwd/harness.html`,
    reuseExistingServer: !process.env.CI,
    timeout: 300_000,
  },
  // Under a sub-path, as on GitHub Pages: pages are opened relative to it ("./").
  use: { baseURL: `http://localhost:${port}/wwd/` },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        permissions: ["microphone"],
        launchOptions: {
          args: [
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            `--use-file-for-fake-audio-capture=${speech}`,
          ],
        },
      },
    },
    {
      // Mobile Safari's engine at phone size, with touch. Playwright's WebKit has no fake audio
      // device on Linux, so microphone steps are skipped there (see e2e specs).
      name: "webkit",
      use: { ...devices["iPhone 15"], permissions: ["microphone"] },
    },
    {
      name: "firefox",
      use: {
        ...devices["Desktop Firefox"],
        launchOptions: {
          firefoxUserPrefs: {
            "media.navigator.streams.fake": true,
            "media.navigator.permission.disabled": true,
          },
        },
      },
    },
  ],
});
