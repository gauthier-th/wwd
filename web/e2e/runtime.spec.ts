import { expect, test, type Page } from "@playwright/test";

async function runHarness(page: Page, query: string) {
  const logs: string[] = [];
  page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
  await page.goto(`harness.html?${query}`);
  await page.waitForFunction(() => window.harness && window.harness.status !== "running", null, {
    timeout: 170_000,
  });
  const state = await page.evaluate(() => window.harness);
  expect(state.status, JSON.stringify({ state, logs }, null, 2)).toBe("done");
  return state as Record<string, unknown>;
}

test("golden parity through a Worker in the browser wasm engine", async ({ page }) => {
  const state = await runHarness(page, "mode=golden");
  const runs = state.runs as { name: string; ok: boolean; maxDiff: number; meanDiff: number; ms: number }[];
  for (const r of runs) {
    console.log(`  ${r.name}: max |diff| ${r.maxDiff.toExponential(2)}, mean ${r.meanDiff.toExponential(2)}, ${r.ms} ms`);
  }
  for (const run of runs) expect(run.ok, JSON.stringify(run, null, 2)).toBe(true);
});

test("microphone -> AudioWorklet -> Worker pipeline", async ({ page, browserName }) => {
  const state = await runHarness(page, "mode=mic&seconds=6");
  const { inputRate, ticks, vadMax, gateOpenTicks } = state;
  console.log(`  input ${inputRate} Hz, ${ticks} ticks, VAD max ${vadMax}, gate open ${gateOpenTicks}`);
  expect(state.errors).toEqual([]);
  expect(state.inputRate as number).toBeGreaterThan(0);
  // 6 s at one tick per 96 ms is ~62 ticks; allow for startup.
  expect(state.ticks as number).toBeGreaterThan(40);
  if (browserName === "chromium") {
    // Chromium's fake mic plays speech.wav, so the VAD must hear speech.
    expect(state.vadMax as number).toBeGreaterThan(0.5);
  }
});
