// The whole app as static files, on a tiny base model (e2e/static-server.sh): create a wake
// word, record and import takes, train in the browser, listen live with the trained file,
// download, delete.

import { fileURLToPath } from "node:url";

import { expect, test, type Page } from "@playwright/test";

const fixture = (name: string) => fileURLToPath(new URL(`../.e2e/fixtures/${name}`, import.meta.url));
const ticks = async (page: Page) => Number(await page.getByTestId("live-chart").getAttribute("data-ticks"));

test("record, train, test live and download a wake word", async ({ page, browserName }) => {
  test.setTimeout(300_000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await page.goto("./");
  await page.getByLabel("Wake word").fill("abad");
  await page.getByLabel("Language").selectOption("fr");
  await page.getByRole("button", { name: "Start recording" }).click();
  await expect(page).toHaveURL(/\/w\/[0-9a-f]{32}$/);
  await expect(page.getByRole("heading", { name: "“abad”" })).toBeVisible();
  await expect(page.locator(".paragraph")).toContainText("La matinée était calme");

  // The wake word: a take recorded elsewhere (a Pi, another app), cut into 6 clips by the upload checks.
  await page.getByTestId("import-positive").setInputFiles(fixture("take.wav"));
  await expect(page.getByTestId("clip")).toHaveCount(6);

  // Other speech, recorded live. Chromium's fake microphone plays speech, which must be kept;
  // Firefox's plays a tone, which the app must refuse before uploading; WebKit's mock signal may
  // go either way. A refused take is then replaced by an imported file.
  const other = page.getByTestId("recorder-other_speech");
  const rows = page.getByTestId("samples-other_speech").getByTestId("sample-row");
  const refused = other.getByTestId("recorder-error");
  await page.getByTestId("record-other_speech").click();
  await expect(other.locator(".meter")).toBeVisible();
  if (browserName === "chromium") await expect(other.locator(".light.on")).toBeVisible({ timeout: 8000 });
  await page.waitForTimeout(8000); // speech.wav: 15 words with pauses, from the start
  await page.getByTestId("record-other_speech").click();
  await expect(rows.or(refused)).toBeVisible();
  if (browserName === "chromium") await expect(rows).toHaveCount(1);
  if (browserName === "firefox") await expect(refused).toContainText("of speech detected");
  if (await refused.isVisible()) {
    await page.getByTestId("import-other_speech").setInputFiles(fixture("other_speech.wav"));
    await expect(rows).toHaveCount(1);
  }
  await page.getByTestId("import-ambience").setInputFiles(fixture("ambience.wav"));
  await expect(page.getByTestId("samples-ambience").getByTestId("sample-row")).toHaveCount(1);

  await page.getByTestId("train").click();
  await expect(page).toHaveURL(/\/t\/[0-9a-f]{32}$/);
  await expect(page.getByTestId("results")).toBeVisible({ timeout: 240_000 });
  await expect(page.getByTestId("results")).toContainText("of your recordings detected");

  // Live test on the trained files, then swap model, gate and sensitivity while listening.
  await page.getByTestId("live-toggle").click();
  await expect.poll(() => ticks(page), { timeout: 30_000 }).toBeGreaterThan(20);
  if (browserName === "chromium") { // its fake microphone plays speech: the VAD must hear it
    await expect.poll(async () => Number(await page.getByTestId("live-chart").getAttribute("data-vad-max")),
      { timeout: 15_000 }).toBeGreaterThan(0.5);
  }
  // Each change starts a fresh view (the new detector numbers its ticks from 0).
  const shown = async () => Number(await page.getByTestId("live-chart").getAttribute("data-shown"));
  await expect.poll(shown, { timeout: 30_000 }).toBeGreaterThanOrEqual(40);
  await expect(page.getByRole("button", { name: "Small (int8)" })).toHaveCount(0); // the browser builds fp32 only
  await page.getByTestId("gate").uncheck();
  await page.getByTestId("sensitivity").fill("5");
  await expect.poll(shown, { timeout: 5_000 }).toBeLessThan(40);
  await expect(page.getByTestId("detections")).toHaveText(/^\d+ detected$/);
  const before = await ticks(page);
  await expect.poll(() => ticks(page), { timeout: 30_000 }).toBeGreaterThan(before + 20);
  await expect.poll(shown, { timeout: 30_000 }).toBeGreaterThan(20);
  await page.getByTestId("live-toggle").click();
  await expect(page.getByTestId("live-toggle")).toHaveText("Start listening");

  // Download at the chosen sensitivity: the same file with its threshold rewritten, zipped here.
  const [download] = await Promise.all([page.waitForEvent("download"), page.getByTestId("bundle-link").click()]);
  expect(download.suggestedFilename()).toBe("wake-word.zip");
  const fs = await import("node:fs");
  const zip = fs.readFileSync(await download.path());
  expect(zip.subarray(0, 2).toString()).toBe("PK");
  expect(zip.includes(Buffer.from("wake-word/model.onnx"))).toBe(true);

  // Delete everything.
  await page.getByRole("link", { name: "← Recordings" }).click();
  page.once("dialog", (d) => void d.accept());
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByTestId("recent")).toHaveCount(0);
  expect(errors).toEqual([]);
});
