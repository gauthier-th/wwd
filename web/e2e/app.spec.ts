// The whole app against a real API on a tiny base model (e2e/api-server.sh): create a wake word,
// record and import takes, train, listen live with the trained files, download, delete.

import { fileURLToPath } from "node:url";

import { expect, test, type Page } from "@playwright/test";

const fixture = (name: string) => fileURLToPath(new URL(`../.e2e/fixtures/${name}`, import.meta.url));
const ticks = async (page: Page) => Number(await page.getByTestId("live-chart").getAttribute("data-ticks"));

test("record, train, test live and download a wake word", async ({ page, browserName }) => {
  test.setTimeout(300_000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await page.goto("/");
  await page.getByLabel("Wake word").fill("abad");
  await page.getByLabel("Language").selectOption("fr");
  await page.getByRole("button", { name: "Start recording" }).click();
  await expect(page).toHaveURL(/\/w\/[0-9a-f]{32}$/);
  await expect(page.getByRole("heading", { name: "“abad”" })).toBeVisible();
  await expect(page.locator(".paragraph")).toContainText("La matinée était calme");

  // The wake word: a take recorded elsewhere (a Pi, another app), cut into 6 clips by the server.
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
  await page.getByRole("button", { name: "Small (int8)" }).click();
  await page.getByTestId("gate").uncheck();
  await page.getByTestId("sensitivity").fill("5");
  const before = await ticks(page);
  await expect.poll(() => ticks(page), { timeout: 30_000 }).toBeGreaterThan(before + 20);
  await page.getByTestId("live-toggle").click();
  await expect(page.getByTestId("live-toggle")).toHaveText("Start listening");

  // Download at the chosen sensitivity: the same files with their thresholds rewritten.
  const href = await page.getByTestId("bundle-link").getAttribute("href");
  expect(href).toContain("fa_per_hour=5");
  const zip = await page.request.get(href!);
  expect(zip.status()).toBe(200);
  expect((await zip.body()).subarray(0, 2).toString()).toBe("PK");

  // Delete everything.
  await page.getByRole("link", { name: "← Recordings" }).click();
  page.once("dialog", (d) => void d.accept());
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByTestId("recent")).toHaveCount(0);
  expect(errors).toEqual([]);
});
