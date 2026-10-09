// The device test page (PLAN.md M4): benchmark, then a short field test that saves its log in
// the browser as it runs. The model is trained through the app first, as a user would.

import { fileURLToPath } from "node:url";

import { expect, test, type Page } from "@playwright/test";

const fixture = (name: string) => fileURLToPath(new URL(`../.e2e/fixtures/${name}`, import.meta.url));

async function trainedModel(page: Page): Promise<string> {
  await page.goto("/");
  await page.getByLabel("Wake word").fill("abad");
  await page.getByLabel("Language").selectOption("fr");
  await page.getByRole("button", { name: "Start recording" }).click();
  await page.getByTestId("import-positive").setInputFiles(fixture("take.wav"));
  await expect(page.getByTestId("clip")).toHaveCount(6);
  await page.getByTestId("train").click();
  await expect(page.getByTestId("results")).toBeVisible({ timeout: 240_000 });
  return page.url();
}

test("benchmark and field test on this device", async ({ page }) => {
  test.setTimeout(300_000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const trainingUrl = await trainedModel(page);

  await page.getByTestId("device-link").click();
  await expect(page).toHaveURL(/\/device$/);
  await page.goto(`${trainingUrl}/device?upload_s=3`);

  await page.getByTestId("bench-run").click();
  await expect(page.getByTestId("bench-results").locator("tbody tr")).toHaveCount(2, { timeout: 120_000 }); // fp32, gate on and off
  await expect(page.getByTestId("benchmark")).toContainText("Saved");

  await page.getByTestId("field-toggle").click();
  await expect(page.getByTestId("field-elapsed")).toHaveText(/0:00:0[2-9]/, { timeout: 30_000 });
  await expect(page.getByTestId("field-upload")).toContainText("Saved", { timeout: 15_000 }); // in the background
  await page.waitForTimeout(3000);
  await page.getByTestId("field-toggle").click();
  const summary = page.getByTestId("field-summary");
  await expect(summary).toBeVisible({ timeout: 30_000 });
  await expect(summary).toContainText("listened");

  // Both runs are kept with the model, in this browser.
  await page.reload();
  await expect(page.getByTestId("past-runs").locator("li")).toHaveCount(2);
  expect(errors).toEqual([]);
});
