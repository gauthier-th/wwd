// The device test page (PLAN.md M4): benchmark, then a short field test that uploads its log in
// the background, against a real API on a tiny base model.

import { fileURLToPath } from "node:url";

import { expect, test, type APIRequestContext } from "@playwright/test";

const fixture = (name: string) => fileURLToPath(new URL(`../.e2e/fixtures/${name}`, import.meta.url));

async function trainedModel(request: APIRequestContext) {
  const ww = await (await request.post("/api/wakewords", { data: { name: "abad", language: "fr" } })).json();
  const fs = await import("node:fs");
  await request.post(`/api/wakewords/${ww.id}/samples`, {
    multipart: { kind: "positive", file: { name: "take.wav", mimeType: "audio/wav", buffer: fs.readFileSync(fixture("take.wav")) } },
  });
  const training = await (await request.post(`/api/wakewords/${ww.id}/trainings`, { data: { target_fa_per_hour: 1 } })).json();
  await expect.poll(async () => (await (await request.get(`/api/trainings/${training.id}`)).json()).status,
    { timeout: 120_000 }).toBe("done");
  return { wid: ww.id as string, tid: training.id as string };
}

test("benchmark and field test on this device", async ({ page, request }) => {
  test.setTimeout(240_000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const { wid, tid } = await trainedModel(request);

  await page.goto(`/w/${wid}/t/${tid}`);
  await page.getByTestId("device-link").click();
  await expect(page).toHaveURL(new RegExp(`/w/${wid}/t/${tid}/device$`));
  await page.goto(`/w/${wid}/t/${tid}/device?upload_s=3`);

  await page.getByTestId("bench-run").click();
  await expect(page.getByTestId("bench-results").locator("tbody tr")).toHaveCount(4, { timeout: 120_000 });
  await expect(page.getByTestId("benchmark")).toContainText("Saved");

  await page.getByTestId("field-toggle").click();
  await expect(page.getByTestId("field-elapsed")).toHaveText(/0:00:0[2-9]/, { timeout: 30_000 });
  await expect(page.getByTestId("field-upload")).toContainText("Uploaded", { timeout: 15_000 }); // in the background
  await page.waitForTimeout(3000);
  await page.getByTestId("field-toggle").click();
  const summary = page.getByTestId("field-summary");
  await expect(summary).toBeVisible({ timeout: 30_000 });
  await expect(summary).toContainText("listened");

  const runs = await (await request.get(`/api/trainings/${tid}/device-runs`)).json();
  expect(runs.map((r: { kind: string }) => r.kind)).toEqual(["benchmark", "fieldtest"]);
  const field = runs[1];
  expect(field.finished).toBe(true);
  expect(field.summary.hours * 3600).toBeGreaterThan(4); // ticks were uploaded
  expect(field.summary.busy_pct.mean).toBeGreaterThan(0);
  await expect(page.getByTestId("past-runs").locator("li")).toHaveCount(2);
  expect(errors).toEqual([]);
  await request.delete(`/api/wakewords/${wid}`);
});
