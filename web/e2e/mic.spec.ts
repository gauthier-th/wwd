// The top bar's microphone dialog: permission state, device choice (kept across reloads) and
// the level test, which opens the chosen device the same way recordings do.

import { expect, test } from "@playwright/test";

test("microphone settings", async ({ page, browserName }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("./");
  await page.getByTestId("mic-button").click();
  const dialog = page.getByTestId("mic-dialog");
  await expect(dialog).toBeVisible();
  const state = dialog.getByTestId("mic-state");
  if (browserName === "chromium") await expect(state).toHaveAttribute("data-state", "granted");

  await dialog.getByTestId("mic-test").click();
  const hearing = dialog.getByTestId("mic-hearing");
  await expect(hearing.or(dialog.getByRole("alert"))).toBeVisible();
  if (browserName !== "webkit") await expect(hearing).toBeVisible(); // fake devices: always one
  if (browserName === "chromium") { // its fake microphone plays speech: the level must move
    await expect.poll(async () => Number(await dialog.getByTestId("mic-meter").getAttribute("data-peak")),
      { timeout: 10_000 }).toBeGreaterThan(0.01);
  }
  if (await hearing.isVisible()) {
    // a working capture shows it's allowed (Firefox's test profile grants per visit, not for good)
    await expect(state).toHaveAttribute("data-state", browserName === "firefox" ? /granted|session/ : "granted");
    const device = dialog.getByLabel("Device");
    const ids = await device.locator("option").evaluateAll((os) => os.map((o) => (o as HTMLOptionElement).value));
    const real = ids.find((v) => v !== "");
    expect(real).toBeTruthy(); // after access, the microphones are listed
    await device.selectOption(real!);
    await expect(hearing).toBeVisible(); // the test restarts on the new choice

    await page.reload();
    await page.getByTestId("mic-button").click();
    await expect(dialog.getByLabel("Device")).toHaveValue(real!);
    await dialog.getByLabel("Device").selectOption("");
    expect(await page.evaluate(() => localStorage.getItem("wwd.micDevice"))).toBeNull();
  }

  await dialog.getByRole("button", { name: "Close" }).click();
  await expect(dialog).toBeHidden();
  expect(errors).toEqual([]);
});
