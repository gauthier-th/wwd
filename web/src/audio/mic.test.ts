import { describe, expect, it } from "vitest";

import { browserOf, micErrorText } from "./mic";

describe("micErrorText", () => {
  it("explains capture failures and leaves other errors alone", () => {
    expect(micErrorText(new DOMException("denied", "NotAllowedError"))).toBe("Microphone access was refused.");
    expect(micErrorText(new DOMException("gone", "NotFoundError"))).toBe("No microphone found.");
    expect(micErrorText(new DOMException("busy", "NotReadableError"))).toMatch(/another app/);
    expect(micErrorText(new TypeError("fetch failed"))).toBeNull();
    expect(micErrorText(undefined)).toBeNull();
  });
});

describe("browserOf", () => {
  const ua = {
    chrome: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    android: "Mozilla/5.0 (Linux; Android 15; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Mobile Safari/537.36",
    firefox: "Mozilla/5.0 (X11; Linux x86_64; rv:144.0) Gecko/20100101 Firefox/144.0",
    safari: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/19.0 Safari/605.1.15",
    iphone: "Mozilla/5.0 (iPhone; CPU iPhone OS 19_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/141.0 Mobile/15E148 Safari/604.1",
  };

  it("picks the settings to describe", () => {
    expect(browserOf(ua.chrome)).toBe("chrome");
    expect(browserOf(ua.android)).toBe("chrome-android");
    expect(browserOf(ua.firefox)).toBe("firefox");
    expect(browserOf(ua.safari)).toBe("safari");
    expect(browserOf(ua.iphone)).toBe("ios"); // Chrome on iOS is Safari underneath
    expect(browserOf(ua.safari, true)).toBe("ios"); // an iPad asks for the desktop site
  });
});
